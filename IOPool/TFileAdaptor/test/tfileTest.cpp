#include "TFile.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TPluginManager.h>
#include <TEnv.h>

#include <array>
#include <iostream>
#include <string>
#include <string_view>

#include "boost/system/system_error.hpp"
#include "boost/filesystem/operations.hpp"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"

int main(int argc, char* argv[]) {
  // This list should replicate the addHandler calls in TFileAdaptor
  std::array<char const*, 10> const protocols = {
      {"^file:", "^http:", "^http[s]?:", "^ftp:", "^web:", "^dcache:", "^dcap:", "^gsidcap:", "^root:", "^[x]?root:"}};
  std::array<char const*, 10> const uris{{"file:foo",
                                          "http://foo",
                                          "https://foo",
                                          "ftp://foo",
                                          "web://foo",
                                          "dcache://foo",
                                          "dcap://foo",
                                          "gsidcap://foo",
                                          "root://foo",
                                          "xroot://foo"}};

  char const* tStorageFactoryFileFunc = "TStorageFactoryFile(char const*, Option_t*, char const*, Int_t)";
  char const* tStorageFactoryFileFuncNet =
      "TStorageFactoryFile(char const*, Option_t*, char const*, Int_t, Int_t, Bool_t)";

  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  } catch (cms::Exception const& e) {
    std::cout << e.explainSelf() << std::endl;
    return 1;
  } catch (boost::system::system_error const& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  gEnv->SetValue("Root.Stacktrace", "0");

  // Make sure ROOT parses system directories first.
  // Then our AddHandler() calls will also remove an existing handler
  // that was registered with the same regex
  gROOT->GetPluginManager()->LoadHandlersFromPluginDirs("TFile");
  gROOT->GetPluginManager()->LoadHandlersFromPluginDirs("TSystem");

  // set our own root plugin
  for (char const* p : protocols) {
    if (std::string_view(p).find("root:") != std::string_view::npos) {
      // xrootd arguments
      gROOT->GetPluginManager()->AddHandler(
          "TFile", p, "TStorageFactoryFile", "pluginIOPoolTFileAdaptor", tStorageFactoryFileFuncNet);
      gROOT->GetPluginManager()->AddHandler("TSystem",
                                            p,
                                            "TStorageFactorySystem",
                                            "pluginIOPoolTFileAdaptor",
                                            "TStorageFactorySystem(const char*, Bool_t)");
    } else {
      // regular case
      gROOT->GetPluginManager()->AddHandler(
          "TFile", p, "TStorageFactoryFile", "pluginIOPoolTFileAdaptor", tStorageFactoryFileFunc);
      gROOT->GetPluginManager()->AddHandler(
          "TSystem", p, "TStorageFactorySystem", "pluginIOPoolTFileAdaptor", "TStorageFactorySystem()");
    }
  }
  gROOT->GetPluginManager()->Print();  // use option="a" to see ctors

  for (char const* u : uris) {
    using namespace std::literals;
    TPluginHandler const* handler = gROOT->GetPluginManager()->FindHandler("TFile", u);
    if (not handler) {
      std::cout << "No TFile handler for " << u << std::endl;
      return 1;
    }
    if (handler->GetClass() != "TStorageFactoryFile"sv) {
      std::cout << "TFile handler for " << u << " was unexpected " << handler->GetClass() << std::endl;
      return 1;
    }

    handler = gROOT->GetPluginManager()->FindHandler("TSystem", u);
    if (not handler) {
      std::cout << "No TSystem handler for " << u << std::endl;
      return 1;
    }
    if (handler->GetClass() != "TStorageFactorySystem"sv) {
      std::cout << "TSystem handler for " << u << " was unexpected " << handler->GetClass() << std::endl;
      return 1;
    }
  }

  std::string fname("file:bha.root");

  {
    std::unique_ptr<TFile> g(TFile::Open(fname.c_str(), "recreate", "", 1));
    g->Close();
  }
  try {
    Bool_t result = gSystem->AccessPathName(fname.c_str(), kFileExists);
    std::cout << "file " << fname << (result ? " does not exist\n" : " exists\n");
    if (result) {
      char const* err = gSystem->GetErrorStr();
      if (err != 0 && *err != '\0')
        std::cout << "error was " << err << "\n";
      return 1;
    }

    std::unique_ptr<TFile> f(TFile::Open(fname.c_str()));
    std::cout << "file size " << f->GetSize() << std::endl;
    f->ls();
  } catch (cms::Exception& e) {
    std::cout << "*ERROR*: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cout << "*ERROR*\n";
    return 1;
  }

  return 0;
}
