#include "TFile.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TPluginManager.h>
#include <TEnv.h>

#include <string>
#include <iostream>

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"

int main(int argc, char* argv[]) {
  /* TestMain tfileTest zip-member:file:bha.zip#bha
     TestMain tfileTest rfio:suncmsc.cern.ch:/data/valid/test/vincenzo/testPool/evData/EVD0_EventData.56709894d26a11d78dd20040f45cca94.1.h300eemm.TestSignalHits
     TestMain tfileTest zip-member:rfio:suncmsc.cern.ch:/data/valid/test/vincenzo/testZip/test1.zip#file.5 */

  char const* protocols[] = {"^file:", "^http:", "^ftp:", "^web:", "^gsiftp:", "^sfn:", "^rfio:", "^dcache:", "^dcap:", "^gsidcap:"};

  char const* tStorageFactoryFileFunc = "TStorageFactoryFile(char const*, Option_t*, char const*, Int_t)"; 

  edmplugin::PluginManager::configure(edmplugin::standard::config());

  gEnv->SetValue("Root.Stacktrace", "0");
  // set our own root plugin
  typedef char const* Cp;
  Cp* begin = protocols;
  Cp* end = &protocols[sizeof(protocols)/sizeof(protocols[0])];
  for (Cp* i = begin; i != end; ++i) {
    gROOT->GetPluginManager()->AddHandler("TFile", *i, "TStorageFactoryFile", "pluginIOPoolTFileAdaptor", tStorageFactoryFileFunc);   
    gROOT->GetPluginManager()->AddHandler("TSystem", *i, "TStorageFactorySystem", "pluginIOPoolTFileAdaptor", "TStorageFactorySystem()");
  }

  gROOT->GetPluginManager()->Print(); // use option="a" to see ctors 
 
  std::string fname("file:bha.root");

  if (argc > 1) fname = argv[1];

  {
    std::auto_ptr<TFile> g(TFile::Open(fname.c_str(), "recreate", "", 1));
    g->Close();
  }
  try {
    Bool_t result = gSystem->AccessPathName(fname.c_str(), kFileExists);
    std::cout << "file " << fname << (result ? " does not exist\n" : " exists\n");
    char const* err = gSystem->GetErrorStr();
    if (err != 0 && *err != '\0')
      std::cout << "error was " << err << "\n";

    if (!result) {
      std::auto_ptr<TFile> f(TFile::Open(fname.c_str()));
      std::cout << "file size " << f->GetSize() << std::endl;
      f->ls();
    }
  }
  catch (cms::Exception& e) {
    std::cout << "*ERROR*: " << e.what() << std::endl;
  }
  catch (...) {
    std::cout << "*ERROR*\n";
  }

  return 0;
}
