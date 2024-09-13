#include "FWCore/Utilities/interface/Exception.h"

#include "BareRootProductGetter.h"

#include "TROOT.h"
#include "TFile.h"

TFile* BareRootProductGetter::currentFile() const {
  TFile* file = dynamic_cast<TFile*>(gROOT->GetListOfFiles()->Last());
  if (nullptr == file) {
    throw cms::Exception("FileNotFound") << "unable to find the TFile '" << gROOT->GetListOfFiles()->Last() << "'\n"
                                         << "retrieved by calling 'gROOT->GetListOfFiles()->Last()'\n"
                                         << "Please check the list of files.";
  }
  return file;
}
