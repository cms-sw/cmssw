#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "TFile.h"
#include "TROOT.h"

namespace fwlite {

TFileService::TFileService(const std::string& fileName) :
  TFileDirectory("", "", TFile::Open(fileName.c_str() , "RECREATE"), ""),
  file_(TFileDirectory::file_),
  fileName_(fileName)
{
}


TFileService::TFileService(TFile * aFile) :
  TFileDirectory("", "", aFile, ""),
  file_(TFileDirectory::file_),
  fileName_(aFile->GetName())
{
}

TFileService::~TFileService() {
  file_->Write();
  file_->Close();
  delete file_;
}

}
