#ifndef MuScleFitTreeProvenance_cc
#define MuScleFitTreeProvenance_cc

#include <iostream>

#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitProvenance.h"

int main(int argc, char* argv[]) 
{
  if( argc != 2 ) {
    std::cout << "Please provide the name of the file with file: or rfio: as needed" << std::endl;
    return 1;
  }
  std::string fileName(argv[1]);
  if( fileName.find("file:") != 0 && fileName.find("rfio:") != 0 ) {
    std::cout << "Please provide the name of the file with file: or rfio: as needed" << std::endl;
    return 1;
  }

  std::cout << "Reading provenance information from the tree:" << std::endl;

  // open input file (can be located on castor)
  TFile* inFile = TFile::Open(fileName.c_str());
  
  MuScleFitProvenance * provenance = (MuScleFitProvenance*)(inFile->Get("MuScleFitProvenance"));
  std::cout << "MuonType = " << provenance->muonType << std::endl;

  return 0;
}

#endif
