#include "RecoLuminosity/ROOTSchema/interface/ROOTFileMerger.h"

#include <iostream>
#include <iomanip>
#include <cstdio>

#include <TFile.h>
#include <TChain.h>

HCAL_HLX::ROOTFileMerger::ROOTFileMerger(){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    minSectionNumber = 99999999;

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

HCAL_HLX::ROOTFileMerger::~ROOTFileMerger(){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::ROOTFileMerger::Merge(const unsigned int runNumber, bool bCMSLive){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
    // TChain::Merge and TTree::CloneTree leak because we used TTree::Bronch to create the tree.

    HCAL_HLX::LUMI_SECTION lumiSection;

    ROOTFileReader RFR;
    RFR.ReplaceFile(CreateInputFileName(runNumber));
    
    SetFileName(CreateRunFileName(runNumber, 0));
    CreateTree(lumiSection);
    
    int nentries = RFR.GetNumEntries();
  
    for(int i = 0; i < nentries; i++){
      RFR.GetEntry(i);
      RFR.GetLumiSection(lumiSection);
      if( minSectionNumber > lumiSection.hdr.sectionNumber ){
	std::cout << minSectionNumber << ":" << lumiSection.hdr.sectionNumber << std::endl;
	minSectionNumber = lumiSection.hdr.sectionNumber;
      }    

      // Must fill Threshold eventually  right now it contains fake data.
      FillTree(lumiSection);
    }
    
    CloseTree();

    // Rename file so that it includes the minimum lumi section number.
    rename( CreateRunFileName(runNumber, 0).c_str(), CreateRunFileName(runNumber, minSectionNumber).c_str() ); 

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

std::string HCAL_HLX::ROOTFileMerger::CreateInputFileName(const unsigned int runNumber){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    std::ostringstream outputString;
    
    outputString << outputDir_
		 << "/"
		 << TimeStampYYYYMM()
		 << "/"
		 << std::setfill('0') << std::setw(9) << runNumber
		 << "/"
		 << outputFilePrefix_
		 << "_"
		 << TimeStampYYYYMMDD()
		 << "_"
		 << std::setfill('0') << std::setw(9) << runNumber
		 << "_*.root";

#ifdef DEBUG
    std::cout << "Input file name: " << outputString.str() << std::endl;
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
    return outputString.str();
}

