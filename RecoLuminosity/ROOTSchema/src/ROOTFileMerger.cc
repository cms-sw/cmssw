#include "RecoLuminosity/ROOTSchema/interface/ROOTFileMerger.h"

#include <iostream>
#include <iomanip>

#include <TFile.h>
#include <TChain.h>

HCAL_HLX::ROOTFileMerger::ROOTFileMerger(){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

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

    //MergeMethodOne(runNumber, bCMSLive);
    //MergeMethodTwo(runNumber, bCMSLive);
    MergeMethodThree(runNumber, bCMSLive);

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::ROOTFileMerger::MergeMethodOne(const unsigned int runNumber, const  bool bCMSLive){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
    
    TChain *Temp = new TChain("LumiTree");

    Temp->Add(CreateInputFileName(runNumber).c_str());
    Temp->Merge(CreateRunFileName(runNumber, bCMSLive).c_str()); // causes leaks
    // source of leak stems from CloneTree

    delete Temp;


#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::ROOTFileMerger::MergeMethodTwo(const unsigned int runNumber, const bool bCMSLive){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
    
    TChain Temp("LumiTree");
    Temp.Add(CreateInputFileName(runNumber).c_str());
    
    TFile* OutputFile = new TFile(CreateRunFileName(runNumber, bCMSLive).c_str(),"RECREATE");
    TTree* OutputTree = Temp.CloneTree();  // causes leak
    // no better than Method One


    OutputFile->Write();
    OutputFile->Close();
    delete OutputFile;

#ifdef DEBUG
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::ROOTFileMerger::MergeMethodThree(const unsigned int runNumber, bool bCMSLive){
#ifdef DEBUG
    std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

    // "Manually fill the tree

    HCAL_HLX::LUMI_SECTION lumiSection;

    ROOTFileReader RFR;
    RFR.ReplaceFile(CreateInputFileName(runNumber));
    
    
    SetFileName(CreateRunFileName(runNumber, bCMSLive));
    CreateTree(lumiSection);
    
    int nentries = RFR.GetNumEntries();
  
    for(int i = 0; i < nentries; i++){
      RFR.GetEntry(i);
      RFR.GetLumiSection(lumiSection);
    
      FillTree(lumiSection);
    }
    
    CloseTree();

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
		 << std::setfill('0') << std::setw(9) << runNumber
		 << "/"
		 << outputFilePrefix_
		 << "_"
		 << TimeStampShort()
		 << "_"
		 << std::setfill('0') << std::setw(9) << runNumber
		 << "_*.root";

#ifdef DEBUG
    std::cout << "Input file name: " << outputString.str() << std::endl;
    std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
    return outputString.str();
}

