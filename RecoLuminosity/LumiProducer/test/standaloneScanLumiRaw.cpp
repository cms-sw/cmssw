#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "LumiRawDataStructures.h"
#include <iostream>
int main(int argc, char** argv){
  const char* filename="test.root";
  //default file to read. file name is taken from command argument
  if(argc>1){
    filename=argv[1];
  }
  TFile *myfile=new TFile("test.root","READ");

  HCAL_HLX::RUN_SUMMARY *myRunSummary = new HCAL_HLX::RUN_SUMMARY;
  TTree *runsummaryTree = (TTree *) myfile->Get("RunSummary");
  if(!runsummaryTree) std::cout<<"no run summary data"<<std::endl;
  runsummaryTree->SetBranchAddress("RunSummary.",&myRunSummary);
  if(!runsummaryTree)  std::cout<<"no run summary data"<<std::endl;
  size_t runsummaryentries=runsummaryTree->GetEntries();
  for(size_t i=0;i<runsummaryentries;++i){
    runsummaryTree->GetEntry(i);
    std::cout<<"run summary runnumber"<<myRunSummary->runNumber<<std::endl;
    std::cout<<"run summary timestamp"<<myRunSummary->timestamp<<std::endl;
  }

  HCAL_HLX::LEVEL1_TRIGGER *myTRG = new HCAL_HLX::LEVEL1_TRIGGER;
  TTree *trgTree = (TTree *) myfile->Get("L1Trigger");
  if(!trgTree) std::cout<<"no trg data"<<std::endl;
  trgTree->SetBranchAddress("L1Trigger.",&myTRG);
  size_t trgentries=trgTree->GetEntries();
  for(size_t i=0;i<trgentries;++i){
    trgTree->GetEntry(i);
    std::cout<<"trg runnumber "<<myTRG->runNumber<<std::endl;
    std::cout<<"trg lsnumber "<<myTRG->sectionNumber<<std::endl;
  }

  HCAL_HLX::HLTRIGGER *myHLT = new HCAL_HLX::HLTRIGGER;
  TTree *hltTree = (TTree *) myfile->Get("HLTrigger");
  if(!hltTree) std::cout<<"no hlt data"<<std::endl;
  hltTree->SetBranchAddress("HLTrigger.",&myHLT);
  size_t hltentries=hltTree->GetEntries();
  for(size_t i=0;i<hltentries;++i){
    hltTree->GetEntry(i);
    std::cout<<"hlt runnumber "<<myHLT->runNumber<<std::endl;
    std::cout<<"hlt lsnumber "<<myHLT->sectionNumber<<std::endl;
  }
  HCAL_HLX::LUMI_SECTION *myLumiSection=new HCAL_HLX::LUMI_SECTION;
  HCAL_HLX::LUMI_SECTION_HEADER *myLumiHeader = &(myLumiSection->hdr);
  HCAL_HLX::LUMI_SUMMARY *myLumiSummary = &(myLumiSection->lumiSummary);
  HCAL_HLX::LUMI_DETAIL *myLumiDetail = &(myLumiSection->lumiDetail);
  
  TTree *hlxTree = (TTree *) myfile->Get("HLXData");
  if(!hlxTree) std::cout<<"no hlx data"<<std::endl;
  hlxTree->SetBranchAddress("Header.",&myLumiHeader);
  hlxTree->SetBranchAddress("Summary.",&myLumiSummary);
  hlxTree->SetBranchAddress("Detail.",&myLumiDetail);
  size_t hlxentries=hlxTree->GetEntries();
  std::cout<<"hlxentries "<<hlxentries<<std::endl;
  for(size_t i=0;i<hlxentries;++i){

    hlxTree->GetEntry(i);
    std::cout<<"lumi header runnumber "<<myLumiHeader->runNumber<<std::endl;
    std::cout<<"lumi header lsnumber "<<myLumiHeader->sectionNumber<<std::endl;
    std::cout<<"lumi summary instantlumi "<<myLumiSummary->InstantLumi<<std::endl;
    std::cout<<"lumi summary instantlumiErr "<<myLumiSummary->InstantLumiErr<<std::endl;
    std::cout<<"lumi detail LHCLumi[0] "<<myLumiDetail->LHCLumi[0]<<std::endl;
  }
}

