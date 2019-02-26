#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "RecoLuminosity/LumiProducer/interface/LumiRawDataStructures.h"
#include "RecoLuminosity/LumiProducer/interface/LumiCorrector.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <map>
//#include <cmath>
#include <algorithm>
#include <fstream>

/** This programm scans a given lumi raw data file and print out the content
 **/
struct beaminfo{
  unsigned int bxidx;
  unsigned int timestamp;
  float lumival;
  float beam1_intensity;
  float beam2_intensity;
};

int main(int argc, char** argv){
  const char* filename="file:test.root";
  //default file to read. file name is taken from command argument
  if(argc>1){
    filename=argv[1];
  }
  //TFile *myfile=new TFile(filename,"READ");
  TFile * myfile=TFile::Open(filename);
  
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
  //std::cout<<"hlxentries "<<hlxentries<<std::endl;
  std::map< unsigned int,std::vector<beaminfo> > bxlumis;
  std::vector<beaminfo> tmpbx;
  unsigned int ncollidingbx=0;
  for(size_t i=0;i<hlxentries;++i){
    hlxTree->GetEntry(i);
    ncollidingbx=myLumiHeader->numBunches;
    //std::cout<<"Lumi summary for run : "<<myLumiHeader->runNumber<<" : LS : "<<myLumiHeader->sectionNumber<<" "<<myLumiHeader->timestamp<<" "<<myLumiHeader->numBunches<<std::endl;
    
    //std::cout<<std::setw(20)<<"lumi details : "<<std::endl;
    unsigned int hlxls=myLumiHeader->sectionNumber;
    unsigned int ts=myLumiHeader->timestamp;
    tmpbx.clear();
    for(size_t j=0;j<3564;++j){
      //std::cout<<std::setw(20)<<"    BX : "<<j<<" : OccLumi : "<<myLumiDetail->OccLumi[0][j]<<std::endl;
      beaminfo b;
      b.timestamp=ts;
      b.bxidx=j;
      b.lumival=myLumiDetail->OccLumi[0][j];
      b.beam1_intensity=0.;
      b.beam2_intensity=0.;
      tmpbx.push_back(b);
    }
    bxlumis.insert(std::make_pair(hlxls,tmpbx));
  }

  LumiCorrector corr;
  std::map< unsigned int,std::vector<beaminfo> >::iterator mapIt;
  std::map< unsigned int,std::vector<beaminfo> >::iterator itBeg=bxlumis.begin();
  std::map< unsigned int,std::vector<beaminfo> >::iterator itEnd=bxlumis.end();
  
  for(mapIt=itBeg;mapIt!=itEnd;++mapIt){
    float totlumi=0.;
    std::vector<beaminfo>::iterator thislslumisBeg=mapIt->second.begin();
    std::vector<beaminfo>::iterator thislslumisEnd=mapIt->second.end();
    for(std::vector<beaminfo>::iterator it=thislslumisBeg;it!=thislslumisEnd;++it){
      totlumi+=it->lumival;
    }
    for(std::vector<beaminfo>::iterator it=thislslumisBeg;it!=thislslumisEnd;++it){
      float thecorrector=corr.TotalNormOcc1(totlumi*1.0e-3,ncollidingbx);	
      float correctedbxlumi=thecorrector*(it->lumival);
      it->lumival=correctedbxlumi;
    } 
  }
  //lsnum++;
  //std::cout<<std::setw(20)<<"#LS "<<lsnum<<" timestamp "<<thislstimestamp<<std::endl;
   
  TTree *diptree=(TTree*)myfile->Get("DIPCombined");
  
  if(diptree){
    std::unique_ptr<HCAL_HLX::DIP_COMBINED_DATA> dipdata(new HCAL_HLX::DIP_COMBINED_DATA);
    diptree->SetBranchAddress("DIPCombined.",&dipdata);
    size_t ndipentries=diptree->GetEntries();
    unsigned int dipls=0;
    if(ndipentries>0){
      for(size_t i=0;i<1;++i){
	diptree->GetEntry(i);
	//unsigned int fillnumber=dipdata->FillNumber;
	dipls=dipdata->sectionNumber;
	std::map< unsigned int,std::vector<beaminfo> >::iterator dipIt=bxlumis.end();
	if(bxlumis.find(dipls)!=dipIt){
	  dipIt=bxlumis.find(dipls);
	}
	for(unsigned int i=0;i<3564;++i){
	  float beam1in=dipdata->Beam[0].averageBunchIntensities[i];
	  float beam2in=dipdata->Beam[1].averageBunchIntensities[i];
	  if(dipIt!=bxlumis.end()){
	    dipIt->second[i].beam1_intensity=beam1in;
	    dipIt->second[i].beam2_intensity=beam2in;
	  }
	}
      }
    }
  }
  
  std::ofstream outfile;
  outfile.open ("out.txt");
  for(mapIt=itBeg;mapIt!=itEnd;++mapIt){
    outfile <<"# "<<mapIt->first<<std::endl;
    std::vector<beaminfo>::iterator thislslumisBeg=mapIt->second.begin();
    std::vector<beaminfo>::iterator thislslumisEnd=mapIt->second.end();
    for(std::vector<beaminfo>::iterator it=thislslumisBeg;it!=thislslumisEnd;++it){
      outfile <<it->timestamp<<","<<it->bxidx<<","<<it->lumival<<","<<it->beam1_intensity<<","<<it->beam2_intensity<<std::endl;
    }
  }
  outfile.close();
}
//TFile * myfile=TFile::Open("rfio:/castor/cern.ch/cms/store/lumi/200912/CMS_LUMI_RAW_20091212_000124025_0001_1.root");
  //HCAL_HLX::RUN_SUMMARY *myRunSummary = new HCAL_HLX::RUN_SUMMARY;
  //TTree *runsummaryTree = (TTree *) myfile->Get("RunSummary");
  //if(!runsummaryTree) std::cout<<"no run summary data"<<std::endl;
  //runsummaryTree->SetBranchAddress("RunSummary.",&myRunSummary);
  //size_t runsummaryentries=runsummaryTree->GetEntries();
  //std::cout<<"n run summary entries "<<runsummaryentries<<std::endl;
  //for(size_t i=0;i<runsummaryentries;++i){
  //runsummaryTree->GetEntry(i);
  // std::cout<<"Summary for run : "<<myRunSummary->runNumber<<std::endl;
  // std::cout<<std::setw(20)<<"timestamp : "<<myRunSummary->timestamp<<" : timestamp micros : "<<myRunSummary->timestamp_micros<<" : start orbit : "<<myRunSummary->startOrbitNumber<<" : end orbit : "<<myRunSummary->endOrbitnumber<<" : fill number : "<<myRunSummary->fillNumber<<" : number CMS LS : "<<myRunSummary->numberCMSLumiSections<<" : number DAQ LS : "<<myRunSummary->numberLumiDAQLumiSections<<std::endl;
  //}
  /**
  HCAL_HLX::LEVEL1_TRIGGER *myTRG = new HCAL_HLX::LEVEL1_TRIGGER;
  TTree *trgTree = (TTree *) myfile->Get("L1Trigger");
  if(!trgTree) std::cout<<"no trg data"<<std::endl;
  trgTree->SetBranchAddress("L1Trigger.",&myTRG);
  size_t trgentries=trgTree->GetEntries();
  for(size_t i=0;i<trgentries;++i){
    trgTree->GetEntry(i);
    //std::cout<<"trg runnumber "<<myTRG->runNumber<<std::endl;
    std::cout<<"TRG for run : "<< myTRG->runNumber<<" : LS : "<<myTRG->sectionNumber<<" : deadtime : "<< myTRG->deadtiecount<<std::endl;
    for( unsigned int j=0; j<128; ++j){
      std::cout<<std::setw(20)<<"GT Algo  "<<j;
      std::cout<<" : path : "<< myTRG->GTAlgo[j].pathName<<" : counts : "<< myTRG->GTAlgo[j].counts<<" : prescale : "<< myTRG->GTAlgo[j].prescale<<std::endl;
    }
    for( unsigned int k=0; k<64; ++k){
      std::cout<<std::setw(20)<<"GT Tech : "<<k;
      std::cout<<" : path : "<< myTRG->GTTech[k].pathName<<" : counts : "<< myTRG->GTTech[k].counts<<" : prescale : "<< myTRG->GTTech[k].prescale<<std::endl;
    }
  }
  **/
  /**
  HCAL_HLX::HLTRIGGER *myHLT = new HCAL_HLX::HLTRIGGER;
  TTree *hltTree = (TTree *) myfile->Get("HLTrigger");
  if(!hltTree) std::cout<<"no hlt data"<<std::endl;
  hltTree->SetBranchAddress("HLTrigger.",&myHLT);
  size_t hltentries=hltTree->GetEntries();
  for(size_t i=0;i<hltentries;++i){
    hltTree->GetEntry(i);
    std::cout<<"HLT for run : "<< myHLT->runNumber<<":  LS  : "<<myHLT->sectionNumber<<" : total hlt paths : "<<myHLT->numPaths<<std::endl;
    for( unsigned int j=0; j<myHLT->numPaths; ++j){
      std::cout<<std::setw(20)<<"HLTConfigId : "<<myHLT->HLTPaths[j].HLTConfigId<<"path : "<<myHLT->HLTPaths[j].PathName<<" : L1Pass : "<<myHLT->HLTPaths[j].L1Pass<<" : PSPass : "<<myHLT->HLTPaths[j].PSPass<<" : PAccept : "<<myHLT->HLTPaths[j].PAccept<<" : PExcept : "<<myHLT->HLTPaths[j].PExcept<<" : PReject : "<<myHLT->HLTPaths[j].PReject<<" : PSIndex : "<<myHLT->HLTPaths[j].PSIndex<<" : Prescale : "<<myHLT->HLTPaths[j].Prescale<<std::endl;
    }
  }
  **/
/**
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
    std::cout<<"Lumi summary for run : "<<myLumiHeader->runNumber<<" : LS : "<<myLumiHeader->sectionNumber<<" cmsalive value: "<<myLumiHeader->bCMSLive<<std::endl;
    bool a=true;
    if(typeid(myLumiHeader->bCMSLive)==typeid(a)){
      std::cout<<"is bool type"<<std::endl;
      std::cout<<"normal bool "<<a<<std::endl;
      std::cout<<"cms alive bool "<< myLumiHeader->bCMSLive<< std::endl;
    }else{
      std::cout<<"not bool type"<<std::endl;
      std::cout<<"cms alive"<< myLumiHeader->bCMSLive<< std::endl;
    }
    //std::cout<<std::setw(20)<<"deadtime norm : "<<myLumiSummary->DeadTimeNormalization<<" : LHC norm : "<<myLumiSummary->LHCNormalization<<" : instantlumi : "<<myLumiSummary->InstantLumi<<" : instantlumiErr : "<<myLumiSummary->InstantLumiErr<<" : instantlumiQlty : "<<myLumiSummary->InstantLumiQlty<<std::endl;
    //std::cout<<std::setw(20)<<"lumi details : "<<std::endl;
    //for(size_t j=0;j<HCAL_HLX_MAX_BUNCHES;++j){
    //  std::cout<<std::setw(20)<<"    LHCLumi : "<<myLumiDetail->LHCLumi[j]<<" : ETLumi : "<<myLumiDetail->ETLumi[j]<<" : ETLumiErr : "<<myLumiDetail->ETLumiErr[j]<<" : ETLumiQlty : "<<myLumiDetail->ETLumiQlty[j]<<" : ETBXNormalization : "<<myLumiDetail->ETBXNormalization[j]<<std::endl;
    //}
  }
  **/
//}

