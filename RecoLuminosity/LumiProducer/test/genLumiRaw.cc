#ifndef RecoLuminosity_LumiProducer_genLumiRaw_h
#define RecoLuminosity_LumiProducer_genLumiRaw_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "LumiRawDataStructures.h"
#include "TFile.h"
#include "TTree.h"
//#include "TChain.h"
#include "TBranch.h"
#include <iostream>
#include <cstring>
/**
   this program is to generate fake lumi raw data samples with desired run/lumi section parameters controlled by EmptySource parameters.
   one job can generate data for at most 1 run and unlimited number of LS
**/

class genLumiRaw : public edm::EDAnalyzer{
public: 
  explicit genLumiRaw(edm::ParameterSet const&);
  virtual ~genLumiRaw();

private:  
  virtual void beginJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, 
				  edm::EventSetup const& c);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void endJob();
  
  void generateRunSummary(unsigned int runnumber,unsigned int totalCMSls);
  void generateHLT(unsigned int runnumber,unsigned int lsnumber);
  void generateTRG(unsigned int runnumber,unsigned int lsnumber);
  void generateHLX(unsigned int runnumber,unsigned int lsnumber);
private:

  static const std::string s_filetype;
  static const std::string s_fileprefix;
  
  static const std::string s_runsummaryTree;
  static const std::string s_runsummaryBranch;
  static const std::string s_runsummaryName;  
  
  static const std::string s_hltTree;
  static const std::string s_hltBranch;
  static const std::string s_hltName;
  
  static const std::string s_trgTree;
  static const std::string s_trgBranch;
  static const std::string s_trgName;
  
  static const std::string s_hlxTree;
  static const std::string s_lumiHeaderBranch;  
  static const std::string s_lumiHeaderName;
  static const std::string s_lumiSummaryBranch;  
  static const std::string s_lumiSummaryName;
  static const std::string s_lumiDetailBranch;  
  static const std::string s_lumiDetailName;  

  unsigned int m_run;
  unsigned int m_firstls;
  unsigned int m_nls;
  int m_bsize;
  int m_splitlevel;
  TFile* m_file;
  //run summary tree&data structures
  TTree* m_runsummaryTree; 
  HCAL_HLX::RUN_SUMMARY* m_runsummary;
  
  //hlx tree& data structures
  TTree* m_hlxTree;
  HCAL_HLX::LUMI_SECTION_HEADER* m_lumiheader; 
  HCAL_HLX::LUMI_SUMMARY* m_lumisummary;
  HCAL_HLX::LUMI_DETAIL*  m_lumidetail;

  //trg tree & data structures
  TTree* m_trgTree;
  HCAL_HLX::LEVEL1_TRIGGER* m_trg;

  //hlt data structures
  TTree* m_hltTree;
  HCAL_HLX::HLTRIGGER* m_hlt;


};//end class

const std::string genLumiRaw::s_filetype="LUMI";
const std::string genLumiRaw::s_fileprefix="CMS_LUMI_";
  
const std::string genLumiRaw::s_runsummaryTree="RunSummary";
const std::string genLumiRaw::s_runsummaryBranch="RunSummary.";
const std::string genLumiRaw::s_runsummaryName="HCAL_HLX::RUN_SUMMARY";  
  
const std::string genLumiRaw::s_hltTree="HLTrigger";
const std::string genLumiRaw::s_hltBranch="HLTrigger.";
const std::string genLumiRaw::s_hltName="HCAL_HLX::HLTRIGGER";
  
const std::string genLumiRaw::s_trgTree="L1Trigger";
const std::string genLumiRaw::s_trgBranch="L1Trigger.";
const std::string genLumiRaw::s_trgName="HCAL_HLX::LEVEL1_TRIGGER";
  
const std::string genLumiRaw::s_hlxTree="HLXData";
const std::string genLumiRaw::s_lumiHeaderBranch="Header.";  
const std::string genLumiRaw::s_lumiHeaderName="HCAL_HLX::LUMI_SECTION_HEADER";
const std::string genLumiRaw::s_lumiSummaryBranch="Summary.";  
const std::string genLumiRaw::s_lumiSummaryName="HCAL_HLX::LUMI_SUMMARY";
const std::string genLumiRaw::s_lumiDetailBranch="Detail.";  
const std::string genLumiRaw::s_lumiDetailName="HCAL_HLX::LUMI_DETAIL";

// -----------------------------------------------------------------

genLumiRaw::genLumiRaw(edm::ParameterSet const& iConfig):m_bsize(64000),m_splitlevel(2),m_runsummary(new HCAL_HLX::RUN_SUMMARY),m_lumiheader(new HCAL_HLX::LUMI_SECTION_HEADER),m_lumisummary(new HCAL_HLX::LUMI_SUMMARY),m_lumidetail(new HCAL_HLX::LUMI_DETAIL),m_trg(new HCAL_HLX::LEVEL1_TRIGGER),m_hlt(new HCAL_HLX::HLTRIGGER)
{  
}

// -----------------------------------------------------------------

genLumiRaw::~genLumiRaw(){
  delete m_runsummary;
  delete m_lumiheader;
  delete m_lumisummary;
  delete m_lumidetail;
  delete m_trg;
  delete m_hlt;
}

// -----------------------------------------------------------------

void genLumiRaw::analyze(edm::Event const& e,edm::EventSetup const&){
  //std::cout<<"testEvtLoop::analyze"<<std::endl;
}

// -----------------------------------------------------------------
void genLumiRaw::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, 
				    edm::EventSetup const& c){
  std::cout<<"I'm in run "<<lumiBlock.run()<<" lumi block "<<lumiBlock.id().luminosityBlock()<<std::endl;
  generateHLT(lumiBlock.run(),lumiBlock.id().luminosityBlock());
  generateTRG(lumiBlock.run(),lumiBlock.id().luminosityBlock());
  generateHLX(lumiBlock.run(),lumiBlock.id().luminosityBlock());
}

// -----------------------------------------------------------------

void genLumiRaw::beginJob(){
  //
  //prepare file name, open file,  book root trees
  //
  const std::string filename="test.root";
  m_file=new TFile(filename.c_str(),"RECREATE");

  //book run summary tree
  m_runsummaryTree = new TTree(s_runsummaryTree.c_str(),s_runsummaryTree.c_str());
  m_runsummaryTree->Branch(s_runsummaryBranch.c_str(),s_runsummaryName.c_str(),&m_runsummary,m_bsize,m_splitlevel);
  //runsummaryBranch->SetAddress(m_runsummary);

  //book hlx tree
  m_hlxTree = new TTree(s_hlxTree.c_str(),s_hlxTree.c_str());
  m_hlxTree->Branch(s_lumiHeaderBranch.c_str(),s_lumiHeaderName.c_str(),&m_lumiheader,m_bsize,m_splitlevel);
  m_hlxTree->Branch(s_lumiSummaryBranch.c_str(),s_lumiSummaryName.c_str(),&m_lumisummary,m_bsize,m_splitlevel);
  m_hlxTree->Branch(s_lumiDetailBranch.c_str(),s_lumiDetailName.c_str(),&m_lumidetail,m_bsize,m_splitlevel);

  //book trg tree
  m_trgTree = new TTree(s_trgTree.c_str(),s_trgTree.c_str());
  m_trgTree->Branch(s_trgBranch.c_str(),s_trgName.c_str(),&m_trg,m_bsize,m_splitlevel);

  //book hlt tree
  m_hltTree = new TTree(s_hltTree.c_str(),s_hltTree.c_str());
  m_hltTree->Branch(s_hltBranch.c_str(),s_hltName.c_str(),&m_hlt,m_bsize,m_splitlevel);

}

// -----------------------------------------------------------------

void genLumiRaw::beginRun(const edm::Run& run, const edm::EventSetup& c){
  //generateRunSummary(1,20);
  
}
 
// -----------------------------------------------------------------
void genLumiRaw::endRun(edm::Run const& run, edm::EventSetup const& c){
  std::cout<<"genLumiRaw::endRun filling runsummary tree"<<std::endl;
  generateRunSummary(run.run(),20);
}

// -----------------------------------------------------------------
void genLumiRaw::endJob(){
  std::cout<<"genLumiRaw::endJob"<<std::endl;
  if(m_file) {
    m_file->Write();
    m_file->Close();
    delete m_file;
    m_file=0;
  }
}

// -----------------------------------------------------------------
void genLumiRaw::generateRunSummary(unsigned int runnumber,
				    unsigned int totalCMSls){
  HCAL_HLX::RUN_SUMMARY localrunsummary;
  const char* runsequence="Fake Run Summary";
  localrunsummary.runNumber=runnumber;
  localrunsummary.timestamp=2;
  localrunsummary.timestamp_micros=3;
  localrunsummary.startOrbitNumber=4;
  localrunsummary.endOrbitnumber=5;
  localrunsummary.fillNumber=6;
  localrunsummary.numberCMSLumiSections=totalCMSls;
  localrunsummary.numberLumiDAQLumiSections=totalCMSls+2;
  localrunsummary.HLTConfigId=7792;
  //std::strncpy(localrunsummary.runSequenceName,runsequence,128);
  std::strcpy(localrunsummary.runSequenceName,runsequence);
  std::cout<<"copied "<<std::memmove(m_runsummary,&localrunsummary,sizeof(HCAL_HLX::RUN_SUMMARY))<<std::endl;
  m_runsummaryTree->Fill();
}
// -----------------------------------------------------------------
void genLumiRaw::generateHLT(unsigned int runnumber,
			     unsigned int lsnumber){
  HCAL_HLX::HLTRIGGER localhlt;
  localhlt.runNumber=runnumber;
  localhlt.sectionNumber=lsnumber;
  const char* pathname="Fake Path";
  const char* modulename="Fake Module";
  for (unsigned int iHLT=0; iHLT < 256; ++iHLT){
    std::strncpy(localhlt.HLTPaths[iHLT].PathName,pathname,128);
    localhlt.HLTPaths[iHLT].L1Pass = iHLT;
    localhlt.HLTPaths[iHLT].PSPass = iHLT*2;
    localhlt.HLTPaths[iHLT].PAccept = iHLT*3;
    localhlt.HLTPaths[iHLT].PExcept = iHLT*4;
    localhlt.HLTPaths[iHLT].PReject = iHLT*5;
    std::strncpy(localhlt.HLTPaths[iHLT].PrescalerModule,modulename,64);
    localhlt.HLTPaths[iHLT].PSIndex = iHLT;
    localhlt.HLTPaths[iHLT].Prescale = iHLT;
  }
  std::memmove(m_hlt,&localhlt,sizeof(HCAL_HLX::HLTRIGGER));
  m_hltTree->Fill();
}
void genLumiRaw::generateTRG(unsigned int runnumber,
			     unsigned int lsnumber){
  HCAL_HLX::LEVEL1_TRIGGER localtrg;
  const char* algoname="Fake";
  const char* techname="11";
  localtrg.runNumber=runnumber;
  localtrg.sectionNumber=lsnumber;
  localtrg.deadtimecount=3;
  for(unsigned int iAlgo=0; iAlgo<128; ++iAlgo){
    std::strncpy(localtrg.GTAlgo[iAlgo].pathName,algoname,128);
    localtrg.GTAlgo[iAlgo].counts=iAlgo;
    localtrg.GTAlgo[iAlgo].prescale=iAlgo;
  }
  for(unsigned int iTech=0; iTech<64; ++iTech){
    std::strncpy(localtrg.GTTech[iTech].pathName,techname,128);
    localtrg.GTTech[iTech].counts=iTech;
    localtrg.GTTech[iTech].prescale=iTech;
  }
  std::memmove(m_trg,&localtrg,sizeof(HCAL_HLX::LEVEL1_TRIGGER));
  m_trgTree->Fill();
}
void genLumiRaw::generateHLX(unsigned int runnumber,
			     unsigned int lsnumber){
  HCAL_HLX::LUMI_SECTION_HEADER locallumiheader;
  locallumiheader.timestamp=1;
  locallumiheader.timestamp_micros=2;
  locallumiheader.runNumber=runnumber;
  locallumiheader.sectionNumber=lsnumber;
  locallumiheader.startOrbit=3;
  locallumiheader.numOrbits=4;
  locallumiheader.numBunches=5;
  locallumiheader.numHLXs=6;
  locallumiheader.bCMSLive=true;
  locallumiheader.bOC0=false;

  HCAL_HLX::LUMI_SUMMARY locallumisummary;
  locallumisummary.DeadTimeNormalization = 1; 
  locallumisummary.LHCNormalization = 2; 
  locallumisummary.InstantLumi = 3;
  locallumisummary.InstantLumiErr = 4;
  locallumisummary.InstantLumiQlty = 5;

  locallumisummary.InstantETLumi = 6;
  locallumisummary.InstantETLumiErr = 7;
  locallumisummary.InstantETLumiQlty = 8;
  locallumisummary.ETNormalization = 9; 
  
  locallumisummary.InstantOccLumi[0] = 10;
  locallumisummary.InstantOccLumiErr[0] = 11;
  locallumisummary.InstantOccLumiQlty[0] = 12;
  locallumisummary.OccNormalization[0] = 13;
  
  locallumisummary.lumiNoise[0] = 14;
  
  locallumisummary.InstantOccLumi[1] = 10;
  locallumisummary.InstantOccLumiErr[1] = 11;
  locallumisummary.InstantOccLumiQlty[1] = 12;
  locallumisummary.OccNormalization[1] = 13;
  
  locallumisummary.lumiNoise[1] = 14;
  
  HCAL_HLX::LUMI_DETAIL locallumidetail;  
  for (unsigned int iBX = 0; iBX < HCAL_HLX_MAX_BUNCHES; ++iBX) {
    locallumidetail.LHCLumi[iBX] = 143; 
    locallumidetail.ETLumi[iBX] = 143;
    locallumidetail.ETLumiErr[iBX] = 143;
    locallumidetail.ETLumiQlty[iBX] = 143;
    locallumidetail.ETBXNormalization[iBX] = 143; 
    locallumidetail.OccLumi[0][iBX] = 143;
    locallumidetail.OccLumiErr[0][iBX] = 143;
    locallumidetail.OccLumiQlty[0][iBX] = 143;
    locallumidetail.OccBXNormalization[0][iBX] = 143;
    locallumidetail.OccLumi[1][iBX] = 143;
    locallumidetail.OccLumiErr[1][iBX] = 143;
    locallumidetail.OccLumiQlty[1][iBX] = 143;
    locallumidetail.OccBXNormalization[1][iBX] = 143;
  }

  std::memmove(m_lumiheader,&locallumiheader,sizeof(HCAL_HLX::LUMI_SECTION_HEADER));
  std::memmove(m_lumisummary,&locallumisummary,sizeof(HCAL_HLX::LUMI_SUMMARY));
  std::memmove(m_lumidetail,&locallumidetail,sizeof(HCAL_HLX::LUMI_DETAIL));
  m_hlxTree->Fill();
}

// -----------------------------------------------------------------
DEFINE_FWK_MODULE(genLumiRaw);
#endif
