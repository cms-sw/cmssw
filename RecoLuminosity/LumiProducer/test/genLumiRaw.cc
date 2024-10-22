#ifndef RecoLuminosity_LumiProducer_genLumiRaw_h
#define RecoLuminosity_LumiProducer_genLumiRaw_h
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLuminosity/LumiProducer/interface/LumiRawDataStructures.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include <iostream>
#include <cstring>
#include <cstdio>
/**
   this program is to generate fake lumi raw data samples with desired run/lumi section parameters controlled by EmptySource parameters.
   one job can generate data for at most 1 run and unlimited number of LS
**/

class genLumiRaw : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit genLumiRaw(edm::ParameterSet const&);
  ~genLumiRaw();

private:
  void beginJob() override;
  void beginRun(const edm::Run& run, const edm::EventSetup& c) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override {}
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endJob() override;

  void generateRunSummary(unsigned int runnumber, unsigned int totalCMSls);
  void generateHLT(unsigned int runnumber, unsigned int lsnumber);
  void generateTRG(unsigned int runnumber, unsigned int lsnumber);
  void generateHLX(unsigned int runnumber, unsigned int lsnumber);

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

  unsigned int m_nls;
  int m_bsize;
  int m_splitlevel;
  TFile* m_file;
  //run summary tree&data structures
  TTree* m_runsummaryTree;
  HCAL_HLX::RUN_SUMMARY* m_runsummary;

  //hlx tree& data structures
  TTree* m_hlxTree;
  HCAL_HLX::LUMI_SECTION* m_lumisection;
  HCAL_HLX::LUMI_SECTION_HEADER* m_lumiheader;
  HCAL_HLX::LUMI_SUMMARY* m_lumisummary;
  HCAL_HLX::LUMI_DETAIL* m_lumidetail;
  //trg tree & data structures
  TTree* m_trgTree;
  HCAL_HLX::LEVEL1_TRIGGER* m_trg;

  //hlt data structures
  TTree* m_hltTree;
  HCAL_HLX::HLTRIGGER* m_hlt;

};  //end class

const std::string genLumiRaw::s_fileprefix = "CMS_LUMI_RAW_00000000_";

const std::string genLumiRaw::s_runsummaryTree = "RunSummary";
const std::string genLumiRaw::s_runsummaryBranch = "RunSummary.";
const std::string genLumiRaw::s_runsummaryName = "HCAL_HLX::RUN_SUMMARY";

const std::string genLumiRaw::s_hltTree = "HLTrigger";
const std::string genLumiRaw::s_hltBranch = "HLTrigger.";
const std::string genLumiRaw::s_hltName = "HCAL_HLX::HLTRIGGER";

const std::string genLumiRaw::s_trgTree = "L1Trigger";
const std::string genLumiRaw::s_trgBranch = "L1Trigger.";
const std::string genLumiRaw::s_trgName = "HCAL_HLX::LEVEL1_TRIGGER";

const std::string genLumiRaw::s_hlxTree = "HLXData";
const std::string genLumiRaw::s_lumiHeaderBranch = "Header.";
const std::string genLumiRaw::s_lumiHeaderName = "HCAL_HLX::LUMI_SECTION_HEADER";
const std::string genLumiRaw::s_lumiSummaryBranch = "Summary.";
const std::string genLumiRaw::s_lumiSummaryName = "HCAL_HLX::LUMI_SUMMARY";
const std::string genLumiRaw::s_lumiDetailBranch = "Detail.";
const std::string genLumiRaw::s_lumiDetailName = "HCAL_HLX::LUMI_DETAIL";

// -----------------------------------------------------------------

genLumiRaw::genLumiRaw(edm::ParameterSet const& iConfig)
    : m_nls(0),
      m_bsize(64000),
      m_splitlevel(2),
      m_file(0),
      m_runsummary(new HCAL_HLX::RUN_SUMMARY),
      m_lumisection(new HCAL_HLX::LUMI_SECTION),
      m_lumiheader(0),
      m_lumisummary(0),
      m_lumidetail(0),
      m_trg(new HCAL_HLX::LEVEL1_TRIGGER),
      m_hlt(new HCAL_HLX::HLTRIGGER) {}

// -----------------------------------------------------------------

genLumiRaw::~genLumiRaw() {
  delete m_runsummary;
  delete m_lumisection;
  delete m_trg;
  delete m_hlt;
}

// -----------------------------------------------------------------

void genLumiRaw::analyze(edm::Event const& e, edm::EventSetup const&) {
  //std::cout<<"testEvtLoop::analyze"<<std::endl;
}

// -----------------------------------------------------------------
void genLumiRaw::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) {
  ++m_nls;
  std::cout << "I'm in run " << lumiBlock.run() << " lumi block " << lumiBlock.id().luminosityBlock() << std::endl;
  generateHLT(lumiBlock.run(), lumiBlock.id().luminosityBlock());
  generateTRG(lumiBlock.run(), lumiBlock.id().luminosityBlock());
  generateHLX(lumiBlock.run(), lumiBlock.id().luminosityBlock());
}

// -----------------------------------------------------------------

void genLumiRaw::beginJob() {}

// -----------------------------------------------------------------

void genLumiRaw::beginRun(const edm::Run& run, const edm::EventSetup& c) {
  //generate file name
  std::cout << "in generate filename " << run.run() << std::endl;
  char runnumber[15];
  ::snprintf(runnumber, sizeof(runnumber), "%09d", run.run());
  std::string filename = s_fileprefix + runnumber + "_0000" + "_0" + ".root";
  //std::cout<<"filename "<<filename<<std::endl;
  //
  //prepare file name, open file,  book root trees
  //
  //const std::string filename="test.root";
  //
  m_file = new TFile(filename.c_str(), "RECREATE");

  //book run summary tree
  m_runsummaryTree = new TTree(s_runsummaryTree.c_str(), s_runsummaryTree.c_str());
  m_runsummaryTree->Branch(s_runsummaryBranch.c_str(), s_runsummaryName.c_str(), &m_runsummary, m_bsize, m_splitlevel);
  //runsummaryBranch->SetAddress(m_runsummary);

  //book hlx tree
  m_hlxTree = new TTree(s_hlxTree.c_str(), s_hlxTree.c_str());

  m_lumiheader = &(m_lumisection->hdr);
  m_hlxTree->Branch(s_lumiHeaderBranch.c_str(), s_lumiHeaderName.c_str(), &m_lumiheader, m_bsize, m_splitlevel);
  m_lumisummary = &(m_lumisection->lumiSummary);
  m_hlxTree->Branch(s_lumiSummaryBranch.c_str(), s_lumiSummaryName.c_str(), &m_lumisummary, m_bsize, m_splitlevel);
  m_lumidetail = &(m_lumisection->lumiDetail);
  m_hlxTree->Branch(s_lumiDetailBranch.c_str(), s_lumiDetailName.c_str(), &m_lumidetail, m_bsize, m_splitlevel);

  //book trg tree
  m_trgTree = new TTree(s_trgTree.c_str(), s_trgTree.c_str());
  m_trgTree->Branch(s_trgBranch.c_str(), s_trgName.c_str(), &m_trg, m_bsize, m_splitlevel);

  //book hlt tree
  m_hltTree = new TTree(s_hltTree.c_str(), s_hltTree.c_str());
  m_hltTree->Branch(s_hltBranch.c_str(), s_hltName.c_str(), &m_hlt, m_bsize, m_splitlevel);
}

// -----------------------------------------------------------------
void genLumiRaw::endRun(edm::Run const& run, edm::EventSetup const& c) {
  std::cout << "run number in endRun " << run.run() << " " << m_nls << std::endl;
  //std::cout<<"nls "<<m_nls<<std::endl;
  generateRunSummary(run.run(), m_nls);
}

// -----------------------------------------------------------------
void genLumiRaw::endJob() {
  std::cout << "genLumiRaw::endJob" << std::endl;
  if (m_file) {
    m_file->Write();
    m_file->Close();
    delete m_file;
    m_file = 0;
  }
}

// -----------------------------------------------------------------
void genLumiRaw::generateRunSummary(unsigned int runnumber, unsigned int totalCMSls) {
  std::cout << "inside generateRunSummary " << runnumber << " " << totalCMSls << std::endl;
  HCAL_HLX::RUN_SUMMARY localrunsummary;
  const char* runsequence = "Fake Run Summary";
  std::strncpy(localrunsummary.runSequenceName, runsequence, 128);
  localrunsummary.HLTConfigId = 7792;
  localrunsummary.timestamp = 2;
  localrunsummary.timestamp_micros = 3;
  localrunsummary.startOrbitNumber = 4;
  localrunsummary.endOrbitnumber = 5;
  localrunsummary.runNumber = runnumber;
  localrunsummary.fillNumber = 6;
  localrunsummary.numberCMSLumiSections = totalCMSls;
  localrunsummary.numberLumiDAQLumiSections = totalCMSls + 2;
  std::memmove(m_runsummary, &localrunsummary, sizeof(HCAL_HLX::RUN_SUMMARY));
  m_runsummaryTree->Fill();
}
// -----------------------------------------------------------------
void genLumiRaw::generateHLT(unsigned int runnumber, unsigned int lsnumber) {
  HCAL_HLX::HLTRIGGER localhlt;
  localhlt.runNumber = runnumber;
  localhlt.sectionNumber = lsnumber;
  const char* pathname = "Fake Path";
  const char* modulename = "Fake Module";
  unsigned int npath = 0;
  for (unsigned int iHLT = 0; iHLT < 256; ++iHLT) {
    ++npath;
    std::strncpy(localhlt.HLTPaths[iHLT].PathName, pathname, 128);
    localhlt.HLTPaths[iHLT].L1Pass = iHLT;
    localhlt.HLTPaths[iHLT].PSPass = iHLT * 2;
    localhlt.HLTPaths[iHLT].PAccept = iHLT * 3;
    localhlt.HLTPaths[iHLT].PExcept = iHLT * 4;
    localhlt.HLTPaths[iHLT].PReject = iHLT * 5;
    std::strncpy(localhlt.HLTPaths[iHLT].PrescalerModule, modulename, 64);
    localhlt.HLTPaths[iHLT].PSIndex = iHLT;
    localhlt.HLTPaths[iHLT].Prescale = iHLT;
    localhlt.HLTPaths[iHLT].HLTConfigId = 6785;
  }
  localhlt.numPaths = npath;
  std::memmove(m_hlt, &localhlt, sizeof(HCAL_HLX::HLTRIGGER));
  m_hltTree->Fill();
}
void genLumiRaw::generateTRG(unsigned int runnumber, unsigned int lsnumber) {
  HCAL_HLX::LEVEL1_TRIGGER localtrg;
  const char* algoname = "Fake";
  const char* techname = "11";
  localtrg.runNumber = runnumber;
  localtrg.sectionNumber = lsnumber;
  localtrg.deadtimecount = 3;
  for (unsigned int iAlgo = 0; iAlgo < 128; ++iAlgo) {
    std::strncpy(localtrg.GTAlgo[iAlgo].pathName, algoname, 128);
    localtrg.GTAlgo[iAlgo].counts = iAlgo;
    localtrg.GTAlgo[iAlgo].prescale = iAlgo;
  }
  for (unsigned int iTech = 0; iTech < 64; ++iTech) {
    std::strncpy(localtrg.GTTech[iTech].pathName, techname, 128);
    localtrg.GTTech[iTech].counts = iTech;
    localtrg.GTTech[iTech].prescale = iTech;
  }
  std::memmove(m_trg, &localtrg, sizeof(HCAL_HLX::LEVEL1_TRIGGER));
  m_trgTree->Fill();
}
void genLumiRaw::generateHLX(unsigned int runnumber, unsigned int lsnumber) {
  HCAL_HLX::LUMI_SECTION locallumisection;

  locallumisection.hdr.timestamp = 1;
  locallumisection.hdr.timestamp_micros = 2;
  locallumisection.hdr.runNumber = runnumber;
  locallumisection.hdr.sectionNumber = lsnumber;
  locallumisection.hdr.startOrbit = 3;
  locallumisection.hdr.numOrbits = 4;
  locallumisection.hdr.numBunches = 5;
  locallumisection.hdr.numHLXs = 6;
  locallumisection.hdr.bCMSLive = true;
  locallumisection.hdr.bOC0 = false;

  // HCAL_HLX::LUMI_SUMMARY locallumisummary;
  locallumisection.lumiSummary.DeadTimeNormalization = 1;
  locallumisection.lumiSummary.LHCNormalization = 2;
  locallumisection.lumiSummary.InstantLumi = 3;
  locallumisection.lumiSummary.InstantLumiErr = 4;
  locallumisection.lumiSummary.InstantLumiQlty = 5;

  locallumisection.lumiSummary.InstantETLumi = 6;
  locallumisection.lumiSummary.InstantETLumiErr = 7;
  locallumisection.lumiSummary.InstantETLumiQlty = 8;
  locallumisection.lumiSummary.ETNormalization = 9;

  locallumisection.lumiSummary.InstantOccLumi[0] = 10;
  locallumisection.lumiSummary.InstantOccLumiErr[0] = 11;
  locallumisection.lumiSummary.InstantOccLumiQlty[0] = 12;
  locallumisection.lumiSummary.OccNormalization[0] = 13;

  locallumisection.lumiSummary.lumiNoise[0] = 14;

  locallumisection.lumiSummary.InstantOccLumi[1] = 10;
  locallumisection.lumiSummary.InstantOccLumiErr[1] = 11;
  locallumisection.lumiSummary.InstantOccLumiQlty[1] = 12;
  locallumisection.lumiSummary.OccNormalization[1] = 13;

  locallumisection.lumiSummary.lumiNoise[1] = 14;

  //HCAL_HLX::LUMI_DETAIL localalumidetail;
  for (unsigned int iBX = 0; iBX < HCAL_HLX_MAX_BUNCHES; ++iBX) {
    locallumisection.lumiDetail.LHCLumi[iBX] = 143;
    locallumisection.lumiDetail.ETLumi[iBX] = 143;
    locallumisection.lumiDetail.ETLumiErr[iBX] = 143;
    locallumisection.lumiDetail.ETLumiQlty[iBX] = 143;
    locallumisection.lumiDetail.ETBXNormalization[iBX] = 143;
    locallumisection.lumiDetail.OccLumi[0][iBX] = 143;
    locallumisection.lumiDetail.OccLumiErr[0][iBX] = 143;
    locallumisection.lumiDetail.OccLumiQlty[0][iBX] = 143;
    locallumisection.lumiDetail.OccBXNormalization[0][iBX] = 143;
    locallumisection.lumiDetail.OccLumi[1][iBX] = 143;
    locallumisection.lumiDetail.OccLumiErr[1][iBX] = 143;
    locallumisection.lumiDetail.OccLumiQlty[1][iBX] = 143;
    locallumisection.lumiDetail.OccBXNormalization[1][iBX] = 143;
  }
  std::memmove(m_lumisection, &locallumisection, sizeof(HCAL_HLX::LUMI_SECTION));
  m_hlxTree->Fill();
}

// -----------------------------------------------------------------
DEFINE_FWK_MODULE(genLumiRaw);
#endif
