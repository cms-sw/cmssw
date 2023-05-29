

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "RecoTBCalo/ZDCTBAnalysis/interface/ZdcTBAnalysis.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include <iostream>
#include <memory>

class ZdcTBAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit ZdcTBAnalyzer(const edm::ParameterSet&);
  ~ZdcTBAnalyzer() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  std::string outputFileName;
  std::string inputFileName;
  bool beamDetectorsADCInfo;
  bool beamDetectorsTDCInfo;
  bool wireChambersInfo;
  bool triggerInfo;
  ZdcTBAnalysis zdcTBAnalysis;

  edm::EDGetTokenT<ZDCRecHitCollection> tok_zdc_;
  edm::EDGetTokenT<HcalTBTriggerData> tok_tb_;
  edm::EDGetTokenT<HcalTBTiming> tok_timing_;
  edm::EDGetTokenT<HcalTBBeamCounters> tok_bc_;
  edm::EDGetTokenT<HcalTBEventPosition> tok_pos_;
};

ZdcTBAnalyzer::ZdcTBAnalyzer(const edm::ParameterSet& iConfig) {
  tok_zdc_ = consumes<ZDCRecHitCollection>(iConfig.getParameter<edm::InputTag>("zdcRecHitCollectionTag"));
  tok_tb_ = consumes<HcalTBTriggerData>(iConfig.getParameter<edm::InputTag>("hcalTBTriggerDataTag"));
  tok_timing_ = consumes<HcalTBTiming>(iConfig.getParameter<edm::InputTag>("hcalTBTimingTag"));
  tok_bc_ = consumes<HcalTBBeamCounters>(iConfig.getParameter<edm::InputTag>("hcalTBBeamCountersTag"));
  tok_pos_ = consumes<HcalTBEventPosition>(iConfig.getParameter<edm::InputTag>("hcalTBEventPositionTag"));

  std::cout << "**************** ZdcTBAnalizer Start**************************" << std::endl;
  edm::ParameterSet para = iConfig.getParameter<edm::ParameterSet>("ZdcTBAnalyzer");

  beamDetectorsADCInfo = para.getParameter<bool>("beamDetectorsADCInfoFlag");
  beamDetectorsTDCInfo = para.getParameter<bool>("beamDetectorsTDCInfoFlag");
  wireChambersInfo = para.getParameter<bool>("wireChambersInfoFlag");
  triggerInfo = para.getParameter<bool>("triggerInfoFlag");
  outputFileName = para.getParameter<std::string>("ntupleOutputFileName");
  zdcTBAnalysis.setup(outputFileName);
}

ZdcTBAnalyzer::~ZdcTBAnalyzer() { ; }

void ZdcTBAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  using namespace edm;
  edm::Handle<ZDCRecHitCollection> zdcRecHits;
  edm::Handle<HcalTBTriggerData> triggers;
  edm::Handle<HcalTBTiming> times;
  edm::Handle<HcalTBBeamCounters> bc;
  edm::Handle<HcalTBEventPosition> chpos;

  e.getByToken(tok_zdc_, zdcRecHits);
  if (triggerInfo) {
    e.getByToken(tok_tb_, triggers);
    zdcTBAnalysis.analyze(*triggers);
  }
  if (beamDetectorsTDCInfo) {
    e.getByToken(tok_timing_, times);  // e.getByLabel("tbunpacker2",times);
    zdcTBAnalysis.analyze(*times);
  }
  if (beamDetectorsADCInfo) {
    e.getByToken(tok_bc_, bc);
    zdcTBAnalysis.analyze(*bc);
  }
  if (wireChambersInfo) {
    e.getByToken(tok_pos_, chpos);
    zdcTBAnalysis.analyze(*chpos);
  }
  zdcTBAnalysis.analyze(*zdcRecHits);
  zdcTBAnalysis.fillTree();
}

void ZdcTBAnalyzer::endJob() {
  zdcTBAnalysis.done();
  std::cout << "****************ZdcTBAnalizer End**************************" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZdcTBAnalyzer);
