

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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


class ZdcTBAnalyzer : public edm::EDAnalyzer {

 public:
  explicit ZdcTBAnalyzer(const edm::ParameterSet&);
  ~ZdcTBAnalyzer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

private:
  std::string outputFileName;
  std::string inputFileName;
  bool beamDetectorsADCInfo;
  bool beamDetectorsTDCInfo;
  bool wireChambersInfo;
  bool triggerInfo;
  ZdcTBAnalysis zdcTBAnalysis;

  edm::InputTag zdcRecHitCollectionTag;
  edm::InputTag hcalTBTriggerDataTag;
  edm::InputTag hcalTBTimingTag;
  edm::InputTag hcalTBBeamCountersTag;
  edm::InputTag hcalTBEventPositionTag;
};

ZdcTBAnalyzer::ZdcTBAnalyzer(const edm::ParameterSet& iConfig) :
  zdcRecHitCollectionTag(iConfig.getParameter<edm::InputTag>("zdcRecHitCollectionTag")),
  hcalTBTriggerDataTag(iConfig.getParameter<edm::InputTag>("hcalTBTriggerDataTag")),
  hcalTBTimingTag(iConfig.getParameter<edm::InputTag>("hcalTBTimingTag")),
  hcalTBBeamCountersTag(iConfig.getParameter<edm::InputTag>("hcalTBBeamCountersTag")),
  hcalTBEventPositionTag(iConfig.getParameter<edm::InputTag>("hcalTBEventPositionTag"))
{
  std::cout<<"**************** ZdcTBAnalizer Start**************************"<<std::endl;
  edm::ParameterSet para = iConfig.getParameter<edm::ParameterSet>("ZdcTBAnalyzer");
  
  beamDetectorsADCInfo = para.getParameter<bool>("beamDetectorsADCInfoFlag");
  beamDetectorsTDCInfo = para.getParameter<bool>("beamDetectorsTDCInfoFlag");
  wireChambersInfo = para.getParameter<bool>("wireChambersInfoFlag");
  triggerInfo = para.getParameter<bool>("triggerInfoFlag");
  outputFileName =  para.getParameter<std::string>("ntupleOutputFileName");
  zdcTBAnalysis.setup(outputFileName);
}

ZdcTBAnalyzer::~ZdcTBAnalyzer(){;}

void ZdcTBAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&){
  using namespace edm;
  edm::Handle<ZDCRecHitCollection> zdcRecHits;
  edm::Handle<HcalTBTriggerData> triggers;
  edm::Handle<HcalTBTiming> times;
  edm::Handle<HcalTBBeamCounters> bc;
  edm::Handle<HcalTBEventPosition> chpos;

  e.getByLabel(zdcRecHitCollectionTag, zdcRecHits);
  if(triggerInfo){
    e.getByLabel(hcalTBTriggerDataTag, triggers);
    zdcTBAnalysis.analyze(*triggers);
  }
  if(beamDetectorsTDCInfo){
    e.getByLabel(hcalTBTimingTag, times);  // e.getByLabel("tbunpacker2",times);
    zdcTBAnalysis.analyze(*times);
  }
  if(beamDetectorsADCInfo){
    e.getByLabel(hcalTBBeamCountersTag, bc);
     zdcTBAnalysis.analyze(*bc);
  }
  if(wireChambersInfo){
    e.getByLabel(hcalTBEventPositionTag, chpos);
    zdcTBAnalysis.analyze(*chpos);
  }     
  zdcTBAnalysis.analyze(*zdcRecHits);
  zdcTBAnalysis.fillTree();
}

void ZdcTBAnalyzer::endJob(){
  zdcTBAnalysis.done();
std::cout<<"****************ZdcTBAnalizer End**************************"<<std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZdcTBAnalyzer);
