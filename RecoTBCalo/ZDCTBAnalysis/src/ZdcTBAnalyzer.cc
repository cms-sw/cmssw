

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
};

ZdcTBAnalyzer::ZdcTBAnalyzer(const edm::ParameterSet& iConfig)
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
  
  e.getByType(zdcRecHits);
  if(triggerInfo){
    e.getByType(triggers);
    zdcTBAnalysis.analyze(*triggers);
  }
  if(beamDetectorsTDCInfo){
    e.getByType(times);  // e.getByLabel("tbunpacker2",times);
    zdcTBAnalysis.analyze(*times);
  }
  if(beamDetectorsADCInfo){
    e.getByType(bc);
     zdcTBAnalysis.analyze(*bc);
  }
  if(wireChambersInfo){
    e.getByType(chpos);
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
