// -*- C++ -*-
//
// Package:    EcalDeadCellTriggerPrimitiveFilter
// Class:      EcalDeadCellTriggerPrimitiveFilter
// 
/**\class EcalDeadCellTriggerPrimitiveFilter EcalDeadCellTriggerPrimitiveFilter.cc

 Description: <one line class summary>
 Event filtering for anomalous ECAL events where the energy measured by ECAL is significantly biased due to energy depositions
 in dead cell regions. 
*/
//
// Original Author:  Hongxuan Liu and Kenichi Hatakeyama
//                   in collaboration with Konstantinos Theofilatos and Ulla Gebbert

// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Provenance/interface/ProcessHistory.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "TFile.h"
#include "TTree.h"

using namespace std;

class EcalDeadCellTriggerPrimitiveFilter : public edm::EDFilter {
public:
  explicit EcalDeadCellTriggerPrimitiveFilter(const edm::ParameterSet&);
  ~EcalDeadCellTriggerPrimitiveFilter();

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void beginJob() override;
  virtual void endJob() override;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void envSet(const edm::EventSetup&);

  // ----------member data ---------------------------

  const bool taggingMode_;

  const bool debug_;
  const int verbose_;

  const bool doEEfilter_;

// Channel status related
  edm::ESHandle<EcalChannelStatus>  ecalStatus;
  edm::ESHandle<CaloGeometry>       geometry;

  void loadEcalDigis(edm::Event& iEvent, const edm::EventSetup& iSetup);
  void loadEcalRecHits(edm::Event& iEvent, const edm::EventSetup& iSetup);

  const edm::InputTag ebReducedRecHitCollection_;
  const edm::InputTag eeReducedRecHitCollection_;
  edm::Handle<EcalRecHitCollection> barrelReducedRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapReducedRecHitsHandle;

  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;

  EcalTPGScale ecalScale_;

  const int maskedEcalChannelStatusThreshold_;

// XXX: All the following can be built at the beginning of a job
// Store DetId <==> vector<double> (eta, phi, theta)
  std::map<DetId, std::vector<double> > EcalAllDeadChannelsValMap;
// Store EB: DetId <==> vector<int> (subdet, ieta, iphi, status)
// Store EE: DetId <==> vector<int> (subdet, ix, iy, iz, status)
  std::map<DetId, std::vector<int> >    EcalAllDeadChannelsBitMap;

// Store DetId <==> EcalTrigTowerDetId
  std::map<DetId, EcalTrigTowerDetId> EcalAllDeadChannelsTTMap;

  int getChannelStatusMaps();

// TP filter
  const double etValToBeFlagged_;

  const edm::InputTag tpDigiCollection_;
  edm::Handle<EcalTrigPrimDigiCollection> pTPDigis;

// chnStatus > 0, then exclusive, i.e., only consider status == chnStatus
// chnStatus < 0, then inclusive, i.e., consider status >= abs(chnStatus)
// Return value:  + : positive zside  - : negative zside
  int setEvtTPstatus(const double &tpCntCut, const int &chnStatus);

  int evtProcessedCnt, totFilteredCnt;

  const bool makeProfileRoot_;
  const std::string profileRootName_;
  TFile *profFile;
  TTree *profTree;

  std::vector<int> *cutFlowFlagTmpPtr;
  std::vector<std::string> *cutFlowStrTmpPtr;

  void loadEventInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  unsigned int run, event, ls;

  bool getEventInfoForFilterOnce_;

  std::string releaseVersion_;
  int hastpDigiCollection_, hasReducedRecHits_;

  bool useTPmethod_, useHITmethod_;

  void loadEventInfoForFilter(const edm::Event& iEvent);

// Only for EB since the dead front-end has one-to-one map to TT
  std::map<EcalTrigTowerDetId, double> accuTTetMap;
  std::map<EcalTrigTowerDetId, int> accuTTchnMap;
  std::map<EcalTrigTowerDetId, int> TTzsideMap;

// For EE, the one-to-one map to dead front-end is the SuperCrystal
  std::map<EcalScDetId, double> accuSCetMap;
  std::map<EcalScDetId, int> accuSCchnMap;
  std::map<EcalScDetId, int> SCzsideMap;

// To be used before a bug fix
  std::vector<DetId> avoidDuplicateVec;
  int setEvtRecHitstatus(const double &tpValCut, const int &chnStatus, const int &towerTest);

};

//
// constructors and destructor
//
EcalDeadCellTriggerPrimitiveFilter::EcalDeadCellTriggerPrimitiveFilter(const edm::ParameterSet& iConfig)
  : taggingMode_ (iConfig.getParameter<bool>("taggingMode") ) 
  , debug_ (iConfig.getParameter<bool>("debug") )
  , verbose_ (iConfig.getParameter<int>("verbose") )
  , doEEfilter_ (iConfig.getUntrackedParameter<bool>("doEEfilter") )
  , ebReducedRecHitCollection_ (iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection") )
  , eeReducedRecHitCollection_ (iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection") )
  , maskedEcalChannelStatusThreshold_ (iConfig.getParameter<int>("maskedEcalChannelStatusThreshold") )
  , etValToBeFlagged_ (iConfig.getParameter<double>("etValToBeFlagged") )
  , tpDigiCollection_ (iConfig.getParameter<edm::InputTag>("tpDigiCollection") )
  , makeProfileRoot_ (iConfig.getUntrackedParameter<bool>("makeProfileRoot") )
  , profileRootName_ (iConfig.getUntrackedParameter<std::string>("profileRootName") )
{
  getEventInfoForFilterOnce_ = false;
  hastpDigiCollection_ = 0; hasReducedRecHits_ = 0; 
  useTPmethod_ = true; useHITmethod_ = false;

  if( makeProfileRoot_ ){

     profFile = new TFile(profileRootName_.c_str(), "RECREATE");
     profTree = new TTree("filter", "filter profile");
     profTree->Branch("run", &run, "run/I");
     profTree->Branch("event", &event, "event/I");
     profTree->Branch("lumi", &ls, "lumi/I");
     profTree->Branch("cutFlowFlag", &cutFlowFlagTmpPtr);
     profTree->Branch("cutFlowStr", &cutFlowStrTmpPtr);

  }

  produces<bool>();
}

EcalDeadCellTriggerPrimitiveFilter::~EcalDeadCellTriggerPrimitiveFilter() {

  if( makeProfileRoot_ ){
     profFile->cd();
     profTree->Write();
     delete profTree;
     profFile->Close();
     delete profFile;
  }

}

void EcalDeadCellTriggerPrimitiveFilter::loadEventInfoForFilter(const edm::Event &iEvent){

  std::vector<edm::Provenance const*> provenances;
  iEvent.getAllProvenance(provenances);
  const unsigned int nProvenance = provenances.size();
  for (unsigned int ip = 0; ip < nProvenance; ip++) {
    const edm::Provenance& provenance = *( provenances[ip] );
    if( provenance.moduleLabel().data() ==  tpDigiCollection_.label() ){ hastpDigiCollection_ = 1; }
    if( provenance.moduleLabel().data() == ebReducedRecHitCollection_.label() || provenance.moduleLabel().data() == eeReducedRecHitCollection_.label() ){
       hasReducedRecHits_++;
    }
    if( hastpDigiCollection_ && hasReducedRecHits_>=2 ){ break; }
  }

  if( debug_ ) std::cout<<"\nhastpDigiCollection_ : "<<hastpDigiCollection_<<"  hasReducedRecHits_ : "<<hasReducedRecHits_<<std::endl;

  const edm::ProcessHistory& history = iEvent.processHistory();
  const unsigned int nHist = history.size();
// XXX: the last one is usually a USER process!
  releaseVersion_ = history[nHist-2].releaseVersion();
  TString tmpTstr(releaseVersion_);
  TObjArray * split = tmpTstr.Tokenize("_");
  int majorV = TString(split->At(1)->GetName()).Atoi();
  int minorV = TString(split->At(2)->GetName()).Atoi();

  if( debug_ ) std::cout<<"processName : "<<history[nHist-2].processName().data()<<"  releaseVersion : "<<releaseVersion_<<std::endl; 

// If TP is available, always use TP.
// In RECO file, we always have ecalTPSkim (at least from 38X for data and 39X for MC).
// In AOD file, we can only have recovered rechits in the reduced rechits collection after 42X
// Do NOT expect end-users provide ecalTPSkim or recovered rechits themselves!!
// If they really can provide them, they must be experts to modify this code to suit their own purpose :-)
  if( !hastpDigiCollection_ && !hasReducedRecHits_ ){ useTPmethod_ = false; useHITmethod_ = false; 
     if( debug_ ){
        std::cout<<"\nWARNING ... Cannot find either tpDigiCollection_ or reducedRecHitCollecion_ ?!"<<std::endl;
        std::cout<<"  Will NOT DO ANY FILTERING !"<<std::endl;
     }
  }
  else if( hastpDigiCollection_ ){ useTPmethod_ = true; useHITmethod_ = false; }
//  else if( majorV >=4 && minorV >=2 ){ useTPmethod_ = false; useHITmethod_ = true; }
  else if( majorV >=5 || (majorV==4 && minorV >=2) ){ useTPmethod_ = false; useHITmethod_ = true; }
  else{ useTPmethod_ = false; useHITmethod_ = false; 
     if( debug_ ){
        std::cout<<"\nWARNING ... TP filter can ONLY be used in AOD after 42X"<<std::endl;
        std::cout<<"  Will NOT DO ANY FILTERING !"<<std::endl;
     }
  }

  if( debug_ ) std::cout<<"useTPmethod_ : "<<useTPmethod_<<"  useHITmethod_ : "<<useHITmethod_<<std::endl;

  getEventInfoForFilterOnce_ = true;
 
}


void EcalDeadCellTriggerPrimitiveFilter::loadEventInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup){
   run = iEvent.id().run();
   event = iEvent.id().event();
   ls = iEvent.luminosityBlock();
}


void EcalDeadCellTriggerPrimitiveFilter::loadEcalDigis(edm::Event& iEvent, const edm::EventSetup& iSetup){

  iEvent.getByLabel(tpDigiCollection_, pTPDigis);
  if ( !pTPDigis.isValid() ) { edm::LogWarning("EcalDeadCellTriggerPrimitiveFilter") << "Can't get the product " << tpDigiCollection_.instance()
                                             << " with label " << tpDigiCollection_.label(); return; }
}

void EcalDeadCellTriggerPrimitiveFilter::loadEcalRecHits(edm::Event& iEvent, const edm::EventSetup& iSetup){

  iEvent.getByLabel(ebReducedRecHitCollection_,barrelReducedRecHitsHandle);
  iEvent.getByLabel(eeReducedRecHitCollection_,endcapReducedRecHitsHandle);

}

//
// static data member definitions
//

void EcalDeadCellTriggerPrimitiveFilter::envSet(const edm::EventSetup& iSetup) {

  if (debug_ && verbose_ >=2) std::cout << "***envSet***" << std::endl;

  ecalScale_.setEventSetup( iSetup );
  iSetup.get<IdealGeometryRecord>().get(ttMap_);

  iSetup.get<EcalChannelStatusRcd> ().get(ecalStatus);
  iSetup.get<CaloGeometryRecord>   ().get(geometry);

  if( !ecalStatus.isValid() )  throw "Failed to get ECAL channel status!";
  if( !geometry.isValid()   )  throw "Failed to get the geometry!";

}

// ------------ method called on each new Event  ------------
bool EcalDeadCellTriggerPrimitiveFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  std::vector<int> cutFlowFlagTmpVec; std::vector<std::string> cutFlowStrTmpVec;

  loadEventInfo(iEvent, iSetup);

  if( !getEventInfoForFilterOnce_ ){ loadEventInfoForFilter(iEvent); }

  evtProcessedCnt++;

  bool pass = true;

  int evtTagged = 0;

  if( useTPmethod_ ){
     loadEcalDigis(iEvent, iSetup);
     evtTagged = setEvtTPstatus(etValToBeFlagged_, 13);
  }

  if( useHITmethod_ ){
     loadEcalRecHits(iEvent, iSetup);
     evtTagged = setEvtRecHitstatus(etValToBeFlagged_, 13, 13);
  }

  if( evtTagged ){ pass = false; totFilteredCnt++; }

  if( makeProfileRoot_ ){

     cutFlowFlagTmpVec.push_back(evtTagged); cutFlowStrTmpVec.push_back("TP");

     cutFlowFlagTmpPtr = &cutFlowFlagTmpVec;
     cutFlowStrTmpPtr = &cutFlowStrTmpVec;

     profTree->Fill();
  }

  if(debug_ && verbose_ >=2){
     int evtstatusABS = abs(evtTagged);
     printf("\nrun : %8d  event : %10d  lumi : %4d  evtTPstatus  ABS : %d  13 : % 2d\n", run, event, ls, evtstatusABS, evtTagged);
  }

  std::auto_ptr<bool> pOut( new bool(pass) ); 
  iEvent.put( pOut );

  if (taggingMode_) return true;
  else return pass;
}

// ------------ method called once each job just before starting event loop  ------------
void EcalDeadCellTriggerPrimitiveFilter::beginJob() {
  evtProcessedCnt = 0;
  totFilteredCnt = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void EcalDeadCellTriggerPrimitiveFilter::endJob() {
}

// ------------ method called once each run just before starting event loop  ------------
void EcalDeadCellTriggerPrimitiveFilter::beginRun(const edm::Run &run, const edm::EventSetup& iSetup) {
// Channel status might change for each run (data)
// Event setup
  envSet(iSetup);
  getChannelStatusMaps();
  if( debug_ && verbose_ >=2) std::cout<< "EcalAllDeadChannelsValMap.size() : "<<EcalAllDeadChannelsValMap.size()<<"  EcalAllDeadChannelsBitMap.size() : "<<EcalAllDeadChannelsBitMap.size()<<std::endl;
  return ;
}

int EcalDeadCellTriggerPrimitiveFilter::setEvtRecHitstatus(const double &tpValCut, const int &chnStatus, const int &towerTest){
        
  if( debug_ && verbose_ >=2) std::cout<<"***begin setEvtTPstatusRecHits***"<<std::endl;
        
  accuTTetMap.clear(); accuTTchnMap.clear(); TTzsideMap.clear();
  accuSCetMap.clear(); accuSCchnMap.clear(); SCzsideMap.clear();
  avoidDuplicateVec.clear();
        
/*         
  const EBRecHitCollection HitecalEB = *(barrelRecHitsHandle.product());
  const EERecHitCollection HitecalEE = *(endcapRecHitsHandle.product());
*/         
  const EBRecHitCollection HitecalEB = *(barrelReducedRecHitsHandle.product());
  const EERecHitCollection HitecalEE = *(endcapReducedRecHitsHandle.product());

  int isPassCut =0;

  EBRecHitCollection::const_iterator ebrechit;
  for (ebrechit = HitecalEB.begin(); ebrechit != HitecalEB.end(); ebrechit++) {

     EBDetId det = ebrechit->id();

     std::map<DetId, vector<double> >::iterator valItor = EcalAllDeadChannelsValMap.find(det);
     if( valItor == EcalAllDeadChannelsValMap.end() ) continue;

     double theta = valItor->second.back();

     std::map<DetId, vector<int> >::iterator bitItor = EcalAllDeadChannelsBitMap.find(det);
     if( bitItor == EcalAllDeadChannelsBitMap.end() ) continue;

     std::map<DetId, EcalTrigTowerDetId>::iterator ttItor = EcalAllDeadChannelsTTMap.find(det);
     if( ttItor == EcalAllDeadChannelsTTMap.end() ) continue;

     int status = bitItor->second.back();

     bool toDo = false;
     if( chnStatus >0 && status == chnStatus ) toDo = true;
     if( chnStatus <0 && status >= abs(chnStatus) ) toDo = true;
// This might be suitable for channels with status other than 13, 
// since this function is written as a general one ...
     if( !ebrechit->isRecovered() ) toDo = false;
//     if( !ebrechit->checkFlag(EcalRecHit::kTowerRecovered) ) toDo = false;

     if( toDo ){

        EcalTrigTowerDetId ttDetId = ttItor->second;
        int ttzside = ttDetId.zside();

        std::vector<DetId> vid = ttMap_->constituentsOf(ttDetId);
        int towerTestCnt =0;
        for(std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
           std::map<DetId, vector<int> >::iterator bit2Itor = EcalAllDeadChannelsBitMap.find( (*dit) );
           if( bit2Itor == EcalAllDeadChannelsBitMap.end() ){ towerTestCnt ++; continue; }
           if( towerTest >0 && bit2Itor->second.back() == towerTest ) continue;
           if( towerTest <0 && bit2Itor->second.back() >= abs(towerTest) ) continue;
           towerTestCnt ++;
        }
        if( towerTestCnt !=0 && debug_ && verbose_ >=2) std::cout<<"towerTestCnt : "<<towerTestCnt<<"  for towerTest : "<<towerTest<<std::endl;

        std::vector<DetId>::iterator avoidItor; avoidItor = find( avoidDuplicateVec.begin(), avoidDuplicateVec.end(), det);
        if( avoidItor == avoidDuplicateVec.end() ){
           avoidDuplicateVec.push_back(det);
        }else{
           continue;
        }

        std::map<EcalTrigTowerDetId, double>::iterator ttetItor = accuTTetMap.find(ttDetId);
        if( ttetItor == accuTTetMap.end() ){
           accuTTetMap[ttDetId] = ebrechit->energy()*sin(theta);
           accuTTchnMap[ttDetId] = 1;
           TTzsideMap[ttDetId] = ttzside;
        }else{
           accuTTetMap[ttDetId] += ebrechit->energy()*sin(theta);
           accuTTchnMap[ttDetId] ++;
        }
     }
  } // loop over EB

  EERecHitCollection::const_iterator eerechit;
  for (eerechit = HitecalEE.begin(); eerechit != HitecalEE.end(); eerechit++) {

     EEDetId det = eerechit->id();

     std::map<DetId, vector<double> >::iterator valItor = EcalAllDeadChannelsValMap.find(det);
     if( valItor == EcalAllDeadChannelsValMap.end() ) continue;

     double theta = valItor->second.back();

     std::map<DetId, vector<int> >::iterator bitItor = EcalAllDeadChannelsBitMap.find(det);
     if( bitItor == EcalAllDeadChannelsBitMap.end() ) continue;

     std::map<DetId, EcalTrigTowerDetId>::iterator ttItor = EcalAllDeadChannelsTTMap.find(det);
     if( ttItor == EcalAllDeadChannelsTTMap.end() ) continue;

     int status = bitItor->second.back();

     bool toDo = false;
     if( chnStatus >0 && status == chnStatus ) toDo = true;
     if( chnStatus <0 && status >= abs(chnStatus) ) toDo = true;
// This might be suitable for channels with status other than 13, 
// since this function is written as a general one ...
     if( !eerechit->isRecovered() ) toDo = false;
//     if( !eerechit->checkFlag(EcalRecHit::kTowerRecovered) ) toDo = false;

     if( toDo ){

// vvvv= Only for debuging or testing purpose =vvvv
        EcalTrigTowerDetId ttDetId = ttItor->second;
//        int ttzside = ttDetId.zside();

        std::vector<DetId> vid = ttMap_->constituentsOf(ttDetId);
        int towerTestCnt =0;
        for(std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
           std::map<DetId, vector<int> >::iterator bit2Itor = EcalAllDeadChannelsBitMap.find( (*dit) );
           if( bit2Itor == EcalAllDeadChannelsBitMap.end() ){ towerTestCnt ++; continue; }
           if( towerTest >0 && bit2Itor->second.back() == towerTest ) continue;
           if( towerTest <0 && bit2Itor->second.back() >= abs(towerTest) ) continue;
           towerTestCnt ++;
        }
        if( towerTestCnt !=0 && debug_ && verbose_ >=2) std::cout<<"towerTestCnt : "<<towerTestCnt<<"  for towerTest : "<<towerTest<<std::endl;
// ^^^^=                  END                 =^^^^

        EcalScDetId sc( (det.ix()-1)/5+1, (det.iy()-1)/5+1, det.zside() );

        std::vector<DetId>::iterator avoidItor; avoidItor = find( avoidDuplicateVec.begin(), avoidDuplicateVec.end(), det);
        if( avoidItor == avoidDuplicateVec.end() ){
           avoidDuplicateVec.push_back(det);
        }else{
           continue;
        }

        std::map<EcalScDetId, double>::iterator scetItor = accuSCetMap.find(sc);
        if( scetItor == accuSCetMap.end() ){
           accuSCetMap[sc] = eerechit->energy()*sin(theta);
           accuSCchnMap[sc] = 1;
           SCzsideMap[sc] = sc.zside();
        }else{
           accuSCetMap[sc] += eerechit->energy()*sin(theta);
           accuSCchnMap[sc] ++;
        }
     }
  } // loop over EE

// Checking for EB
  std::map<EcalTrigTowerDetId, double>::iterator ttetItor;
  for( ttetItor = accuTTetMap.begin(); ttetItor != accuTTetMap.end(); ttetItor++){

     EcalTrigTowerDetId ttDetId = ttetItor->first;

     double ttetVal = ttetItor->second;

     std::map<EcalTrigTowerDetId, int>::iterator ttchnItor = accuTTchnMap.find(ttDetId);
     if( ttchnItor == accuTTchnMap.end() ){ cout<<"\nERROR  cannot find ttDetId : "<<ttDetId<<" in accuTTchnMap?!"<<endl<<endl; }

     std::map<EcalTrigTowerDetId, int>::iterator ttzsideItor = TTzsideMap.find(ttDetId);
     if( ttzsideItor == TTzsideMap.end() ){ cout<<"\nERROR  cannot find ttDetId : "<<ttDetId<<" in TTzsideMap?!"<<endl<<endl; }

     if( ttchnItor->second != 25 && debug_ && verbose_ >=2) cout<<"WARNING ... ttchnCnt : "<<ttchnItor->second<<"  NOT equal  25!"<<endl;

     if( ttetVal >= tpValCut ){ isPassCut = 1; isPassCut *= ttzsideItor->second; }

  }

// Checking for EE
  std::map<EcalScDetId, double>::iterator scetItor;
  for( scetItor = accuSCetMap.begin(); scetItor != accuSCetMap.end(); scetItor++){

     EcalScDetId scDetId = scetItor->first;

     double scetVal = scetItor->second;

     std::map<EcalScDetId, int>::iterator scchnItor = accuSCchnMap.find(scDetId);
     if( scchnItor == accuSCchnMap.end() ){ cout<<"\nERROR  cannot find scDetId : "<<scDetId<<" in accuSCchnMap?!"<<endl<<endl; }

     std::map<EcalScDetId, int>::iterator sczsideItor = SCzsideMap.find(scDetId);
     if( sczsideItor == SCzsideMap.end() ){ cout<<"\nERROR  cannot find scDetId : "<<scDetId<<" in SCzsideMap?!"<<endl<<endl; }

     if( scchnItor->second != 25 && debug_ && verbose_ >=2) cout<<"WARNING ... scchnCnt : "<<scchnItor->second<<"  NOT equal  25!"<<endl;

     if( scetVal >= tpValCut ){ isPassCut = 1; isPassCut *= sczsideItor->second; }

  }

  if( debug_ && verbose_ >=2) std::cout<<"***end setEvtTPstatusRecHits***"<<std::endl;

  return isPassCut;

}


int EcalDeadCellTriggerPrimitiveFilter::setEvtTPstatus(const double &tpValCut, const int &chnStatus){
 
  if( debug_ && verbose_ >=2) std::cout<<"***begin setEvtTPstatus***"<<std::endl;

  int isPassCut =0;

  std::map<DetId, std::vector<int> >::iterator bitItor;
  for(bitItor = EcalAllDeadChannelsBitMap.begin(); bitItor != EcalAllDeadChannelsBitMap.end(); bitItor++){
        
     DetId maskedDetId = bitItor->first;
     int subdet = bitItor->second.front(), status = bitItor->second.back();

// if NOT filtering on EE, skip EE subdet
     if( !doEEfilter_ && subdet != 1 ) continue;
           
     std::map<DetId, EcalTrigTowerDetId>::iterator ttItor = EcalAllDeadChannelsTTMap.find(maskedDetId);
     if( ttItor == EcalAllDeadChannelsTTMap.end() ) continue;

     bool toDo = false;
     if( chnStatus >0 && status == chnStatus ) toDo = true;
     if( chnStatus <0 && status >= abs(chnStatus) ) toDo = true;

     if( toDo ){

        EcalTrigTowerDetId ttDetId = ttItor->second;
        int ttzside = ttDetId.zside();

        const EcalTrigPrimDigiCollection * tpDigis = 0;
        tpDigis = pTPDigis.product();
        EcalTrigPrimDigiCollection::const_iterator tp = tpDigis->find( ttDetId );
        if( tp != tpDigis->end() ){
           double tpEt = ecalScale_.getTPGInGeV( tp->compressedEt(), tp->id() );
           if(tpEt >= tpValCut ){ isPassCut = 1; isPassCut *= ttzside; }
        }
     }
  } // loop over EB + EE

  if( debug_ && verbose_ >=2) std::cout<<"***end setEvtTPstatus***"<<std::endl;

  return isPassCut;
}


int EcalDeadCellTriggerPrimitiveFilter::getChannelStatusMaps(){

  EcalAllDeadChannelsValMap.clear(); EcalAllDeadChannelsBitMap.clear();

// Loop over EB ...
  for( int ieta=-85; ieta<=85; ieta++ ){
     for( int iphi=0; iphi<=360; iphi++ ){
        if(! EBDetId::validDetId( ieta, iphi ) )  continue;
            
        const EBDetId detid = EBDetId( ieta, iphi, EBDetId::ETAPHIMODE );
        EcalChannelStatus::const_iterator chit = ecalStatus->find( detid );
// refer https://twiki.cern.ch/twiki/bin/viewauth/CMS/EcalChannelStatus
        int status = ( chit != ecalStatus->end() ) ? chit->getStatusCode() & 0x1F : -1; 

        const CaloSubdetectorGeometry*  subGeom = geometry->getSubdetectorGeometry (detid);
        const CaloCellGeometry*        cellGeom = subGeom->getGeometry (detid);
        double eta = cellGeom->getPosition ().eta ();
        double phi = cellGeom->getPosition ().phi ();
        double theta = cellGeom->getPosition().theta();

        if(status >= maskedEcalChannelStatusThreshold_){
           std::vector<double> valVec; std::vector<int> bitVec;
           valVec.push_back(eta); valVec.push_back(phi); valVec.push_back(theta);
           bitVec.push_back(1); bitVec.push_back(ieta); bitVec.push_back(iphi); bitVec.push_back(status);
           EcalAllDeadChannelsValMap.insert( std::make_pair(detid, valVec) );
           EcalAllDeadChannelsBitMap.insert( std::make_pair(detid, bitVec) );
        }
     } // end loop iphi
  } // end loop ieta

// Loop over EE detid
  for( int ix=0; ix<=100; ix++ ){
     for( int iy=0; iy<=100; iy++ ){
        for( int iz=-1; iz<=1; iz++ ){
           if(iz==0)  continue;
           if(! EEDetId::validDetId( ix, iy, iz ) )  continue;

           const EEDetId detid = EEDetId( ix, iy, iz, EEDetId::XYMODE );
           EcalChannelStatus::const_iterator chit = ecalStatus->find( detid );
           int status = ( chit != ecalStatus->end() ) ? chit->getStatusCode() & 0x1F : -1;

           const CaloSubdetectorGeometry*  subGeom = geometry->getSubdetectorGeometry (detid);
           const CaloCellGeometry*        cellGeom = subGeom->getGeometry (detid);
           double eta = cellGeom->getPosition ().eta () ;
           double phi = cellGeom->getPosition ().phi () ;
           double theta = cellGeom->getPosition().theta();

           if(status >= maskedEcalChannelStatusThreshold_){
              std::vector<double> valVec; std::vector<int> bitVec;
              valVec.push_back(eta); valVec.push_back(phi); valVec.push_back(theta);
              bitVec.push_back(2); bitVec.push_back(ix); bitVec.push_back(iy); bitVec.push_back(iz); bitVec.push_back(status);
              EcalAllDeadChannelsValMap.insert( std::make_pair(detid, valVec) );
              EcalAllDeadChannelsBitMap.insert( std::make_pair(detid, bitVec) );
           }
        } // end loop iz
     } // end loop iy
  } // end loop ix

  EcalAllDeadChannelsTTMap.clear();
  std::map<DetId, std::vector<int> >::iterator bitItor; 
  for(bitItor = EcalAllDeadChannelsBitMap.begin(); bitItor != EcalAllDeadChannelsBitMap.end(); bitItor++){
     const DetId id = bitItor->first;
     EcalTrigTowerDetId ttDetId = ttMap_->towerOf(id);
     EcalAllDeadChannelsTTMap.insert(std::make_pair(id, ttDetId) );
  }

  return 1;
}


//define this as a plug-in
DEFINE_FWK_MODULE(EcalDeadCellTriggerPrimitiveFilter);
