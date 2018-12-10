// -*- C++ -*-
//
// Package:    EcalDeadCellDeltaRFilter
// Class:      EcalDeadCellDeltaRFilter
//
/**\class EcalDeadCellDeltaRFilter EcalDeadCellDeltaRFilter.cc

 Description: <one line class summary>
 Event filtering for RA2 analysis (filtering status is stored in the event)
*/
//
// Original Author:  Hongxuan Liu
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// XXX: Must BEFORE Frameworkfwd.h
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/strbitset.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Math/interface/deltaR.h"

// HCAL
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1.h"

class EcalDeadCellDeltaRFilter : public edm::EDFilter {
public:
  explicit EcalDeadCellDeltaRFilter(const edm::ParameterSet&);
  ~EcalDeadCellDeltaRFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, 
                            const edm::EventSetup&) override; 
  virtual void envSet(const edm::EventSetup&);

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
  edm::Handle<edm::View<reco::Jet> > jets;
// jet selection cut: pt, eta
// default (pt=-1, eta= 9999) means no cut
  const std::vector<double> jetSelCuts_;

  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
  edm::Handle<edm::View<reco::MET> > met;

  const bool debug_, printSkimInfo_;

  bool isPrintedOnce;

  void loadEventInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void loadJets(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void loadMET(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  edm::RunNumber_t run;
  edm::EventNumber_t event;
  edm::LuminosityBlockNumber_t ls;
  bool isdata;

  double calomet, calometPhi, tcmet, tcmetPhi, pfmet, pfmetPhi;

// Channel status related
  edm::ESHandle<EcalChannelStatus>  ecalStatus; // these come from EventSetup
  edm::ESHandle<HcalChannelQuality> hcalStatus; // these may come per LS
  edm::ESHandle<CaloGeometry>       geometry;

  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;

  EcalTPGScale ecalScale_;

  const int maskedEcalChannelStatusThreshold_;
  const int chnStatusToBeEvaluated_;

// XXX: All the following can be built at the beginning of a run
// Store DetId <==> std::vector<double> (eta, phi, theta)
  std::map<DetId, std::vector<double> > EcalAllDeadChannelsValMap;
// Store EB: DetId <==> std::vector<int> (subdet, ieta, iphi, status)
// Store EE: DetId <==> std::vector<int> (subdet, ix, iy, iz, status)
  std::map<DetId, std::vector<int> >    EcalAllDeadChannelsBitMap;

// Store DetId <==> EcalTrigTowerDetId
  std::map<DetId, EcalTrigTowerDetId> EcalAllDeadChannelsTTMap;

  int getChannelStatusMaps();

  int evtProcessedCnt, totTPFilteredCnt;
  double wtdEvtProcessed, wtdTPFiltered;

  const bool makeProfileRoot_;
  const std::string profileRootName_;
  TFile *profFile;
  TH1F *h1_dummy;

  const bool isProd_;
  const int verbose_;

  const bool doCracks_;
// Cracks definition
  const std::vector<double> cracksHBHEdef_, cracksHEHFdef_;

// Simple dR filter
  const std::vector<double> EcalDeadCellDeltaRFilterInput_;

  int dPhiToMETfunc(const std::vector<reco::Jet> &jetTVec, const double &dPhiCutVal, std::vector<reco::Jet> &closeToMETjetsVec);
  int dRtoMaskedChnsEvtFilterFunc(const std::vector<reco::Jet> &jetTVec, const int &chnStatus, const double &dRCutVal);

  int etaToBoundary(const std::vector<reco::Jet> &jetTVec);

  int isCloseToBadEcalChannel(const reco::Jet &jet, const double &deltaRCut, const int &chnStatus, std::map<double, DetId> &deltaRdetIdMap);

  const bool taggingMode_;
};

void EcalDeadCellDeltaRFilter::loadMET(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  iEvent.getByToken(metToken_, met);

}

void EcalDeadCellDeltaRFilter::loadEventInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup){
   run = iEvent.id().run();
   event = iEvent.id().event();
   ls = iEvent.luminosityBlock();
   isdata = iEvent.isRealData();

   if( !isPrintedOnce ){
      if( debug_ ){
         if( isdata ) std::cout<<"\nInput dataset is DATA"<<std::endl<<std::endl;
         else std::cout<<"\nInput dataset is MC"<<std::endl<<std::endl;
      }
      isPrintedOnce = true;
   }

}

void EcalDeadCellDeltaRFilter::loadJets(const edm::Event& iEvent, const edm::EventSetup& iSetup ){

  iEvent.getByToken(jetToken_, jets);

}

//
// static data member definitions
//

//
// constructors and destructor
//
EcalDeadCellDeltaRFilter::EcalDeadCellDeltaRFilter(const edm::ParameterSet& iConfig)
  : jetToken_ (consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("jetInputTag")))
  , jetSelCuts_ (iConfig.getParameter<std::vector<double> >("jetSelCuts"))

  , metToken_ (consumes<edm::View<reco::MET> >(iConfig.getParameter<edm::InputTag>("metInputTag")))

  , debug_ (iConfig.getUntrackedParameter<bool>("debug",false))
  , printSkimInfo_ (iConfig.getUntrackedParameter<bool>("printSkimInfo",false))

  , maskedEcalChannelStatusThreshold_ (iConfig.getParameter<int>("maskedEcalChannelStatusThreshold"))
  , chnStatusToBeEvaluated_ (iConfig.getParameter<int>("chnStatusToBeEvaluated"))

  , makeProfileRoot_ (iConfig.getUntrackedParameter<bool>("makeProfileRoot", true))
  , profileRootName_ (iConfig.getUntrackedParameter<std::string>("profileRootName", "EcalDeadCellDeltaRFilter.root"))

  , isProd_ (iConfig.getUntrackedParameter<bool>("isProd"))
  , verbose_ (iConfig.getParameter<int>("verbose"))

  , doCracks_ (iConfig.getUntrackedParameter<bool>("doCracks"))
  , cracksHBHEdef_ (iConfig.getParameter<std::vector<double> > ("cracksHBHEdef"))
  , cracksHEHFdef_ (iConfig.getParameter<std::vector<double> > ("cracksHEHFdef"))

  , EcalDeadCellDeltaRFilterInput_ (iConfig.getParameter<std::vector<double> >("EcalDeadCellDeltaRFilterInput"))

  , taggingMode_ (iConfig.getParameter<bool>("taggingMode"))
{
  produces<int> ("deadCellStatus"); produces<int> ("boundaryStatus");
  produces<bool>();

  if( makeProfileRoot_ ){
     profFile = new TFile(profileRootName_.c_str(), "RECREATE");
     h1_dummy = new TH1F("dummy", "dummy", 500, 0, 500);
  }
}

EcalDeadCellDeltaRFilter::~EcalDeadCellDeltaRFilter() {
  if( makeProfileRoot_ ){
     profFile->cd();

     h1_dummy->Write();

     profFile->Close();
     delete profFile;
  }
}

void EcalDeadCellDeltaRFilter::envSet(const edm::EventSetup& iSetup) {

  if (debug_) std::cout << "***envSet***" << std::endl;

  ecalScale_.setEventSetup( iSetup );
  iSetup.get<IdealGeometryRecord>().get(ttMap_);

  iSetup.get<EcalChannelStatusRcd> ().get(ecalStatus);
  iSetup.get<CaloGeometryRecord>   ().get(geometry);

  if( !ecalStatus.isValid() )  throw "Failed to get ECAL channel status!";
  if( !geometry.isValid()   )  throw "Failed to get the geometry!";

}

// ------------ method called on each new Event  ------------
bool EcalDeadCellDeltaRFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  loadEventInfo(iEvent, iSetup);
  loadJets(iEvent, iSetup);
  loadMET(iEvent, iSetup);

// XXX: In the following, never assign pass to true again
// Currently, always true
  bool pass = true;

  using namespace edm;

  std::vector<reco::Jet> seledJets;

  for( edm::View<reco::Jet>::const_iterator ij = jets->begin(); ij != jets->end(); ij++){
     if( ij->pt() > jetSelCuts_[0] && std::abs(ij->eta()) < jetSelCuts_[1] ){
        seledJets.push_back(reco::Jet(*ij));
     }
  }

  if( seledJets.empty() ) return pass;

  double dPhiToMET = EcalDeadCellDeltaRFilterInput_[0], dRtoDeadCell = EcalDeadCellDeltaRFilterInput_[1];

  std::vector<reco::Jet> closeToMETjetsVec;

  int dPhiToMETstatus = dPhiToMETfunc(seledJets, dPhiToMET, closeToMETjetsVec);

// Get event filter for simple dR cut
  int deadCellStatus = dRtoMaskedChnsEvtFilterFunc(closeToMETjetsVec, chnStatusToBeEvaluated_, dRtoDeadCell);

  int boundaryStatus = etaToBoundary(closeToMETjetsVec);

  if(debug_ ){
     printf("\nrun : %8u  event : %12llu  ls : %8u  dPhiToMETstatus : %d  deadCellStatus : %d  boundaryStatus : %d\n", run, event, ls, dPhiToMETstatus, deadCellStatus, boundaryStatus);
     printf("met : %6.2f  metphi : % 6.3f  dPhiToMET : %5.3f  dRtoDeadCell : %5.3f\n", (*met)[0].pt(), (*met)[0].phi(), dPhiToMET, dRtoDeadCell);
  }

  if( makeProfileRoot_ ){
//     h1_dummy->Fill(xxx);
  }

  iEvent.put(std::make_unique<int>(deadCellStatus), "deadCellStatus");
  iEvent.put(std::make_unique<int>(boundaryStatus), "boundaryStatus");

  if( deadCellStatus || (doCracks_ && boundaryStatus) ) pass = false;

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass;
}

// ------------ method called once each job just before starting event loop  ------------
void EcalDeadCellDeltaRFilter::beginJob() {
  if (debug_) std::cout << "beginJob" << std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void EcalDeadCellDeltaRFilter::endJob() {
  if (debug_) std::cout << "endJob" << std::endl;
}

// ------------ method called once each run just before starting event loop  ------------
void EcalDeadCellDeltaRFilter::beginRun(const edm::Run &run, const edm::EventSetup& iSetup) {
  if (debug_) std::cout << "beginRun" << std::endl;
// Channel status might change for each run (data)
// Event setup
  envSet(iSetup);
  getChannelStatusMaps();
  if( debug_) std::cout<< "EcalAllDeadChannelsValMap.size() : "<<EcalAllDeadChannelsValMap.size()<<"  EcalAllDeadChannelsBitMap.size() : "<<EcalAllDeadChannelsBitMap.size()<<std::endl;
  return;
}

// ------------ method called once each run just after starting event loop  ------------
void EcalDeadCellDeltaRFilter::endRun(const edm::Run &run, const edm::EventSetup& iSetup) {
  if (debug_) std::cout << "endRun" << std::endl;
  return;
}

// ------------ method called at lumi block start
void EcalDeadCellDeltaRFilter::beginLuminosityBlock(const edm::LuminosityBlock& iLSblock, const edm::EventSetup& iSetup) {

  // needs per-LS access, if used at all
  iSetup.get<HcalChannelQualityRcd>().get("withTopo",hcalStatus);
  if( !hcalStatus.isValid() )  throw "Failed to get HCAL channel status!";
  return; 
}

int EcalDeadCellDeltaRFilter::etaToBoundary(const std::vector<reco::Jet> &jetTVec){

  int isClose = 0;

  int cntOrder10 = 0;
  for(unsigned int ij=0; ij<jetTVec.size(); ij++){

     double recoJetEta = jetTVec[ij].eta();

     if( std::abs(recoJetEta)>cracksHBHEdef_[0] && std::abs(recoJetEta)<cracksHBHEdef_[1] ) isClose += (cntOrder10*10 + 1);
     if( std::abs(recoJetEta)>cracksHEHFdef_[0] && std::abs(recoJetEta)<cracksHEHFdef_[1] ) isClose += (cntOrder10*10 + 2);

     if( isClose/pow(10, cntOrder10) >=3 ) cntOrder10 = isClose/10 + 1;
  }

  return isClose;

}


// Cache all jets that are close to the MET within a dphi of dPhiCutVal
int EcalDeadCellDeltaRFilter::dPhiToMETfunc(const std::vector<reco::Jet> &jetTVec, const double &dPhiCutVal, std::vector<reco::Jet> &closeToMETjetsVec){

  closeToMETjetsVec.clear();

  double minDphi = 999.0;
  int minIdx = -1;
  for(unsigned int ii=0; ii<jetTVec.size(); ii++){

     const reco::Jet& jet = jetTVec[ii];

     double deltaPhi = std::abs(reco::deltaPhi( jet.phi(), (*met)[0].phi() ) );
     if( deltaPhi > dPhiCutVal ) continue;

     closeToMETjetsVec.push_back(jetTVec[ii]);

     if( deltaPhi < minDphi ){
        minDphi = deltaPhi;
        minIdx = ii;
     }
  }

  if( minIdx == -1 ){} // removing a stupid compiling WARNING that minIdx NOT used.
//  if( minIdx == -1 ) return 0;
//  closeToMETjetsVec.push_back(jetTVec[minIdx]);

  return (int)closeToMETjetsVec.size();
}


int EcalDeadCellDeltaRFilter::dRtoMaskedChnsEvtFilterFunc(const std::vector<reco::Jet> &jetTVec, const int &chnStatus, const double &dRCutVal){

  int isClose = 0;

  for(unsigned int ii=0; ii<jetTVec.size(); ii++){

     const reco::Jet& jet = jetTVec[ii];

     std::map<double, DetId> dummy;
     int isPerJetClose = isCloseToBadEcalChannel(jet, dRCutVal, chnStatus, dummy);
//     if( isPerJetClose ){ isClose = 1; break; }
     if( isPerJetClose ){ isClose ++; }
  }

  return isClose;

}


int EcalDeadCellDeltaRFilter::isCloseToBadEcalChannel(const reco::Jet &jet, const double &deltaRCut, const int &chnStatus, std::map<double, DetId> &deltaRdetIdMap){

   double jetEta = jet.eta(), jetPhi = jet.phi();

   deltaRdetIdMap.clear();

   double min_dist = 999;
   DetId min_detId;

   std::map<DetId, std::vector<int> >::iterator bitItor;
   for(bitItor = EcalAllDeadChannelsBitMap.begin(); bitItor != EcalAllDeadChannelsBitMap.end(); bitItor++){

      DetId maskedDetId = bitItor->first;
//      int subdet = bitItor->second.front();
      int status = bitItor->second.back();

      if( chnStatus >0 && status != chnStatus ) continue;
      if( chnStatus <0 && status < abs(chnStatus) ) continue;

      std::map<DetId, std::vector<double> >::iterator valItor = EcalAllDeadChannelsValMap.find(maskedDetId);
      if( valItor == EcalAllDeadChannelsValMap.end() ){ std::cout<<"Error cannot find maskedDetId in EcalAllDeadChannelsValMap ?!"<<std::endl; continue; }

      double eta = (valItor->second)[0], phi = (valItor->second)[1];

      double dist = reco::deltaR(eta, phi, jetEta, jetPhi);

      if( min_dist > dist ){ min_dist = dist; min_detId = maskedDetId; }
   }

   if( min_dist > deltaRCut && deltaRCut >0 ) return 0;

   deltaRdetIdMap.insert( std::make_pair(min_dist, min_detId) );

   return 1;
}


int EcalDeadCellDeltaRFilter::getChannelStatusMaps(){

  EcalAllDeadChannelsValMap.clear(); EcalAllDeadChannelsBitMap.clear();

// Loop over EB ...
  for( int ieta=-85; ieta<=85; ieta++ ){
     for( int iphi=0; iphi<=360; iphi++ ){
        if(! EBDetId::validDetId( ieta, iphi ) )  continue;

        const EBDetId detid = EBDetId( ieta, iphi, EBDetId::ETAPHIMODE );
        EcalChannelStatus::const_iterator chit = ecalStatus->find( detid );
// refer https://twiki.cern.ch/twiki/bin/viewauth/CMS/EcalChannelStatus
        int status = ( chit != ecalStatus->end() ) ? chit->getStatusCode() & 0x1F : -1;

	const CaloSubdetectorGeometry* subGeom = geometry->getSubdetectorGeometry (detid);
        auto cellGeom = subGeom->getGeometry (detid);
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

           const CaloSubdetectorGeometry* subGeom = geometry->getSubdetectorGeometry (detid);
           auto cellGeom = subGeom->getGeometry (detid);
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
DEFINE_FWK_MODULE(EcalDeadCellDeltaRFilter);
