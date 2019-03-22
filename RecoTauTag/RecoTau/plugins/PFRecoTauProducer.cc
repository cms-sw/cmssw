/* class PFRecoTauProducer
 * EDProducer of the PFTauCollection, starting from the PFTauTagInfoCollection, 
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 */

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"
#include "RecoTauTag/RecoTau/interface/HPSPFRecoTauAlgorithm.h"

#include "CLHEP/Random/RandGauss.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFRecoTauProducer : public EDProducer {
 public:
  explicit PFRecoTauProducer(const edm::ParameterSet& iConfig);
  ~PFRecoTauProducer() override;
  void produce(edm::Event&,const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
 private:
  edm::EDGetTokenT<PFTauTagInfoCollection> PFTauTagInfoProducer_;
  edm::InputTag ElectronPreIDProducer_;
  edm::EDGetTokenT<VertexCollection> PVProducer_;
  std::string Algorithm_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;
  double JetMinPt_;
  PFRecoTauAlgorithmBase* PFRecoTauAlgo_;
};

PFRecoTauProducer::PFRecoTauProducer(const edm::ParameterSet& iConfig){
  PFTauTagInfoProducer_   = consumes<PFTauTagInfoCollection>(iConfig.getParameter<edm::InputTag>("PFTauTagInfoProducer") );
  ElectronPreIDProducer_  = iConfig.getParameter<edm::InputTag>("ElectronPreIDProducer");
  PVProducer_             = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("PVProducer") );
  Algorithm_              = iConfig.getParameter<std::string>("Algorithm");
  smearedPVsigmaX_        = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_        = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_        = iConfig.getParameter<double>("smearedPVsigmaZ");	
  JetMinPt_               = iConfig.getParameter<double>("JetPtMin");

  if(Algorithm_ =="ConeBased") {
    PFRecoTauAlgo_=new PFRecoTauAlgorithm(iConfig);
  }
  else if(Algorithm_ =="HPS") {
    PFRecoTauAlgo_=new HPSPFRecoTauAlgorithm(iConfig);
  }
  else {    //Add inside out Algorithm here

    //If no Algorithm found throw exception
    throw cms::Exception("") << "Unknown Algorithkm" << std::endl;
  }
    

  produces<PFTauCollection>();      
}
PFRecoTauProducer::~PFRecoTauProducer(){
  delete PFRecoTauAlgo_;
}

void PFRecoTauProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  auto resultPFTau = std::make_unique<PFTauCollection>();
  
  edm::ESHandle<TransientTrackBuilder> myTransientTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",myTransientTrackBuilder);
  PFRecoTauAlgo_->setTransientTrackBuilder(myTransientTrackBuilder.product());

  //edm::ESHandle<MagneticField> myMF;
  //iSetup.get<IdealMagneticFieldRecord>().get(myMF);
  //PFRecoTauAlgo_->setMagneticField(myMF.product());

  // Electron PreID tracks: Temporary until integrated to PFCandidate
  /*
  edm::Handle<PFRecTrackCollection> myPFelecTk; 
  iEvent.getByLabel(ElectronPreIDProducer_,myPFelecTk); 
  const PFRecTrackCollection theElecTkCollection=*(myPFelecTk.product()); 
  */
  // query a rec/sim PV
  edm::Handle<VertexCollection> thePVs;
  iEvent.getByToken(PVProducer_,thePVs);
  const VertexCollection vertCollection=*(thePVs.product());
  Vertex thePV;
  if(!vertCollection.empty()) thePV=*(vertCollection.begin());
  else{
    Vertex::Error SimPVError;
    SimPVError(0,0)=smearedPVsigmaX_*smearedPVsigmaX_;
    SimPVError(1,1)=smearedPVsigmaY_*smearedPVsigmaY_;
    SimPVError(2,2)=smearedPVsigmaZ_*smearedPVsigmaZ_;
    Vertex::Point SimPVPoint(CLHEP::RandGauss::shoot(0.,smearedPVsigmaX_),  
			     CLHEP::RandGauss::shoot(0.,smearedPVsigmaY_),  
			     CLHEP::RandGauss::shoot(0.,smearedPVsigmaZ_));
    thePV=Vertex(SimPVPoint,SimPVError,1,1,1);    
  }
  
  edm::Handle<PFTauTagInfoCollection> thePFTauTagInfoCollection;
  iEvent.getByToken(PFTauTagInfoProducer_,thePFTauTagInfoCollection);
  int iinfo=0;
  for(PFTauTagInfoCollection::const_iterator i_info=thePFTauTagInfoCollection->begin();i_info!=thePFTauTagInfoCollection->end();i_info++) { 
    if((*i_info).pfjetRef()->pt()>JetMinPt_){
      //        PFTau myPFTau=PFRecoTauAlgo_->buildPFTau(Ref<PFTauTagInfoCollection>(thePFTauTagInfoCollection,iinfo),thePV,theElecTkCollection);
        PFTau myPFTau=PFRecoTauAlgo_->buildPFTau(Ref<PFTauTagInfoCollection>(thePFTauTagInfoCollection,iinfo),thePV);
       resultPFTau->push_back(myPFTau);
    }
    ++iinfo;
  }
  iEvent.put(std::move(resultPFTau));
}

void
PFRecoTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauProducer
  edm::ParameterSetDescription desc;
  desc.add<double>("Rphi", 2.0);
  desc.add<double>("LeadTrack_minPt", 0.0);
  desc.add<edm::InputTag>("PVProducer", edm::InputTag("offlinePrimaryVertices"));
  desc.add<std::string>("ECALSignalConeSizeFormula", "0.15");
  desc.add<std::string>("TrackerIsolConeMetric", "DR");
  desc.add<std::string>("TrackerSignalConeMetric", "DR");
  desc.add<double>("EcalStripSumE_deltaPhiOverQ_minValue", -0.1);
  desc.add<double>("smearedPVsigmaX", 0.0015);
  desc.add<double>("smearedPVsigmaY", 0.0015);
  desc.add<std::string>("MatchingConeMetric", "DR");
  desc.add<std::string>("TrackerSignalConeSizeFormula", "0.07");
  desc.add<std::string>("MatchingConeSizeFormula", "0.1");
  desc.add<double>("TrackerIsolConeSize_min", 0.0);
  desc.add<double>("MatchingConeSize_min", 0.0);
  desc.add<edm::InputTag>("ElectronPreIDProducer", edm::InputTag("elecpreid"));
  desc.add<double>("ChargedHadrCandLeadChargedHadrCand_tksmaxDZ", 1.0);
  desc.add<double>("TrackerIsolConeSize_max", 0.6);
  desc.add<double>("TrackerSignalConeSize_max", 0.07);
  desc.add<std::string>("HCALIsolConeMetric", "DR");
  desc.add<bool>("AddEllipseGammas", false);
  desc.add<double>("maximumForElectrionPreIDOutput", -0.1);
  desc.add<double>("TrackerSignalConeSize_min", 0.0);
  desc.add<double>("JetPtMin", 0.0);
  desc.add<std::string>("HCALIsolConeSizeFormula", "0.50");
  desc.add<double>("AreaMetric_recoElements_maxabsEta", 2.5);
  desc.add<double>("HCALIsolConeSize_max", 0.6);
  desc.add<unsigned int>("Track_IsolAnnulus_minNhits", 3);
  desc.add<std::string>("HCALSignalConeMetric", "DR");
  desc.add<double>("ElecPreIDLeadTkMatch_maxDR", 0.01);
  desc.add<edm::InputTag>("PFTauTagInfoProducer", edm::InputTag("pfRecoTauTagInfoProducer"));
  desc.add<std::string>("ECALIsolConeMetric", "DR");
  desc.add<std::string>("ECALIsolConeSizeFormula", "0.50");
  desc.add<bool>("UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint", true);
  desc.add<std::string>("Algorithm", "ConeBased");
  desc.add<double>("ECALIsolConeSize_max", 0.6);
  desc.add<std::string>("ECALSignalConeMetric", "DR");
  desc.add<double>("EcalStripSumE_deltaPhiOverQ_maxValue", 0.5);
  desc.add<double>("HCALSignalConeSize_max", 0.6);
  desc.add<double>("ECALSignalConeSize_min", 0.0);
  desc.add<double>("EcalStripSumE_minClusEnergy", 0.1);
  desc.add<double>("EcalStripSumE_deltaEta", 0.03);
  desc.add<std::string>("TrackerIsolConeSizeFormula", "0.50");
  desc.add<double>("LeadPFCand_minPt", 5.0);
  desc.add<double>("HCALSignalConeSize_min", 0.0);
  desc.add<double>("ECALSignalConeSize_max", 0.6);
  desc.add<std::string>("HCALSignalConeSizeFormula", "0.10");
  desc.add<bool>("putNeutralHadronsInP4", false);
  desc.add<double>("TrackLeadTrack_maxDZ", 1.0);
  desc.add<unsigned int>("ChargedHadrCand_IsolAnnulus_minNhits", 0);
  desc.add<double>("ECALIsolConeSize_min", 0.0);
  desc.add<bool>("UseTrackLeadTrackDZconstraint", true);
  desc.add<double>("MaxEtInEllipse", 2.0);
  desc.add<std::string>("DataType", "AOD");
  desc.add<double>("smearedPVsigmaZ", 0.005);
  desc.add<double>("MatchingConeSize_max", 0.6);
  desc.add<double>("HCALIsolConeSize_min", 0.0);
  descriptions.add("pfRecoTauProducer", desc);
}

DEFINE_FWK_MODULE(PFRecoTauProducer);
