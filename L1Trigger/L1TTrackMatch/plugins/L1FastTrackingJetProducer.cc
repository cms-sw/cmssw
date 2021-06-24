///////////////////////////////////////////////////////////////////////////
//
// Producer of L1FastTrackingJets                                               
//
// FastTracking Jets: Jets created by running the FastJet clustering algorithm on L1 tracks that have been matched to a Tracking Particle
// Author: G.Karathanasis , CU Boulder 
///////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// L1 objects
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"


// geometry
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

//mc 
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <string>
#include "TMath.h"
#include "TH1.h"

using namespace l1t;
using namespace edm;
using namespace std;

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1FastTrackingJetProducer : public edm::stream::EDProducer<>
{
public:

  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

  explicit L1FastTrackingJetProducer(const edm::ParameterSet&);
  ~L1FastTrackingJetProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // track selection criteria
  float trkZMax_;            // in [cm]
  float trkChi2dofMax_;      // maximum track chi2dof
  double trkBendChi2Max_;    // maximum track bendchi2
  float trkPtMin_;           // in [GeV]
  float trkEtaMax_;          // in [rad]
  int trkNStubMin_; // minimum number of stubs
  int trkNPSStubMin_;        // minimum number of PS stubs
  double deltaZ0Cut_;         // save with |L1z-z0| < maxZ0
  double coneSize_;           // Use anti-kt with this cone size
  bool doTightChi2_;
  bool displaced_;            //use prompt/displaced tracks
  bool selectTrkMatchGenTight_;
  bool selectTrkMatchGenLoose_;
  bool selectTrkMatchGenOrPU_;

  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken_;
  edm::EDGetTokenT<std::vector<l1t::Vertex>> pvToken_;
  const edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > genToken_;
  edm::ESGetToken<TrackerTopology,TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
};

// constructor
L1FastTrackingJetProducer::L1FastTrackingJetProducer(const edm::ParameterSet& iConfig) :
trackToken_(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
pvToken_(consumes<std::vector<l1t::Vertex>>(iConfig.getParameter<edm::InputTag>("L1PrimaryVertexTag"))),
genToken_(consumes< TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > (iConfig.getParameter<edm::InputTag>("GenInfo"))),
tTopoToken_(esConsumes<TrackerTopology,TrackerTopologyRcd>(edm::ESInputTag("",""))),
tGeomToken_(esConsumes<TrackerGeometry,TrackerDigiGeometryRecord>(edm::ESInputTag("","")))
{
  trkZMax_    = (float)iConfig.getParameter<double>("trk_zMax");
  trkChi2dofMax_ = (float)iConfig.getParameter<double>("trk_chi2dofMax");
  trkBendChi2Max_ = iConfig.getParameter<double>("trk_bendChi2Max");
  trkPtMin_   = (float)iConfig.getParameter<double>("trk_ptMin");
  trkEtaMax_  = (float)iConfig.getParameter<double>("trk_etaMax");
  trkNStubMin_   = (int)iConfig.getParameter<int>("trk_nStubMin");
  trkNPSStubMin_ = (int)iConfig.getParameter<int>("trk_nPSStubMin");
  deltaZ0Cut_ = (float)iConfig.getParameter<double>("deltaZ0Cut");
  coneSize_ = (float)iConfig.getParameter<double>("coneSize");
  doTightChi2_ = iConfig.getParameter<bool>("doTightChi2");
  displaced_ = iConfig.getParameter<bool>("displaced");
  selectTrkMatchGenTight_ = iConfig.getParameter<bool> ("selectTrkMatchGenTight");
  selectTrkMatchGenLoose_ = iConfig.getParameter<bool> ("selectTrkMatchGenLoose");
  selectTrkMatchGenOrPU_ = iConfig.getParameter<bool> ("selectTrkMatchGenOrPU");
  if (displaced_) produces<TkJetCollection>("L1FastTrackingJetsExtended");
  else produces<TkJetCollection>("L1FastTrackingJets");
}

// destructor
L1FastTrackingJetProducer::~L1FastTrackingJetProducer() { }

// producer
void L1FastTrackingJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<TkJetCollection> L1FastTrackingJets(new TkJetCollection);

  // L1 tracks
  edm::Handle< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);
  std::vector< TTTrack< Ref_Phase2TrackerDigi_ > >::const_iterator iterL1Track;

  // Gen
  edm::Handle< TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > MCTrkAssociation;
  if (!iEvent.isRealData())
      iEvent.getByToken(genToken_,MCTrkAssociation);

  // Tracker Topology
  const TrackerTopology &tTopo = iSetup.getData(tTopoToken_);
  const TrackerGeometry & tGeom = iSetup.getData(tGeomToken_);

  edm::Handle<std::vector<l1t::Vertex>> L1VertexHandle;
  iEvent.getByToken(pvToken_, L1VertexHandle);

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, coneSize_);
  std::vector<fastjet::PseudoJet>  JetInputs;

 

  float recoVtx = L1VertexHandle->begin()->z0();
  unsigned int this_l1track = 0;
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    this_l1track++;
    float trk_pt = iterL1Track->momentum().perp();
    float trk_z0 = iterL1Track->z0();
    float trk_chi2dof = iterL1Track->chi2Red();
    float trk_bendchi2 = iterL1Track->stubPtConsistency();
    std::vector< edm::Ref<edmNew::DetSetVector< TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > > theStubs = iterL1Track -> getStubRefs() ;
    int trk_nstub = (int)theStubs.size();

    if (fabs(trk_z0) > trkZMax_) continue;
    if (fabs(iterL1Track->momentum().eta()) > trkEtaMax_) continue;
    if (trk_pt < trkPtMin_) continue;
    if (trk_nstub < trkNStubMin_) continue;
    if (trk_chi2dof>trkChi2dofMax_) continue;
    if (trk_bendchi2 > trkBendChi2Max_) continue;
    if (doTightChi2_ && (trk_pt>20.0 && trk_chi2dof>5.0)) continue;

    int trk_nPS = 0;
    for (int istub=0; istub<trk_nstub; istub++) {
      DetId detId( theStubs.at(istub)->getDetId() );
      bool tmp_isPS = false;
      if (detId.det() == DetId::Detector::Tracker) {
        if (detId.subdetId() == StripSubdetector::TOB && tTopo.tobLayer(detId) <= 3)     tmp_isPS = true;
        else if (detId.subdetId() == StripSubdetector::TID && tTopo.tidRing(detId) <= 9) tmp_isPS = true;
      }
      if (tmp_isPS) trk_nPS++;
    }
    if (trk_nPS < trkNPSStubMin_) continue;
    if (fabs(recoVtx-trk_z0) > deltaZ0Cut_) continue;
    if (!iEvent.isRealData()){
       edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> trk_ptr(TTTrackHandle, this_l1track);
      
       if ( !(MCTrkAssociation->isGenuine(trk_ptr)) && selectTrkMatchGenTight_ )
          continue;
       if ( !(MCTrkAssociation->isLooselyGenuine(trk_ptr)
           || MCTrkAssociation->isGenuine(trk_ptr)) 
           && selectTrkMatchGenLoose_ )
          continue;
       if ( !(MCTrkAssociation->isLooselyGenuine(trk_ptr) 
           || MCTrkAssociation->isGenuine(trk_ptr) 
           || MCTrkAssociation->isCombinatoric(trk_ptr)) 
           && selectTrkMatchGenOrPU_ )
          continue;
    }

    fastjet::PseudoJet psuedoJet(iterL1Track->momentum().x(), iterL1Track->momentum().y(), iterL1Track->momentum().z(), iterL1Track->momentum().mag());
    JetInputs.push_back(psuedoJet);	// input tracks for clustering
    JetInputs.back().set_user_index(this_l1track-1); // save track index in the collection
  } // end loop over tracks

  fastjet::ClusterSequence cs(JetInputs,jet_def); // define the output jet collection
  std::vector<fastjet::PseudoJet> JetOutputs=fastjet::sorted_by_pt(cs.inclusive_jets(0)); // output jet collection, pT-ordered

  for (unsigned int ijet=0;ijet<JetOutputs.size();++ijet) {
    math::XYZTLorentzVector jetP4(JetOutputs[ijet].px(),JetOutputs[ijet].py(),JetOutputs[ijet].pz(),JetOutputs[ijet].modp());
    float sumpt = 0;
    float avgZ = 0;
    std::vector< edm::Ptr< L1TTTrackType > > L1TrackPtrs;
    std::vector<fastjet::PseudoJet> fjConstituents =fastjet::sorted_by_pt(cs.constituents(JetOutputs[ijet]));

    for (unsigned int i=0; i<fjConstituents.size(); ++i) {
      auto index = fjConstituents[i].user_index();
      edm::Ptr< L1TTTrackType > trkPtr(TTTrackHandle, index) ;
      L1TrackPtrs.push_back(trkPtr); // L1Tracks in the jet
      sumpt = sumpt+trkPtr->momentum().perp();
      avgZ = avgZ+trkPtr->momentum().perp()*trkPtr->z0();
    }
    avgZ = avgZ/sumpt;
    edm::Ref<JetBxCollection> jetRef;
    TkJet trkJet(jetP4, jetRef, L1TrackPtrs, avgZ);
    L1FastTrackingJets->push_back(trkJet);
  }//end loop over Jet Outputs

  if (displaced_) iEvent.put(std::move(L1FastTrackingJets), "L1FastTrackingJetsExtended");
  else iEvent.put(std::move(L1FastTrackingJets), "L1FastTrackingJets");
}

void L1FastTrackingJetProducer::beginJob() { }

void L1FastTrackingJetProducer::endJob() { }

void L1FastTrackingJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1FastTrackingJetProducer);
