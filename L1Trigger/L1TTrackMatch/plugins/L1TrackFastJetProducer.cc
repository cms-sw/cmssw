///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Producer of TkJet,                                                    //
// Cluster L1 tracks using fastjet                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

// geometry
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

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

class L1TrackFastJetProducer : public edm::EDProducer
{
public:

  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

  explicit L1TrackFastJetProducer(const edm::ParameterSet&);
  ~L1TrackFastJetProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // track selection criteria
  float trk_zMax;            // in [cm]
  float trk_chi2dofMax;      // maximum track chi2dof
  double trk_bendChi2Max;    // maximum track bendchi2
  float trk_ptMin;           // in [GeV]
  float trk_etaMax;          // in [rad]
  int trk_nStubMin; // minimum number of stubs
  int trk_nPSStubMin;        // minimum number of PS stubs
  double deltaZ0Cut;         // save with |L1z-z0| < maxZ0
  double coneSize;           // Use anti-kt with this cone size
  bool doTightChi2;
  bool displaced;            //use prompt/displaced tracks

  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
  edm::EDGetTokenT<TkPrimaryVertexCollection> PVertexToken;
};

// constructor
L1TrackFastJetProducer::L1TrackFastJetProducer(const edm::ParameterSet& iConfig) :
trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
PVertexToken(consumes<TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1PrimaryVertexTag")))
{
  trk_zMax    = (float)iConfig.getParameter<double>("trk_zMax");
  trk_chi2dofMax = (float)iConfig.getParameter<double>("trk_chi2dofMax");
  trk_bendChi2Max = iConfig.getParameter<double>("trk_bendChi2Max");
  trk_ptMin   = (float)iConfig.getParameter<double>("trk_ptMin");
  trk_etaMax  = (float)iConfig.getParameter<double>("trk_etaMax");
  trk_nStubMin   = (int)iConfig.getParameter<int>("trk_nStubMin");
  trk_nPSStubMin = (int)iConfig.getParameter<int>("trk_nPSStubMin");
  deltaZ0Cut = (float)iConfig.getParameter<double>("deltaZ0Cut");
  coneSize = (float)iConfig.getParameter<double>("coneSize");
  doTightChi2 = iConfig.getParameter<bool>("doTightChi2");
  displaced = iConfig.getParameter<bool>("displaced");
  if (displaced) produces<TkJetCollection>("L1TrackFastJetsExtended");
  else produces<TkJetCollection>("L1TrackFastJets");
}

// destructor
L1TrackFastJetProducer::~L1TrackFastJetProducer() { }

// producer
void L1TrackFastJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<TkJetCollection> L1TrackFastJets(new TkJetCollection);

  // L1 tracks
  edm::Handle< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > TTTrackHandle;
  iEvent.getByToken(trackToken, TTTrackHandle);
  std::vector< TTTrack< Ref_Phase2TrackerDigi_ > >::const_iterator iterL1Track;

  // Tracker Topology
  edm::ESHandle<TrackerTopology> tTopoHandle_;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* tTopo = tTopoHandle_.product();
  ESHandle<TrackerGeometry> tGeomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);

  edm::Handle<TkPrimaryVertexCollection>TkPrimaryVertexHandle;
  iEvent.getByToken(PVertexToken, TkPrimaryVertexHandle);

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, coneSize);
  std::vector<fastjet::PseudoJet>  JetInputs;

  float recoVtx = TkPrimaryVertexHandle->begin()->zvertex();
  unsigned int this_l1track = 0;
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    this_l1track++;
    float trk_pt = iterL1Track->momentum().perp();
    float trk_z0 = iterL1Track->z0();
    float trk_chi2dof = iterL1Track->chi2Red();
    float trk_bendchi2 = iterL1Track->stubPtConsistency();
    std::vector< edm::Ref<edmNew::DetSetVector< TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > > theStubs = iterL1Track -> getStubRefs() ;
    int trk_nstub = (int)theStubs.size();

    if (fabs(trk_z0) > trk_zMax) continue;
    if (fabs(iterL1Track->momentum().eta()) > trk_etaMax) continue;
    if (trk_pt < trk_ptMin) continue;
    if (trk_nstub < trk_nStubMin) continue;
    if (trk_chi2dof>trk_chi2dofMax) continue;
    if (trk_bendchi2 > trk_bendChi2Max) continue;
    if (doTightChi2 && (trk_pt>20.0 && trk_chi2dof>5.0)) continue;

    int trk_nPS = 0;
    for (int istub=0; istub<trk_nstub; istub++) {
      DetId detId( theStubs.at(istub)->getDetId() );
      bool tmp_isPS = false;
      if (detId.det() == DetId::Detector::Tracker) {
        if (detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3)     tmp_isPS = true;
        else if (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9) tmp_isPS = true;
      }
      if (tmp_isPS) trk_nPS++;
    }
    if (trk_nPS < trk_nPSStubMin) continue;
    if (fabs(recoVtx-trk_z0) > deltaZ0Cut) continue;

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
    L1TrackFastJets->push_back(trkJet);
  }//end loop over Jet Outputs

  if (displaced) iEvent.put(std::move(L1TrackFastJets), "L1TrackFastJetsExtended");
  else iEvent.put(std::move(L1TrackFastJets), "L1TrackFastJets");
}

void L1TrackFastJetProducer::beginJob() { }

void L1TrackFastJetProducer::endJob() { }

void L1TrackFastJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackFastJetProducer);
