///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Producer of TrackFastJet,                                             //
// Cluster L1 tracks using fastjet                                       //
//                                                                       //
// Updates: Claire Savard (claire.savard@colorado.edu), Nov. 2023        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>

// user include files
#include "DataFormats/Common/interface/RefVector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

#include <fastjet/JetDefinition.hh>

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

class L1TrackFastJetProducer : public edm::stream::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;
  typedef edm::RefVector<L1TTTrackCollectionType> L1TTTrackRefCollectionType;

  explicit L1TrackFastJetProducer(const edm::ParameterSet&);
  ~L1TrackFastJetProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // jet configurations
  const double coneSize_;  // Use anti-kt with this cone size
  const bool displaced_;   //use prompt/displaced tracks

  const EDGetTokenT<L1TTTrackRefCollectionType> trackToken_;
};

// constructor
L1TrackFastJetProducer::L1TrackFastJetProducer(const edm::ParameterSet& iConfig)
    : coneSize_((float)iConfig.getParameter<double>("coneSize")),
      displaced_(iConfig.getParameter<bool>("displaced")),
      trackToken_(consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<InputTag>("L1TrackInputTag"))) {
  if (displaced_)
    produces<TkJetCollection>("L1TrackFastJetsExtended");
  else
    produces<TkJetCollection>("L1TrackFastJets");
}

// destructor
L1TrackFastJetProducer::~L1TrackFastJetProducer() {}

// producer
void L1TrackFastJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<TkJetCollection> L1TrackFastJets(new TkJetCollection);

  // L1 tracks
  edm::Handle<L1TTTrackRefCollectionType> TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, coneSize_);
  std::vector<fastjet::PseudoJet> JetInputs;

  for (unsigned int this_l1track = 0; this_l1track < TTTrackHandle->size(); this_l1track++) {
    edm::Ptr<L1TTTrackType> iterL1Track(TTTrackHandle, this_l1track);

    fastjet::PseudoJet psuedoJet(iterL1Track->momentum().x(),
                                 iterL1Track->momentum().y(),
                                 iterL1Track->momentum().z(),
                                 iterL1Track->momentum().mag());
    JetInputs.push_back(psuedoJet);                 // input tracks for clustering
    JetInputs.back().set_user_index(this_l1track);  // save track index in the collection
  }                                                 // end loop over tracks

  fastjet::ClusterSequence cs(JetInputs, jet_def);  // define the output jet collection
  std::vector<fastjet::PseudoJet> JetOutputs =
      fastjet::sorted_by_pt(cs.inclusive_jets(0));  // output jet collection, pT-ordered

  for (unsigned int ijet = 0; ijet < JetOutputs.size(); ++ijet) {
    math::XYZTLorentzVector jetP4(
        JetOutputs[ijet].px(), JetOutputs[ijet].py(), JetOutputs[ijet].pz(), JetOutputs[ijet].modp());
    float sumpt = 0;
    float avgZ = 0;
    std::vector<edm::Ptr<L1TTTrackType> > L1TrackPtrs;
    std::vector<fastjet::PseudoJet> fjConstituents = fastjet::sorted_by_pt(cs.constituents(JetOutputs[ijet]));

    for (unsigned int i = 0; i < fjConstituents.size(); ++i) {
      auto index = fjConstituents[i].user_index();
      edm::Ptr<L1TTTrackType> trkPtr(TTTrackHandle, index);
      L1TrackPtrs.push_back(trkPtr);  // L1Tracks in the jet
      sumpt = sumpt + trkPtr->momentum().perp();
      avgZ = avgZ + trkPtr->momentum().perp() * trkPtr->z0();
    }
    avgZ = avgZ / sumpt;
    edm::Ref<JetBxCollection> jetRef;
    TkJet trkJet(jetP4, jetRef, L1TrackPtrs, avgZ);
    L1TrackFastJets->push_back(trkJet);
  }  //end loop over Jet Outputs

  if (displaced_)
    iEvent.put(std::move(L1TrackFastJets), "L1TrackFastJetsExtended");
  else
    iEvent.put(std::move(L1TrackFastJets), "L1TrackFastJets");
}

void L1TrackFastJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(
      "L1PVertexInputTag",
      edm::InputTag("l1tTrackVertexAssociationProducerForJets", "Level1TTTracksSelectedAssociated"));
  desc.add<double>("coneSize", 0.5);
  desc.add<bool>("displaced", false);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackFastJetProducer);
