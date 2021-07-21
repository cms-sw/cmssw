#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/Conversion.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"
#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <memory>
#include <string>
#include <vector>

namespace pat {

  class PATConversionProducer : public edm::global::EDProducer<> {
  public:
    explicit PATConversionProducer(const edm::ParameterSet &iConfig);
    ~PATConversionProducer() override;

    void produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  private:
    // configurables
    const edm::EDGetTokenT<edm::View<reco::GsfElectron> > electronToken_;
    const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
    const edm::EDGetTokenT<reco::ConversionCollection> conversionsToken_;
  };

}  // namespace pat

using namespace pat;
using namespace std;

PATConversionProducer::PATConversionProducer(const edm::ParameterSet &iConfig)
    : electronToken_(consumes<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electronSource"))),
      bsToken_(consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"))),
      conversionsToken_(consumes<reco::ConversionCollection>(edm::InputTag("allConversions"))) {
  // produces vector of muons
  produces<std::vector<Conversion> >();
}

PATConversionProducer::~PATConversionProducer() {}

void PATConversionProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  // Get the collection of electrons from the event
  edm::Handle<edm::View<reco::GsfElectron> > electrons;
  iEvent.getByToken(electronToken_, electrons);

  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(bsToken_, bsHandle);
  const reco::BeamSpot &beamspot = *bsHandle.product();

  // for conversion veto selection
  edm::Handle<reco::ConversionCollection> hConversions;
  iEvent.getByToken(conversionsToken_, hConversions);

  std::vector<Conversion> *patConversions = new std::vector<Conversion>();

  for (reco::ConversionCollection::const_iterator conv = hConversions->begin(); conv != hConversions->end(); ++conv) {
    reco::Vertex vtx = conv->conversionVertex();

    int index = 0;
    for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end();
         ++itElectron) {
      //find matched conversions with electron and save those conversions with matched electron index
      if (ConversionTools::matchesConversion(*itElectron, *conv)) {
        double vtxProb = TMath::Prob(vtx.chi2(), vtx.ndof());
        math::XYZVector mom(conv->refittedPairMomentum());
        double dbsx = vtx.x() - beamspot.position().x();
        double dbsy = vtx.y() - beamspot.position().y();
        double lxy = (mom.x() * dbsx + mom.y() * dbsy) / mom.rho();
        int nHitsMax = 0;

        for (std::vector<uint8_t>::const_iterator it = conv->nHitsBeforeVtx().begin();
             it != conv->nHitsBeforeVtx().end();
             ++it) {
          if ((*it) > nHitsMax)
            nHitsMax = (*it);
        }

        pat::Conversion anConversion(index);
        anConversion.setVtxProb(vtxProb);
        anConversion.setLxy(lxy);
        anConversion.setNHitsMax(nHitsMax);

        patConversions->push_back(anConversion);
        break;
      }
      index++;
    }
  }

  // add the electrons to the event output
  std::unique_ptr<std::vector<Conversion> > ptr(patConversions);
  iEvent.put(std::move(ptr));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATConversionProducer);
