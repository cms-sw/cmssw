#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

// Vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

// Transient Track and IP
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include <cmath>
#include <vector>

// SoftPFElectronTagInfoProducer:  the SoftPFElectronTagInfoProducer takes
// a PFCandidateCollection as input and produces a RefVector
// to the likely soft electrons in this collection.

class SoftPFElectronTagInfoProducer : public edm::global::EDProducer<> {
public:
  SoftPFElectronTagInfoProducer(const edm::ParameterSet& conf);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const final;
  static bool isElecClean(edm::Event&, const reco::GsfElectron*);
  static float boostedPPar(const math::XYZVector&, const math::XYZVector&);

  // service used to make transient tracks from tracks
  const edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
  const edm::EDGetTokenT<edm::View<reco::Jet> > token_jets;
  const edm::EDGetTokenT<edm::View<reco::GsfElectron> > token_elec;
  const edm::EDGetTokenT<reco::BeamSpot> token_BeamSpot;
  const edm::EDGetTokenT<reco::ConversionCollection> token_allConversions;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> token_builder;
  const edm::EDPutTokenT<reco::CandSoftLeptonTagInfoCollection> token_put;
  const float DeltaRElectronJet, MaxSip3Dsig;
};

SoftPFElectronTagInfoProducer::SoftPFElectronTagInfoProducer(const edm::ParameterSet& conf)
    : token_primaryVertex(consumes(conf.getParameter<edm::InputTag>("primaryVertex"))),
      token_jets(consumes(conf.getParameter<edm::InputTag>("jets"))),
      token_elec(consumes(conf.getParameter<edm::InputTag>("electrons"))),
      token_BeamSpot(consumes(edm::InputTag("offlineBeamSpot"))),
      token_allConversions(consumes(edm::InputTag("allConversions"))),
      token_builder(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      token_put(produces()),
      DeltaRElectronJet(conf.getParameter<double>("DeltaRElectronJet")),
      MaxSip3Dsig(conf.getParameter<double>("MaxSip3Dsig")) {}

void SoftPFElectronTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("primaryVertex");
  desc.add<edm::InputTag>("jets");
  desc.add<edm::InputTag>("electrons");
  desc.add<double>("DeltaRElectronJet");
  desc.add<double>("MaxSip3Dsig");
  descriptions.addDefault(desc);
}

void SoftPFElectronTagInfoProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  reco::CandSoftLeptonTagInfoCollection theElecTagInfo;
  const auto& transientTrackBuilder = iSetup.getData(token_builder);

  edm::Handle<reco::VertexCollection> PVCollection = iEvent.getHandle(token_primaryVertex);
  if (!PVCollection.isValid()) {
    iEvent.emplace(token_put, std::move(theElecTagInfo));
    return;
  }
  reco::ConversionCollection const& hConversions = iEvent.get(token_allConversions);

  edm::View<reco::Jet> const& theJetCollection = iEvent.get(token_jets);

  edm::View<reco::GsfElectron> const& theGEDGsfElectronCollection = iEvent.get(token_elec);

  if (PVCollection->empty() and not theJetCollection.empty() and not theGEDGsfElectronCollection.empty()) {
    //we would need to access a vertex from the collection but there isn't one.
    iEvent.emplace(token_put, std::move(theElecTagInfo));
    return;
  }

  const reco::BeamSpot& beamspot = iEvent.get(token_BeamSpot);

  for (unsigned int i = 0; i < theJetCollection.size(); i++) {
    edm::RefToBase<reco::Jet> jetRef = theJetCollection.refAt(i);
    reco::CandSoftLeptonTagInfo tagInfo;
    tagInfo.setJetRef(jetRef);
    for (unsigned int ie = 0, ne = theGEDGsfElectronCollection.size(); ie < ne; ++ie) {
      //Get the edm::Ptr and the GsfElectron
      edm::Ptr<reco::Candidate> lepPtr = theGEDGsfElectronCollection.ptrAt(ie);
      const reco::GsfElectron* recoelectron = theGEDGsfElectronCollection.refAt(ie).get();
      const pat::Electron* patelec = dynamic_cast<const pat::Electron*>(recoelectron);
      if (patelec) {
        if (!patelec->passConversionVeto())
          continue;
      } else {
        if (ConversionTools::hasMatchedConversion(*(recoelectron), hConversions, beamspot.position()))
          continue;
      }
      //Make sure that the electron is inside the jet
      if (reco::deltaR2((*recoelectron), (*jetRef)) > DeltaRElectronJet * DeltaRElectronJet)
        continue;
      // Need a gsfTrack
      if (recoelectron->gsfTrack().get() == nullptr)
        continue;
      reco::SoftLeptonProperties properties;
      // reject if it has issues with the track
      if (!isElecClean(iEvent, recoelectron))
        continue;
      //Compute the TagInfos members
      math::XYZVector pel = recoelectron->p4().Vect();
      math::XYZVector pjet = jetRef->p4().Vect();
      reco::TransientTrack transientTrack = transientTrackBuilder.build(recoelectron->gsfTrack());
      auto const& vertex = PVCollection->front();
      Measurement1D ip2d = IPTools::signedTransverseImpactParameter(
                               transientTrack, GlobalVector(jetRef->px(), jetRef->py(), jetRef->pz()), vertex)
                               .second;
      Measurement1D ip3d = IPTools::signedImpactParameter3D(
                               transientTrack, GlobalVector(jetRef->px(), jetRef->py(), jetRef->pz()), vertex)
                               .second;
      properties.sip2dsig = ip2d.significance();
      properties.sip3dsig = ip3d.significance();
      properties.sip2d = ip2d.value();
      properties.sip3d = ip3d.value();
      properties.deltaR = reco::deltaR((*jetRef), (*recoelectron));
      properties.ptRel = ((pjet - pel).Cross(pel)).R() / pjet.R();
      float mag = pel.R() * pjet.R();
      float dot = recoelectron->p4().Dot(jetRef->p4());
      properties.etaRel = -log((mag - dot) / (mag + dot)) / 2.;
      properties.ratio = recoelectron->pt() / jetRef->pt();
      properties.ratioRel = recoelectron->p4().Dot(jetRef->p4()) / pjet.Mag2();
      properties.p0Par = boostedPPar(recoelectron->momentum(), jetRef->momentum());
      properties.elec_mva = recoelectron->mva_e_pi();
      properties.charge = recoelectron->charge();
      if (std::abs(properties.sip3dsig) > MaxSip3Dsig)
        continue;
      // Fill the TagInfos
      tagInfo.insert(lepPtr, properties);
    }
    theElecTagInfo.push_back(tagInfo);
  }
  iEvent.emplace(token_put, std::move(theElecTagInfo));
}

bool SoftPFElectronTagInfoProducer::isElecClean(edm::Event& iEvent, const reco::GsfElectron* candidate) {
  using namespace reco;
  const HitPattern& hitPattern = candidate->gsfTrack().get()->hitPattern();
  uint32_t hit = hitPattern.getHitPattern(HitPattern::TRACK_HITS, 0);
  bool hitCondition =
      !(HitPattern::validHitFilter(hit) && ((HitPattern::pixelBarrelHitFilter(hit) && HitPattern::getLayer(hit) < 3) ||
                                            HitPattern::pixelEndcapHitFilter(hit)));
  if (hitCondition)
    return false;

  return true;
}

float SoftPFElectronTagInfoProducer::boostedPPar(const math::XYZVector& vector, const math::XYZVector& axis) {
  static const double lepton_mass = 0.00;
  static const double jet_mass = 5.279;
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > lepton(
      vector.Dot(axis) / axis.r(), ROOT::Math::VectorUtil::Perp(vector, axis), 0., lepton_mass);
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > jet(axis.r(), 0., 0., jet_mass);
  ROOT::Math::BoostX boost(-jet.Beta());
  return boost(lepton).x();
}

DEFINE_FWK_MODULE(SoftPFElectronTagInfoProducer);
