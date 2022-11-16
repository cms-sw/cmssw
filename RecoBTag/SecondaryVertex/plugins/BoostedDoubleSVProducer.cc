// -*- C++ -*-
//
// Package:    ​RecoBTag/​SecondaryVertex
// Class:      BoostedDoubleSVProducer
//
/**\class BoostedDoubleSVProducer BoostedDoubleSVProducer.cc ​RecoBTag/​SecondaryVertex/plugins/BoostedDoubleSVProducer.cc
  *
  * Description: EDProducer that produces collection of BoostedDoubleSVTagInfos
  *
  * Implementation:
  *    A collection of SecondaryVertexTagInfos is taken as input and a collection of BoostedDoubleSVTagInfos
  *    is produced as output.
  */
//
// Original Author:  Dinko Ferencek
//         Created:  Thu, 06 Oct 2016 14:02:30 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/contrib/Njettiness.hh"

#include <map>

//
// class declaration
//

class BoostedDoubleSVProducer : public edm::stream::EDProducer<> {
public:
  explicit BoostedDoubleSVProducer(const edm::ParameterSet&);
  ~BoostedDoubleSVProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  void calcNsubjettiness(const reco::JetBaseRef& jet,
                         float& tau1,
                         float& tau2,
                         std::vector<fastjet::PseudoJet>& currentAxes) const;
  void setTracksPVBase(const reco::TrackRef& trackRef, const reco::VertexRef& vertexRef, float& PVweight) const;
  void setTracksPV(const reco::CandidatePtr& trackRef, const reco::VertexRef& vertexRef, float& PVweight) const;
  void etaRelToTauAxis(const reco::VertexCompositePtrCandidate& vertex,
                       const fastjet::PseudoJet& tauAxis,
                       std::vector<float>& tau_trackEtaRel) const;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<std::vector<reco::CandSecondaryVertexTagInfo>> svTagInfos_;

  const double beta_;
  const double R0_;

  const double maxSVDeltaRToJet_;
  const double maxDistToAxis_;
  const double maxDecayLen_;
  reco::V0Filter trackPairV0Filter;
  reco::TrackSelector trackSelector;

  edm::EDGetTokenT<edm::ValueMap<float>> weightsToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> trackBuilderToken_;
  edm::Handle<edm::ValueMap<float>> weightsHandle_;

  // static variables
  static constexpr float dummyZ_ratio = -3.0f;
  static constexpr float dummyTrackSip3dSig = -50.0f;
  static constexpr float dummyTrackSip2dSigAbove = -19.0f;
  static constexpr float dummyTrackEtaRel = -1.0f;
  static constexpr float dummyVertexMass = -1.0f;
  static constexpr float dummyVertexEnergyRatio = -1.0f;
  static constexpr float dummyVertexDeltaR = -1.0f;
  static constexpr float dummyFlightDistance2dSig = -1.0f;

  static constexpr float charmThreshold = 1.5f;
  static constexpr float bottomThreshold = 5.2f;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BoostedDoubleSVProducer::BoostedDoubleSVProducer(const edm::ParameterSet& iConfig)
    : svTagInfos_(
          consumes<std::vector<reco::CandSecondaryVertexTagInfo>>(iConfig.getParameter<edm::InputTag>("svTagInfos"))),
      beta_(iConfig.getParameter<double>("beta")),
      R0_(iConfig.getParameter<double>("R0")),
      maxSVDeltaRToJet_(iConfig.getParameter<double>("maxSVDeltaRToJet")),
      maxDistToAxis_(iConfig.getParameter<edm::ParameterSet>("trackSelection").getParameter<double>("maxDistToAxis")),
      maxDecayLen_(iConfig.getParameter<edm::ParameterSet>("trackSelection").getParameter<double>("maxDecayLen")),
      trackPairV0Filter(iConfig.getParameter<edm::ParameterSet>("trackPairV0Filter")),
      trackSelector(iConfig.getParameter<edm::ParameterSet>("trackSelection")) {
  edm::InputTag srcWeights = iConfig.getParameter<edm::InputTag>("weights");
  trackBuilderToken_ =
      esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"));
  if (!srcWeights.label().empty())
    weightsToken_ = consumes<edm::ValueMap<float>>(srcWeights);
  produces<std::vector<reco::BoostedDoubleSVTagInfo>>();
}

BoostedDoubleSVProducer::~BoostedDoubleSVProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void BoostedDoubleSVProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the track builder
  edm::ESHandle<TransientTrackBuilder> trackBuilder = iSetup.getHandle(trackBuilderToken_);

  // get input secondary vertex TagInfos
  edm::Handle<std::vector<reco::CandSecondaryVertexTagInfo>> svTagInfos;
  iEvent.getByToken(svTagInfos_, svTagInfos);

  if (!weightsToken_.isUninitialized())
    iEvent.getByToken(weightsToken_, weightsHandle_);

  // create the output collection
  auto tagInfos = std::make_unique<std::vector<reco::BoostedDoubleSVTagInfo>>();

  // loop over TagInfos
  for (std::vector<reco::CandSecondaryVertexTagInfo>::const_iterator iterTI = svTagInfos->begin();
       iterTI != svTagInfos->end();
       ++iterTI) {
    // get TagInfos
    const reco::CandIPTagInfo& ipTagInfo = *(iterTI->trackIPTagInfoRef().get());
    const reco::CandSecondaryVertexTagInfo& svTagInfo = *(iterTI);

    // default variable values
    float z_ratio = dummyZ_ratio;
    float trackSip3dSig_3 = dummyTrackSip3dSig, trackSip3dSig_2 = dummyTrackSip3dSig,
          trackSip3dSig_1 = dummyTrackSip3dSig, trackSip3dSig_0 = dummyTrackSip3dSig;
    float tau2_trackSip3dSig_0 = dummyTrackSip3dSig, tau1_trackSip3dSig_0 = dummyTrackSip3dSig,
          tau2_trackSip3dSig_1 = dummyTrackSip3dSig, tau1_trackSip3dSig_1 = dummyTrackSip3dSig;
    float trackSip2dSigAboveCharm_0 = dummyTrackSip2dSigAbove, trackSip2dSigAboveBottom_0 = dummyTrackSip2dSigAbove,
          trackSip2dSigAboveBottom_1 = dummyTrackSip2dSigAbove;
    float tau1_trackEtaRel_0 = dummyTrackEtaRel, tau1_trackEtaRel_1 = dummyTrackEtaRel,
          tau1_trackEtaRel_2 = dummyTrackEtaRel;
    float tau2_trackEtaRel_0 = dummyTrackEtaRel, tau2_trackEtaRel_1 = dummyTrackEtaRel,
          tau2_trackEtaRel_2 = dummyTrackEtaRel;
    float tau1_vertexMass = dummyVertexMass, tau1_vertexEnergyRatio = dummyVertexEnergyRatio,
          tau1_vertexDeltaR = dummyVertexDeltaR, tau1_flightDistance2dSig = dummyFlightDistance2dSig;
    float tau2_vertexMass = dummyVertexMass, tau2_vertexEnergyRatio = dummyVertexEnergyRatio,
          tau2_vertexDeltaR = dummyVertexDeltaR, tau2_flightDistance2dSig = dummyFlightDistance2dSig;
    float jetNTracks = 0, nSV = 0, tau1_nSecondaryVertices = 0, tau2_nSecondaryVertices = 0;

    // get the jet reference
    const reco::JetBaseRef jet = svTagInfo.jet();

    std::vector<fastjet::PseudoJet> currentAxes;
    float tau2, tau1;
    // calculate N-subjettiness
    calcNsubjettiness(jet, tau1, tau2, currentAxes);

    const reco::VertexRef& vertexRef = ipTagInfo.primaryVertex();
    GlobalPoint pv(0., 0., 0.);
    if (ipTagInfo.primaryVertex().isNonnull())
      pv = GlobalPoint(vertexRef->x(), vertexRef->y(), vertexRef->z());

    const std::vector<reco::CandidatePtr>& selectedTracks = ipTagInfo.selectedTracks();
    const std::vector<reco::btag::TrackIPData>& ipData = ipTagInfo.impactParameterData();
    size_t trackSize = selectedTracks.size();

    reco::TrackKinematics allKinematics;
    std::vector<float> IP3Ds, IP3Ds_1, IP3Ds_2;
    int contTrk = 0;

    // loop over tracks associated to the jet
    for (size_t itt = 0; itt < trackSize; ++itt) {
      const reco::CandidatePtr trackRef = selectedTracks[itt];

      float track_PVweight = 0.;
      setTracksPV(trackRef, vertexRef, track_PVweight);
      if (track_PVweight > 0.5)
        allKinematics.add(trackRef);

      const reco::btag::TrackIPData& data = ipData[itt];
      bool isSelected = false;
      if (trackSelector(trackRef, data, *jet, pv))
        isSelected = true;

      // check if the track is from V0
      bool isfromV0 = false, isfromV0Tight = false;
      std::vector<reco::CandidatePtr> trackPairV0Test(2);

      trackPairV0Test[0] = trackRef;

      for (size_t jtt = 0; jtt < trackSize; ++jtt) {
        if (itt == jtt)
          continue;

        const reco::btag::TrackIPData& pairTrackData = ipData[jtt];
        const reco::CandidatePtr pairTrackRef = selectedTracks[jtt];

        trackPairV0Test[1] = pairTrackRef;

        if (!trackPairV0Filter(trackPairV0Test)) {
          isfromV0 = true;

          if (trackSelector(pairTrackRef, pairTrackData, *jet, pv))
            isfromV0Tight = true;
        }

        if (isfromV0 && isfromV0Tight)
          break;
      }

      if (isSelected && !isfromV0Tight)
        jetNTracks += 1.;

      reco::TransientTrack transientTrack = trackBuilder->build(trackRef);
      GlobalVector direction(jet->px(), jet->py(), jet->pz());

      int index = 0;
      if (currentAxes.size() > 1 &&
          reco::deltaR2(trackRef->momentum(), currentAxes[1]) < reco::deltaR2(trackRef->momentum(), currentAxes[0]))
        index = 1;
      direction = GlobalVector(currentAxes[index].px(), currentAxes[index].py(), currentAxes[index].pz());

      // decay distance and track distance wrt to the closest tau axis
      float decayLengthTau = -1;
      float distTauAxis = -1;

      TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(
          transientTrack.impactPointState(), *vertexRef, direction, transientTrack.field());
      if (closest.isValid())
        decayLengthTau = (closest.globalPosition() - RecoVertex::convertPos(vertexRef->position())).mag();

      distTauAxis = std::abs(IPTools::jetTrackDistance(transientTrack, direction, *vertexRef).second.value());

      float IP3Dsig = ipTagInfo.impactParameterData()[itt].ip3d.significance();

      if (!isfromV0 && decayLengthTau < maxDecayLen_ && distTauAxis < maxDistToAxis_) {
        IP3Ds.push_back(IP3Dsig < -50. ? -50. : IP3Dsig);
        ++contTrk;
        if (currentAxes.size() > 1) {
          if (reco::deltaR2(trackRef->momentum(), currentAxes[0]) < reco::deltaR2(trackRef->momentum(), currentAxes[1]))
            IP3Ds_1.push_back(IP3Dsig < -50. ? -50. : IP3Dsig);
          else
            IP3Ds_2.push_back(IP3Dsig < -50. ? -50. : IP3Dsig);
        } else
          IP3Ds_1.push_back(IP3Dsig < -50. ? -50. : IP3Dsig);
      }
    }

    std::vector<size_t> indices = ipTagInfo.sortedIndexes(reco::btag::IP2DSig);
    bool charmThreshSet = false;

    reco::TrackKinematics kin;
    for (size_t i = 0; i < indices.size(); ++i) {
      size_t idx = indices[i];
      const reco::btag::TrackIPData& data = ipData[idx];
      const reco::CandidatePtr trackRef = selectedTracks[idx];

      kin.add(trackRef);

      if (kin.vectorSum().M() > charmThreshold  // charm cut
          && !charmThreshSet) {
        trackSip2dSigAboveCharm_0 = data.ip2d.significance();

        charmThreshSet = true;
      }

      if (kin.vectorSum().M() > bottomThreshold)  // bottom cut
      {
        trackSip2dSigAboveBottom_0 = data.ip2d.significance();
        if ((i + 1) < indices.size())
          trackSip2dSigAboveBottom_1 = (ipData[indices[i + 1]]).ip2d.significance();

        break;
      }
    }

    float dummyTrack = -50.;

    std::sort(IP3Ds.begin(), IP3Ds.end(), std::greater<float>());
    std::sort(IP3Ds_1.begin(), IP3Ds_1.end(), std::greater<float>());
    std::sort(IP3Ds_2.begin(), IP3Ds_2.end(), std::greater<float>());
    int num_1 = IP3Ds_1.size();
    int num_2 = IP3Ds_2.size();

    switch (contTrk) {
      case 0:

        trackSip3dSig_0 = dummyTrack;
        trackSip3dSig_1 = dummyTrack;
        trackSip3dSig_2 = dummyTrack;
        trackSip3dSig_3 = dummyTrack;

        break;

      case 1:

        trackSip3dSig_0 = IP3Ds.at(0);
        trackSip3dSig_1 = dummyTrack;
        trackSip3dSig_2 = dummyTrack;
        trackSip3dSig_3 = dummyTrack;

        break;

      case 2:

        trackSip3dSig_0 = IP3Ds.at(0);
        trackSip3dSig_1 = IP3Ds.at(1);
        trackSip3dSig_2 = dummyTrack;
        trackSip3dSig_3 = dummyTrack;

        break;

      case 3:

        trackSip3dSig_0 = IP3Ds.at(0);
        trackSip3dSig_1 = IP3Ds.at(1);
        trackSip3dSig_2 = IP3Ds.at(2);
        trackSip3dSig_3 = dummyTrack;

        break;

      default:

        trackSip3dSig_0 = IP3Ds.at(0);
        trackSip3dSig_1 = IP3Ds.at(1);
        trackSip3dSig_2 = IP3Ds.at(2);
        trackSip3dSig_3 = IP3Ds.at(3);
    }

    switch (num_1) {
      case 0:

        tau1_trackSip3dSig_0 = dummyTrack;
        tau1_trackSip3dSig_1 = dummyTrack;

        break;

      case 1:

        tau1_trackSip3dSig_0 = IP3Ds_1.at(0);
        tau1_trackSip3dSig_1 = dummyTrack;

        break;

      default:

        tau1_trackSip3dSig_0 = IP3Ds_1.at(0);
        tau1_trackSip3dSig_1 = IP3Ds_1.at(1);
    }

    switch (num_2) {
      case 0:

        tau2_trackSip3dSig_0 = dummyTrack;
        tau2_trackSip3dSig_1 = dummyTrack;

        break;

      case 1:
        tau2_trackSip3dSig_0 = IP3Ds_2.at(0);
        tau2_trackSip3dSig_1 = dummyTrack;

        break;

      default:

        tau2_trackSip3dSig_0 = IP3Ds_2.at(0);
        tau2_trackSip3dSig_1 = IP3Ds_2.at(1);
    }

    math::XYZVector jetDir = jet->momentum().Unit();
    reco::TrackKinematics tau1Kinematics;
    reco::TrackKinematics tau2Kinematics;
    std::vector<float> tau1_trackEtaRels, tau2_trackEtaRels;

    std::map<double, size_t> VTXmap;
    for (size_t vtx = 0; vtx < svTagInfo.nVertices(); ++vtx) {
      const reco::VertexCompositePtrCandidate& vertex = svTagInfo.secondaryVertex(vtx);
      // get the vertex kinematics
      reco::TrackKinematics vertexKinematic(vertex);

      if (currentAxes.size() > 1) {
        if (reco::deltaR2(svTagInfo.flightDirection(vtx), currentAxes[1]) <
            reco::deltaR2(svTagInfo.flightDirection(vtx), currentAxes[0])) {
          tau2Kinematics = tau2Kinematics + vertexKinematic;
          if (tau2_flightDistance2dSig < 0) {
            tau2_flightDistance2dSig = svTagInfo.flightDistance(vtx, true).significance();
            tau2_vertexDeltaR = reco::deltaR(svTagInfo.flightDirection(vtx), currentAxes[1]);
          }
          etaRelToTauAxis(vertex, currentAxes[1], tau2_trackEtaRels);
          tau2_nSecondaryVertices += 1.;
        } else {
          tau1Kinematics = tau1Kinematics + vertexKinematic;
          if (tau1_flightDistance2dSig < 0) {
            tau1_flightDistance2dSig = svTagInfo.flightDistance(vtx, true).significance();
            tau1_vertexDeltaR = reco::deltaR(svTagInfo.flightDirection(vtx), currentAxes[0]);
          }
          etaRelToTauAxis(vertex, currentAxes[0], tau1_trackEtaRels);
          tau1_nSecondaryVertices += 1.;
        }

      } else if (!currentAxes.empty()) {
        tau1Kinematics = tau1Kinematics + vertexKinematic;
        if (tau1_flightDistance2dSig < 0) {
          tau1_flightDistance2dSig = svTagInfo.flightDistance(vtx, true).significance();
          tau1_vertexDeltaR = reco::deltaR(svTagInfo.flightDirection(vtx), currentAxes[0]);
        }
        etaRelToTauAxis(vertex, currentAxes[0], tau1_trackEtaRels);
        tau1_nSecondaryVertices += 1.;
      }

      const GlobalVector& flightDir = svTagInfo.flightDirection(vtx);
      if (reco::deltaR2(flightDir, jetDir) < (maxSVDeltaRToJet_ * maxSVDeltaRToJet_))
        VTXmap[svTagInfo.flightDistance(vtx).error()] = vtx;
    }
    nSV = VTXmap.size();

    math::XYZTLorentzVector allSum = allKinematics.weightedVectorSum();
    if (tau1_nSecondaryVertices > 0.) {
      const math::XYZTLorentzVector& tau1_vertexSum = tau1Kinematics.weightedVectorSum();
      if (allSum.E() > 0.)
        tau1_vertexEnergyRatio = tau1_vertexSum.E() / allSum.E();
      if (tau1_vertexEnergyRatio > 50.)
        tau1_vertexEnergyRatio = 50.;

      tau1_vertexMass = tau1_vertexSum.M();
    }

    if (tau2_nSecondaryVertices > 0.) {
      const math::XYZTLorentzVector& tau2_vertexSum = tau2Kinematics.weightedVectorSum();
      if (allSum.E() > 0.)
        tau2_vertexEnergyRatio = tau2_vertexSum.E() / allSum.E();
      if (tau2_vertexEnergyRatio > 50.)
        tau2_vertexEnergyRatio = 50.;

      tau2_vertexMass = tau2_vertexSum.M();
    }

    float dummyEtaRel = -1.;

    std::sort(tau1_trackEtaRels.begin(), tau1_trackEtaRels.end());
    std::sort(tau2_trackEtaRels.begin(), tau2_trackEtaRels.end());

    switch (tau2_trackEtaRels.size()) {
      case 0:

        tau2_trackEtaRel_0 = dummyEtaRel;
        tau2_trackEtaRel_1 = dummyEtaRel;
        tau2_trackEtaRel_2 = dummyEtaRel;

        break;

      case 1:

        tau2_trackEtaRel_0 = tau2_trackEtaRels.at(0);
        tau2_trackEtaRel_1 = dummyEtaRel;
        tau2_trackEtaRel_2 = dummyEtaRel;

        break;

      case 2:

        tau2_trackEtaRel_0 = tau2_trackEtaRels.at(0);
        tau2_trackEtaRel_1 = tau2_trackEtaRels.at(1);
        tau2_trackEtaRel_2 = dummyEtaRel;

        break;

      default:

        tau2_trackEtaRel_0 = tau2_trackEtaRels.at(0);
        tau2_trackEtaRel_1 = tau2_trackEtaRels.at(1);
        tau2_trackEtaRel_2 = tau2_trackEtaRels.at(2);
    }

    switch (tau1_trackEtaRels.size()) {
      case 0:

        tau1_trackEtaRel_0 = dummyEtaRel;
        tau1_trackEtaRel_1 = dummyEtaRel;
        tau1_trackEtaRel_2 = dummyEtaRel;

        break;

      case 1:

        tau1_trackEtaRel_0 = tau1_trackEtaRels.at(0);
        tau1_trackEtaRel_1 = dummyEtaRel;
        tau1_trackEtaRel_2 = dummyEtaRel;

        break;

      case 2:

        tau1_trackEtaRel_0 = tau1_trackEtaRels.at(0);
        tau1_trackEtaRel_1 = tau1_trackEtaRels.at(1);
        tau1_trackEtaRel_2 = dummyEtaRel;

        break;

      default:

        tau1_trackEtaRel_0 = tau1_trackEtaRels.at(0);
        tau1_trackEtaRel_1 = tau1_trackEtaRels.at(1);
        tau1_trackEtaRel_2 = tau1_trackEtaRels.at(2);
    }

    int cont = 0;
    GlobalVector flightDir_0, flightDir_1;
    reco::Candidate::LorentzVector SV_p4_0, SV_p4_1;
    double vtxMass = 0.;

    for (std::map<double, size_t>::iterator iVtx = VTXmap.begin(); iVtx != VTXmap.end(); ++iVtx) {
      ++cont;
      const reco::VertexCompositePtrCandidate& vertex = svTagInfo.secondaryVertex(iVtx->second);
      if (cont == 1) {
        flightDir_0 = svTagInfo.flightDirection(iVtx->second);
        SV_p4_0 = vertex.p4();
        vtxMass = SV_p4_0.mass();

        if (vtxMass > 0.)
          z_ratio = reco::deltaR(currentAxes[1], currentAxes[0]) * SV_p4_0.pt() / vtxMass;
      }
      if (cont == 2) {
        flightDir_1 = svTagInfo.flightDirection(iVtx->second);
        SV_p4_1 = vertex.p4();
        vtxMass = (SV_p4_1 + SV_p4_0).mass();

        if (vtxMass > 0.)
          z_ratio = reco::deltaR(flightDir_0, flightDir_1) * SV_p4_1.pt() / vtxMass;

        break;
      }
    }

    // when only one tau axis has SVs assigned, they are all assigned to the 1st tau axis
    // in the special case below need to swap values
    if ((tau1_vertexMass < 0 && tau2_vertexMass > 0)) {
      float temp = tau1_trackEtaRel_0;
      tau1_trackEtaRel_0 = tau2_trackEtaRel_0;
      tau2_trackEtaRel_0 = temp;

      temp = tau1_trackEtaRel_1;
      tau1_trackEtaRel_1 = tau2_trackEtaRel_1;
      tau2_trackEtaRel_1 = temp;

      temp = tau1_trackEtaRel_2;
      tau1_trackEtaRel_2 = tau2_trackEtaRel_2;
      tau2_trackEtaRel_2 = temp;

      temp = tau1_flightDistance2dSig;
      tau1_flightDistance2dSig = tau2_flightDistance2dSig;
      tau2_flightDistance2dSig = temp;

      tau1_vertexDeltaR = tau2_vertexDeltaR;

      temp = tau1_vertexEnergyRatio;
      tau1_vertexEnergyRatio = tau2_vertexEnergyRatio;
      tau2_vertexEnergyRatio = temp;

      temp = tau1_vertexMass;
      tau1_vertexMass = tau2_vertexMass;
      tau2_vertexMass = temp;
    }

    reco::TaggingVariableList vars;

    vars.insert(reco::btau::jetNTracks, jetNTracks, true);
    vars.insert(reco::btau::jetNSecondaryVertices, nSV, true);
    vars.insert(reco::btau::trackSip3dSig_0, trackSip3dSig_0, true);
    vars.insert(reco::btau::trackSip3dSig_1, trackSip3dSig_1, true);
    vars.insert(reco::btau::trackSip3dSig_2, trackSip3dSig_2, true);
    vars.insert(reco::btau::trackSip3dSig_3, trackSip3dSig_3, true);
    vars.insert(reco::btau::tau1_trackSip3dSig_0, tau1_trackSip3dSig_0, true);
    vars.insert(reco::btau::tau1_trackSip3dSig_1, tau1_trackSip3dSig_1, true);
    vars.insert(reco::btau::tau2_trackSip3dSig_0, tau2_trackSip3dSig_0, true);
    vars.insert(reco::btau::tau2_trackSip3dSig_1, tau2_trackSip3dSig_1, true);
    vars.insert(reco::btau::trackSip2dSigAboveCharm, trackSip2dSigAboveCharm_0, true);
    vars.insert(reco::btau::trackSip2dSigAboveBottom_0, trackSip2dSigAboveBottom_0, true);
    vars.insert(reco::btau::trackSip2dSigAboveBottom_1, trackSip2dSigAboveBottom_1, true);
    vars.insert(reco::btau::tau1_trackEtaRel_0, tau1_trackEtaRel_0, true);
    vars.insert(reco::btau::tau1_trackEtaRel_1, tau1_trackEtaRel_1, true);
    vars.insert(reco::btau::tau1_trackEtaRel_2, tau1_trackEtaRel_2, true);
    vars.insert(reco::btau::tau2_trackEtaRel_0, tau2_trackEtaRel_0, true);
    vars.insert(reco::btau::tau2_trackEtaRel_1, tau2_trackEtaRel_1, true);
    vars.insert(reco::btau::tau2_trackEtaRel_2, tau2_trackEtaRel_2, true);
    vars.insert(reco::btau::tau1_vertexMass, tau1_vertexMass, true);
    vars.insert(reco::btau::tau1_vertexEnergyRatio, tau1_vertexEnergyRatio, true);
    vars.insert(reco::btau::tau1_flightDistance2dSig, tau1_flightDistance2dSig, true);
    vars.insert(reco::btau::tau1_vertexDeltaR, tau1_vertexDeltaR, true);
    vars.insert(reco::btau::tau2_vertexMass, tau2_vertexMass, true);
    vars.insert(reco::btau::tau2_vertexEnergyRatio, tau2_vertexEnergyRatio, true);
    vars.insert(reco::btau::tau2_flightDistance2dSig, tau2_flightDistance2dSig, true);
    vars.insert(reco::btau::z_ratio, z_ratio, true);

    vars.finalize();

    tagInfos->push_back(reco::BoostedDoubleSVTagInfo(
        vars, edm::Ref<std::vector<reco::CandSecondaryVertexTagInfo>>(svTagInfos, iterTI - svTagInfos->begin())));
  }

  // put the output in the event
  iEvent.put(std::move(tagInfos));
}

void BoostedDoubleSVProducer::calcNsubjettiness(const reco::JetBaseRef& jet,
                                                float& tau1,
                                                float& tau2,
                                                std::vector<fastjet::PseudoJet>& currentAxes) const {
  std::vector<fastjet::PseudoJet> fjParticles;

  // loop over jet constituents and push them in the vector of FastJet constituents
  for (const reco::CandidatePtr& daughter : jet->daughterPtrVector()) {
    if (daughter.isNonnull() && daughter.isAvailable()) {
      const reco::Jet* subjet = dynamic_cast<const reco::Jet*>(daughter.get());
      // if the daughter is actually a subjet
      if (subjet && daughter->numberOfDaughters() > 1) {
        // loop over subjet constituents and push them in the vector of FastJet constituents
        for (size_t i = 0; i < daughter->numberOfDaughters(); ++i) {
          const reco::CandidatePtr& constit = subjet->daughterPtr(i);

          if (constit.isNonnull() && constit->pt() > std::numeric_limits<double>::epsilon()) {
            // Check if any values were nan or inf
            float valcheck = constit->px() + constit->py() + constit->pz() + constit->energy();
            if (edm::isNotFinite(valcheck)) {
              edm::LogWarning("FaultyJetConstituent")
                  << "Jet constituent required for N-subjettiness computation contains Nan/Inf values!";
              continue;
            }
            if (subjet->isWeighted()) {
              float w = 0.0;
              if (!weightsToken_.isUninitialized())
                w = (*weightsHandle_)[constit];
              else {
                throw cms::Exception("MissingConstituentWeight")
                    << "BoostedDoubleSVProducer: No weights (e.g. PUPPI) given for weighted jet collection"
                    << std::endl;
              }
              if (w > 0) {
                fjParticles.push_back(
                    fastjet::PseudoJet(constit->px() * w, constit->py() * w, constit->pz() * w, constit->energy() * w));
              }
            } else {
              fjParticles.push_back(fastjet::PseudoJet(constit->px(), constit->py(), constit->pz(), constit->energy()));
            }
          } else
            edm::LogWarning("MissingJetConstituent")
                << "Jet constituent required for N-subjettiness computation is missing!";
        }
      } else {
        // Check if any values were nan or inf
        float valcheck = daughter->px() + daughter->py() + daughter->pz() + daughter->energy();
        if (edm::isNotFinite(valcheck)) {
          edm::LogWarning("FaultyJetConstituent")
              << "Jet constituent required for N-subjettiness computation contains Nan/Inf values!";
          continue;
        }
        if (jet->isWeighted()) {
          float w = 0.0;
          if (!weightsToken_.isUninitialized())
            w = (*weightsHandle_)[daughter];
          else {
            throw cms::Exception("MissingConstituentWeight")
                << "BoostedDoubleSVProducer: No weights (e.g. PUPPI) given for weighted jet collection" << std::endl;
          }
          if (w > 0 && daughter->pt() > std::numeric_limits<double>::epsilon()) {
            fjParticles.push_back(
                fastjet::PseudoJet(daughter->px() * w, daughter->py() * w, daughter->pz() * w, daughter->energy() * w));
          }
        } else {
          fjParticles.push_back(fastjet::PseudoJet(daughter->px(), daughter->py(), daughter->pz(), daughter->energy()));
        }
      }
    } else
      edm::LogWarning("MissingJetConstituent") << "Jet constituent required for N-subjettiness computation is missing!";
  }

  // N-subjettiness calculator
  fastjet::contrib::Njettiness njettiness(fastjet::contrib::OnePass_KT_Axes(),
                                          fastjet::contrib::NormalizedMeasure(beta_, R0_));

  // calculate N-subjettiness
  tau1 = njettiness.getTau(1, fjParticles);
  tau2 = njettiness.getTau(2, fjParticles);
  currentAxes = njettiness.currentAxes();
}

void BoostedDoubleSVProducer::setTracksPVBase(const reco::TrackRef& trackRef,
                                              const reco::VertexRef& vertexRef,
                                              float& PVweight) const {
  PVweight = 0.;

  const reco::TrackBaseRef trackBaseRef(trackRef);

  typedef reco::Vertex::trackRef_iterator IT;

  const reco::Vertex& vtx = *(vertexRef);
  // loop over tracks in vertices
  for (IT it = vtx.tracks_begin(); it != vtx.tracks_end(); ++it) {
    const reco::TrackBaseRef& baseRef = *it;
    // one of the tracks in the vertex is the same as the track considered in the function
    if (baseRef == trackBaseRef) {
      PVweight = vtx.trackWeight(baseRef);
      break;
    }
  }
}

void BoostedDoubleSVProducer::setTracksPV(const reco::CandidatePtr& trackRef,
                                          const reco::VertexRef& vertexRef,
                                          float& PVweight) const {
  PVweight = 0.;

  const pat::PackedCandidate* pcand = dynamic_cast<const pat::PackedCandidate*>(trackRef.get());

  if (pcand)  // MiniAOD case
  {
    if (pcand->fromPV() == pat::PackedCandidate::PVUsedInFit) {
      PVweight = 1.;
    }
  } else {
    const reco::PFCandidate* pfcand = dynamic_cast<const reco::PFCandidate*>(trackRef.get());

    setTracksPVBase(pfcand->trackRef(), vertexRef, PVweight);
  }
}

void BoostedDoubleSVProducer::etaRelToTauAxis(const reco::VertexCompositePtrCandidate& vertex,
                                              const fastjet::PseudoJet& tauAxis,
                                              std::vector<float>& tau_trackEtaRel) const {
  math::XYZVector direction(tauAxis.px(), tauAxis.py(), tauAxis.pz());
  const std::vector<reco::CandidatePtr>& tracks = vertex.daughterPtrVector();

  for (std::vector<reco::CandidatePtr>::const_iterator track = tracks.begin(); track != tracks.end(); ++track)
    tau_trackEtaRel.push_back(std::abs(reco::btau::etaRel(direction.Unit(), (*track)->momentum())));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void BoostedDoubleSVProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void BoostedDoubleSVProducer::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BoostedDoubleSVProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("beta", 1.0);
  desc.add<double>("R0", 0.8);
  desc.add<double>("maxSVDeltaRToJet", 0.7);
  {
    edm::ParameterSetDescription trackSelection;
    trackSelection.setAllowAnything();
    desc.add<edm::ParameterSetDescription>("trackSelection", trackSelection);
  }
  {
    edm::ParameterSetDescription trackPairV0Filter;
    trackPairV0Filter.add<double>("k0sMassWindow", 0.03);
    desc.add<edm::ParameterSetDescription>("trackPairV0Filter", trackPairV0Filter);
  }
  desc.add<edm::InputTag>("svTagInfos", edm::InputTag("pfInclusiveSecondaryVertexFinderAK8TagInfos"));
  desc.add<edm::InputTag>("weights", edm::InputTag(""));
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BoostedDoubleSVProducer);
