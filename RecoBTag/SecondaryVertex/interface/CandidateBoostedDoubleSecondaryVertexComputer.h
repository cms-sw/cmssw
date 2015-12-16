#ifndef RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CommonTools/Utils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "fastjet/PseudoJet.hh"

class CandidateBoostedDoubleSecondaryVertexComputer : public JetTagComputer {

  public:
    CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters);

    void  initialize(const JetTagComputerRecord &) override;
    float discriminator(const TagInfoHelper & tagInfos) const override;

  private:
    void calcNsubjettiness(const reco::JetBaseRef & jet, float & tau1, float & tau2, std::vector<fastjet::PseudoJet> & currentAxes) const;
    void setTracksPVBase(const reco::TrackRef & trackRef, const reco::VertexRef & vertexRef, float & PVweight) const;
    void setTracksPV(const reco::CandidatePtr & trackRef, const reco::VertexRef & vertexRef, float & PVweight) const;
    void vertexKinematics(const reco::VertexCompositePtrCandidate & vertex, reco::TrackKinematics & vertexKinematics) const;
    void etaRelToTauAxis(const reco::VertexCompositePtrCandidate & vertex, fastjet::PseudoJet & tauAxis, std::vector<float> & tau_trackEtaRel) const;

    const double beta_;
    const double R0_;

    const double maxSVDeltaRToJet_;
    const bool useCondDB_;
    const std::string gbrForestLabel_;
    const edm::FileInPath weightFile_;
    const bool useGBRForest_;
    const bool useAdaBoost_;
    const double maxDistToAxis_;
    const double maxDecayLen_;
    reco::V0Filter trackPairV0Filter;
    reco::TrackSelector trackSelector;

    edm::ESHandle<TransientTrackBuilder> trackBuilder;
    std::unique_ptr<TMVAEvaluator> mvaID;

    // static variables
    static constexpr float dummyZ_ratio             = -3.0f;
    static constexpr float dummyTrackSip3dSig       = -50.0f;
    static constexpr float dummyTrackSip2dSigAbove  = -19.0f;
    static constexpr float dummyTrackEtaRel         = -1.0f;
    static constexpr float dummyVertexMass          = -1.0f;
    static constexpr float dummyVertexEnergyRatio   = -1.0f;
    static constexpr float dummyVertexDeltaR        = -1.0f;
    static constexpr float dummyFlightDistance2dSig = -1.0f;

    static constexpr float charmThreshold  = 1.5f;
    static constexpr float bottomThreshold = 5.2f;
};

#endif // RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
