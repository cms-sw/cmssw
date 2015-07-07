#ifndef RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/contrib/Njettiness.hh"

#include <mutex>


class CandidateBoostedDoubleSecondaryVertexComputer : public JetTagComputer {

  public:
    CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters);

    float discriminator(const TagInfoHelper & tagInfos) const override;

  private:
    void calcNsubjettiness(const reco::JetBaseRef & jet, float & tau1, float & tau2, std::vector<fastjet::PseudoJet> & currentAxes) const;
    void setTracksPVBase(const reco::TrackRef & trackRef, const reco::VertexRef & vertexRef, float & PVweight) const;
    void setTracksPV(const reco::CandidatePtr & trackRef, const reco::VertexRef & vertexRef, float & PVweight) const;
    void vertexKinematics(const reco::VertexCompositePtrCandidate & vertex, reco::TrackKinematics & vertexKinematics) const;

    const double beta_ ;
    const double R0_;
    // N-subjettiness calculator
    fastjet::contrib::Njettiness njettiness_;

    const double maxSVDeltaRToJet_;

    edm::FileInPath weightFile_;
    mutable std::mutex m_mutex;
    [[cms::thread_guard("m_mutex")]] std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif // RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
