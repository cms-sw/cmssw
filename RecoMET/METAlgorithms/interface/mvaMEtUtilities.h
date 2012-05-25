#ifndef RecoMET_METAlgorithms_mvaMEtUtilities_h
#define RecoMET_METAlgorithms_mvaMEtUtilities_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

// Definition of JetInfo data format
#include "DataFormats/METReco/interface/MVAMETData.h"
#include "DataFormats/METReco/interface/MVAMETDataFwd.h"

#include <vector>
#include <utility>

class mvaMEtUtilities
{
 public:

  typedef reco::JetInfo JetInfo;

  mvaMEtUtilities(const edm::ParameterSet& cfg);
  virtual ~mvaMEtUtilities();

  bool passesMVA(const reco::Candidate::LorentzVector&, double);

  reco::Candidate::LorentzVector leadJetP4(const std::vector<JetInfo>&);
  reco::Candidate::LorentzVector subleadJetP4(const std::vector<JetInfo>&);
  unsigned numJetsAboveThreshold(const std::vector<JetInfo>&, double);

  std::vector<JetInfo> cleanJets(const std::vector<JetInfo>&,
				 const std::vector<reco::Candidate::LorentzVector>&, double, double);

  struct pfCandInfo
  {
    pfCandInfo()
      : p4_(0.,0.,0.,0.),
	dZ_(0.)
    {}
    ~pfCandInfo() {}
    reco::Candidate::LorentzVector p4_;
    double dZ_;
  };

  std::vector<pfCandInfo> cleanPFCands(const std::vector<pfCandInfo>&,
				       const std::vector<reco::Candidate::LorentzVector>&, double, bool);

  CommonMETData computePFCandSum(const std::vector<pfCandInfo>&, double, int);
  CommonMETData computeJetSum_neutral(const std::vector<JetInfo>&, bool);

  CommonMETData computePUMEt(const std::vector<pfCandInfo>&, const std::vector<JetInfo>&, double);

  CommonMETData computeNegPFRecoil(const CommonMETData&, const std::vector<pfCandInfo>&, double);
  CommonMETData computeNegTrackRecoil(const CommonMETData&, const std::vector<pfCandInfo>&, double);
  CommonMETData computeNegNoPURecoil(const CommonMETData&, const std::vector<pfCandInfo>&, const std::vector<JetInfo>&, double);
  CommonMETData computeNegPUCRecoil(const CommonMETData&, const std::vector<pfCandInfo>&, const std::vector<JetInfo>&, double);

 protected:

  reco::Candidate::LorentzVector jetP4(const std::vector<JetInfo>&, unsigned);

  // cuts on jet Id. MVA output in bins of jet Pt and eta
  double mvaCut_[3][4][4];
};

#endif
