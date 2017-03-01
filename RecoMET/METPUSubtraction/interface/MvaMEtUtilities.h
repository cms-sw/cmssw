#ifndef RecoMET_METPUSubtraction_mvaMEtUtilities_h
#define RecoMET_METPUSubtraction_mvaMEtUtilities_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/PUSubMETData.h"

#include <vector>
#include <utility>

class MvaMEtUtilities 
{
 public:

  enum {kPFCands=0,kLeptons,kJets};
  enum {kPF=0, kChHS, kHS, kPU, kHSMinusNeutralPU};

 private:

  CommonMETData leptonsSum_;
  CommonMETData leptonsChSum_;
  CommonMETData pfCandSum_;
  CommonMETData pfCandChHSSum_;
  CommonMETData pfCandChPUSum_;
  CommonMETData neutralJetHSSum_;
  CommonMETData neutralJetPUSum_;

  std::vector<reco::PUSubMETCandInfo> cleanedJets_;

  double dzCut_;
  double ptThreshold_;

 public:
  
  MvaMEtUtilities(const edm::ParameterSet& cfg);
  virtual ~MvaMEtUtilities();

  reco::Candidate::LorentzVector leadJetP4(const std::vector<reco::PUSubMETCandInfo>&);
  reco::Candidate::LorentzVector subleadJetP4(const std::vector<reco::PUSubMETCandInfo>&);
  unsigned numJetsAboveThreshold(const std::vector<reco::PUSubMETCandInfo>&, double);

  const std::vector<reco::PUSubMETCandInfo>& getCleanedJets() const;

  //access functions for lepton suns ============
  double getLeptonsSumMEX() const;
  double getLeptonsSumMEY() const;

  double getLeptonsChSumMEX() const;
  double getLeptonsChSumMEY() const; 

  //recoil and sum computing functions ========
  void computeAllSums(const std::vector<reco::PUSubMETCandInfo>& jets, 
		      const std::vector<reco::PUSubMETCandInfo>& leptons,
		      const std::vector<reco::PUSubMETCandInfo>& pfCandidates);
  
  CommonMETData computeRecoil(int metType);


 protected:

  reco::Candidate::LorentzVector jetP4(const std::vector<reco::PUSubMETCandInfo>&, unsigned);

  // cuts on jet Id. MVA output in bins of jet Pt and eta
  double mvaCut_[3][4][4]; 

 private:

  //utilities functions for jets ===============
  bool passesMVA(const reco::Candidate::LorentzVector&, double);

  std::vector<reco::PUSubMETCandInfo> cleanJets(const std::vector<reco::PUSubMETCandInfo>&, 
						const std::vector<reco::PUSubMETCandInfo>&, double, double);
  
  //utilities functions for pf candidate ====== 
  std::vector<reco::PUSubMETCandInfo> cleanPFCands(const std::vector<reco::PUSubMETCandInfo>&, 
						   const std::vector<reco::PUSubMETCandInfo>&, double, bool);

  CommonMETData computeCandSum( int compKey, double dZmax, int dZflag,
				bool iCharged,  bool mvaPassFlag,
				const std::vector<reco::PUSubMETCandInfo>& objects );


  void finalize(CommonMETData& metData);

};

#endif
