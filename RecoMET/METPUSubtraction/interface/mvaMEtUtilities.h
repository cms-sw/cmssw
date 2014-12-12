#ifndef RecoMET_METPUSubtraction_mvaMEtUtilities_h
#define RecoMET_METPUSubtraction_mvaMEtUtilities_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/PUSubMETData.h"

#include <vector>
#include <utility>

class mvaMEtUtilities 
{
 public:

  enum {kPFCands=0,kLeptons,kJets};
  enum {kPF=0, kChHS, kHS, kPU, kHSMinusNeutralPU};

 private:

  CommonMETData _leptonsSum;
  CommonMETData _leptonsChSum;
  CommonMETData _pfCandSum;
  CommonMETData _pfCandChHSSum;
  CommonMETData _pfCandChPUSum;
  CommonMETData _neutralJetHSSum;
  CommonMETData _neutralJetPUSum;

  std::vector<reco::PUSubMETCandInfo> _cleanedJets;

  double _dzCut;
  double _ptThreshold;

 public:
  
  mvaMEtUtilities(const edm::ParameterSet& cfg);
  virtual ~mvaMEtUtilities();

  reco::Candidate::LorentzVector leadJetP4(const std::vector<reco::PUSubMETCandInfo>&);
  reco::Candidate::LorentzVector subleadJetP4(const std::vector<reco::PUSubMETCandInfo>&);
  unsigned numJetsAboveThreshold(const std::vector<reco::PUSubMETCandInfo>&, double);

  std::vector<reco::PUSubMETCandInfo> getCleanedJets();

  //access functions for lepton suns ============
  double getLeptonsSumMEX();
  double getLeptonsSumMEY();

  double getLeptonsChSumMEX();
  double getLeptonsChSumMEY(); 

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
