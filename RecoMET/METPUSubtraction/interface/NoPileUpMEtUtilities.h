#ifndef RecoMET_METPUSubtraction_noPileUpMEtUtilities_h
#define RecoMET_METPUSubtraction_noPileUpMEtUtilities_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/PUSubMETData.h"
#include "DataFormats/METReco/interface/PUSubMETDataFwd.h"

#include <vector>

class NoPileUpMEtUtilities
{

 public:

  enum {kOutsideJet=0,kWithin, kAll};
  enum {kChHSMET=0, kChPUMET, kNeutralUncMET, kHadronicHSMET, kHadronicPUMET};

  NoPileUpMEtUtilities();
  ~NoPileUpMEtUtilities();

 
  // general auxiliary functions
  void finalizeMEtData(CommonMETData&);
  
  //-------------------------------------------------------------------------------
  // auxiliary functions for jets
  reco::PUSubMETCandInfoCollection cleanJets(const reco::PUSubMETCandInfoCollection&, 
					     const std::vector<reco::Candidate::LorentzVector>&, 
					     double, bool);
  
  // auxiliary functions for PFCandidates
  reco::PUSubMETCandInfoCollection cleanPFCandidates(const reco::PUSubMETCandInfoCollection&,
						     const std::vector<reco::Candidate::LorentzVector>&,
						     double, bool);

 
  // common internal functions for jets and pfCandidates
  void computeAllSums( const reco::PUSubMETCandInfoCollection& jets,
		       const reco::PUSubMETCandInfoCollection& pfCandidates);

  CommonMETData computeRecoil(int metType, double& sumAbsPx, double& sumAbsPy);
  //-------------------------------------------------------------------------------

 private:
 
  // common internal functions for jets and pfCandidates, to compute the different object sums
  CommonMETData computeCandidateSum(const reco::PUSubMETCandInfoCollection& cands,
				    bool neutralFracOnly, double& sumAbsPx, double& sumAbsPy);
  
  reco::PUSubMETCandInfoCollection selectCandidates(const reco::PUSubMETCandInfoCollection& cands, 
						    double minPt, double maxPt, int type, 
						    bool isCharged, int isWithinJet);


 private:


  double minPtDef_;
  double maxPtDef_;

  CommonMETData chHSPfcSum_;
  CommonMETData chPUPfcSum_;
  CommonMETData nUncPfcSum_;
  CommonMETData nHSJetSum_;
  CommonMETData nPUJetSum_;

  double chHSPfcSumAbsPx_;
  double chPUPfcSumAbsPx_;
  double nUncPfcSumAbsPx_;
  double nHSJetSumAbsPx_;
  double nPUJetSumAbsPx_;

  double chHSPfcSumAbsPy_;
  double chPUPfcSumAbsPy_;
  double nUncPfcSumAbsPy_;
  double nHSJetSumAbsPy_;
  double nPUJetSumAbsPy_;

};

#endif
