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

  /* struct candInfo */
  /* { */
  /*   candInfo() */
  /*   : p4_(0.,0.,0.,0.), */
  /*     dZ_(0.), */
  /*     chargedFrac_(0.), */
  /*     mva_(0.) */
  /*   {} */
  /*   ~candInfo() {};     */
  /*   reco::Candidate::LorentzVector p4_; */
  /*   double dZ_; */
  /*   double chargedFrac_; */
  /*   double mva_; */
  /* }; */

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

  bool passesMVA(const reco::Candidate::LorentzVector&, double);

  reco::Candidate::LorentzVector leadJetP4(const std::vector<reco::PUSubMETCandInfo>&);
  reco::Candidate::LorentzVector subleadJetP4(const std::vector<reco::PUSubMETCandInfo>&);
  unsigned numJetsAboveThreshold(const std::vector<reco::PUSubMETCandInfo>&, double);

  void setDzCut(double dzCut);
  void setPtThreshold(double _ptThreshold);

  double getLeptonsSumMEX();
  double getLeptonsSumMEY();

  double getLeptonsChSumMEX();
  double getLeptonsChSumMEY(); 

  std::vector<reco::PUSubMETCandInfo> getCleanedJets();

  std::vector<reco::PUSubMETCandInfo> cleanPFCands(const std::vector<reco::PUSubMETCandInfo>&, 
						   const std::vector<reco::PUSubMETCandInfo>&, double, bool);

  void computeAllSums(const std::vector<reco::PUSubMETCandInfo>& jets, 
		      const std::vector<reco::PUSubMETCandInfo>& leptons,
		      const std::vector<reco::PUSubMETCandInfo>& pfCandidates);
  
  CommonMETData computeRecoil(int metType);

  void finalize(CommonMETData& metData);
 protected:

  reco::Candidate::LorentzVector jetP4(const std::vector<reco::PUSubMETCandInfo>&, unsigned);

  // cuts on jet Id. MVA output in bins of jet Pt and eta
  double mvaCut_[3][4][4]; 

 private:

  std::vector<reco::PUSubMETCandInfo> cleanJets(const std::vector<reco::PUSubMETCandInfo>&, 
						const std::vector<reco::PUSubMETCandInfo>&, double, double);
  
  CommonMETData computeCandSum( int compKey, double dZmax, int dZflag,
				bool iCharged,  bool mvaPassFlag,
				const std::vector<reco::PUSubMETCandInfo>& objects );
};

#endif
