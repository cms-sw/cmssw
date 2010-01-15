/*
Hadrons + Strips Tau Identification Algorithm
---------------------------------------------
Michail Bachtis
University of Wisconsin-Madison 
bachtis@cern.ch
*/

#ifndef RecoTauTag_RecoTau_HPSPFTauAlgorithm
#define RecoTauTag_RecoTau_HPSPFTauAlgorithm

#include "RecoTauTag/TauTagTools/interface/PFCandidateStripMerger.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithmBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

class HPSPFRecoTauAlgorithm : public PFRecoTauAlgorithmBase
{
 public:
  //  Constructors are following the PFRecoTauProducer scheme. 
  //  The input is a ParameterSet of the algorithm configuration
  HPSPFRecoTauAlgorithm();
  HPSPFRecoTauAlgorithm(const edm::ParameterSet&);
  ~HPSPFRecoTauAlgorithm();

  //Basic Method that creates the taus 
  reco::PFTau buildPFTau(const reco::PFTauTagInfoRef&,const reco::Vertex&); 

 private:   
  //*Private Members *//
  PFCandidateMergerBase * candidateMerger_;


  //* Helper Methods *//

  //Creators of the Decay Modes
  reco::PFTauCollection buildOneProng(const reco::PFTauTagInfoRef&);
  reco::PFTauCollection buildOneProngStrip(const reco::PFTauTagInfoRef&);
  reco::PFTauCollection buildOneProngTwoStrips(const reco::PFTauTagInfoRef&);
  reco::PFTauCollection buildThreeProngs(const reco::PFTauTagInfoRef&);

  //Narrowness selection
  bool isNarrowTau(const reco::PFTau&,double);

  //Associate Isolation Candidates
  void associateIsolationCandidates(reco::PFTau&,double);

  //Apply muon Rejection
  void applyMuonRejection(reco::PFTau&);

  //Apply electron Rejection
  void applyElectronRejection(reco::PFTau&,double);


  //Sort a reference vector(THIS NEEDS TO GO AWAY!)
  void sortRefVector(reco::PFCandidateRefVector&);

  //Method to create a candidate from the merged EM Candidates vector;
  math::XYZTLorentzVector createMergedLorentzVector(const reco::PFCandidateRefVector&);

  void removeCandidateFromRefVector(const reco::PFCandidateRef&,reco::PFCandidateRefVector&); 

  void applyMassConstraint(math::XYZTLorentzVector&,double );

  //Configure the algorithm!
  void configure(const edm::ParameterSet&);

  //* Variables for configuration*//
  
  //Which merging algorithm to use
  std::string emMerger_; 

  //Overlap Removal Criterion for overlaping decay modes
  std::string overlapCriterion_;

  //Decay Mode activation
  bool doOneProngs_;
  bool doOneProngStrips_;
  bool doOneProngTwoStrips_;
  bool doThreeProngs_;


  //Minimum Tau Pt for the reconstruction
  double tauThreshold_;
  //Lead Pion Threshold
  double leadPionThreshold_;

  //strip threshold
  double stripPtThreshold_;
  //Isolation Cone sizes for different particles
  double chargeIsolationCone_;
  double gammaIsolationCone_;
  double neutrHadrIsolationCone_;

  //Use a solid cone or an annulus?
  bool   useIsolationAnnulus_;

  //Mass Windows 
  std::vector<double> oneProngStripMassWindow_;
  std::vector<double> oneProngTwoStripsMassWindow_;
  std::vector<double> threeProngMassWindow_;

  //Matching Cone between PFTau , PFJet 
  double matchingCone_;


  //Cone Narrowness. Unfortunately we have to remove TFormula! 
  //The code supports A(DeltaR or Angle) < coneParameter_/(ET or ENERGY)
  std::string coneMetric_; //DeltaR or Angle
  std::string narrownessMetric_; //ET or energy
  double coneParameter_;  
  double minSignalCone_;
  double maxSignalCone_;


//SORTERS
  class HPSSorterByIsolation
  {
  public:
    HPSSorterByIsolation() {}
    bool operator()(reco::PFTau t1,reco::PFTau t2)
    {
      double iso1 =t1.isolationPFGammaCandsEtSum()+t1.isolationPFChargedHadrCandsPtSum();
      double iso2 =t2.isolationPFGammaCandsEtSum()+t2.isolationPFChargedHadrCandsPtSum();
      return iso1<iso2;
    }
  };
  
  class HPSSorterByPt
  {
  public:
    HPSSorterByPt() {}
    bool operator()(reco::PFTau t1,reco::PFTau t2)
    {
      return t1.pt()>t2.pt();
    }
  };

  
};
#endif



