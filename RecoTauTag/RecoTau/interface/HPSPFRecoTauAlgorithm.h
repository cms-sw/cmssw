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
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"
#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"


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
  void buildOneProng(const reco::PFTauTagInfoRef&,const std::vector<reco::PFCandidatePtr>& );
  void buildOneProngStrip(const reco::PFTauTagInfoRef&,const std::vector<std::vector<reco::PFCandidatePtr>>&,const std::vector<reco::PFCandidatePtr>&);
  void buildOneProngTwoStrips(const reco::PFTauTagInfoRef&,const std::vector<std::vector<reco::PFCandidatePtr>>&,const std::vector<reco::PFCandidatePtr>&);
  void buildThreeProngs(const reco::PFTauTagInfoRef&,const std::vector<reco::PFCandidatePtr>&);

  //Narrowness selection
  bool isNarrowTau(const reco::PFTau&,double);

  //Associate Isolation Candidates
  void associateIsolationCandidates(reco::PFTau&,double);

  reco::PFTau getBestTauCandidate(reco::PFTauCollection&); //get the best tau if we have overlap 
  void applyMuonRejection(reco::PFTau&);

  //Apply electron Rejection
  void applyElectronRejection(reco::PFTau&,double);

  //Method to create a candidate from the merged EM Candidates vector;
  math::XYZTLorentzVector createMergedLorentzVector(const std::vector<reco::PFCandidatePtr>&);

  void removeCandidateFromRefVector(const reco::PFCandidatePtr&,std::vector<reco::PFCandidatePtr>&); 
  void applyMassConstraint(math::XYZTLorentzVector&,double );

  bool refitThreeProng(reco::PFTau&); 



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
  std::vector<double> oneProngTwoStripsPi0MassWindow_;
  std::vector<double> threeProngMassWindow_;

  //Matching Cone between PFTau , PFJet 
  double matchingCone_;


  //Cone Narrowness. Unfortunately we have to remove TFormula! 
  //The code supports A(DeltaR or Angle) < coneParameter_/(ET or ENERGY)
  std::string coneSizeFormula_;
  std::string coneMetric_;
  double minSignalCone_;
  double maxSignalCone_;


  TFormula coneSizeFormula;

  reco::PFTauCollection pfTaus_;



  class HPSTauPtSorter {
  public:

    HPSTauPtSorter()
      {
      }


    ~HPSTauPtSorter()
      {}

    bool operator()(const reco::PFTau& a , const reco::PFTau& b) {
      return (a.pt() > b.pt());
    }
  };


  class HPSTauIsolationSorter {
  public:

    HPSTauIsolationSorter()
      {
      }


    ~HPSTauIsolationSorter()
      {}

    bool operator()(const reco::PFTau& a , const reco::PFTau& b) {
      return (a.isolationPFGammaCandsEtSum()+a.isolationPFChargedHadrCandsPtSum())<
	(b.isolationPFGammaCandsEtSum()+b.isolationPFChargedHadrCandsPtSum());
    }
  };

  
};
#endif



