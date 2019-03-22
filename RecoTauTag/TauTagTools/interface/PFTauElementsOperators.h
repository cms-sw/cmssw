#ifndef RecoTauTag_RecoTau_PFTauElementsOperators_H_
#define RecoTauTag_RecoTau_PFTauElementsOperators_H_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"
#include "RecoTauTag/TauTagTools/interface/TauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInCone.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInAnnulus.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInEllipse.h"
#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/Utils/interface/Angle.h"

#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include <algorithm>

class PFTauElementsOperators : public TauElementsOperators {
 public:
  PFTauElementsOperators(reco::PFTau& thePFTau);
  ~PFTauElementsOperators(){}   
  void setAreaMetricrecoElementsmaxabsEta( double x);
  //return the leading PFCandidate in a given cone around the jet axis or a given direction
  reco::CandidatePtr leadCand(const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  reco::CandidatePtr leadCand(const math::XYZVector& myVector,const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  reco::CandidatePtr leadChargedHadrCand(const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  reco::CandidatePtr leadChargedHadrCand(const math::XYZVector& myVector,const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  reco::CandidatePtr leadNeutrHadrCand(const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  reco::CandidatePtr leadNeutrHadrCand(const math::XYZVector& myVector,const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  reco::CandidatePtr leadGammaCand(const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  reco::CandidatePtr leadGammaCand(const math::XYZVector& myVector,const std::string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  
  // return all PFCandidates in a cone of metric* "cone_metric" and size "conesize" around a direction "myVector" 
  std::vector<reco::CandidatePtr> PFCandsInCone(const std::vector<reco::CandidatePtr>& PFCands,const math::XYZVector& myVector,const std::string conemetric,const double conesize,const double minPt)const;
  std::vector<reco::CandidatePtr> PFCandsInCone(const math::XYZVector& myVector,const std::string conemetric,const double conesize,const double minPt)const;
  std::vector<reco::CandidatePtr> PFChargedHadrCandsInCone(const math::XYZVector& myVector,const std::string conemetric,const double conesize,const double minPt)const;
  std::vector<reco::CandidatePtr> PFChargedHadrCandsInCone(const math::XYZVector& myVector,const std::string conemetric,const double conesize,const double minPt,const double PFChargedHadrCand_tracktorefpoint_maxDZ,const double refpoint_Z, const reco::Vertex &mPV)const;
  std::vector<reco::CandidatePtr> PFNeutrHadrCandsInCone(const math::XYZVector& myVector,const std::string conemetric,const double conesize,const double minPt)const;
  std::vector<reco::CandidatePtr> PFGammaCandsInCone(const math::XYZVector& myVector,const std::string conemetric,const double conesize,const double minPt)const;
  
  // return all PFCandidates in a annulus defined by inner(metric* "innercone_metric" and size "innercone_size") and outer(metric* "outercone_metric" and size "outercone_size") cones around a direction "myVector" 
  std::vector<reco::CandidatePtr> PFCandsInAnnulus(const std::vector<reco::CandidatePtr>& PFCands,const math::XYZVector& myVector,const std::string innercone_metric,const double innercone_size,const std::string outercone_metric,const double outercone_size,const double minPt)const;
  std::vector<reco::CandidatePtr> PFCandsInAnnulus(const math::XYZVector& myVector,const std::string innercone_metric,const double innercone_size,const std::string outercone_metric,const double outercone_size,const double minPt)const;
  std::vector<reco::CandidatePtr> PFChargedHadrCandsInAnnulus(const math::XYZVector& myVector,const std::string innercone_metric,const double innercone_size,const std::string outercone_metric,const double outercone_size,const double minPt)const;
  std::vector<reco::CandidatePtr> PFChargedHadrCandsInAnnulus(const math::XYZVector& myVector,const std::string innercone_metric,const double innercone_size,const std::string outercone_metric,const double outercone_size,const double minPt,const double PFChargedHadrCand_tracktorefpoint_maxDZ,const double refpoint_Z, const reco::Vertex &myPV)const;
  std::vector<reco::CandidatePtr> PFNeutrHadrCandsInAnnulus(const math::XYZVector& myVector,const std::string innercone_metric,const double innercone_size,const std::string outercone_metric,const double outercone_size,const double minPt)const;
  std::vector<reco::CandidatePtr> PFGammaCandsInAnnulus(const math::XYZVector& myVector,const std::string innercone_metric,const double innercone_size,const std::string outercone_metric,const double outercone_size,const double minPt)const;
  //Put function to get elements inside ellipse here ... EELL
  std::pair<std::vector<reco::CandidatePtr>,std::vector<reco::CandidatePtr>> PFGammaCandsInOutEllipse(const std::vector<reco::CandidatePtr>&, const reco::Candidate&, double rPhi, double rEta, double maxPt) const;
  //EELL

  /// append elements of theInputCands that pass Pt requirement to the end of theOutputCands
  void                 copyCandRefsFilteredByPt(const std::vector<reco::CandidatePtr>& theInputCands, std::vector<reco::CandidatePtr>& theOutputCands, const double minPt);

  /// compute size of cone using the Inside-Out cone (Author Evan Friis, UC Davis)
  void                 computeInsideOutContents(const std::vector<reco::CandidatePtr>& theChargedCands, const std::vector<reco::CandidatePtr>& theGammaCands, const math::XYZVector& leadTrackVector, 
                               const TFormula& coneSizeFormula, double (*ptrToMetricFunction)(const math::XYZVector&, const math::XYZVector&),  // determines grow function, and the metric to compare the opening angle to
                               const double minChargedSize, const double maxChargedSize, const double minNeutralSize, const double maxNeutralSize,
                               const double minChargedPt, const double minNeutralPt,
                               const std::string& outlierCollectorConeMetric, const double outlierCollectorConeSize,
                               std::vector<reco::CandidatePtr>& signalChargedObjects, std::vector<reco::CandidatePtr>& outlierChargedObjects,
                               std::vector<reco::CandidatePtr>& signalGammaObjects, std::vector<reco::CandidatePtr>& outlierGammaObjects, bool useScanningAxis); //these last two quantities are the return values

  // return 1 if no/low PFCandidates activity in an isolation annulus around a leading PFCandidate, 0 otherwise; 
  // different possible metrics* for the matching, signal and isolation cones; 
  double discriminatorByIsolPFCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFCandsN(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFCandsN(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFChargedHadrCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFChargedHadrCandsN(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFChargedHadrCandsN(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFNeutrHadrCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFNeutrHadrCandsN(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFNeutrHadrCandsN(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFGammaCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFGammaCandsN(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFGammaCandsN(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFCandsEtSum(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFCandsEtSum(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFChargedHadrCandsEtSum(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFChargedHadrCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFChargedHadrCandsEtSum(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFNeutrHadrCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFNeutrHadrCandsEtSum(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFNeutrHadrCandsEtSum(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFGammaCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFGammaCandsEtSum(std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFGammaCandsEtSum(const math::XYZVector& myVector,std::string matchingcone_metric,double matchingcone_size,std::string signalcone_metric,double signalcone_size,std::string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);  
 private:
  reco::JetBaseRef PFJetRef_;
  double AreaMetric_recoElements_maxabsEta_;
  std::vector<reco::CandidatePtr> PFCands_;
  std::vector<reco::CandidatePtr> IsolPFCands_;
  std::vector<reco::CandidatePtr> PFChargedHadrCands_;
  std::vector<reco::CandidatePtr> IsolPFChargedHadrCands_;
  std::vector<reco::CandidatePtr> PFNeutrHadrCands_;
  std::vector<reco::CandidatePtr> IsolPFNeutrHadrCands_;
  std::vector<reco::CandidatePtr> PFGammaCands_;
  std::vector<reco::CandidatePtr> IsolPFGammaCands_;
  // template objects for DR and Angle metrics
  DeltaR<math::XYZVector> metricDR_;  
  Angle<math::XYZVector> metricAngle_;
  double computeDeltaR(const math::XYZVector& vec1, const math::XYZVector& vec2);
  double computeAngle(const math::XYZVector& vec1, const math::XYZVector& vec2);
  ElementsInCone<math::XYZVector,DeltaR<math::XYZVector>,reco::Candidate> PFCandsinCone_DRmetric_;
  ElementsInCone<math::XYZVector,Angle<math::XYZVector>,reco::Candidate> PFCandsinCone_Anglemetric_; 
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector>,DeltaR<math::XYZVector>,reco::Candidate> PFCandsinAnnulus_innerDRouterDRmetrics_;
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector>,Angle<math::XYZVector>,reco::Candidate> PFCandsinAnnulus_innerDRouterAnglemetrics_; 
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector>,Angle<math::XYZVector>,reco::Candidate> PFCandsinAnnulus_innerAngleouterAnglemetrics_;
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector>,DeltaR<math::XYZVector>,reco::Candidate> PFCandsinAnnulus_innerAngleouterDRmetrics_; 
  ElementsInEllipse<reco::Candidate, reco::Candidate> PFCandidatesInEllipse_;
};
#endif

// * different possible metrics for a cone : "DR", "angle", "area"; 




