#ifndef PFTauElementsOperators_H_
#define PFTauElementsOperators_H_

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/Tau.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "RecoTauTag/TauTagTools/interface/TauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInCone.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInAnnulus.h"
#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "PhysicsTools/Utilities/interface/Angle.h"

#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace edm;
using namespace std;
using namespace reco;

class PFTauElementsOperators : public TauElementsOperators {
 public:
  PFTauElementsOperators(Tau& theTau): TauElementsOperators(theTau),AreaMetric_recoElements_maxabsEta_(2.5){
    PFJetRef_=theTau.getpfjetRef();
    PFCands_=theTau.getSelectedPFCands();
    IsolPFCands_=theTau.getIsolationPFCands();
    PFChargedHadrCands_=theTau.getSelectedPFChargedHadrCands();
    IsolPFChargedHadrCands_=theTau.getIsolationPFChargedHadrCands();
    PFNeutrHadrCands_=theTau.getSelectedPFNeutrHadrCands();
    IsolPFNeutrHadrCands_=theTau.getIsolationPFNeutrHadrCands();
    PFGammaCands_=theTau.getSelectedPFGammaCands();
    IsolPFGammaCands_=theTau.getIsolationPFGammaCands();
  }
  ~PFTauElementsOperators(){}   
  void setAreaMetricrecoElementsmaxabsEta( double x) {AreaMetric_recoElements_maxabsEta_=x;}
  //return the leading PFCandidate in a given cone around the jet axis or a given direction
  PFCandidateRef leadPFCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  PFCandidateRef leadPFCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  PFCandidateRef leadPFChargedHadrCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  PFCandidateRef leadPFChargedHadrCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  PFCandidateRef leadPFNeutrHadrCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  PFCandidateRef leadPFNeutrHadrCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  PFCandidateRef leadPFGammaCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const;
  PFCandidateRef leadPFGammaCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const;  
  
  // return all PFCandidate's in a cone of metric* "cone_metric" and size "conesize" around a direction "myVector" 
  PFCandidateRefVector PFCandsInCone(const PFCandidateRefVector PFCands,const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const;
  PFCandidateRefVector PFCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const;
  PFCandidateRefVector PFChargedHadrCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const;
  PFCandidateRefVector PFNeutrHadrCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const;
  PFCandidateRefVector PFGammaCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const;
  
  // return all PFCandidate's in a annulus defined by inner(metric* "innercone_metric" and size "innercone_size") and outer(metric* "outercone_metric" and size "outercone_size") cones around a direction "myVector" 
  PFCandidateRefVector PFCandsInAnnulus(const PFCandidateRefVector PFCands,const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const;
  PFCandidateRefVector PFCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const;
  PFCandidateRefVector PFChargedHadrCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt);
  PFCandidateRefVector PFNeutrHadrCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const;
  PFCandidateRefVector PFGammaCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const;
  
  // return 1 if no/low PFCandidate's activity in an isolation annulus around a leading PFCandidate, 0 otherwise; 
  // different possible metrics* for the matching, signal and isolation cones; 
  double discriminatorByIsolPFCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFChargedHadrCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFChargedHadrCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFChargedHadrCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFNeutrHadrCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFNeutrHadrCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFNeutrHadrCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFGammaCandsN(int IsolPFCands_maxN=0);
  double discriminatorByIsolPFGammaCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFGammaCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN=0);
  double discriminatorByIsolPFCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFChargedHadrCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFChargedHadrCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFChargedHadrCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFNeutrHadrCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFNeutrHadrCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFNeutrHadrCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFGammaCandsEtSum(double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFGammaCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);
  double discriminatorByIsolPFGammaCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum=0);  
 private:
  PFJetRef PFJetRef_;
  double AreaMetric_recoElements_maxabsEta_;
  PFCandidateRefVector PFCands_;
  PFCandidateRefVector IsolPFCands_;
  PFCandidateRefVector PFChargedHadrCands_;
  PFCandidateRefVector IsolPFChargedHadrCands_;
  PFCandidateRefVector PFNeutrHadrCands_;
  PFCandidateRefVector IsolPFNeutrHadrCands_;
  PFCandidateRefVector PFGammaCands_;
  PFCandidateRefVector IsolPFGammaCands_;
  // template objects for DR and Angle metrics
  DeltaR<math::XYZVector> metricDR_;  
  Angle<math::XYZVector> metricAngle_;
  ElementsInCone<math::XYZVector,DeltaR<math::XYZVector>,reco::PFCandidateCollection> PFCandsinCone_DRmetric_;
  ElementsInCone<math::XYZVector,Angle<math::XYZVector>,reco::PFCandidateCollection> PFCandsinCone_Anglemetric_; 
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector>,DeltaR<math::XYZVector>,reco::PFCandidateCollection> PFCandsinAnnulus_innerDRouterDRmetrics_;
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector>,Angle<math::XYZVector>,reco::PFCandidateCollection> PFCandsinAnnulus_innerDRouterAnglemetrics_; 
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector>,Angle<math::XYZVector>,reco::PFCandidateCollection> PFCandsinAnnulus_innerAngleouterAnglemetrics_;
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector>,DeltaR<math::XYZVector>,reco::PFCandidateCollection> PFCandsinAnnulus_innerAngleouterDRmetrics_; 
};
#endif

// * different possible metrics for a cone : "DR", "angle", "area"; 




