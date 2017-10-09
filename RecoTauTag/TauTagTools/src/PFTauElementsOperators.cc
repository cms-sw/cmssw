#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"

using namespace reco;
using std::string;

 PFTauElementsOperators::PFTauElementsOperators(PFTau& thePFTau): TauElementsOperators(thePFTau),AreaMetric_recoElements_maxabsEta_(2.5){
   PFJetRef_=thePFTau.pfTauTagInfoRef()->pfjetRef();
   PFCands_=thePFTau.pfTauTagInfoRef()->PFCands();
   PFChargedHadrCands_=thePFTau.pfTauTagInfoRef()->PFChargedHadrCands();
   PFNeutrHadrCands_=thePFTau.pfTauTagInfoRef()->PFNeutrHadrCands();
   PFGammaCands_=thePFTau.pfTauTagInfoRef()->PFGammaCands();
   IsolPFCands_=thePFTau.isolationPFCands();
   IsolPFChargedHadrCands_=thePFTau.isolationPFChargedHadrCands();
   IsolPFNeutrHadrCands_=thePFTau.isolationPFNeutrHadrCands();
   IsolPFGammaCands_=thePFTau.isolationPFGammaCands();
   Tracks_=thePFTau.pfTauTagInfoRef()->Tracks();
}
void PFTauElementsOperators::setAreaMetricrecoElementsmaxabsEta( double x) {AreaMetric_recoElements_maxabsEta_=x;}

std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFCandsInCone(const std::vector<reco::PFCandidatePtr>& thePFCands,const math::XYZVector& myVector,const string conemetric,const double conesize,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCands;
  for (std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=thePFCands.begin();iPFCand!=thePFCands.end();++iPFCand) {
    if ((**iPFCand).pt()>minPt)theFilteredPFCands.push_back(*iPFCand);
  }  
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone;
  if (conemetric=="DR"){
    theFilteredPFCandsInCone=PFCandsinCone_DRmetric_(myVector,metricDR_,conesize,theFilteredPFCands);
  }else if(conemetric=="angle"){
    theFilteredPFCandsInCone=PFCandsinCone_Anglemetric_(myVector,metricAngle_,conesize,theFilteredPFCands);
  }else if(conemetric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaCone;
    theFixedAreaCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    double coneangle=theFixedAreaCone(myVector.theta(),myVector.phi(),0,conesize,errorFlag); 
    if (errorFlag!=0)return theFilteredPFCandsInCone;   
    theFilteredPFCandsInCone=PFCandsinCone_Anglemetric_(myVector,metricAngle_,coneangle,theFilteredPFCands);
  }else return std::vector<reco::PFCandidatePtr>();
  return theFilteredPFCandsInCone;
}
std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFCandsInCone(const math::XYZVector& myVector,const string conemetric,const double conesize,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFCandsInCone(PFCands_,myVector,conemetric,conesize,minPt);
  return theFilteredPFCandsInCone;
}
std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFChargedHadrCandsInCone(const math::XYZVector& myVector,const string conemetric,const double conesize,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFCandsInCone(PFChargedHadrCands_,myVector,conemetric,conesize,minPt);
  return theFilteredPFCandsInCone;
}
std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFChargedHadrCandsInCone(const math::XYZVector& myVector,const string conemetric,const double conesize,const double minPt,const double PFChargedHadrCand_tracktorefpoint_maxDZ,const double refpoint_Z, const Vertex &myPV)const{     
  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands;
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=PFChargedHadrCands_.begin();iPFCand!=PFChargedHadrCands_.end();iPFCand++){
    TrackRef PFChargedHadrCand_track=(**iPFCand).trackRef();
    if (!PFChargedHadrCand_track)continue;
    if (fabs((*PFChargedHadrCand_track).dz(myPV.position())-refpoint_Z)<=PFChargedHadrCand_tracktorefpoint_maxDZ) filteredPFChargedHadrCands.push_back(*iPFCand);
  }
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFCandsInCone(filteredPFChargedHadrCands,myVector,conemetric,conesize,minPt);
  return theFilteredPFCandsInCone;
}
std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFNeutrHadrCandsInCone(const math::XYZVector& myVector,const string conemetric,const double conesize,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFCandsInCone(PFNeutrHadrCands_,myVector,conemetric,conesize,minPt);
  return theFilteredPFCandsInCone;
}
 std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFGammaCandsInCone(const math::XYZVector& myVector,const string conemetric,const double conesize,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFCandsInCone(PFGammaCands_,myVector,conemetric,conesize,minPt);
  return theFilteredPFCandsInCone;
}

// Function to get elements inside ellipse here ... EELL
std::pair<std::vector<reco::PFCandidatePtr>, std::vector<reco::PFCandidatePtr>> PFTauElementsOperators::PFGammaCandsInOutEllipse(const std::vector<reco::PFCandidatePtr>& PFGammaCands_, const PFCandidate& leadCand_, double rPhi, double rEta, double maxPt) const{
  std::pair<std::vector<reco::PFCandidatePtr>,std::vector<reco::PFCandidatePtr>> myPFGammaCandsInEllipse = PFCandidatesInEllipse_(leadCand_, rPhi, rEta, PFGammaCands_);
  std::vector<reco::PFCandidatePtr> thePFGammaCandsInEllipse = myPFGammaCandsInEllipse.first;
  std::vector<reco::PFCandidatePtr> thePFGammaCandsOutEllipse = myPFGammaCandsInEllipse.second;
  std::vector<reco::PFCandidatePtr> theFilteredPFGammaCandsInEllipse;
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFGammaCand = thePFGammaCandsInEllipse.begin(); iPFGammaCand != thePFGammaCandsInEllipse.end(); ++iPFGammaCand){
    if((**iPFGammaCand).pt() <= maxPt) theFilteredPFGammaCandsInEllipse.push_back(*iPFGammaCand);
    else thePFGammaCandsOutEllipse.push_back(*iPFGammaCand);
  }
  std::pair<std::vector<reco::PFCandidatePtr>, std::vector<reco::PFCandidatePtr>> theFilteredPFGammaCandsInOutEllipse(theFilteredPFGammaCandsInEllipse, thePFGammaCandsOutEllipse);
  
  return theFilteredPFGammaCandsInOutEllipse;
}
// EELL


 std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFCandsInAnnulus(const std::vector<reco::PFCandidatePtr>& thePFCands,const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCands;
  for (std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=thePFCands.begin();iPFCand!=thePFCands.end();++iPFCand) {
    if ((**iPFCand).pt()>minPt)theFilteredPFCands.push_back(*iPFCand);
  }  
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInAnnulus;
  if (outercone_metric=="DR"){
    if (innercone_metric=="DR"){
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerDRouterDRmetrics_(myVector,metricDR_,innercone_size,metricDR_,outercone_size,theFilteredPFCands);
    }else if(innercone_metric=="angle"){
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_size,metricDR_,outercone_size,theFilteredPFCands);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return theFilteredPFCandsInAnnulus;
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_angle,metricDR_,outercone_size,theFilteredPFCands);
    }else return std::vector<reco::PFCandidatePtr>();
  }else if(outercone_metric=="angle"){
    if (innercone_metric=="DR"){
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerDRouterAnglemetrics_(myVector,metricDR_,innercone_size,metricAngle_,outercone_size,theFilteredPFCands);
    }else if(innercone_metric=="angle"){
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_size,theFilteredPFCands);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return theFilteredPFCandsInAnnulus;
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_size,theFilteredPFCands);
    }else return std::vector<reco::PFCandidatePtr>();
  }else if(outercone_metric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaSignalCone;
    theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    if (innercone_metric=="DR"){
      // not implemented yet
    }else if(innercone_metric=="angle"){
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_size,outercone_size,errorFlag);    
      if (errorFlag!=0)return theFilteredPFCandsInAnnulus;
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_angle,theFilteredPFCands);
    }else if(innercone_metric=="area"){
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);    
      if (errorFlag!=0)return theFilteredPFCandsInAnnulus;
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_angle,outercone_size,errorFlag);
      if (errorFlag!=0)return theFilteredPFCandsInAnnulus;
      theFilteredPFCandsInAnnulus=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_angle,theFilteredPFCands);
    }else return std::vector<reco::PFCandidatePtr>();
  }
  return theFilteredPFCandsInAnnulus;
}
 std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFCandsInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInAnnulus=PFCandsInAnnulus(PFCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return theFilteredPFCandsInAnnulus;
}
 std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFChargedHadrCandsInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInAnnulus=PFCandsInAnnulus(PFChargedHadrCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return theFilteredPFCandsInAnnulus;
}
std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFChargedHadrCandsInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt,const double PFChargedHadrCand_tracktorefpoint_maxDZ,const double refpoint_Z, const Vertex &myPV)const{     
  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands;
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=PFChargedHadrCands_.begin();iPFCand!=PFChargedHadrCands_.end();iPFCand++){
    TrackRef PFChargedHadrCand_track=(**iPFCand).trackRef();
    if (!PFChargedHadrCand_track)continue;
    if (fabs((*PFChargedHadrCand_track).dz(myPV.position())-refpoint_Z)<=PFChargedHadrCand_tracktorefpoint_maxDZ) filteredPFChargedHadrCands.push_back(*iPFCand);
  }
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInAnnulus=PFCandsInAnnulus(filteredPFChargedHadrCands,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return theFilteredPFCandsInAnnulus;
}
 std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFNeutrHadrCandsInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInAnnulus=PFCandsInAnnulus(PFNeutrHadrCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return theFilteredPFCandsInAnnulus;
}
 std::vector<reco::PFCandidatePtr> PFTauElementsOperators::PFGammaCandsInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  std::vector<reco::PFCandidatePtr> theFilteredPFCandsInAnnulus=PFCandsInAnnulus(PFGammaCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return theFilteredPFCandsInAnnulus;
}
PFCandidatePtr PFTauElementsOperators::leadPFCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0.){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand =theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidatePtr PFTauElementsOperators::leadPFCand(const math::XYZVector& myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidatePtr PFTauElementsOperators::leadPFChargedHadrCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFChargedHadrCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0.){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand =theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidatePtr PFTauElementsOperators::leadPFChargedHadrCand(const math::XYZVector& myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFChargedHadrCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidatePtr PFTauElementsOperators::leadPFNeutrHadrCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFNeutrHadrCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0.){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand =theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidatePtr PFTauElementsOperators::leadPFNeutrHadrCand(const math::XYZVector& myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFNeutrHadrCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidatePtr PFTauElementsOperators::leadPFGammaCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFGammaCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0.){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand =theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidatePtr PFTauElementsOperators::leadPFGammaCand(const math::XYZVector& myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidatePtr myleadPFCand;
  const std::vector<reco::PFCandidatePtr> theFilteredPFCandsInCone=PFGammaCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (theFilteredPFCandsInCone.size()>0){
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theFilteredPFCandsInCone.begin();iPFCand!=theFilteredPFCandsInCone.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}

void
PFTauElementsOperators::copyCandRefsFilteredByPt(const std::vector<reco::PFCandidatePtr>& theInputCands, std::vector<reco::PFCandidatePtr>& theOutputCands, const double minPt)
{
   for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand  = theInputCands.begin();
                                            iPFCand != theInputCands.end();
                                            ++iPFCand)
   {
      if ( (*iPFCand)->pt() > minPt )
         theOutputCands.push_back(*iPFCand);
   }
}

// Inside Out Signal contents determination algorithm
// Determines tau signal content by building from seed track
void
PFTauElementsOperators::computeInsideOutContents(const std::vector<reco::PFCandidatePtr>& theChargedCands, const std::vector<reco::PFCandidatePtr>& theGammaCands,
       const math::XYZVector& leadTrackVector, const TFormula& coneSizeFormula, double (*ptrToMetricFunction)(const math::XYZVector&, const math::XYZVector&),  
       const double minChargedSize, const double maxChargedSize, const double minNeutralSize, const double maxNeutralSize,
       const double minChargedPt, const double minNeutralPt,
       const string& outlierCollectorConeMetric, const double outlierCollectionMaxSize,
       std::vector<reco::PFCandidatePtr>& signalChargedObjects, std::vector<reco::PFCandidatePtr>& outlierChargedObjects,
       std::vector<reco::PFCandidatePtr>& signalGammaObjects, std::vector<reco::PFCandidatePtr>& outlierGammaObjects, bool useScanningAxis) 
{
   if (theChargedCands.empty() && theGammaCands.empty()) 
      return;
   //copy the vector of PFCands filtering by Pt
   std::vector<reco::PFCandidatePtr> filteredCands;
   filteredCands.reserve(theChargedCands.size() + theGammaCands.size());

   copyCandRefsFilteredByPt(theChargedCands, filteredCands, minChargedPt);
   copyCandRefsFilteredByPt(theGammaCands, filteredCands, minNeutralPt);

   if (filteredCands.empty())
      return;


   //create vector of indexes as RefVectors can't use STL algos?
   unsigned int numberOfCandidates = filteredCands.size();
   std::vector<uint32_t> filteredCandIndexes(numberOfCandidates);
   for(uint32_t theIndex = 0; theIndex < numberOfCandidates; theIndex++)
      filteredCandIndexes[theIndex] = theIndex;
   
   TauTagTools::sortRefsByOpeningDistance myAngularSorter(leadTrackVector, ptrToMetricFunction, filteredCands);

   //sort the remaining candidates by angle to seed track
   sort(filteredCandIndexes.begin(), filteredCandIndexes.end(), myAngularSorter);

   std::vector<reco::PFCandidatePtr> sortedCands;
   for(std::vector<uint32_t>::const_iterator theSortedIndex = filteredCandIndexes.begin();
                                        theSortedIndex != filteredCandIndexes.end();
                                        ++theSortedIndex)
   {
      sortedCands.push_back(filteredCands.at(*theSortedIndex));
   }

   //get first candidate (seed trk by definition)
   std::vector<reco::PFCandidatePtr>::const_iterator signalObjectCandidate = sortedCands.begin();
   double totalEnergySoFar                                    = (**signalObjectCandidate).energy();
   double totalEtSoFar                                        = (**signalObjectCandidate).et();
   math::XYZVector axisVectorSoFar                            = leadTrackVector;
   double currentDelToCenter = ptrToMetricFunction(leadTrackVector, axisVectorSoFar);
   //advance to next object
   ++signalObjectCandidate;
   bool doneBuilding = false;
   while (!doneBuilding && signalObjectCandidate != sortedCands.end())
   {
      //compute cutoff quanity
      math::XYZVector testAxis       = axisVectorSoFar;
      if (useScanningAxis)
         testAxis           += (**signalObjectCandidate).momentum();
      currentDelToCenter    = ptrToMetricFunction((**signalObjectCandidate).momentum(), testAxis);
      double testEnergy     = totalEnergySoFar + (**signalObjectCandidate).energy();
      double testEt         = totalEtSoFar     + (**signalObjectCandidate).et();
      bool isCharged        = (**signalObjectCandidate).charge();
      bool addThisObject    = true;
      if (currentDelToCenter > ((isCharged)?maxChargedSize:maxNeutralSize) ) {
         //max conesize is reached
         addThisObject = false;
      }
      else if (currentDelToCenter > ((isCharged)?minChargedSize:minNeutralSize) )
      {
         //only do calculation if we are in the region above the minimum size
         double cutOffQuantity = coneSizeFormula.Eval(testEnergy, testEt);
         if (currentDelToCenter > cutOffQuantity)
            addThisObject = false;
         else if (useScanningAxis)
         {
            //make sure we don't make the seed track kinematically inconsistent  
            //(stop growth if axis-lead track distance greater than computed opening angle
            if (ptrToMetricFunction(testAxis, leadTrackVector) > cutOffQuantity)
               addThisObject = false;
         }
      }
      if (addThisObject)
      {
         axisVectorSoFar  = testAxis;
         totalEnergySoFar = testEnergy;
         totalEtSoFar     = testEt;
         ++signalObjectCandidate;  //signal object candidate now points to one past the
      }
      else
      {
         doneBuilding = true;
      }
   }
   // split the collection into two collections
   double largest3DOpeningAngleSignal = 0.;  //used for area outlier collection cone
   for (std::vector<reco::PFCandidatePtr>::const_iterator iterCand =  sortedCands.begin();
                                             iterCand != signalObjectCandidate;
                                             ++iterCand)
   {
      double angleToAxis = TauTagTools::computeAngle((*iterCand)->momentum(), axisVectorSoFar);
      if (angleToAxis > largest3DOpeningAngleSignal)
         largest3DOpeningAngleSignal = angleToAxis;

      if ((*iterCand)->charge())
         signalChargedObjects.push_back(*iterCand);
      else
         signalGammaObjects.push_back(*iterCand);
   }

   //function pointer to outlier collection cutoff quantity
   double (*outlierCollectionCalculator)(const math::XYZVector&, const math::XYZVector&) = TauTagTools::computeAngle;
   double outlierCollectionComputedMaxSize = 0;

   if (!outlierCollectorConeMetric.compare("angle"))
   {
      outlierCollectionCalculator       = TauTagTools::computeAngle;
      outlierCollectionComputedMaxSize  = outlierCollectionMaxSize;
   } else if (!outlierCollectorConeMetric.compare("area"))
   {
      //determine opening angle (in 3D space of signal objects)
      
      outlierCollectionCalculator       = TauTagTools::computeAngle;  //the area cone outside angle is defined in 3D
      FixedAreaIsolationCone theFixedAreaCone;
      theFixedAreaCone.setAcceptanceLimit(2.5);
      //calculate new iso cone size
      int errorFlagForConeSizeCalc = 0;
      outlierCollectionComputedMaxSize = theFixedAreaCone(axisVectorSoFar.theta(), axisVectorSoFar.phi(), largest3DOpeningAngleSignal, outlierCollectionMaxSize, errorFlagForConeSizeCalc);
      if (errorFlagForConeSizeCalc != 0)
      {
         edm::LogError("PFRecoTauAlgorithm") << "Error: " << errorFlagForConeSizeCalc << " when calculated const area annulus.  Taking all non-signal PFJet consituents!";
         outlierCollectionComputedMaxSize = 1000.;  //takes everything...
      }
   } else if (!outlierCollectorConeMetric.compare("DR"))
   {
      outlierCollectionCalculator       = TauTagTools::computeDeltaR;
      outlierCollectionComputedMaxSize  = outlierCollectionMaxSize;
   } 
   else //warn if metric choice for outlier collection is not consistent
   {
      edm::LogWarning("PFRecoTauAlgorithm") << "Error in computeInsideOutContents(...):  Outlier (isolation) collector cone metric (" << outlierCollectorConeMetric 
                                            << ") is not recognized! All non-signal associated PFJet constituents will be included in the outliers!";
   }

   for (std::vector<reco::PFCandidatePtr>::const_iterator iterCand =  signalObjectCandidate;
                                             iterCand != sortedCands.end();
                                             ++iterCand)
   {
      //stop adding objects if we have reached the maximum outleir collector size
      double outlierOpeningQuantity = outlierCollectionCalculator(axisVectorSoFar, (*iterCand)->momentum());
      if (outlierOpeningQuantity > outlierCollectionComputedMaxSize) break;

      if ((*iterCand)->charge())
         outlierChargedObjects.push_back(*iterCand);
      else
         outlierGammaObjects.push_back(*iterCand);
   }
   //done
}


// ***
double PFTauElementsOperators::discriminatorByIsolPFCandsN(int IsolPFCands_maxN){
  double myDiscriminator=0.;
  if ((int)IsolPFCands_.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const std::vector<reco::PFCandidatePtr> isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsN(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const std::vector<reco::PFCandidatePtr> isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsN(int IsolPFChargedHadrCands_maxN){
  double myDiscriminator=0.;
  if ((int)IsolPFChargedHadrCands_.size()<=IsolPFChargedHadrCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFChargedHadrCands_maxN){
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  std::vector<reco::PFCandidatePtr> isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFChargedHadrCands.size()<=IsolPFChargedHadrCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsN(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFChargedHadrCands_maxN){
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFChargedHadrCands.size()<=IsolPFChargedHadrCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsN(int IsolPFNeutrHadrCands_maxN){
  double myDiscriminator=0.;
  if ((int)IsolPFNeutrHadrCands_.size()<=IsolPFNeutrHadrCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFNeutrHadrCands_maxN){
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFNeutrHadrCands.size()<=IsolPFNeutrHadrCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsN(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFNeutrHadrCands_maxN){
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFNeutrHadrCands.size()<=IsolPFNeutrHadrCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsN(int IsolPFGammaCands_maxN){
  double myDiscriminator=0.;
  if ((int)IsolPFGammaCands_.size()<=IsolPFGammaCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFGammaCands_maxN){
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFGammaCands.size()<=IsolPFGammaCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsN(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFGammaCands_maxN){
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFGammaCands.size()<=IsolPFGammaCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsEtSum(double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=IsolPFCands_.begin();iPFCand!=IsolPFCands_.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFCands.begin();iPFCand!=isolPFCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsEtSum(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFCands.begin();iPFCand!=isolPFCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsEtSum(double IsolPFChargedHadrCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=IsolPFChargedHadrCands_.begin();iPFCand!=IsolPFChargedHadrCands_.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFChargedHadrCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFChargedHadrCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFChargedHadrCands.begin();iPFCand!=isolPFChargedHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFChargedHadrCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsEtSum(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFChargedHadrCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFChargedHadrCands.begin();iPFCand!=isolPFChargedHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFChargedHadrCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsEtSum(double IsolPFNeutrHadrCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=IsolPFNeutrHadrCands_.begin();iPFCand!=IsolPFNeutrHadrCands_.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFNeutrHadrCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFNeutrHadrCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFNeutrHadrCands.begin();iPFCand!=isolPFNeutrHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFNeutrHadrCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsEtSum(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFNeutrHadrCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFNeutrHadrCands.begin();iPFCand!=isolPFNeutrHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFNeutrHadrCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsEtSum(double IsolPFGammaCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=IsolPFGammaCands_.begin();iPFCand!=IsolPFGammaCands_.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFGammaCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFGammaCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFGammaCands.begin();iPFCand!=isolPFGammaCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFGammaCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsEtSum(const math::XYZVector& myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFGammaCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidatePtr myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   std::vector<reco::PFCandidatePtr> isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=isolPFGammaCands.begin();iPFCand!=isolPFGammaCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFGammaCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
