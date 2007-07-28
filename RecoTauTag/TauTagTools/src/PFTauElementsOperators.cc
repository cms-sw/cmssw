#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"

 PFCandidateRefVector PFTauElementsOperators::PFCandsInCone(const PFCandidateRefVector thePFCands,const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const{     
  PFCandidateRefVector tmp;
  if (conemetric=="DR"){
    tmp=PFCandsinCone_DRmetric_(myVector,metricDR_,conesize,thePFCands);
  }else if(conemetric=="angle"){
    tmp=PFCandsinCone_Anglemetric_(myVector,metricAngle_,conesize,thePFCands);
  }else if(conemetric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaCone;
    theFixedAreaCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    double coneangle=theFixedAreaCone(myVector.theta(),myVector.phi(),0,conesize,errorFlag); 
    if (errorFlag!=0)return tmp;   
    tmp=PFCandsinCone_Anglemetric_(myVector,metricAngle_,coneangle,thePFCands);
  }else return PFCandidateRefVector();
  PFCandidateRefVector selectedPFCands;
  for (PFCandidateRefVector::const_iterator iPFCand=tmp.begin();iPFCand!=tmp.end();++iPFCand) {
    if ((**iPFCand).pt()>minPt)selectedPFCands.push_back(*iPFCand);
  }  
  return selectedPFCands;
}
 PFCandidateRefVector PFTauElementsOperators::PFCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const{     
  PFCandidateRefVector tmp=PFCandsInCone(PFCands_,myVector,conemetric,conesize,minPt);
  return tmp;
}
 PFCandidateRefVector PFTauElementsOperators::PFChargedHadrCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const{     
  PFCandidateRefVector tmp=PFCandsInCone(PFChargedHadrCands_,myVector,conemetric,conesize,minPt);
  return tmp;
  }
 PFCandidateRefVector PFTauElementsOperators::PFNeutrHadrCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const{     
  PFCandidateRefVector tmp=PFCandsInCone(PFNeutrHadrCands_,myVector,conemetric,conesize,minPt);
  return tmp;
}
 PFCandidateRefVector PFTauElementsOperators::PFGammaCandsInCone(const math::XYZVector myVector,const string conemetric,const double conesize,const double minPt)const{     
  PFCandidateRefVector tmp=PFCandsInCone(PFGammaCands_,myVector,conemetric,conesize,minPt);
  return tmp;
}
 PFCandidateRefVector PFTauElementsOperators::PFCandsInAnnulus(const PFCandidateRefVector thePFCands,const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  PFCandidateRefVector tmp;
  if (outercone_metric=="DR"){
    if (innercone_metric=="DR"){
      tmp=PFCandsinAnnulus_innerDRouterDRmetrics_(myVector,metricDR_,innercone_size,metricDR_,outercone_size,thePFCands);
    }else if(innercone_metric=="angle"){
      tmp=PFCandsinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_size,metricDR_,outercone_size,thePFCands);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return tmp;
      tmp=PFCandsinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_angle,metricDR_,outercone_size,thePFCands);
    }else return PFCandidateRefVector();
  }else if(outercone_metric=="angle"){
    if (innercone_metric=="DR"){
      tmp=PFCandsinAnnulus_innerDRouterAnglemetrics_(myVector,metricDR_,innercone_size,metricAngle_,outercone_size,thePFCands);
    }else if(innercone_metric=="angle"){
      tmp=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_size,thePFCands);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return tmp;
      tmp=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_size,thePFCands);
    }else return PFCandidateRefVector();
  }else if(outercone_metric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaSignalCone;
    theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    if (innercone_metric=="DR"){
      // not implemented yet
    }else if(innercone_metric=="angle"){
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_size,outercone_size,errorFlag);    
      if (errorFlag!=0)return tmp;
      tmp=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_angle,thePFCands);
    }else if(innercone_metric=="area"){
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);    
      if (errorFlag!=0)return tmp;
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_angle,outercone_size,errorFlag);
      if (errorFlag!=0)return tmp;
      tmp=PFCandsinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_angle,thePFCands);
    }else return PFCandidateRefVector();
  }
  PFCandidateRefVector selectedPFCands;
  for (PFCandidateRefVector::const_iterator iPFCand=tmp.begin();iPFCand!=tmp.end();++iPFCand) {
    if ((**iPFCand).pt()>minPt)selectedPFCands.push_back(*iPFCand);
  }  
  return selectedPFCands;
}
 PFCandidateRefVector PFTauElementsOperators::PFCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  PFCandidateRefVector tmp=PFCandsInAnnulus(PFCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return tmp;
}
 PFCandidateRefVector PFTauElementsOperators::PFChargedHadrCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt){     
  PFCandidateRefVector tmp=PFCandsInAnnulus(PFChargedHadrCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return tmp;
}
 PFCandidateRefVector PFTauElementsOperators::PFNeutrHadrCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  PFCandidateRefVector tmp=PFCandsInAnnulus(PFNeutrHadrCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return tmp;
}
 PFCandidateRefVector PFTauElementsOperators::PFGammaCandsInAnnulus(const math::XYZVector myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  PFCandidateRefVector tmp=PFCandsInAnnulus(PFGammaCands_,myVector,innercone_metric,innercone_size,outercone_metric,outercone_size,minPt);
  return tmp;
}
PFCandidateRef PFTauElementsOperators::leadPFCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0.){
    for(PFCandidateRefVector::const_iterator iPFCand =tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidateRef PFTauElementsOperators::leadPFCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0){
    for(PFCandidateRefVector::const_iterator iPFCand=tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidateRef PFTauElementsOperators::leadPFChargedHadrCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFChargedHadrCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0.){
    for(PFCandidateRefVector::const_iterator iPFCand =tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidateRef PFTauElementsOperators::leadPFChargedHadrCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFChargedHadrCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0){
    for(PFCandidateRefVector::const_iterator iPFCand=tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidateRef PFTauElementsOperators::leadPFNeutrHadrCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFNeutrHadrCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0.){
    for(PFCandidateRefVector::const_iterator iPFCand =tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidateRef PFTauElementsOperators::leadPFNeutrHadrCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFNeutrHadrCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0){
    for(PFCandidateRefVector::const_iterator iPFCand=tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidateRef PFTauElementsOperators::leadPFGammaCand(const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFGammaCandsInCone(PFJet_XYZVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0.){
    for(PFCandidateRefVector::const_iterator iPFCand =tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
PFCandidateRef PFTauElementsOperators::leadPFGammaCand(const math::XYZVector myVector,const string matchingcone_metric,const double matchingcone_size,const double minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFGammaCandsInCone(myVector,matchingcone_metric,matchingcone_size,minPt);
  double pt_cut=minPt;
  if (tmp.size()>0){
    for(PFCandidateRefVector::const_iterator iPFCand=tmp.begin();iPFCand!=tmp.end();iPFCand++){
      if((*iPFCand)->pt()>pt_cut) {
	myleadPFCand=*iPFCand;
	pt_cut=(**iPFCand).pt();
      }
    }
  }
  return myleadPFCand;
}
// ***
double PFTauElementsOperators::discriminatorByIsolPFCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFChargedHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFChargedHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFNeutrHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFNeutrHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsN(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  if ((int)isolPFGammaCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsN(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,int IsolPFCands_maxN){
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  if ((int)isolPFGammaCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFCands.begin();iPFCand!=isolPFCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFCands=PFCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFCands.begin();iPFCand!=isolPFCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFChargedHadrCands.begin();iPFCand!=isolPFChargedHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFChargedHadrCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFChargedHadrCands.begin();iPFCand!=isolPFChargedHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFNeutrHadrCands.begin();iPFCand!=isolPFNeutrHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFNeutrHadrCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFNeutrHadrCands.begin();iPFCand!=isolPFNeutrHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsEtSum(string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFGammaCands.begin();iPFCand!=isolPFGammaCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFTauElementsOperators::discriminatorByIsolPFGammaCandsEtSum(math::XYZVector myVector,string matchingcone_metric,double matchingcone_size,string signalcone_metric,double signalcone_size,string isolcone_metric,double isolcone_size,bool useOnlyChargedHadrforleadPFCand,double minPt_leadPFCand,double minPt_PFCand,double IsolPFCands_maxEtSum){
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_metric,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  //if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
   PFCandidateRefVector isolPFGammaCands=PFGammaCandsInAnnulus(leadPFCand_XYZVector,signalcone_metric,signalcone_size,isolcone_metric,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFGammaCands.begin();iPFCand!=isolPFGammaCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
