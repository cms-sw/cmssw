#include "DataFormats/BTauReco/interface/PFIsolatedTauTagInfo.h"

using namespace edm;
using namespace reco;
using namespace std;

const PFCandidateRefVector PFIsolatedTauTagInfo::PFCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFCands_.begin();iPFCand!=PFCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR<conesize) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRefVector PFIsolatedTauTagInfo::PFChargedHadrCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFChargedHadrCands_.begin();iPFCand!=PFChargedHadrCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR<conesize) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRefVector PFIsolatedTauTagInfo::PFNeutrHadrCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFNeutrHadrCands_.begin();iPFCand!=PFNeutrHadrCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR<conesize) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRefVector PFIsolatedTauTagInfo::PFGammaCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFGammaCands_.begin();iPFCand!=PFGammaCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR<conesize) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRefVector PFIsolatedTauTagInfo::PFCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFCands_.begin();iPFCand!=PFCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR>=innercone_size && deltaR<outercone_size) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRefVector PFIsolatedTauTagInfo::PFChargedHadrCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFChargedHadrCands_.begin();iPFCand!=PFChargedHadrCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR>=innercone_size && deltaR<outercone_size) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRefVector PFIsolatedTauTagInfo::PFNeutrHadrCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFNeutrHadrCands_.begin();iPFCand!=PFNeutrHadrCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR>=innercone_size && deltaR<outercone_size) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRefVector PFIsolatedTauTagInfo::PFGammaCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const{     
  PFCandidateRefVector tmp;
  for(PFCandidateRefVector::const_iterator iPFCand=PFGammaCands_.begin();iPFCand!=PFGammaCands_.end();iPFCand++){
    if((**iPFCand).pt()<minPt)continue;
    const math::XYZVector PFCand_XYZVector=(**iPFCand).momentum() ;
    float deltaR=ROOT::Math::VectorUtil::DeltaR(myVector,PFCand_XYZVector);
    if (deltaR>=innercone_size && deltaR<outercone_size) tmp.push_back(*iPFCand);
  }
  return tmp;
}
const PFCandidateRef PFIsolatedTauTagInfo::leadPFCand(const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFCandsInCone(PFJet_XYZVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
const PFCandidateRef PFIsolatedTauTagInfo::leadPFCand(const math::XYZVector myVector,const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFCandsInCone(myVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
const PFCandidateRef PFIsolatedTauTagInfo::leadPFChargedHadrCand(const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFChargedHadrCandsInCone(PFJet_XYZVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
const PFCandidateRef PFIsolatedTauTagInfo::leadPFChargedHadrCand(const math::XYZVector myVector,const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFChargedHadrCandsInCone(myVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
const PFCandidateRef PFIsolatedTauTagInfo::leadPFNeutrHadrCand(const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFNeutrHadrCandsInCone(PFJet_XYZVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
const PFCandidateRef PFIsolatedTauTagInfo::leadPFNeutrHadrCand(const math::XYZVector myVector,const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFNeutrHadrCandsInCone(myVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
const PFCandidateRef PFIsolatedTauTagInfo::leadPFGammaCand(const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  if (!PFJetRef_) return myleadPFCand;
  math::XYZVector PFJet_XYZVector=(*PFJetRef_).momentum();
  const PFCandidateRefVector tmp=PFGammaCandsInCone(PFJet_XYZVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
const PFCandidateRef PFIsolatedTauTagInfo::leadPFGammaCand(const math::XYZVector myVector,const float matchingcone_size,const float minPt)const{
  PFCandidateRef myleadPFCand;
  const PFCandidateRefVector tmp=PFGammaCandsInCone(myVector,matchingcone_size,minPt);
  float pt_cut=minPt;
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
double PFIsolatedTauTagInfo::discriminatorByIsolPFCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFCands=PFCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  if ((int)isolPFCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFCands=PFCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  if ((int)isolPFCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFChargedHadrCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  if ((int)isolPFChargedHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFChargedHadrCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  if ((int)isolPFChargedHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFNeutrHadrCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  if ((int)isolPFNeutrHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFNeutrHadrCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  if ((int)isolPFNeutrHadrCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFGammaCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFGammaCands=PFGammaCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  if ((int)isolPFGammaCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFGammaCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN)const{
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFGammaCands=PFGammaCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  if ((int)isolPFGammaCands.size()<=IsolPFCands_maxN) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFCands=PFCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFCands.begin();iPFCand!=isolPFCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFCands=PFCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFCands.begin();iPFCand!=isolPFCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFChargedHadrCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFChargedHadrCands.begin();iPFCand!=isolPFChargedHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFChargedHadrCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFChargedHadrCands=PFChargedHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFChargedHadrCands.begin();iPFCand!=isolPFChargedHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFNeutrHadrCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFNeutrHadrCands.begin();iPFCand!=isolPFNeutrHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFNeutrHadrCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFNeutrHadrCands=PFNeutrHadrCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFNeutrHadrCands.begin();iPFCand!=isolPFNeutrHadrCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFGammaCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0.;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(matchingcone_size,minPt_leadPFCand);  
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFGammaCands=PFGammaCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);   
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFGammaCands.begin();iPFCand!=isolPFGammaCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
double PFIsolatedTauTagInfo::discriminatorByIsolPFGammaCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum)const{
  double myIsolPFCandsEtSum=0.;
  double myDiscriminator=0;
  PFCandidateRef myleadPFCand;
  if (useOnlyChargedHadrforleadPFCand) myleadPFCand=leadPFChargedHadrCand(myVector,matchingcone_size,minPt_leadPFCand);  
  else myleadPFCand=leadPFCand(myVector,matchingcone_size,minPt_leadPFCand); 
  if(!myleadPFCand)return myDiscriminator;
  if(signalcone_size>=isolcone_size) return 1.;
  math::XYZVector leadPFCand_XYZVector=(*myleadPFCand).momentum() ;
  const PFCandidateRefVector isolPFGammaCands=PFGammaCandsInBand(leadPFCand_XYZVector,signalcone_size,isolcone_size,minPt_PFCand);  
  for(PFCandidateRefVector::const_iterator iPFCand=isolPFGammaCands.begin();iPFCand!=isolPFGammaCands.end();iPFCand++) myIsolPFCandsEtSum+=(**iPFCand).et();
  if (myIsolPFCandsEtSum<=IsolPFCands_maxEtSum) myDiscriminator=1;
  return myDiscriminator;
}
// ***
void PFIsolatedTauTagInfo::filterPFChargedHadrCands(double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointDZ,bool UsePVconstraint,double PVtx_Z,bool UseOnlyChargedHadr_for_LeadCand,double LeadChargedHadrCandtoJet_MatchingConeSize,double LeadChargedHadrCand_minPt){
  PFCandidateRefVector filteredPFCands;
  PFCandidateRefVector filteredPFChargedHadrCands;
  for(PFCandidateRefVector::const_iterator iPFCand=PFCands_.begin();iPFCand!=PFCands_.end();iPFCand++){
    if ((**iPFCand).particleId()==PFChargedHadrCand_codenumber){
      // *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
      TrackRef PFChargedHadrCand_rectk;
      if ((**iPFCand).block()->elements().size()!=0){
	for (OwnVector<PFBlockElement>::const_iterator iPFBlock=(**iPFCand).block()->elements().begin();iPFBlock!=(**iPFCand).block()->elements().end();iPFBlock++){
	  if ((*iPFBlock).type()==PFRecTrack_codenumber && ROOT::Math::VectorUtil::DeltaR((**iPFCand).momentum(),(*iPFBlock).trackRef()->momentum())<0.001){
	    PFChargedHadrCand_rectk=(*iPFBlock).trackRef();
	  }
	}
      }else continue;
      if (!PFChargedHadrCand_rectk)continue;
      if ((*PFChargedHadrCand_rectk).pt()>=ChargedHadrCand_tkminPt &&
	  (*PFChargedHadrCand_rectk).normalizedChi2()<=ChargedHadrCand_tkmaxChi2 &&
	  fabs((*PFChargedHadrCand_rectk).d0())<=ChargedHadrCand_tkmaxipt &&
	  (*PFChargedHadrCand_rectk).recHitsSize()>=(unsigned int)ChargedHadrCand_tkminTrackerHitsn &&
	  (*PFChargedHadrCand_rectk).hitPattern().numberOfValidPixelHits()>=ChargedHadrCand_tkminPixelHitsn){
	if (UsePVconstraint){
	  if (fabs((*PFChargedHadrCand_rectk).dz()-PVtx_Z)<=ChargedHadrCand_tktorefpointDZ){
	    filteredPFChargedHadrCands.push_back(*iPFCand);
	  }
	}else{
	  filteredPFChargedHadrCands.push_back(*iPFCand);
	}
      }
    }else filteredPFCands.push_back(*iPFCand);
  }
  if (!UsePVconstraint){
    if (UseOnlyChargedHadr_for_LeadCand){
      PFCandidateRef myleadPFChargedHadrCand=leadPFChargedHadrCand(LeadChargedHadrCandtoJet_MatchingConeSize,LeadChargedHadrCand_minPt);  
      if (!myleadPFChargedHadrCand){}
      else{
	PFCandidateRefVector filteredPFChargedHadrCandsbis;        
	TrackRef myleadPFChargedHadrCand_rectk;
	for (OwnVector<PFBlockElement>::const_iterator iPFBlock=(*myleadPFChargedHadrCand).block()->elements().begin();iPFBlock!=(*myleadPFChargedHadrCand).block()->elements().end();iPFBlock++){
	  if ((*iPFBlock).type()==PFRecTrack_codenumber && ROOT::Math::VectorUtil::DeltaR((*myleadPFChargedHadrCand).momentum(),(*iPFBlock).trackRef()->momentum())<0.001) myleadPFChargedHadrCand_rectk=(*iPFBlock).trackRef();
	}
	for(PFCandidateRefVector::const_iterator iPFCand=filteredPFChargedHadrCands.begin();iPFCand!=filteredPFChargedHadrCands.end();iPFCand++){	  
	  TrackRef PFChargedHadrCand_rectk;
	  for (OwnVector<PFBlockElement>::const_iterator iPFBlock=(**iPFCand).block()->elements().begin();iPFBlock!=(**iPFCand).block()->elements().end();iPFBlock++){
	    if ((*iPFBlock).type()==PFRecTrack_codenumber && ROOT::Math::VectorUtil::DeltaR((**iPFCand).momentum(),(*iPFBlock).trackRef()->momentum())<0.001) PFChargedHadrCand_rectk=(*iPFBlock).trackRef();
	  }
	  if (fabs((*PFChargedHadrCand_rectk).dz()-(*myleadPFChargedHadrCand_rectk).dz())<=ChargedHadrCand_tktorefpointDZ){
	    filteredPFChargedHadrCandsbis.push_back(*iPFCand);
	  }
	}
	filteredPFChargedHadrCands=filteredPFChargedHadrCandsbis;
      }
    }
  }
  for(PFCandidateRefVector::const_iterator iPFCand=filteredPFChargedHadrCands.begin();iPFCand!=filteredPFChargedHadrCands.end();iPFCand++)filteredPFCands.push_back(*iPFCand);
  PFCands_=filteredPFCands;
  PFChargedHadrCands_=filteredPFChargedHadrCands;
}
void PFIsolatedTauTagInfo::filterPFNeutrHadrCands(double NeutrHadrCand_HcalclusminE){
  PFCandidateRefVector filteredPFCands;
  PFCandidateRefVector filteredPFNeutrHadrCands;
  for(PFCandidateRefVector::const_iterator iPFCand=PFCands_.begin();iPFCand!=PFCands_.end();iPFCand++){
    if ((**iPFCand).particleId()==PFNeutrHadrCand_codenumber){
      // *** Whether the neutral hadron candidate will be selected or not depends on its rec. HCAL cluster properties. 
      if ((**iPFCand).energy()>=NeutrHadrCand_HcalclusminE){
	filteredPFCands.push_back(*iPFCand);
	filteredPFNeutrHadrCands.push_back(*iPFCand);
      }
    }else filteredPFCands.push_back(*iPFCand);
  }
  PFCands_=filteredPFCands;
  PFNeutrHadrCands_=filteredPFNeutrHadrCands;
}
void PFIsolatedTauTagInfo::filterPFGammaCands(double GammaCand_EcalclusminE){
  PFCandidateRefVector filteredPFCands;
  PFCandidateRefVector filteredPFGammaCands;
  for(PFCandidateRefVector::const_iterator iPFCand=PFCands_.begin();iPFCand!=PFCands_.end();iPFCand++){
    if ((**iPFCand).particleId()==PFGammaCand_codenumber){
      // *** Whether the gamma candidate will be selected or not depends on its rec. ECAL cluster properties. 
      if ((**iPFCand).energy()>=GammaCand_EcalclusminE){
	filteredPFCands.push_back(*iPFCand);
	filteredPFGammaCands.push_back(*iPFCand);
      }
    }else filteredPFCands.push_back(*iPFCand);
  }
  PFCands_=filteredPFCands;
  PFGammaCands_=filteredPFGammaCands;
}
void PFIsolatedTauTagInfo::removefilters(){
  PFCands_=initialPFCands_;
  PFChargedHadrCands_=initialPFChargedHadrCands_;
  PFNeutrHadrCands_=initialPFNeutrHadrCands_;
  PFGammaCands_=initialPFGammaCands_;  
}
