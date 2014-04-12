#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "Math/GenVector/VectorUtil.h"


L2TauIsolationAlgs::L2TauIsolationAlgs()
{

}

L2TauIsolationAlgs::~L2TauIsolationAlgs()
{

}



double 
L2TauIsolationAlgs::isolatedEt(const math::PtEtaPhiELorentzVectorCollection& hits,const math::XYZVector& direction,double innerCone, double outerCone) const
{
  //Compute the Isolation
  double eRMax = 0.;
  double eRMin = 0;
  if(hits.size()>0)
    for (math::PtEtaPhiELorentzVectorCollection::const_iterator mRH = hits.begin();mRH!=hits.end();++mRH)
    {
	  double delta = ROOT::Math::VectorUtil::DeltaR(direction,(*mRH));
	  if(delta <outerCone)
	    eRMax+= mRH->pt();
	  if(delta <innerCone)
	    eRMin+= mRH->pt();
    }
  double etIsol = eRMax - eRMin;
  return etIsol;
}


int 
L2TauIsolationAlgs::nClustersAnnulus(const math::PtEtaPhiELorentzVectorCollection& hits,const math::XYZVector& direction,double innerCone, double outerCone) const
{
  //Compute the Isolation
  int cands=0;
  if(hits.size()>0)
    for (math::PtEtaPhiELorentzVectorCollection::const_iterator mRH = hits.begin();mRH!=hits.end();++mRH)
    {
	  double delta = ROOT::Math::VectorUtil::DeltaR(direction,(*mRH));
	  if(delta <outerCone)
	    if(delta >innerCone)
	      cands++;
    }
  return cands;
}


std::vector<double> 
L2TauIsolationAlgs::clusterShape(const math::PtEtaPhiELorentzVectorCollection& hits,const math::XYZVector& direction,double innerCone,double outerCone) const
{
  double eta_rms=0.;
  double phi_rms=0.;
  double dr_rms=0.;
  double sumpt = 0.;

  std::vector<double> rmsVector; //declare the vector
  if(hits.size()>0)
    {
      for(math::PtEtaPhiELorentzVectorCollection::const_iterator  c = hits.begin();c!=hits.end();++c) //loop on clusters
	{
	  eta_rms+=c->pt()*std::pow(c->eta()-direction.eta(),2);
	  phi_rms+=c->pt()*std::pow(ROOT::Math::VectorUtil::DeltaPhi(*c,direction),2);
	  dr_rms+=c->pt()*std::pow(ROOT::Math::VectorUtil::DeltaR(*c,direction),2);
	  sumpt+=c->pt();			   
	}
    }
  else
    {
      eta_rms=0.;
      phi_rms=0.;
      dr_rms =0.;
      sumpt=1.;
    }

  rmsVector.push_back(eta_rms/sumpt);
  rmsVector.push_back(phi_rms/sumpt);
  rmsVector.push_back(dr_rms/sumpt);

  return rmsVector;
}

