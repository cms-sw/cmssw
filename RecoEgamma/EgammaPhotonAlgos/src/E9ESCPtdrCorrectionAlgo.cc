#include "RecoEgamma/EgammaPhotonAlgos/interface/E9ESCPtdrCorrectionAlgo.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"


double E9ESCPtdrCorrectionAlgo::barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  float x=ph.r9();
  if (x>1.) x=1.;

  double par[7] = {0};
  
  par[0]=-6.56543e-02; //A0
  par[1]= 4.22053e-02; //A1
  par[2]=-1.09911e-02; //A2
  par[3]=-1.26122e+02; //B0
  par[4]= 1.44254e+02; //B1
  par[5]=-5.47766e+01; //B2
  par[6]= 8.90000e-01; //Step
  
  //float uncorrectedE = ph.superCluster()->energy();
  float uncorrectedE = ph.energy();
  float f; 
  
  if(x>par[6]) {
    double aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
    f = aprime +par[3]*x+par[4]*x*x+par[5]*x*x*x; }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  if(f>0.1111) f=0.1111;
  if(f<-0.0909) f=-0.0909;
  float correctedSC = uncorrectedE/(1.+f);
      
  return correctedSC/ph.energy();    
}


double E9ESCPtdrCorrectionAlgo::endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  float x=ph.r9();
  
  double par[7] = {0};
  
  par[0]=-4.24042e-02; //A0
  par[1]=-2.94145e-02; //A1
  par[2]= 6.27221e-02; //A2
  par[3]=-9.61526e+01; //B0
  par[4]= 1.11165e+02; //B1
  par[5]=-4.26139e+01; //B2
  par[6]= 8.90000e-01; //Step
  

  //ENDCAPS SHOULD BE FIXED at the first level, in the photon producer
  float uncorrectedE = ph.energy();
  
  float f;
  if(x>par[6]) {
    double aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
    f = aprime +par[3]*x+par[4]*x*x+par[5]*x*x*x; }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  if(f>0.1111) f=0.1111;
  if(f<-0.0909) f=-0.0909;

  float correctedSC = uncorrectedE/(1.+f);

  return correctedSC/ph.energy();  
}


