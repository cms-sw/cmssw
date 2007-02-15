#include "RecoEgamma/EgammaPhotonAlgos/interface/E9ESCCorrectionAlgo.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"


double E9ESCCorrectionAlgo::barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //translate from E9ESCEBRY in ORCA
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
  
 
  //      float uncorrectedE = ph->scEnergy();
  float uncorrectedE = ph.superCluster()->energy();
  float f; 
  
  
  if(x>par[6]) {
    double aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
    f = aprime +par[3]*x+par[4]*x*x+par[5]*x*x*x; }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  //5x5
  float CUTR9_BAR_5x5 = 0.94;
  double par5x5[7] = {0};
  
  if(x>CUTR9_BAR_5x5)
    {
      par5x5[0]=0.970778; //A0      
    }
  else
    {
      par5x5[0]=1.;//A0
	
    }
  
  
  float f5x5=par5x5[0]; 
  //      float uncorrectedE25 =  ph->s25Energy();
  float uncorrectedE25 =  ph.e5x5();  
  float correctedSC = uncorrectedE/(1.+f);
  float correctedS25 = uncorrectedE25/f5x5;  
  float bestEstimate = (x>CUTR9_BAR_5x5) ? correctedS25 : correctedSC;
      
  return bestEstimate/ph.energy();    
}


double E9ESCCorrectionAlgo::endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //translate from E9ESCEFRY in ORCA
  float x=ph.r9();
  
  double par[7] = {0};
  
  par[0]=-4.24042e-02; //A0
  par[1]=-2.94145e-02; //A1
  par[2]= 6.27221e-02; //A2
  par[3]=-9.61526e+01; //B0
  par[4]= 1.11165e+02; //B1
  par[5]=-4.26139e+01; //B2
  par[6]= 8.90000e-01; //Step
  
  
  //      float uncorrectedE = ph->scEnergy();
  float uncorrectedE = ph.superCluster()->energy();
  
  float f;
  if(x>par[6]) {
    double aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
    f = aprime +par[3]*x+par[4]*x*x+par[5]*x*x*x; }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  float correctedSC = uncorrectedE/(1.+f);
  //      float correctedS25 = ph->s25Energy();
  //float correctedS25 = ph.superCluster()->energy();
  
  float bestEstimate =  correctedSC;
  
  return bestEstimate/ph.energy(); 
}


