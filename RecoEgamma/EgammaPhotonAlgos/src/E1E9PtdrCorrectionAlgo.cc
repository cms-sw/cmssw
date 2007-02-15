#include "RecoEgamma/EgammaPhotonAlgos/interface/E1E9PtdrCorrectionAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

double E1E9PtdrCorrectionAlgo::barrelCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  float CUTR9_BAR_1 = 0.95;
  float CUTR9_BAR_2 = 0.88;

  float r9 = ph.r9();

  float x=ph.r19();
  if(x>0.84) x = 0.84;
  
  double par[7] = {0};

  if(r9>CUTR9_BAR_1)
    {
      par[0]=-2.32636e-02; //A0
      par[1]= 2.05843e-02; //A1
      par[2]= 1.74828e-02; //A2
      par[3]=-1.04690e+03; //B0
      par[4]= 1.29310e+03; //B1
      par[5]=-5.32320e+02; //B2
      par[6]= 8.00000e-01; //Step
    }
  else if(r9>CUTR9_BAR_2)
    {
      par[0]=-2.45606e-02; //A0
      par[1]= 0.00000e+00; //A1
      par[2]= 0.00000e+00; //A2
      par[3]= 4.70311e-02; //B0
      par[4]=-8.93206e-03; //B1
      par[5]=-4.53717e-03; //B2
      par[6]= 0.00000e+00; //Step
    }
  else
    {
      par[0]=-9.90714e-02; //A0
      par[1]= 0.00000e+00; //A1
      par[2]= 0.00000e+00; //A2
      par[3]= 4.85494e-01; //B0
      par[4]=-7.94980e-01; //B1
      par[5]= 4.35771e-01; //B2
      par[6]= 0.00000e+00; //Step      
    }

  float f; 
  float oldCorr =  ph.energy();
  
  if(x>par[6])
    {
      float aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
      f = aprime+par[3]*x+par[4]*x*x+par[5]*x*x*x;
    }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  if(f>0.1111) f=0.1111;
  if(f<-0.0909) f=-0.0909;
  float correctedSC = oldCorr/(1.+f);
        
  return correctedSC/ph.energy();    
}


double E1E9PtdrCorrectionAlgo::endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  float CUTR9_END_1 = 0.95;
  float CUTR9_END_2 = 0.88;

  float r9 = ph.r9();

  float x=ph.r19();
  if(x>0.865) x=0.865;
  
  double par[7] = {0};
  
  if(r9>CUTR9_END_1)
    {
      par[0]=-2.59239e-02; //A0
      par[1]= 2.18800e-02; //A1
      par[2]= 1.93251e-02; //A2
      par[3]=-2.58468e+02; //B0
      par[4]= 3.17632e+02; //B1
      par[5]=-1.30021e+02; //B2
      par[6]= 8.00000e-01; //Step
    }
  else if(r9>CUTR9_END_2)
    {
      par[0]=-5.70955e-02; //A0
      par[1]= 0.00000e+00; //A1
      par[2]= 0.00000e+00; //A2
      par[3]= 2.27551e-01; //B0
      par[4]=-3.46722e-01; //B1
      par[5]= 2.10467e-01; //B2
      par[6]= 0.00000e+00; //Step
    }
  else
    {
      par[0]=-1.57218e-02; //A0
      par[1]= 4.04455e-02; //A1
      par[2]=-2.19763e-02; //A2
      par[3]=-1.96940e+02; //B0
      par[4]= 5.31627e-02; //B1
      par[5]= 9.82211e+01; //B2
      par[6]= 8.10000e-01; //Step
    }

  
  float f; 
  float oldCorr =  ph.energy();

  if(x>par[6])
    {
      float aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
      f = aprime+par[3]*x+par[4]*x*x+par[5]*x*x*x;
    }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  if(f>0.1111) f=0.1111;
  if(f<-0.0909) f=-0.0909;

  float correctedSC = oldCorr/(1.+f);
  
  return correctedSC/ph.energy();  
}

