#include "RecoEgamma/EgammaPhotonAlgos/interface/E1E9CorrectionAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

reco::Photon E1E9CorrectionAlgo::applyBarrelCorrection(const reco::Photon& ph,const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //translate from E1E9EBRY in ORCA

  float CUTR9_BAR_1 = 0.95;
  float CUTR9_BAR_2 = 0.88;

  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr = clshpMap.find(ph.superCluster()->seed());
  assert(seedShpItr != clshpMap.end());
  const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;
  //      float e1 = ph->GetEGCandidate()->data()->seed()->getS1();
  float e1 = seedShapeRef->eMax();
  //      float e9 = ph->GetEGCandidate()->data()->seed()->getS9();
  float e9 = seedShapeRef->e3x3();
  //      float r9 = e9/ph->GetEGCandidate()->data()->energyUncorrected();
  float r9 = e9/unCorrectedEnergy(ph.superCluster());

  float x=e1/e9;
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
  //      float oldCorr =  ph->scEnergy();
  float oldCorr =  ph.superCluster()->energy();
  
  if(x>par[6])
    {
      float aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
      f = aprime+par[3]*x+par[4]*x*x+par[5]*x*x*x;
    }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  //5x5
  float CUTR9_BAR_5x5 = 0.94;
  double par5x5[7] = {0};
  
  if(r9>CUTR9_BAR_5x5)
    {
      par5x5[0]=0.976260; //A0
      par5x5[1]= 0.035083; //A1      
    }
  else
    {
      par5x5[0]=1.; //A0
      par5x5[1]=0.; //A1
    }
      
  
  float f5x5= par5x5[0]+par5x5[1]*x; 
  //      float oldCorrS25 =  ph->s25Energy();     
  float oldCorrS25 =  seedShapeRef->e5x5();                
           
  float correctedS25 = oldCorrS25/f5x5;
            
  float correctedSC = oldCorr/(1.+f);
        
  float bestEstimate = (r9>CUTR9_BAR_5x5) ? correctedS25 : correctedSC;
  
  
  math::XYZTLorentzVector new4p = ph.p4();
  new4p *= (bestEstimate/ph.energy());
  
  reco::Photon correctedPhoton(ph.charge(),new4p, ph.vertex());
  correctedPhoton.setSuperCluster(ph.superCluster());
  return correctedPhoton;    
}


reco::Photon E1E9CorrectionAlgo::applyEndcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //translate from E1E9EFRY in ORCA
  float CUTR9_END_1 = 0.95;
  float CUTR9_END_2 = 0.88;

  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr = clshpMap.find(ph.superCluster()->seed());
  assert(seedShpItr != clshpMap.end());
  const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;  
  //      float e1 = ph->GetEGCandidate()->data()->seed()->getS1();
  float e1 = seedShapeRef->eMax();
  //      float e9 = ph->GetEGCandidate()->data()->seed()->getS9();
  float e9 = seedShapeRef->e3x3();
  //      float r9 = e9/ph->GetEGCandidate()->data()->energyUncorrected();
  float r9 = e9/unCorrectedEnergy(ph.superCluster());

  float x=e1/e9;
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
  //      float oldCorr =  ph->scEnergy();
  float oldCorr =  ph.superCluster()->energy();

  if(x>par[6])
    {
      float aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
      f = aprime+par[3]*x+par[4]*x*x+par[5]*x*x*x;
    }
  else
    f = par[0]+par[1]*x+par[2]*x*x;
  
  float correctedSC = oldCorr/(1.+f);
  //      float correctedS25 = ph->s25Energy();
  //  float correctedS25 = ph.superCluster()->e5x5();
  float bestEstimate = correctedSC;

  
  math::XYZTLorentzVector new4p = ph.p4();
  new4p *= (bestEstimate/ph.energy());
  
  reco::Photon correctedPhoton(ph.charge(),new4p, ph.vertex());
  correctedPhoton.setSuperCluster(ph.superCluster());
  return correctedPhoton;  
}

