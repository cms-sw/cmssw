#include "RecoEgamma/EgammaPhotonAlgos/interface/EtaCorrectionAlgo.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

double EtaCorrectionAlgo::barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //translate from EtaEBRY in ORCA
  
  float CUTR9_BAR_1 = 0.95;
  float CUTR9_BAR_2 = 0.88;
  
  float x = fabs(ph.superCluster()->position().eta());
  float r9=ph.r9();
  
  double par[7] = {0};
  
  if (r9>CUTR9_BAR_1) {
    par[0]= 4.94402e-04; //A0
    par[1]= 3.34142e-04; //A1
    par[2]=-6.13033e-04; //A2    
  } 
  else if (r9>CUTR9_BAR_2) {
    par[0]= 3.71886e-03; //A0
    par[1]= 4.31974e-03; //A1
    par[2]=-7.43623e-03; //A2
  }
  else {
    par[0]= 6.78413e-03; //A0
    par[1]= 7.77232e-03; //A1
    par[2]=-1.42335e-02; //A2    
  }

  //      float oldCorr =  ph->scEnergy();
  float oldCorr =  ph.superCluster()->energy();
  float f = par[0]+par[1]*x+par[2]*pow(x,2);        
  float newCorr = oldCorr/(1.+f);
  
  //plus a small overall shift
  if(r9>CUTR9_BAR_1) {
    par[1]= 1.33791e-03; //Mean
  }
  else if(r9>CUTR9_BAR_2) {
    par[1]= 3.03155e-04; //Mean
  }
  else {
    par[1]= 3.05011e-03; //Mean
  }
  
  float g = par[1];
  
  //5x5
  float CUTR9_BAR_5x5 = 0.94;
  double par5x5[7] = {0};
  
  if (r9>CUTR9_BAR_5x5) {
    par5x5[0]= 1.001881; //A0
    par5x5[1]= 0.000904; //A1
    par5x5[2]=-0.004132; //A2    
  } 
  else {
    //no correction
    par5x5[0]= 1.;
    par5x5[1]= 0.; //A1
    par5x5[2]=0.; //A2
  }
  
  //      float oldCorrS25 =  ph->s25Energy();
  float oldCorrS25 =  ph.e5x5();
  float f5x5 = par5x5[0]+par5x5[1]*x+par5x5[2]*pow(x,2);        
  float newCorrS25 = oldCorrS25/f5x5;
  
  float correctedSC = newCorr/(1.+g);
  float correctedS25 = newCorrS25;
  
  float bestEstimate = (r9>CUTR9_BAR_5x5) ? correctedS25 : correctedSC;
  
  return bestEstimate/ph.energy();    
}


double EtaCorrectionAlgo::endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //translate from EtaEFRY in ORCA
  float CUTR9_END_1 = 0.95;
  float CUTR9_END_2 = 0.88;

  float x = fabs(ph.superCluster()->position().eta());
  float r9=ph.r9();
  
  double par[7] = {0};
  
  if(r9>CUTR9_END_1)
    {
      par[0]=-5.05252e-02; //A0
      par[1]= 0.00000e+00; //A1
      par[2]= 0.00000e+00; //A2
      par[3]= 5.47901e-02; //B0
      par[4]=-2.10686e-02; //B1
      par[5]= 3.07066e-03; //B2
      par[6]= 0.00000e+00; //Step
      
    }
  else if(r9>CUTR9_END_2)
    {
      par[0]=-3.45755e-02; //A0
      par[1]= 0.00000e+00; //A1
      par[2]= 0.00000e+00; //A2
      par[3]= 4.07299e-02; //B0
      par[4]=-2.07674e-02; //B1
      par[5]= 4.49499e-03; //B2
      par[6]= 0.00000e+00; //Step
    }
  else
    {
      par[0]=-1.13141e+00; //A0
      par[1]= 0.00000e+00; //A1
      par[2]= 0.00000e+00; //A2
      par[3]= 1.61507e+00; //B0
      par[4]=-7.60165e-01; //B1
      par[5]= 1.18162e-01; //B2
      par[6]= 0.00000e+00; //Step
    }
  
  float aprime=par[0]+par[1]*par[6]+par[2]*par[6]*par[6]-(par[3]*par[6]+par[4]*par[6]*par[6]+par[5]*par[6]*par[6]*par[6]);
  float f =    aprime+par[3]*x+par[4]*x*x+par[5]*x*x*x;
  
  
  //       float oldCorr =  ph->scEnergy();
  float oldCorr =  ph.superCluster()->energy();
  float newCorr = oldCorr/(1.+f);

  //plus a small overall shift
  if(r9>CUTR9_END_1) {
    par[1]= 5.95200e-04; //Mean
  }
  else if(r9>CUTR9_END_2) {
    par[1]= 7.66434e-04; //Mean
  }
  else {
    par[1]=-1.22161e-04; //Mean
  }
  
  float g = par[1];
  float correctedSC = newCorr/(1.+g);
  //       float correctedS25 = ph->s25Energy();
  //float correctedS25 = ph.superCluster()->e5x5();
  float bestEstimate = correctedSC;

  return bestEstimate/ph.energy();  
}

