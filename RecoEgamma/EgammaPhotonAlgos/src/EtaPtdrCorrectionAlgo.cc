#include "RecoEgamma/EgammaPhotonAlgos/interface/EtaPtdrCorrectionAlgo.h"
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

double EtaPtdrCorrectionAlgo::barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //translate from EtaEBRY in ORCA
  
  float CUTR9_BAR_1 = 0.95;
  float CUTR9_BAR_2 = 0.88;
  
  float x = fabs(ph.superCluster()->position().eta());

  float r9 = ph.r9();
  
  double par[18] = {0};
  
  if (r9>CUTR9_BAR_1) {
	  par[0]= 4.94402e-04; //A0
	  par[1]= 3.34142e-04; //A1
	  par[2]=-6.13033e-04; //A2
	  par[3]= 1.00000e-01; //B0
	  par[4]= 1.00000e-01; //B1
	  par[5]= 1.00000e-01; //B2
	  par[6]= -2.27442e-02; //G0nor
	  par[7]= 8.92159e-03; //G0sig
	  par[8]= -3.42997e-02; //G1nor
	  par[9]= 4.42518e-01; //G1mea
	  par[10]= 6.68076e-03; //G1sig
	  par[11]= -3.03381e-02; //G2nor
	  par[12]= 7.91046e-01; //G2mea
	  par[13]= 7.61364e-03; //G2sig
	  par[14]= -2.96372e-02; //G3nor
	  par[15]= 1.14048e+00; //G3mea
	  par[16]= 7.55460e-03; //G3sig
	  par[17]= 0.00000e+00; //Step
  } 
  else if (r9>CUTR9_BAR_2) {
	  par[0]= 3.71886e-03; //A0
	  par[1]= 4.31974e-03; //A1
	  par[2]=-7.43623e-03; //A2
	  par[3]= 1.00000e-01; //B0
	  par[4]= 1.00000e-01; //B1
	  par[5]= 1.00000e-01; //B2
	  par[6]= -5.00000e-02; //G0nor
	  par[7]= 6.70600e-03; //G0sig
	  par[8]= -3.99968e-02; //G1nor
	  par[9]= 4.40866e-01; //G1mea
	  par[10]= 7.47417e-03; //G1sig
	  par[11]= -3.52818e-02; //G2nor
	  par[12]= 7.91479e-01; //G2mea
	  par[13]= 7.77186e-03; //G2sig
	  par[14]= -2.83228e-02; //G3nor
	  par[15]= 1.13842e+00; //G3mea
	  par[16]= 9.72887e-03; //G3sig
	  par[17]= 0.00000e+00; //Step
  }
  else {
	  par[0]= 6.78413e-03; //A0
	  par[1]= 7.77232e-03; //A1
	  par[2]=-1.42335e-02; //A2
	  par[3]= 1.00000e-01; //B0
	  par[4]= 1.00000e-01; //B1
	  par[5]= 1.00000e-01; //B2
	  par[6]= -4.99999e-02; //G0nor
	  par[7]= 3.52201e-03; //G0sig
	  par[8]= -4.03395e-02; //G1nor
	  par[9]= 4.43275e-01; //G1mea
	  par[10]= 6.81329e-03; //G1sig
	  par[11]= -4.99989e-02; //G2nor
	  par[12]= 7.97909e-01; //G2mea
	  par[13]= 3.01145e-03; //G2sig
	  par[14]= -3.66766e-02; //G3nor
	  par[15]= 1.13918e+00; //G3mea
	  par[16]= 7.88372e-03; //G3sig
	  par[17]= 0.00000e+00; //Step
  }

  //      float oldCorr =  ph->scEnergy();
  float oldCorr =  ph.energy();


  float g00 =  x            / par[7];
  float g04 = (x - par[9])  / par[10];
  float g08 = (x - par[12]) / par[13];
  float g12 = (x - par[15]) / par[16];
  float f= par[6]*exp(-0.5*g00*g00)
    + par[8]*exp(-0.5*g04*g04)
    + par[11]*exp(-0.5*g08*g08)
    + par[14]*exp(-0.5*g12*g12)
    ;
  
  if(x>par[17])
    f+=par[0]+par[1]*x+par[2]*pow(x,2);
  else
    f+=par[0]+par[1]*par[17]+par[2]*pow(par[17],2)-(par[3]*par[17]+par[4]*pow(par[17],2)+par[5]*pow(par[17],3))
      +par[3]*x+par[4]*x*x+par[5]*pow(x,3)
      ;


  if(f>0.1111) f=0.1111;
  if(f<-0.0909) f=-0.0909;

  float newCorr = oldCorr/(1.+f);
  
  //plus a small overall shift  //CHANGED from ORCA to CMSSW
  if(r9>CUTR9_BAR_1) {
    par[1]= -0.0035; //Mean
  }
  else if(r9>CUTR9_BAR_2) {
    par[1]= -0.006; //Mean
  }
  else {
    par[1]= -0.001; //Mean
  }

  float g = par[1];
  
  float correctedSC = newCorr/(1.+g);

  return (correctedSC/ph.energy())*0.965;    
}


double EtaPtdrCorrectionAlgo::endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  float CUTR9_END_1 = 0.95;
  float CUTR9_END_2 = 0.88;
  
  float x = fabs(ph.superCluster()->position().eta());

  float r9 = ph.r9();
  
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


  if(f>0.1111) f=0.1111;
  if(f<-0.0909) f=-0.0909;

  
  float oldCorr =  ph.energy();
  float newCorr = oldCorr/(1.+f);

  //plus a small overall shift //CHANGED from ORCA to CMSSW
  if(r9>CUTR9_END_1) {
    par[1]= -0.0044; //Mean
  }
  else if(r9>CUTR9_END_2) {
    par[1]= -0.006; //Mean
  }
  else {
    par[1]= -0.005; //Mean
  }
  

  float g = par[1];
  float correctedSC = newCorr/(1.+g);

  return (correctedSC/ph.energy())*0.975;  
}
