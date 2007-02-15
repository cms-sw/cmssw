#include "RecoEgamma/EgammaPhotonAlgos/interface/PhiPtdrCorrectionAlgo.h"
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

double PhiPtdrCorrectionAlgo::barrelCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  float CUTR9_BAR_1 = 0.95;
  float CUTR9_BAR_2 = 0.88;
  
  float x = fmod(ph.superCluster()->position().phi()*360./twopi,20.);

  float r9 = ph.r9();
  
  double par[7] = {0};
      
  if(r9>CUTR9_BAR_1)
    {
	  par[0]=-3.54215e-02; //gaus Norm
	  par[1]= 1.00000e+01; //Mean
	  par[2]= 3.11864e-01; //Sigma
	  par[3]= 0.00000e+00; //Agaus Norm
	  par[4]= 1.00000e+01; //AMean
	  par[5]= 1.00000e-01; //ASigma
	  par[6]= 4.93000e-04; //Constant
    }
  else if(r9>CUTR9_BAR_2)
    {
	  par[0]=-8.86087e-02; //gaus Norm
	  par[1]= 1.00000e+01; //Mean
	  par[2]= 2.41492e-01; //Sigma
	  par[3]= 0.00000e+00; //Agaus Norm
	  par[4]= 1.00000e+01; //AMean
	  par[5]= 1.00000e-01; //ASigma
	  par[6]= 2.83604e-04; //Constant
    }
  else
    {
	  par[0]=-1.34181e-02; //gaus Norm
	  par[1]= 1.00000e+01; //Mean
	  par[2]= 1.20060e+00; //Sigma
	  par[3]= 0.00000e+00; //Agaus Norm
	  par[4]= 1.00000e+01; //AMean
	  par[5]= 1.00000e-01; //ASigma
	  par[6]= 2.17577e-03; //Constant
    }
  float arg2 = (x - par[1])/par[2];
  float arg3 = (x - par[4])/par[5];
  float f = par[6]+par[0]*exp(-0.5*arg2*arg2)+par[3]*exp(-0.5*arg3*arg3);      

  if(f>0.1111) f=0.1111;
  if(f<-0.0909) f=-0.0909;

  float oldCorr =  ph.energy();
  float correctedSC = oldCorr/(1.+f);

  return correctedSC/ph.energy();    
}


double PhiPtdrCorrectionAlgo::endcapCorrection(const reco::Photon& ph, const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //NO CORRECTION
  return 1;  
}

