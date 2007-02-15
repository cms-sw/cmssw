#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonBasketBorderCorrectionAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/EcalBarrelAlgo/interface/EcalBarrelGeometry.h"

#include <map>
#include <vector>
#include <algorithm>


double PhotonBasketBorderCorrectionAlgo::barrelCorrection(const reco::Photon&ph,  const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //From EgammaPhoton/src/PhotonEscale.h
  float correctedSC(0);
  
  reco::basicCluster_iterator bcItr;  
  for(bcItr = ph.superCluster()->clustersBegin(); bcItr != ph.superCluster()->clustersEnd(); ++bcItr){
    reco::BasicClusterShapeAssociationCollection::const_iterator shapeItr = clshpMap.find(*bcItr);
    assert(shapeItr != clshpMap.end());
    const reco::ClusterShapeRef& shapeRef = shapeItr->val;    
    correctedSC += basicClusterCorrEnergy(*(*bcItr), *shapeRef);
  }
  correctedSC *= ph.superCluster()->energy()/ph.superCluster()->rawEnergy();

  float r9 = ph.r9();
  float CUTR9_BAR_5x5 = 0.97;
  float correctedS25 = ph.e5x5(); 
  float bestEstimate = (r9>CUTR9_BAR_5x5) ? correctedS25 : correctedSC;

  return bestEstimate/ph.energy();  
}

float PhotonBasketBorderCorrectionAlgo::basicClusterCorrEnergy(const reco::BasicCluster&bc,  const reco::ClusterShape& shape )
{  
  
  float logREta=10.;
  float logRPhi=10.; 
  std::vector<double> energyFractionInPhi_v  = shape.energyBasketFractionPhi();
  std::vector<double> energyFractionInEta_v = shape.energyBasketFractionEta();

  if(energyFractionInPhi_v.size() > 1)
    { 
      //cross border on phi direction
      logRPhi =  log(energyFractionInPhi_v[0]/(1 - energyFractionInPhi_v[0]));
    
      //check the cluster located in east or west side
      if(bc.position().Eta() < 0)logRPhi *= -1.0;
    }
  
  bool crossCenter(false) ;
  if(energyFractionInEta_v.size() > 1)
    {
      //cross border on eta direction
      logREta = log(energyFractionInEta_v[0]/(1-energyFractionInEta_v[0]));
      
      //check the cluster located in east or west side
      //if cross the center of barrel, Cluster postion direct to 
      //center crystal zone(+-5 crystal), 
      crossCenter = ((bc.position().Eta() * logREta) < 0 )&&fabs(bc.position().Eta())< 0.08 ? true:false;
      if(bc.position().Eta() < 0 && !crossCenter)logREta *= -1.0;            
    }
  
  float corPhi = crackCorPhi(logRPhi);
  float corEta = crossCenter ? zeroCrackCor(logREta) : crackCorEta(logREta);
  
  return bc.energy() * corPhi * corEta;
}      

float PhotonBasketBorderCorrectionAlgo::crackCorPhi(float &logphi)
{
  //Copy from ORCA EGBBasketBorderCorr.cc
  float a0; float a1; float a2; float a3; float shift = 1.;

  if (logphi > -4. && logphi < -1.2) {
    a0=0.851531;
    a1=-0.092526;
    a2=-0.017085;
    a3=-0.000692;
  } else if (logphi > 0.3 && logphi < 3.) {
    a0=0.930298;
    a1=0.008094;
    a2=0.019431;
    a3=-0.005055;
  } else {
    a0 = 1.;
    a1 = 0.;
    a2 = 0.;
    a3 = 0.;
    shift = 1.;
  }
    
  float corrfac = 1./(a0 + a1*logphi + a2*pow(logphi,2) + a3*pow(logphi,3));

  return corrfac/shift;
}    

float PhotonBasketBorderCorrectionAlgo::crackCorEta(float &logeta)
{
  //Copy from ORCA EGBBasketBorderCorr.cc
  float a0; float a1; float a2; float a3; float shift = 1.;

  if (logeta > -4. && logeta < -1.2) {
    a0=0.470389;
    a1=-0.437314;
    a2=-0.123885;
    a3=-0.011883;
  } else if (logeta > 0. && logeta < 3.) {
    a0=0.850576;
    a1=0.084013;
    a2=-0.011688;
    a3=-0.000162;
  } else {
    a0 = 1.;
    a1 = 0.;
    a2 = 0.;
    a3 = 0.;
    shift = 1.;
  }
    
  float corrfac = 1./(a0 + a1*logeta + a2*pow(logeta,2) + a3*pow(logeta,3));

  return corrfac/shift;
}

float PhotonBasketBorderCorrectionAlgo::zeroCrackCor(float &logeta)
{
  //Copy from EGBBasketBorderCorr.cc
  float a0; float a1; float a2; float a3;float shift = 1.;

  if (fabs(logeta) > 1. && fabs(logeta) < 3.) {
    a0=0.877007;
    a1=0.020403;
    a2=0.021756;
    a3=-0.005136;
  } else {
    a0 = 1.;
    a1 = 0.;
    a2 = 0.;
    a3 = 0.;
    shift = 1.;
  }
    
  float corrfac = 1./(a0 + a1*fabs(logeta) + a2*pow(logeta,2) + a3*pow(fabs(logeta),3));

  return corrfac/shift;
}


double PhotonBasketBorderCorrectionAlgo::endcapCorrection(const reco::Photon&ph,  const reco::BasicClusterShapeAssociationCollection &clshpMap)
{
  //dummy correction now
  return 1.; 
}

