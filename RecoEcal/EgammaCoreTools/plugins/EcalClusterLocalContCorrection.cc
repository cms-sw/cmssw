#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterLocalContCorrection.h"
#include "TVector2.h"
#include "TMath.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

//////From DummyHepMCAnalyzer.cc:
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace std;
using namespace edm;

float EcalClusterLocalContCorrection::getValue( const reco::BasicCluster & basicCluster, const EcalRecHitCollection & recHit) const
{
        checkInit();
        // private member params_ = EcalClusterLocalContCorrectionParameters
        // (see in CondFormats/EcalObjects/interface)
        EcalFunctionParameters::const_iterator it;
        std::cout << "[[EcalClusterLocalContCorrection::getValue]] " 
                << params_->params().size() << " parameters:";
        for ( it = params_->params().begin(); it != params_->params().end(); ++it ) {
                std::cout << " " << *it;
        }
        std::cout << "\n";
        return 1;
}


float EcalClusterLocalContCorrection::getValue( const reco::SuperCluster & superCluster, const int mode ) const
{
  checkInit();
  
  //correction factor to be returned, and to be calculated in this present function:
  double correction_factor=1.;
  double fetacor=1.; //eta dependent part of the correction factor
  double fphicor=1.; //phi dependent part of the correction factor

  //********************************************************************************************************************//
  //These local containment corrections correct a photon energy for leakage outside a 5x5 crystal cluster. They  depend on the local position in the hit crystal. The local position coordinates, called later EtaCry and PhiCry in the code, are comprised between -0.5 and 0.5 and correspond to the distance between the photon supercluster position and the center of the hit crystal, expressed in number of  crystal widthes. The correction parameters (that should be filled in CalibCalorimetry/EcalTrivialCondModules/python/EcalTrivialCondRetriever_cfi.py) were calculated using simulaion and thus take into account the effect of the magnetic field. They  only apply to unconverted photons in the barrel, but a use for non brem electrons could be considered (not tested yet). For more details, cf the CMS internal note 2009-013 by S. Tourneur and C. Seez

  //Beware: The user should make sure it only uses this correction factor for unconverted photons (or not breming electrons)


  const reco::CaloClusterPtr & seedbclus =  superCluster.seed();
  
  //If not barrel, return 1:
  if (TMath::Abs(seedbclus->eta()) >1.4442 ) return 1.;
  
  edm::ESHandle<CaloGeometry> pG;
  es_->get<CaloGeometryRecord>().get(pG); 
  
  const CaloSubdetectorGeometry* geom=pG->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);//EcalBarrel = 1
  
  const math::XYZPoint position_ = seedbclus->position(); 
  double Theta = -position_.theta()+0.5*TMath::Pi();
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());
  
  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  const float X0 = 0.89; const float T0 = 7.4;
  double depth = X0 * (T0 + log(seedbclus->energy()));
  
  
  //search which crystal is closest to the cluster position and call it crystalseed:
  //std::vector<DetId> crystals_vector = seedbclus->getHitsByDetId();   //deprecated
  std::vector< std::pair<DetId, float> > crystals_vector = seedbclus->hitsAndFractions();
  float dphimin=999.;
  float detamin=999.;
  int ietaclosest = 0;
  int iphiclosest = 0;
  for (unsigned int icry=0; icry!=crystals_vector.size(); ++icry) {    
    EBDetId crystal(crystals_vector[icry].first);
    const CaloCellGeometry* cell=geom->getGeometry(crystal);
    GlobalPoint center_pos = (dynamic_cast<const TruncatedPyramid*>(cell))->getPosition(depth);
    double EtaCentr = center_pos.eta();
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
    if (TMath::Abs(EtaCentr-Eta) < detamin) {
      detamin = TMath::Abs(EtaCentr-Eta); 
      ietaclosest = crystal.ieta();
    }
    if (TMath::Abs(TVector2::Phi_mpi_pi(PhiCentr-Phi)) < dphimin) {
      dphimin = TMath::Abs(TVector2::Phi_mpi_pi(PhiCentr-Phi)); 
      iphiclosest = crystal.iphi();
    }
  }
  EBDetId crystalseed(ietaclosest, iphiclosest);
  
  // Get center cell position from shower depth
  const CaloCellGeometry* cell=geom->getGeometry(crystalseed);
  GlobalPoint center_pos = (dynamic_cast<const TruncatedPyramid*>(cell))->getPosition(depth);
  
  //if the seed crystal is neighbourgh of a supermodule border, don't apply the phi dependent  containment corrections, but use the larger crack corrections instead.
  int iphimod20 = TMath::Abs(iphiclosest%20);
   if ( iphimod20 <=1 ) fphicor=1.;

   else{
      double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
      double PhiWidth = (TMath::Pi()/180.);
      double PhiCry = (TVector2::Phi_mpi_pi(Phi-PhiCentr))/PhiWidth;
      if (PhiCry>0.5) PhiCry=0.5;
      if (PhiCry<-0.5) PhiCry=-0.5;
       //Some flips to take into account ECAL barrel symmetries:
      if (ietaclosest<0) PhiCry *= -1.;
      
      //Fetching parameters of the polynomial (see  CMS IN-2009/013)
      double g[5];
      for (int k=0; k!=5; ++k) g[k] = (params_->params())[k+5];
      
      fphicor=0.;
      for (int k=0; k!=5; ++k) fphicor += g[k]*std::pow(PhiCry,k);
   }
   
   //if the seed crystal is neighbourgh of a module border, don't apply the eta dependent  containment corrections, but use the larger crack corrections instead.
  int ietamod20 = TMath::Abs(ietaclosest%20);
  if (TMath::Abs(ietaclosest) >24 && (ietamod20==5 || ietamod20==6) ) fetacor = 1.;
  
  else
    {      
      double ThetaCentr = -center_pos.theta()+0.5*TMath::Pi();
      double ThetaWidth = (TMath::Pi()/180.)*TMath::Cos(ThetaCentr);
      double EtaCry = (Theta-ThetaCentr)/ThetaWidth;    
      if (EtaCry>0.5) EtaCry=0.5;
      if (EtaCry<-0.5) EtaCry=-0.5;
      //flip to take into account ECAL barrel symmetries:
      if (ietaclosest<0) EtaCry *= -1.;
      
      //Fetching parameters of the polynomial (see  CMS IN-2009/013)
      double f[5];
      for (int k=0; k!=5; ++k) f[k] = (params_->params())[k];
     
      fetacor=0.;
      for (int k=0; k!=5; ++k) fetacor += f[k]*std::pow(EtaCry,k);
    }
  

  correction_factor  = (params_->params())[10]/(fetacor*fphicor);
  
  //*********************************************************************************************************************//
  
  //return the correction factor. Use it to multiply the cluster energy.
  return correction_factor;
}





#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN( EcalClusterFunctionFactory, EcalClusterLocalContCorrection, "EcalClusterLocalContCorrection");
