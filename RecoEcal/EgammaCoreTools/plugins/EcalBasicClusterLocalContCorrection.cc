#include "RecoEcal/EgammaCoreTools/plugins/EcalBasicClusterLocalContCorrection.h"
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace std;
using namespace edm;


float EcalBasicClusterLocalContCorrection::getValue( const reco::SuperCluster & superCluster, const int mode ) const
{
  //checkInit();
  return 1;
}

float EcalBasicClusterLocalContCorrection::getValue( const reco::BasicCluster & basicCluster, const EcalRecHitCollection & recHit ) const
{
  checkInit();

  // number of parameters needed by this parametrization
  size_t nparams = 24;

  //correction factor to be returned, and to be calculated in this present function:
  double correction_factor=1.;
  double fetacor=1.; //eta dependent part of the correction factor
  double fphicor=1.; //phi dependent part of the correction factor


  //--------------if barrel calculate local position wrt xtal center -------------------
  edm::ESHandle<CaloGeometry> caloGeometry;
  es_->get<CaloGeometryRecord>().get(caloGeometry); 
  const CaloSubdetectorGeometry* geom = caloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);//EcalBarrel = 1
  

  const math::XYZPoint position_ = basicCluster.position(); 
  double Theta = -position_.theta()+0.5*TMath::Pi();
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());
  
  
  
  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  const float X0 = 0.89; const float T0 = 7.4;
  double depth = X0 * (T0 + log(basicCluster.energy()));
  
  
  //search which crystal is closest to the cluster position and call it crystalseed:
  //std::vector<DetId> crystals_vector = *scRef.getHitsByDetId();   //deprecated
  std::vector< std::pair<DetId, float> > crystals_vector = basicCluster.hitsAndFractions();
  float dphimin=999.;
  float detamin=999.;
  int ietaclosest = 0;
  int iphiclosest = 0;
  
  
  for (unsigned int icry=0; icry!=crystals_vector.size(); ++icry) 
    {    
      
      EBDetId crystal(crystals_vector[icry].first);
      const CaloCellGeometry* cell=geom->getGeometry(crystal);// problema qui
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

  //PHI
  double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
  double PhiWidth = (TMath::Pi()/180.);
  double PhiCry = (TVector2::Phi_mpi_pi(Phi-PhiCentr))/PhiWidth;
  if (PhiCry>0.5) PhiCry=0.5;
  if (PhiCry<-0.5) PhiCry=-0.5;
  //flip to take into account ECAL barrel symmetries:
  if (ietaclosest<0) PhiCry *= -1.;
  

  //ETA
  double ThetaCentr = -center_pos.theta()+0.5*TMath::Pi();
  double ThetaWidth = (TMath::Pi()/180.)*TMath::Cos(ThetaCentr);
  double EtaCry = (Theta-ThetaCentr)/ThetaWidth;    
  if (EtaCry>0.5) EtaCry=0.5;
  if (EtaCry<-0.5) EtaCry=-0.5;
  //flip to take into account ECAL barrel symmetries:
  if (ietaclosest<0) EtaCry *= -1.;
  


  //-------------- end calculate local position -------------
  

  size_t payloadsize = params_->params().size();
  
  if (payloadsize < nparams  )
    edm::LogError("Invalid Payload") << "Parametrization requires " << nparams << " parameters but only " << payloadsize << " are found in DB. Perhaps incompatible Global Tag"  << std::endl;
  

  if (payloadsize > nparams  )
    edm::LogWarning("Size mismatch ") << "Parametrization requires " << nparams << " parameters but " << payloadsize << " are found in DB. Perhaps incompatible Global Tag"  << std::endl;
  


  std::pair<double,double> localPosition(EtaCry,PhiCry);

  //--- local cluster coordinates 
  float localEta = localPosition.first;
  float localPhi = localPosition.second;
  
  //--- ecal module
  int imod    = getEcalModule(basicCluster.seed());
    
  //-- corrections parameters
  float pe[3], pp[3];
  pe[0]= (params_->params())[0+imod*3];
  pe[1]= (params_->params())[1+imod*3];
  pe[2]= (params_->params())[2+imod*3];
  pp[0]= (params_->params())[12+imod*3];
  pp[1]= (params_->params())[13+imod*3];
  pp[2]= (params_->params())[14+imod*3];  


  //--- correction vs local eta
  fetacor = pe[0]+pe[1]*localEta+pe[2]*localEta*localEta;

  //--- correction vs local phi
  fphicor = pp[0]+pp[1]*localPhi+pp[2]*localPhi*localPhi;


  //if the seed crystal is neighbourgh of a supermodule border, don't apply the phi dependent  containment corrections, but use the larger crack corrections instead.
  int iphimod20 = TMath::Abs(iphiclosest%20);
  if ( iphimod20 <=1 ) fphicor=1.;

  correction_factor  = (1./fetacor)*(1./fphicor);
  
  //return the correction factor. Use it to multiply the cluster energy.
  return correction_factor;
}


//------------------------------------------------------------------------------------------------------
int EcalBasicClusterLocalContCorrection::getEcalModule(DetId id) const
{
  int mod = 0;
  int ieta = (EBDetId(id)).ieta();
  
  if (fabs(ieta) <=25 ) mod = 0;
  if (fabs(ieta) > 25 && fabs(ieta) <=45 ) mod = 1;
  if (fabs(ieta) > 45 && fabs(ieta) <=65 ) mod = 2;
  if (fabs(ieta) > 65 && fabs(ieta) <=85 ) mod = 3;

  return (mod);
 
}
//------------------------------------------------------------------------------------------------------



#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN( EcalClusterFunctionFactory, EcalBasicClusterLocalContCorrection, "EcalBasicClusterLocalContCorrection");
