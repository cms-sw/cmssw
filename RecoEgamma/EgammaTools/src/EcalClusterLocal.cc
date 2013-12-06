#include "../interface/EcalClusterLocal.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "TMath.h"
#include "TVector2.h"

EcalClusterLocal::EcalClusterLocal()
{}

EcalClusterLocal::~EcalClusterLocal()
{}

void EcalClusterLocal::localCoordsEB( const reco::CaloCluster &bclus, const edm::EventSetup &es, float &etacry, float &phicry, int &ieta, int &iphi, float &thetatilt, float &phitilt) const {
  edm::ESHandle<CaloGeometry> pG;
  es.get<CaloGeometryRecord>().get(pG); 
  localCoordsEB( bclus, *pG, etacry, phicry, ieta, iphi, thetatilt, phitilt);
}



 
void EcalClusterLocal::localCoordsEB( const reco::CaloCluster &bclus, const CaloGeometry & caloGeometry,  float &etacry, float &phicry, int &ieta, int &iphi, float &thetatilt, float &phitilt) const
{
  
  assert(bclus.hitsAndFractions().at(0).first.subdetId()==EcalBarrel);
  
  const CaloSubdetectorGeometry* geom=caloGeometry.getSubdetectorGeometry(DetId::Ecal,EcalBarrel);//EcalBarrel = 1
  
  const math::XYZPoint position_ = bclus.position(); 
  double Theta = -position_.theta()+0.5*TMath::Pi();
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());
  
  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  const float X0 = 0.89; const float T0 = 7.4;
  double depth = X0 * (T0 + log(bclus.energy()));
  
  //find max energy crystal
  std::vector< std::pair<DetId, float> > crystals_vector = bclus.hitsAndFractions();
  float drmin = 999.;
  EBDetId crystalseed;
  //printf("starting loop over crystals, etot = %5f:\n",bclus.energy());
  for (unsigned int icry=0; icry!=crystals_vector.size(); ++icry) {    
    
    EBDetId crystal(crystals_vector[icry].first);
        
    const CaloCellGeometry* cell=geom->getGeometry(crystal);
    GlobalPoint center_pos = (dynamic_cast<const TruncatedPyramid*>(cell))->getPosition(depth);
    double EtaCentr = center_pos.eta();
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());

    float dr = reco::deltaR(Eta,Phi,EtaCentr,PhiCentr);
    if (dr<drmin) {
      drmin = dr;
      crystalseed = crystal;
    }
    
  }
  
  ieta = crystalseed.ieta();
  iphi = crystalseed.iphi();
  
  // Get center cell position from shower depth
  const CaloCellGeometry* cell=geom->getGeometry(crystalseed);
  const TruncatedPyramid *cpyr = dynamic_cast<const TruncatedPyramid*>(cell);

  thetatilt = cpyr->getThetaAxis();
  phitilt = cpyr->getPhiAxis();

  GlobalPoint center_pos = cpyr->getPosition(depth);
  
  double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
  double PhiWidth = (TMath::Pi()/180.);
  phicry = (TVector2::Phi_mpi_pi(Phi-PhiCentr))/PhiWidth;
    //Some flips to take into account ECAL barrel symmetries:
  if (ieta<0) phicry *= -1.;  
  
  double ThetaCentr = -center_pos.theta()+0.5*TMath::Pi();
  double ThetaWidth = (TMath::Pi()/180.)*TMath::Cos(ThetaCentr);
  etacry = (Theta-ThetaCentr)/ThetaWidth;    
  //flip to take into account ECAL barrel symmetries:
  if (ieta<0) etacry *= -1.;  
  
  return;

}

void EcalClusterLocal::localCoordsEE( const reco::CaloCluster &bclus, const edm::EventSetup &es, float &xcry, float &ycry, int &ix, int &iy, float &thetatilt, float &phitilt) const
{
  edm::ESHandle<CaloGeometry> pG;
  es.get<CaloGeometryRecord>().get(pG); 
  localCoordsEE( bclus, *pG, xcry, ycry, ix, iy, thetatilt, phitilt);
}

void EcalClusterLocal::localCoordsEE( const reco::CaloCluster &bclus, const CaloGeometry & caloGeometry, float &xcry, float &ycry, int &ix, int &iy, float &thetatilt, float &phitilt) const
{
    
  assert(bclus.hitsAndFractions().at(0).first.subdetId()==EcalEndcap);
  
  const CaloSubdetectorGeometry* geom=caloGeometry.getSubdetectorGeometry(DetId::Ecal,EcalEndcap);//EcalBarrel = 1
  
  const math::XYZPoint position_ = bclus.position(); 
  //double Theta = -position_.theta()+0.5*TMath::Pi();
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());
  double X = position_.x();
  double Y = position_.y();
  
  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  const float X0 = 0.89; float T0 = 1.2;
  //different T0 value if outside of preshower coverage
  if (TMath::Abs(bclus.eta())<1.653) T0 = 3.1;
  
  double depth = X0 * (T0 + log(bclus.energy()));
  
  //find max energy crystal
  std::vector< std::pair<DetId, float> > crystals_vector = bclus.hitsAndFractions();
  float drmin = 999.;
  EEDetId crystalseed;
  //printf("starting loop over crystals, etot = %5f:\n",bclus.energy());
  for (unsigned int icry=0; icry!=crystals_vector.size(); ++icry) {    
    
    EEDetId crystal(crystals_vector[icry].first);
        
    const CaloCellGeometry* cell=geom->getGeometry(crystal);
    GlobalPoint center_pos = (dynamic_cast<const TruncatedPyramid*>(cell))->getPosition(depth);
    double EtaCentr = center_pos.eta();
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());

    float dr = reco::deltaR(Eta,Phi,EtaCentr,PhiCentr);
    if (dr<drmin) {
      drmin = dr;
      crystalseed = crystal;
    }
    
  }
  
  ix = crystalseed.ix();
  iy = crystalseed.iy();
  
  // Get center cell position from shower depth
  const CaloCellGeometry* cell=geom->getGeometry(crystalseed);
  const TruncatedPyramid *cpyr = dynamic_cast<const TruncatedPyramid*>(cell);

  thetatilt = cpyr->getThetaAxis();
  phitilt = cpyr->getPhiAxis();

  GlobalPoint center_pos = cpyr->getPosition(depth);
  
  double XCentr = center_pos.x();
  double XWidth = 2.59;
  xcry = (X-XCentr)/XWidth;
 
  
  double YCentr = center_pos.y();
  double YWidth = 2.59;
  ycry = (Y-YCentr)/YWidth;
  
  return;

}


