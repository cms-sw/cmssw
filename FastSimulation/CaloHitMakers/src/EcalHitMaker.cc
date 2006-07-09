#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include <algorithm>

EcalHitMaker::EcalHitMaker(Calorimeter * theCalo,
			   const HepPoint3D& ecalentrance, 
			   const DetId& cell, int onEcal,
			   unsigned size, unsigned showertype):
  CaloHitMaker(theCalo,DetId::Ecal,((onEcal==1)?EcalBarrel:EcalEndcap),onEcal,showertype),
  EcalEntrance_(ecalentrance),myTrack_(NULL)
{
  X0depthoffset_ = 0. ;
  X0PS1_ = 0.;
  X0PS2_ = 0.; 
  X0ECAL_ = 27.;
  X0EHGAP_ = 3.;
  X0HCAL_ = 900.;
  L0PS1_ = 0.;
  L0PS2_ = 0.;
  L0ECAL_ = 0.;
  L0EHGAP_ = 0.;
  maxX0_ = 30.;
  totalX0_ = 30;
  totalL0_ = 0. ;
  pivot_ = cell;

}

EcalHitMaker::~EcalHitMaker()
{
  ;
}

bool EcalHitMaker::addHitDepth(double r,double phi,double depth)
{
  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.rawId());
  if(itcheck==hitMap_.end())
    {
      hitMap_.insert(std::pair<uint32_t,float>(pivot_.rawId(),spotEnergy));
    }
  else
    {
      itcheck->second+=spotEnergy;
    }
  return true;
}


bool EcalHitMaker::addHit(double r,double phi,unsigned layer)
{
  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.rawId());
  if(itcheck==hitMap_.end())
    {
      hitMap_.insert(std::pair<uint32_t,float>(pivot_.rawId(),spotEnergy));
    }
  else
    {
      itcheck->second+=spotEnergy;
    }
  return true;
}


bool EcalHitMaker::getQuads(double depth)
{
  return true;
}

void EcalHitMaker::setTrackParameters(const HepNormal3D& normal,
				   double X0depthoffset,
				   const FSimTrack& theTrack)
{
  myTrack_=&theTrack;
  normal_=normal.unit();
  X0depthoffset_=X0depthoffset;
}
