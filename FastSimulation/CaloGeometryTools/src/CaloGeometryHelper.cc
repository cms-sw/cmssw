#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

// needed for the debugging
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

#include "FastSimulation/CaloGeometryTools/interface/DistanceToCell.h"
#include "FastSimulation/CaloGeometryTools/interface/Crystal.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

CaloGeometryHelper::CaloGeometryHelper():Calorimeter()
{
  neighbourmapcalculated_= false;
  psLayer1Z_ = 303;
  psLayer2Z_ = 307;
}

CaloGeometryHelper::CaloGeometryHelper(const edm::ParameterSet& fastCalo):Calorimeter(fastCalo)
{
  //  std::cout << " In the constructor with ParameterSet " << std::endl;
  psLayer1Z_ = 303;
  psLayer2Z_ = 307;
}

void CaloGeometryHelper::initialize(double bField)
{
  buildCrystalArray();
  buildNeighbourArray();
  bfield_ = bField;
  preshowerPresent_=(getEcalPreshowerGeometry()!=0);
    
  if(preshowerPresent_)
    {
      ESDetId cps1(getEcalPreshowerGeometry()->getClosestCellInPlane(GlobalPoint(80.,80.,303.),1));
      psLayer1Z_ = getEcalPreshowerGeometry()->getGeometry(cps1)->getPosition().z();
      ESDetId cps2(getEcalPreshowerGeometry()->getClosestCellInPlane(GlobalPoint(80.,80.,307.),2));
      psLayer2Z_ = getEcalPreshowerGeometry()->getGeometry(cps2)->getPosition().z();
      LogDebug("CaloGeometryTools")  << " Preshower layer positions " << psLayer1Z_ << " " << psLayer2Z_ << std::endl;
    }
  else
    LogDebug("CaloGeometryTools")  << " No preshower present" << std::endl;

  //  std::cout << " Preshower layer positions " << psLayer1Z_ << " " << psLayer2Z_ << std::endl;

}

CaloGeometryHelper::~CaloGeometryHelper()
{;
}

DetId CaloGeometryHelper::getClosestCell(const XYZPoint& point, bool ecal, bool central) const
{
  DetId result;
  if(ecal)
    {
      if(central)
	{
	  //	  std::cout << "EcalBarrelGeometry_" << " " << EcalBarrelGeometry_ << std::endl;
	  result = EcalBarrelGeometry_->getClosestCell(GlobalPoint(point.X(),point.Y(),point.Z()));
#ifdef DEBUGGCC
	  if(result.null()) return result;
	  GlobalPoint ip=GlobalPoint(point.X(),point.Y(),point.Z());
	  GlobalPoint cc=EcalBarrelGeometry_->getGeometry(result)->getPosition();
	  float deltaeta2 = ip.eta()-cc.eta();
	  deltaeta2 *= deltaeta2;
	  float deltaphi2 = acos(cos(ip.phi()-cc.phi()));
	  deltaphi2 *= deltaphi2;
	  Histos::instance()->fill("h100",point.eta(),sqrt(deltaeta2+deltaphi2));
#endif
	}
      else
	{
	  result = EcalEndcapGeometry_->getClosestCell(GlobalPoint(point.X(),point.Y(),point.Z()));
#ifdef DEBUGGCC
	  if(result.null()) 
	    {
	      return result;
	    }
	  GlobalPoint ip=GlobalPoint(point.X(),point.Y(),point.Z());
	  GlobalPoint cc=EcalEndcapGeometry_->getGeometry(result)->getPosition();
	  Histos::instance()->fill("h110",point.eta(),(ip-cc).perp());
#endif
	}
    }
  else
    {
      result=HcalGeometry_->getClosestCell(GlobalPoint(point.X(),point.Y(),point.Z()));
      HcalDetId myDetId(result);

      // special patch for HF
      if ( myDetId.subdetId() == HcalForward ) {
	int mylayer;
	if ( fabs(point.Z()) > 1132. ) {
	  mylayer = 2;
	} else {
	  mylayer = 1;
	}
	HcalDetId myDetId2((HcalSubdetector)myDetId.subdetId(),myDetId.ieta(),myDetId.iphi(),mylayer);
	result = myDetId2;
	return result;
      }


      if(result.subdetId()!=HcalEndcap) return result;
      // Special patch to correct the HCAL geometry
      if(myDetId.depth()==3) return result;

      int ieta=myDetId.ietaAbs();
      float azmin=400.458;         /// in sync with BaseParticlePropagator 

      if(ieta<=17) 
        return result;
      else if(ieta>=18 && ieta<=26) 
        azmin += 35.0;    // don't consider ieta=18 nose separately
      else if(ieta>=27)
        azmin += 21.0;

      HcalDetId first(HcalEndcap,myDetId.ieta(),myDetId.iphi(),1);
      bool layer2=(fabs(point.Z())>azmin);
      if(!layer2)
        {
          return first;
        }
      else
        {
          HcalDetId second(HcalEndcap,myDetId.ieta(),myDetId.iphi(),2);
	  if(second!=HcalDetId()) result=second;
	}
#ifdef DEBUGGCC
      if(result.null()) 
	{
	  return result;
	}
      GlobalPoint ip=GlobalPoint(point.x(),point.y(),point.z());
      GlobalPoint cc=HcalGeometry_->getGeometry(result)->getPosition();
      float deltaeta2 = ip.eta()-cc.eta();
      deltaeta2 *= deltaeta2;
      float deltaphi2 = acos(cos(ip.phi()-cc.phi()));
      deltaphi2 *= deltaphi2;

      Histos::instance()->fill("h120",point.eta(),sqrt(deltaeta2+deltaphi2));
#endif
      
    }
  return result;
}

void CaloGeometryHelper::getWindow(const DetId& pivot,int s1,int s2,std::vector<DetId>& vec) const
{
  // currently the getWindow method is the same for EcalBarrelTopology and EndcapTopology
  // (implemented in CaloSubDetectorTopology)
  // optimized versions are foreseen 
  vec=getEcalTopology(pivot.subdetId())->getWindow(pivot,s1,s2);
  DistanceToCell distance(getEcalGeometry(pivot.subdetId()),pivot);
  sort(vec.begin(),vec.end(),distance);
}

void CaloGeometryHelper::buildCrystal(const DetId & cell,Crystal& xtal) const
{
  if(cell.subdetId()==EcalBarrel)
    {
      xtal=Crystal(cell,&barrelCrystals_[EBDetId(cell).hashedIndex()]);
      return;
    }
  if(cell.subdetId()==EcalEndcap)
    {
      xtal=Crystal(cell,&endcapCrystals_[EEDetId(cell).hashedIndex()]);
      return;
    }     
}

// Build the array of (max)8 neighbors
void CaloGeometryHelper::buildNeighbourArray()
{

  static const CaloDirection orderedDir[8]={SOUTHWEST,SOUTH,SOUTHEAST,WEST,EAST,NORTHWEST,NORTH,
					    NORTHEAST};

  const unsigned nbarrel = EBDetId::kSizeForDenseIndexing;
  // Barrel first. The hashed index runs from 0 to 61199
  barrelNeighbours_.resize(nbarrel);
  
  //std::cout << " Building the array of neighbours (barrel) " ;

  const std::vector<DetId>&  vec(EcalBarrelGeometry_->getValidDetIds(DetId::Ecal,EcalBarrel));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      // We get the 9 cells in a square. 
      std::vector<DetId> neighbours(EcalBarrelTopology_->getWindow(vec[ic],3,3));
      //      std::cout << " Cell " << EBDetId(vec[ic]) << std::endl;
      unsigned nneighbours=neighbours.size();

      unsigned hashedindex=EBDetId(vec[ic]).hashedIndex();
      if(hashedindex>=nbarrel)
	{
	  LogDebug("CaloGeometryTools")  << " Array overflow " << std::endl;
	}


      // If there are 9 cells, it is easy, and this order is know:
//      6  7  8
//      3  4  5 
//      0  1  2   (0 = SOUTHWEST)

      if(nneighbours==9)
	{
	  //barrelNeighbours_[hashedindex].reserve(8);
	  unsigned int nn=0;
	  for(unsigned in=0;in<nneighbours;++in)
	    {
	      // remove the centre
	      if(neighbours[in]!=vec[ic]) 
		{
		  barrelNeighbours_[hashedindex][nn]=(neighbours[in]);
		  nn++;
		  //	      std::cout << " Neighbour " << in << " " << EBDetId(neighbours[in]) << std::endl;
		}
	    }
	}
      else
	{
	  DetId central(vec[ic]);
	  //barrelNeighbours_[hashedindex].resize(8,DetId(0));
	  for(unsigned idir=0;idir<8;++idir)
	    {
	      DetId testid=central;
	      bool status=move(testid,orderedDir[idir],false);
	      if(status) barrelNeighbours_[hashedindex][idir]=testid;
	    }

	}
    }

  // Moved to the endcap

  //  std::cout << " done " << size << std::endl;
  //  std::cout << " Building the array of neighbours (endcap) " ;


  const std::vector<DetId> & vece(EcalEndcapGeometry_->getValidDetIds(DetId::Ecal,EcalEndcap));
  size=vece.size();    
  // There are some holes in the hashedIndex for the EE. Hence the array is bigger than the number
  // of crystals
  const unsigned nendcap=EEDetId::kSizeForDenseIndexing;

  endcapNeighbours_.resize(nendcap);
  for(unsigned ic=0; ic<size; ++ic) 
    {
      // We get the 9 cells in a square. 
      std::vector<DetId> neighbours(EcalEndcapTopology_->getWindow(vece[ic],3,3));
      unsigned nneighbours=neighbours.size();
      // remove the centre
      unsigned hashedindex=EEDetId(vece[ic]).hashedIndex();
      
      if(hashedindex>=nendcap)
	{
	  LogDebug("CaloGeometryTools")  << " Array overflow " << std::endl;
	}

      if(nneighbours==9)
	{
	  //endcapNeighbours_[hashedindex].reserve(8);
	  unsigned int nn=0;
	  for(unsigned in=0;in<nneighbours;++in)
	    {	  
	      // remove the centre
	      if(neighbours[in]!=vece[ic]) 
		{
		  endcapNeighbours_[hashedindex][nn]=(neighbours[in]);
		  nn++;
		}
	    }
	}
      else
	{
	  DetId central(vece[ic]);
	  //endcapNeighbours_[hashedindex].resize(8,DetId(0));
	  for(unsigned idir=0;idir<8;++idir)
	    {
	      DetId testid=central;
	      bool status=move(testid,orderedDir[idir],false);
	      if(status) endcapNeighbours_[hashedindex][idir]=testid;
	    }

	}
    }
  //  std::cout << " done " << size <<std::endl;
  neighbourmapcalculated_ = true;
}

const CaloGeometryHelper::NeiVect& CaloGeometryHelper::getNeighbours(const DetId& detid) const
{
  return (detid.subdetId()==EcalBarrel)?barrelNeighbours_[EBDetId(detid).hashedIndex()]:
    endcapNeighbours_[EEDetId(detid).hashedIndex()];
}

bool CaloGeometryHelper::move(DetId& cell, const CaloDirection&dir,bool fast) const
{  
  DetId originalcell = cell; 
  if(dir==NONE || cell==DetId(0)) return false;

  // Conversion CaloDirection and index in the table
  // CaloDirection :NONE,SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST, NORTHEAST,NORTHWEST,NORTH
  // Table : SOUTHWEST,SOUTH,SOUTHEAST,WEST,EAST,NORTHWEST,NORTH, NORTHEAST
  static const int calodirections[9]={-1,1,2,0,4,3,7,5,6};
    
  if(fast&&neighbourmapcalculated_)
    {
      DetId result = (originalcell.subdetId()==EcalBarrel) ? 
	barrelNeighbours_[EBDetId(originalcell).hashedIndex()][calodirections[dir]]:
	endcapNeighbours_[EEDetId(originalcell).hashedIndex()][calodirections[dir]];
      bool status =  !result.null();
      cell = result;
      return status; 
    }
  
  if(dir==NORTH || dir ==SOUTH || dir==EAST || dir==WEST)
    {
      return simplemove(cell,dir);
    }
  else
    {
      if(dir == NORTHEAST || dir==NORTHWEST || dir==SOUTHEAST || dir==SOUTHWEST)
	return diagonalmove(cell,dir);
    }
  
  cell = DetId(0);
  return false;
}


bool CaloGeometryHelper::simplemove(DetId& cell, const CaloDirection& dir) const
{
  std::vector<DetId> neighbours;
  if(cell.subdetId()==EcalBarrel)
    neighbours = EcalBarrelTopology_->getNeighbours(cell,dir);
  else if(cell.subdetId()==EcalEndcap)
    neighbours= EcalEndcapTopology_->getNeighbours(cell,dir);
  
  if(neighbours.size()>0 && !neighbours[0].null())
    {
      cell = neighbours[0];
      return true;
    }
  else 
    {
      cell = DetId(0);
      return false;
    }
}

bool CaloGeometryHelper::diagonalmove(DetId& cell, const CaloDirection& dir) const
{
  bool result; 
  // One has to try both paths
  if(dir==NORTHEAST)
    {
      result = simplemove(cell,NORTH);
      if(result)
	return simplemove(cell,EAST);
      else
	{
	  result = simplemove(cell,EAST);
	  if(result)
	    return simplemove(cell,NORTH);
	  else
	    return false; 
	}
    }
  else if(dir==NORTHWEST)
    {
      result = simplemove(cell,NORTH);
      if(result)
	return simplemove(cell,WEST);
      else
	{
	  result = simplemove(cell,WEST);
	  if(result)
	    return simplemove(cell,NORTH);
	  else
	    return false; 
	}
    }
  else if(dir == SOUTHEAST)
    {
      result = simplemove(cell,SOUTH);
      if(result)
	return simplemove(cell,EAST);
      else
	{
	  result = simplemove(cell,EAST);
	  if(result)
	    return simplemove(cell,SOUTH);
	  else
	    return false; 
	}
    }
  else if(dir == SOUTHWEST)
    {
      result = simplemove(cell,SOUTH);
      if(result)
	return simplemove(cell,WEST);
      else
	{
	  result = simplemove(cell,SOUTH);
	  if(result)
	    return simplemove(cell,WEST);
	  else
	    return false; 
	}
    }
  cell = DetId(0);
  return false;
}

bool CaloGeometryHelper::borderCrossing(const DetId& c1, const DetId& c2) const
{
  if(c1.subdetId()!=c2.subdetId()) return false;

  if(c1.subdetId()==EcalBarrel)
    {
      // there is a crack if the two cells don't belong to the same 
      // module
      EBDetId cc1(c1);
      EBDetId cc2(c2);
      return (cc1.im()!=cc2.im()||cc1.ism()!=cc2.ism() );
    }
  
if(c1.subdetId()==EcalEndcap)
    {
      // there is a crack if the two cells don't belong to the same 
      // module
      return (EEDetId(c1).isc()!=EEDetId(c2).isc());
    }
 return false;
}

void CaloGeometryHelper::buildCrystalArray()
{
  const unsigned nbarrel = EBDetId::kSizeForDenseIndexing;
  // Barrel first. The hashed index runs from 0 to 61199
  barrelCrystals_.resize(nbarrel,BaseCrystal());

  //std::cout << " Building the array of crystals (barrel) " ;
  const std::vector<DetId>&  vec(EcalBarrelGeometry_->getValidDetIds(DetId::Ecal,EcalBarrel));
  unsigned size=vec.size();    
  const CaloCellGeometry * geom=0;
  for(unsigned ic=0; ic<size; ++ic) 
    {
      unsigned hashedindex=EBDetId(vec[ic]).hashedIndex();
      geom = EcalBarrelGeometry_->getGeometry(vec[ic]);
      BaseCrystal xtal(vec[ic]);
      xtal.setCorners(geom->getCorners(),geom->getPosition());
      barrelCrystals_[hashedindex]=xtal;
    }
  
  //  std::cout << " done " << size << std::endl;
  //  std::cout << " Building the array of crystals (endcap) " ;
  

  const std::vector<DetId>&  vece(EcalEndcapGeometry_->getValidDetIds(DetId::Ecal,EcalEndcap));
  size=vece.size();    
  // There are some holes in the hashedIndex for the EE. Hence the array is bigger than the number
  // of crystals
  const unsigned nendcap=EEDetId::kSizeForDenseIndexing;

  endcapCrystals_.resize(nendcap,BaseCrystal());
  for(unsigned ic=0; ic<size; ++ic) 
    {
      unsigned hashedindex=EEDetId(vece[ic]).hashedIndex();
      geom = EcalEndcapGeometry_->getGeometry(vece[ic]);
      BaseCrystal xtal(vece[ic]);
      xtal.setCorners(geom->getCorners(),geom->getPosition());
      endcapCrystals_[hashedindex]=xtal;
    }
  //  std::cout << " done " << size << std::endl;
}
