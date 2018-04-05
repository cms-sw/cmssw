//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/CrystalWindowMap.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"

#include <algorithm>

CrystalWindowMap::CrystalWindowMap(const CaloGeometryHelper *calo,const std::vector<Crystal> & cw):
  myCalorimeter_(calo),originalVector_(cw)
{
  size_=cw.size();
  if(size_==0) return;
  
  // Loop over the crystals in the grid and stores the index number of the 8 neighbours
  myNeighbours_.resize(size_);

  for(unsigned ic=0; ic<size_; ++ic)
    {
      const CaloGeometryHelper::NeiVect& neighbours=myCalorimeter_->getNeighbours(cw[ic].getDetId());
      myNeighbours_[ic].reserve(8);
      for(unsigned in=0; in<8;++in)
	{
	  // Check that the crystal is in the grid

	  Crystal::crystalEqual myCrystal(neighbours[in]);
	  std::vector<Crystal>::const_iterator itcheck;
	  itcheck=find_if(cw.begin(),cw.end(),myCrystal);
	  // The neighbour might not be in the grid
	  if(itcheck==cw.end())
	    {
//	      std::cout << " Ouh la " << std::endl;
//	      for(unsigned ic=0;ic<size_;++ic)
//		{
//		  std::cout << cw[ic].getDetId().rawId()<<  " " ;
//		}
//	      std::cout << std::endl ; 
//	      std::cout << " We are looking for " << neighbours[in].rawId() << std::endl;
//	      edm::LogWarning("CrystalWindowMap") << " Inconsistency in the CellWindow " << std::endl;      
	    }
	  else
	    {
	      myNeighbours_[ic].push_back(itcheck-cw.begin());
	      //	      std::cout << " index " << itcheck-cw.begin() << std::endl;
	    }
	}
    }
}

bool 
CrystalWindowMap::getCrystalWindow(unsigned iq,std::vector<unsigned>&  cw ) const
{
  if(iq<size_) // iq >= 0, since iq is unsigned
    {
      cw=myNeighbours_[iq];
      return true;
    }
  else
    return false;
}

bool 
CrystalWindowMap::getCrystalWindow(unsigned iq,const std::vector<unsigned>*  cw) const
{
  
  if(iq<size_) // iq >= 0, since iq is unsigned
    {
      cw=&myNeighbours_[iq];
      return true;
    }
  else
    return false;
    //  std::map<CrystalID,CrystalWindow>::const_iterator itcheck=myMap.find(cell);
}

const std::vector<unsigned>& 
CrystalWindowMap::getCrystalWindow(unsigned iq, bool& status) const
{
  return myNeighbours_[iq];
}
 
