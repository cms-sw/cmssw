#include "DataFormats/TrackReco/interface/HitPattern.h"


using namespace reco;

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"



HitPattern::HitPattern(const TrackingRecHitRefVector hitlist)
{
  for ( int i = 0 ; i < PatternSize ; i++ ) hitPattern_[i]=0;
  set(hitlist);
}

void HitPattern::set(const TrackingRecHitRefVector & hitlist)
{
  for ( int i = 0 ; i < PatternSize ; i++ ) hitPattern_[i]=0;
  int counter = 0;

  for (trackingRecHit_iterator hit = hitlist.begin() ;
       hit != hitlist.end() && counter < 32*PatternSize/HitSize ;
       hit++, counter++)
    {
       DetId id = (*hit)->geographicalId();
      uint32_t valid = (uint32_t) (*hit)->isValid();
      
      
      uint32_t pattern = 0;
      uint32_t detid=id.det();
      // adding subdetector bit, removing LS bit (wildcard)
      pattern += ((detid)&SubDetectorMask)<<SubDetectorOffset;

      // adding substructure bits, removing LS bit (wildcard)
      uint32_t subdet = id.subdetId();

      pattern += ((subdet)&SubstrMask)<<SubstrOffset;

      uint32_t layer = 0;

      // to understand the layer/disk/wheel number, we need to instantiate each 
      if (detid == DetId::Tracker)
	{
	  if (subdet == PixelSubdetector::PixelBarrel) 
	    layer = PXBDetId(id).layer();
	  else if (subdet == PixelSubdetector::PixelEndcap)
	    layer = PXFDetId(id).disk();
	  else if (subdet == StripSubdetector::TIB)
	    layer = TIBDetId(id).layer();
	  else if (subdet == StripSubdetector::TID)
	    layer = TIDDetId(id).wheel();
	  else if (subdet == StripSubdetector::TOB)
	    layer = TOBDetId(id).layer();
	  else if (subdet == StripSubdetector::TEC)
	    layer = TECDetId(id).wheel();
	}
      else if (detid == DetId::Muon)
	{
	  if (subdet == (uint32_t) MuonSubdetId::DT) 
	    layer = DTLayerId(id.rawId()).layer();
	  else if (subdet == (uint32_t) MuonSubdetId::CSC)
	    layer = CSCDetId(id.rawId()).layer();
	  else if (subdet == (uint32_t) MuonSubdetId::RPC)
	    layer = RPCDetId(id.rawId()).layer();
	}


      pattern += (layer&LayerMask)<<LayerOffset;
      pattern += (valid&ValidMask)<<ValidOffset;

      setHitPattern(counter,pattern);
    }

}
                                                                                        


bool HitPattern::hasValidHitInFirstPixelBarrel() const
{
  
  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
    {
      uint32_t pattern = getHitPattern(i);
      
      if (validHitFilter(pattern) && trackerHitFilter(pattern))
	{
	  
	  uint32_t substructure = (pattern >> SubstrOffset) 
	    & SubstrMask + 1;
	  
	  if (substructure == PixelSubdetector::PixelBarrel)
	    {
	      int layer = (pattern>>LayerOffset) 
		& LayerMask + 1;
		      if (layer == 1) return true;
	    }
	}
      
    }
  return false;
}

uint32_t HitPattern::getHitPattern(int position) const
{
  int offset = position * HitSize;
  uint32_t pattern = 0; 
  for (int i = 0; i < HitSize; i++)
    {
      int pos = offset + i;
      uint32_t word = hitPattern_[pos / 32];
      uint32_t bit = (word >> (pos%32)) & 0x1;
      pattern += bit << i;
    }
  return pattern;
}

void HitPattern::setHitPattern(int position, uint32_t pattern)
{
  int offset = position * HitSize;
  for (int i = 0; i < HitSize; i++)
    {
      int pos = offset + i;
      uint32_t bit = (pattern >> i) & 0x1;
      hitPattern_[pos / 32] += bit << ((offset + i) % 32); 
    }
}


