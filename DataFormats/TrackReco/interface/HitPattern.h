#ifndef TrackReco_HitPattern_h
#define TrackReco_HitPattern_h
/** \class reco::HitPattern HitPattern.h DataFormats/TrackReco/interface/HitPattern.h
 *
 * HitPattern. Summary of the information of the hits associated to the track 
 * in the AOD, when the RecHits are no longer available, 
 * the compact hit pattern should allow basic track selection based 
 * on number of hits in the various subdetectors
 * object stored in the RECO/AOD
 *
 * \author Marcel Vos, INFN Pisa
 *
 * \version $Id: TrackerHitPattern.h,v 1.0 2006/05/16 vos Exp $
 *
 */
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"






namespace reco {


  class HitPattern
    {
    public:

      /// default constructor

      HitPattern() {}
      HitPattern(TrackingRecHitRefVector hitlist); 

      
      void set(const TrackingRecHitRefVector & hitlist);

      bool validHitFilter(uint32_t pattern) const
	{
	  if ((pattern>>ValidOffset) & ValidMask) return true;
	  return false;
	}
      
      bool trackerHitFilter(uint32_t pattern) const
	{
	  if (DetId::Detector(((pattern>>SubDetectorOffset) & SubDetectorMask)) == DetId::Tracker) return true;
	  return false;
	}

      bool muonHitFilter(uint32_t pattern) const
	{
	  if (DetId::Detector(((pattern>>SubDetectorOffset) & SubDetectorMask)) == DetId::Muon) return true;
	  return false;
	}

      
      bool pixelHitFilter(uint32_t pattern) const
	{ 
	  if (!trackerHitFilter(pattern)) return false;
	  
	  uint32_t substructure = (pattern >> SubstrOffset) & SubstrMask;
	  
	  if (substructure == PixelSubdetector::PixelBarrel || 
	      substructure == PixelSubdetector::PixelEndcap) return true; 
	  return false;
	}
      
      int numberOfValidHits() const
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (validHitFilter(pattern)) count++;
	    }
	    
	  return count;
	}

      int numberOfLostHits() const
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (muonHitFilter(pattern) || trackerHitFilter(pattern))
		{
		  if (!validHitFilter(pattern)) count++;
		}
	    }
	  return count;
	}

      
      int numberOfValidMuonHits() const
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (validHitFilter(pattern) 
		  && muonHitFilter(pattern)) count++;
	    }
	    
	  return count;
	}

      int numberOfLostMuonHits() const
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (!validHitFilter(pattern) 
		  && muonHitFilter(pattern)) count++;

	    }
	  return count;
	}

	

      int numberOfValidTrackerHits() const
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (validHitFilter(pattern) 
		  && trackerHitFilter(pattern)) count++;
	    }
	  return count;
	}

      int numberOfLostTrackerHits()
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (!validHitFilter(pattern) 
		  && trackerHitFilter(pattern)) count++;

	    }
	  return count;
	}
      
      int numberOfValidPixelHits() const
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (validHitFilter(pattern) 
		  && pixelHitFilter(pattern)) count++;

	    }
	  return count;
	}

      int numberOfLostPixelHits() const
	{
	  int count=0;
	  for (int i = 0 ; i < (PatternSize * 32) / HitSize ; i++)
	    {
	      uint32_t pattern = getHitPattern(i);
	      if (!validHitFilter(pattern) 
		  && pixelHitFilter(pattern)) count++;
	    }
	  return count;
	}
      
      
      
      bool hasValidHitInFirstPixelBarrel() const; // return true if a valid hit is found in the first pixel barrel layer
      


      uint32_t getHitPattern(int position) const; 
	      
      void setHitPattern(int position, uint32_t pattern);


    private:

            
      const static unsigned short PatternSize = 5; // number of 32 bit integers to store the full pattern
      const static unsigned short HitSize = 9;     // number of bits used for each hit
      const static unsigned short SubDetectorOffset = 8; // 1 bit to distinguish tracker and muon subsystems
      const static unsigned short SubDetectorMask = 0x1; 
      
      const static unsigned short SubstrOffset = 5; // 3 bits identify the tracker (PXB, PXF, TIB, TID, TOB, TEC) or muon chamber (DT, CSC, RPD) substructure        
      const static unsigned short SubstrMask = 0x7; 
      const static unsigned short LayerOffset = 1; // 4 bits identify the layer/wheel within the substructure. Note that this implies that for end-cap structures the "side" is not stored.
      const static unsigned short LayerMask = 0xF;
      const static unsigned short ValidOffset = 0; // Finally, 1 bit is reserved to indicate whether the hit was valid.
      const static unsigned short ValidMask = 0x1;
      
      
      /*
	full hit pattern information is packed in 
	PatternSize 32 bit words
	each hit is described by HitSize bits. 

      */

      uint32_t hitPattern_[PatternSize]; 
    };
} 


#endif
