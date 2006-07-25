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
 * \version $Id: HitPattern.h,v 1.3 2006/06/14 07:20:23 llista Exp $
 *
 */
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

namespace reco {
  class HitPattern {
  public:
    /// default constructor
    HitPattern() {}
    bool validHitFilter( uint32_t pattern ) const;
    bool trackerHitFilter( uint32_t pattern ) const;
    bool muonHitFilter( uint32_t pattern ) const;
    bool pixelHitFilter( uint32_t pattern ) const;
    unsigned int numberOfValidHits() const;    
    unsigned int numberOfLostHits() const;
    unsigned int numberOfValidMuonHits() const;
    unsigned int numberOfLostMuonHits() const;
    unsigned int numberOfValidTrackerHits() const;    
    unsigned int numberOfLostTrackerHits();    
    unsigned int numberOfValidPixelHits() const;
    unsigned int numberOfLostPixelHits() const;    
    /// return true if a valid hit is found in the first pixel barrel layer
    bool hasValidHitInFirstPixelBarrel() const; 
    uint32_t getHitPattern( int position ) const; 
    void setHitPattern( int position, uint32_t pattern );
    void clear();

    /// number of 32 bit integers to store the full pattern
    static const unsigned short patternSize = 5; 
    /// number of bits used for each hit
    static const unsigned short hitSize = 9;     
    /// number or patterns
    static const unsigned short numberOfPatterns = (patternSize * 32) / hitSize;
    /// 1 bit to distinguish tracker and muon subsystems
    static const unsigned short subDetectorOffset = 8; 
    static const unsigned short subDetectorMask = 0x1; 
    /// 3 bits identify the tracker (PXB, PXF, TIB, TID, TOB, TEC) 
    /// or muon chamber (DT, CSC, RPD) substructure        
    static const unsigned short substrOffset = 5; 
    static const unsigned short substrMask = 0x7;

    /// 4 bits identify the layer/wheel within the substructure. 
    /// Note that this implies that for end-cap structures the "side" is not stored. 
    static const unsigned short layerOffset = 1; 
    static const unsigned short layerMask = 0xF;
    /// Finally, 1 bit is reserved to indicate whether the hit was valid.
    static const unsigned short validOffset = 0; 
    static const unsigned short validMask = 0x1;

  private:
    ///  full hit pattern information is packed in  PatternSize 32 bit words
    ///  each hit is described by HitSize bits. 
    uint32_t hitPattern_[ patternSize ]; 
  };
} 


#endif
