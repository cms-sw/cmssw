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
 * \version $Id: HitPattern.h,v 1.7 2006/08/28 11:24:58 llista Exp $
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
    HitPattern();
    /// constructor from iterator (begin, end) pair
    template<typename I>
    HitPattern( const I & begin, const I & end ) { set( begin, end ); }
    /// constructor from hit collection
    template<typename C>
    HitPattern( const C & c ) { set( c ); }
    /// set pattern from iterator (begin, end) pair
    template<typename I>
    void set( const I & begin, const I & end ) {
      for ( int i = 0 ; i < PatternSize ; i++ ) hitPattern_[i] = 0;
      unsigned int counter = 0;
      for ( I hit = begin; hit != end && counter < 32 * PatternSize / HitSize;
	    hit ++, counter ++ ) 
	set( * hit, counter );
    }
    /// set patter for i-th hit
    void set( const TrackingRecHit &, unsigned int i );
    bool validHitFilter( uint32_t pattern ) const;
    bool trackerHitFilter( uint32_t pattern ) const;
    bool muonHitFilter( uint32_t pattern ) const;
    bool pixelHitFilter( uint32_t pattern ) const;
    int numberOfValidHits() const;    
    int numberOfLostHits() const;
    int numberOfValidMuonHits() const;
    int numberOfLostMuonHits() const;
    int numberOfValidTrackerHits() const;    
    int numberOfLostTrackerHits();    
    int numberOfValidPixelHits() const;
    int numberOfLostPixelHits() const;    
    /// return true if a valid hit is found in the first pixel barrel layer
    bool hasValidHitInFirstPixelBarrel() const; 
    uint32_t getHitPattern(int position) const; 

  private:
    /// number of 32 bit integers to store the full pattern
    const static unsigned short PatternSize = 20; 
    /// number of bits used for each hit
    const static unsigned short HitSize = 9;     
    /// 1 bit to distinguish tracker and muon subsystems
    const static unsigned short SubDetectorOffset = 8; 
    const static unsigned short SubDetectorMask = 0x1; 
    /// 3 bits identify the tracker (PXB, PXF, TIB, TID, TOB, TEC) 
    /// or muon chamber (DT, CSC, RPD) substructure        
    const static unsigned short SubstrOffset = 5; 
    const static unsigned short SubstrMask = 0x7;
    /// 4 bits identify the layer/wheel within the substructure. 
    /// Note that this implies that for end-cap structures the "side" is not stored. 
    const static unsigned short LayerOffset = 1; 
    const static unsigned short LayerMask = 0xF;
    /// Finally, 1 bit is reserved to indicate whether the hit was valid.
    const static unsigned short ValidOffset = 0; 
    const static unsigned short ValidMask = 0x1;
    
    ///  full hit pattern information is packed in  PatternSize 32 bit words
    ///  each hit is described by HitSize bits. 
    uint32_t hitPattern_[ PatternSize ]; 

    void setHitPattern(int position, uint32_t pattern);
    /// set patter for i-th hit passign a reference
    void set( const TrackingRecHitRef & ref, unsigned int i ) { set( * ref, i ); }
  };
} 


#endif
