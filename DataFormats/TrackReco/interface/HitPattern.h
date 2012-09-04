// -*- C++ -*-
#ifndef TrackReco_HitPattern_h
#define TrackReco_HitPattern_h

//
// File: DataFormats/TrackReco/interface/HitPattern.h
//
// Marcel Vos, INFN Pisa
// v1.10 2007/05/08 bellan
// Zongru Wan, Kansas State University
// Jean-Roch Vlimant
// Kevin Burkett
// Boris Mangano
//
// Hit pattern is the summary information of the hits associated to track in
// AOD.  When RecHits are no longer available, the compact hit pattern should
// allow basic track selection based on the hits in various subdetectors.  The
// hits of a track are saved in unit32_t hitPattern_[28], initialized as
// 0x00000000, ..., 0x00000000.  Set one hit with 10 bits
//
//      +-----+-----+-----+-----+-----+-----+-----+-----+----------------+-----+-----+
//      |tk/mu|  sub-structure  |   sub-sub-structure   |     stereo     |  hit type |
//      +-----+-----+-----+-----+-----+-----+-----+-----+----------------+-----+-----+
//  ... | 10  |   9    8     7  |   6    5     4     3  |        2       |  1     0  | bit
//
//      |tk = 1      PXB = 1            layer = 1-3                       hit type = 0-3
//      |tk = 1      PXF = 2            disk  = 1-2                       hit type = 0-3
//      |tk = 1      TIB = 3            layer = 1-4      0=rphi,1=stereo  hit type = 0-3
//      |tk = 1      TID = 4            wheel = 1-3      0=rphi,1=stereo  hit type = 0-3
//      |tk = 1      TOB = 5            layer = 1-6      0=rphi,1=stereo  hit type = 0-3
//      |tk = 1      TEC = 6            wheel = 1-9      0=rphi,1=stereo  hit type = 0-3
//      |mu = 0      DT  = 1            4*(stat-1)+superlayer             hit type = 0-3
//      |mu = 0      CSC = 2            4*(stat-1)+(ring-1)               hit type = 0-3
//      |mu = 0      RPC = 3            4*(stat-1)+2*layer+region         hit type = 0-3
//
//      hit type, see DataFormats/TrackingRecHit/interface/TrackingRecHit.h
//      valid    = valid hit                                     = 0
//      missing  = detector is good, but no rec hit found        = 1
//      inactive = detector is off, so there was no hope         = 2
//      bad      = there were many bad strips within the ellipse = 3
//
// The maximum number of hits = 32*28/11 = 81.  It had been shown by Zongru
// using a 100 GeV muon sample with 5000 events uniform in eta and phi, the 
// average (maximum) number of tracker hits is 13 (17) and the average 
// (maximum) number of muon detector hits is about 26 (50).  If the number of 
// hits of a track is larger than 80 then the extra hits are ignored by hit 
// pattern.  The static hit pattern array might be improved to a dynamic one
// in the future.
//
// Because of tracking with/without overlaps and with/without hit-splitting, 
// the final number of hits per track is pretty "variable".  Compared with the
// number of valid hits, the number of crossed layers with measurement should
// be more robust to discriminate between good and fake track.
//
// Since 4-bit for sub-sub-structure is not enough to specify a muon layer,
// the layer case counting methods are implemented for tracker only.  This is
// different from the hit counting methods which are implemented for both 
// tracker and muon detector.
//
// Given a tracker layer, specified by sub-structure and layer, the method
// getTrackerLayerCase(substr, layer) groups all of the hits in the hit pattern
// array for the layer together and returns one of the four cases
//
//      crossed
//        layer case 0: valid + (missing, off, bad) ==> with measurement
//        layer case 1: missing + (off, bad) ==> without measurement
//        layer case 2: off, bad ==> totally off or bad, cannot say much
//      not crossed
//        layer case 999999: track outside acceptance or in gap ==> null
//
// Given a tracker layer, specified by sub-structure and layer, the method
// getTrackerMonoStereo(substr, layer) groups all of the valid hits in the hit
// pattern array for the layer together and returns 
//
//	0:		neither a valid mono nor a valid stereo hit
//      MONO:		valid mono hit
//	STEREO:		valid stereo hit
//	MONO | STEREO:	both
//
// Given a track, here is an example usage of hit pattern
//
//      // hit pattern of the track
//      const reco::HitPattern& p = track->hitPattern();
//
//      // loop over the hits of the track
//      for (int i=0; i<p.numberOfHits(); i++) {
//        uint32_t hit = p.getHitPattern(i);
//
//        // if the hit is valid and in pixel barrel, print out the layer
//        if (p.validHitFilter(hit) && p.pixelBarrelHitFilter(hit))
//	    std::cout << "valid hit found in pixel barrel layer "
//                    << p.getLayer(hit) << std::endl;
//
//        // expert level: printout the hit in 10-bit binary format
//        std::cout << "hit in 10-bit binary format = "; 
//        for (int j=9; j>=0; j--) {
//          int bit = (hit >> j) & 0x1;
//          std::cout << bit;
//        }
//        std::cout << std::endl;
//      }
//
//      // count the number of valid pixel barrel *** hits ***
//      std::cout << "number of of valid pixel barrel hits is "
//                << p.numberOfValidPixelBarrelHits() << std::endl;
//
//      // count the number of pixel barrel *** layers *** with measurement
//      std::cout << "number of of pixel barrel layers with measurement is "
//                << p.pixelBarrelLayersWithMeasurement() << std::endl;
//
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include <algorithm>
#include <ostream>


namespace reco {
  class HitPattern {
  public:
    enum { MONO = 1, STEREO = 2 };

   // number of 32 bit integers to store the full pattern
    const static unsigned short PatternSize = 25;

    // number of bits used for each hit
    const static unsigned short HitSize = 11;    
 
    static const int MaxHits = (PatternSize * 32) / HitSize;

    // default constructor
    // init hit pattern array as 0x00000000, ..., 0x00000000
    HitPattern() { for (int i=0; i<PatternSize; i++) hitPattern_[i] = 0; }

    // constructor from iterator (begin, end) pair
    template<typename I>
    HitPattern(const I & begin, const I & end) { set(begin, end); }

    // constructor from hit collection
    template<typename C>
    HitPattern(const C & c) { set(c); }

    // set pattern from iterator (begin, end) pair
    // init hit pattern array as 0x00000000, ..., 0x00000000
    // loop over the hits and set hit pattern
    template<typename I>
    void set(const I & begin, const I & end) {
      for (int i=0; i<PatternSize; i++) hitPattern_[i] = 0;
      unsigned int counter = 0;
      for (I hit=begin; hit!=end && counter<32*PatternSize/HitSize;
	   hit++, counter++)
	set(*hit, counter);
    }


    // generic count methods
    typedef bool filterType(unsigned int);
    int countHits(filterType filter) const {
      int count = 0;
      for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
	uint32_t pattern = getHitPattern(i);
	if (pattern == 0) break;
	if (filter(pattern)) ++count;
      }
      return count;
    }

    int countTypedHits(filterType typeFilter, filterType filter) const {
      int count = 0;
      for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
	uint32_t pattern = getHitPattern(i);
	if (pattern == 0) break;
	if (typeFilter(pattern)&&filter(pattern)) ++count;
      }
      return count;
    }

    template<typename F>
    void call(filterType typeFilter, F f) const {
     for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
	uint32_t pattern = getHitPattern(i);
	if (pattern == 0) break;
	// f() return false to ask to stop looping 
	if (typeFilter(pattern) && !f(pattern) ) break;
     }
    }

    // print the pattern of the position-th hit
    void printHitPattern (int position, std::ostream &stream) const;
    void print (std::ostream &stream = std::cout) const;

    // set the pattern of the i-th hit
    void set(const TrackingRecHit &hit, unsigned int i){setHitPattern(i, encode(hit,i));}
    
    // append a hit to the hit pattern
    void appendHit(const TrackingRecHit & hit);

    // get the pattern of the position-th hit
    uint32_t getHitPattern(int position) const; 

    static bool trackerHitFilter(uint32_t pattern); // tracker hit
    static bool muonHitFilter(uint32_t pattern);    // muon hit

    static uint32_t getSubStructure(uint32_t pattern);  // sub-structure
    static bool pixelHitFilter(uint32_t pattern);       // pixel
    static bool pixelBarrelHitFilter(uint32_t pattern); // pixel barrel
    static bool pixelEndcapHitFilter(uint32_t pattern); // pixel endcap
    static bool stripHitFilter(uint32_t pattern);       // strip 
    static bool stripTIBHitFilter(uint32_t pattern);    // strip TIB
    static bool stripTIDHitFilter(uint32_t pattern);    // strip TID
    static bool stripTOBHitFilter(uint32_t pattern);    // strip TOB
    static bool stripTECHitFilter(uint32_t pattern);    // strip TEC
    static bool muonDTHitFilter(uint32_t pattern);      // muon DT
    static bool muonCSCHitFilter(uint32_t pattern);     // muon CSC
    static bool muonRPCHitFilter(uint32_t pattern);     // muon RPC

    static uint32_t getLayer(uint32_t pattern); // sub-sub-structure
    static uint32_t getSubSubStructure(uint32_t pattern); // sub-sub-structure

    /// Muon station (1-4). Only valid for muon patterns, of course.
    static uint32_t getMuonStation(uint32_t pattern);  // only for patterns from muon, of course
    /// DT superlayer (1-3). Where the "hit" was a DT segment, superlayer is 0.  Only valid for muon DT patterns, of course.
    static uint32_t getDTSuperLayer(uint32_t pattern); // only for DT patterns
    /// CSC ring (1-4). Only valid for muon CSC patterns, of course.
    static uint32_t getCSCRing(uint32_t pattern) ; 
    /// RPC layer: for station 1 and 2, layer = 1(inner) or 2(outer); for station 3, 4 layer is always 0. Only valid for muon RPC patterns, of course.
    static uint32_t getRPCLayer(uint32_t pattern) ; 
    /// RPC region: 0 = barrel, 1 = endcap. Only valid for muon RPC patterns, of course.
    static uint32_t getRPCregion(uint32_t pattern);

    static uint32_t getHitType(uint32_t pattern);   // hit type
    static bool validHitFilter(uint32_t pattern);   // hit type 0 = valid
    static bool type_1_HitFilter(uint32_t pattern); // hit type 1
    static bool type_2_HitFilter(uint32_t pattern); // hit type 2
    static bool type_3_HitFilter(uint32_t pattern); // hit type 3

    static uint32_t getSide (uint32_t pattern);		// mono (0) or stereo (1)

    bool hasValidHitInFirstPixelBarrel() const; // has valid hit in PXB layer 1
    bool hasValidHitInFirstPixelEndcap() const; // has valid hit in PXF layer 1

    int numberOfHits() const;                 // not-null
    int numberOfValidHits() const;            // not-null, valid
    int numberOfValidTrackerHits() const;     // not-null, valid, tracker
    int numberOfValidMuonHits() const;        // not-null, valid, muon
    int numberOfValidPixelHits() const;       // not-null, valid, pixel
    int numberOfValidPixelBarrelHits() const; // not-null, valid, pixel PXB
    int numberOfValidPixelEndcapHits() const; // not-null, valid, pixel PXF
    int numberOfValidStripHits() const;       // not-null, valid, strip
    int numberOfValidStripTIBHits() const;    // not-null, valid, strip TIB
    int numberOfValidStripTIDHits() const;    // not-null, valid, strip TID
    int numberOfValidStripTOBHits() const;    // not-null, valid, strip TOB
    int numberOfValidStripTECHits() const;    // not-null, valid, strip TEC
    int numberOfValidMuonDTHits() const;      // not-null, valid, muon DT
    int numberOfValidMuonCSCHits() const;     // not-null, valid, muon CSC
    int numberOfValidMuonRPCHits() const;     // not-null, valid, muon RPC
    int numberOfLostHits() const;             // not-null, not valid
    int numberOfLostTrackerHits() const;      // not-null, not valid, tracker
    int numberOfLostMuonHits() const;         // not-null, not valid, muon
    int numberOfLostPixelHits() const;        // not-null, not valid, pixel
    int numberOfLostPixelBarrelHits() const;  // not-null, not valid, pixel PXB
    int numberOfLostPixelEndcapHits() const;  // not-null, not valid, pixel PXF
    int numberOfLostStripHits() const;        // not-null, not valid, strip
    int numberOfLostStripTIBHits() const;     // not-null, not valid, strip TIB
    int numberOfLostStripTIDHits() const;     // not-null, not valid, strip TID
    int numberOfLostStripTOBHits() const;     // not-null, not valid, strip TOB
    int numberOfLostStripTECHits() const;     // not-null, not valid, strip TEC
    int numberOfLostMuonDTHits() const;       // not-null, not valid, muon DT
    int numberOfLostMuonCSCHits() const;      // not-null, not valid, muon CSC
    int numberOfLostMuonRPCHits() const;      // not-null, not valid, muon RPC
    int numberOfBadHits() const;              // not-null, bad (only used in Muon Ch.)
    int numberOfBadMuonHits() const;          // not-null, bad, muon
    int numberOfBadMuonDTHits() const;        // not-null, bad, muon DT
    int numberOfBadMuonCSCHits() const;       // not-null, bad, muon CSC
    int numberOfBadMuonRPCHits() const;       // not-null, bad, muon RPC
    int numberOfInactiveHits() const;         // not-null, inactive
    int numberOfInactiveTrackerHits() const;  // not-null, inactive, tracker


    int numberOfValidStripLayersWithMonoAndStereo () 
      const; // count strip layers that have non-null, valid mono and stereo hits

    uint32_t getTrackerLayerCase(uint32_t substr, uint32_t layer) const;
    uint32_t getTrackerMonoStereo (uint32_t substr, uint32_t layer) const;

    int trackerLayersWithMeasurement() const;        // case 0: tracker
    int pixelLayersWithMeasurement() const;          // case 0: pixel
    int stripLayersWithMeasurement() const;          // case 0: strip
    int pixelBarrelLayersWithMeasurement() const;    // case 0: pixel PXB
    int pixelEndcapLayersWithMeasurement() const;    // case 0: pixel PXF
    int stripTIBLayersWithMeasurement() const;       // case 0: strip TIB
    int stripTIDLayersWithMeasurement() const;       // case 0: strip TID
    int stripTOBLayersWithMeasurement() const;       // case 0: strip TOB
    int stripTECLayersWithMeasurement() const;       // case 0: strip TEC
    int trackerLayersWithoutMeasurement() const;     // case 1: tracker
    int pixelLayersWithoutMeasurement() const;       // case 1: pixel
    int stripLayersWithoutMeasurement() const;       // case 1: strip
    int pixelBarrelLayersWithoutMeasurement() const; // case 1: pixel PXB
    int pixelEndcapLayersWithoutMeasurement() const; // case 1: pixel PXF
    int stripTIBLayersWithoutMeasurement() const;    // case 1: strip TIB
    int stripTIDLayersWithoutMeasurement() const;    // case 1: strip TID
    int stripTOBLayersWithoutMeasurement() const;    // case 1: strip TOB
    int stripTECLayersWithoutMeasurement() const;    // case 1: strip TEC
    int trackerLayersTotallyOffOrBad() const;        // case 2: tracker
    int pixelLayersTotallyOffOrBad() const;          // case 2: pixel
    int stripLayersTotallyOffOrBad() const;          // case 2: strip
    int pixelBarrelLayersTotallyOffOrBad() const;    // case 2: pixel PXB
    int pixelEndcapLayersTotallyOffOrBad() const;    // case 2: pixel PXF
    int stripTIBLayersTotallyOffOrBad() const;       // case 2: strip TIB
    int stripTIDLayersTotallyOffOrBad() const;       // case 2: strip TID
    int stripTOBLayersTotallyOffOrBad() const;       // case 2: strip TOB
    int stripTECLayersTotallyOffOrBad() const;       // case 2: strip TEC
    int trackerLayersNull() const;                   // case 999999: tracker
    int pixelLayersNull() const;                     // case 999999: pixel
    int stripLayersNull() const;                     // case 999999: strip
    int pixelBarrelLayersNull() const;               // case 999999: pixel PXB
    int pixelEndcapLayersNull() const;               // case 999999: pixel PXF
    int stripTIBLayersNull() const;                  // case 999999: strip TIB
    int stripTIDLayersNull() const;                  // case 999999: strip TID
    int stripTOBLayersNull() const;                  // case 999999: strip TOB
    int stripTECLayersNull() const;                  // case 999999: strip TEC



    /// subdet = 0(all), 1(DT), 2(CSC), 3(RPC); hitType=-1(all), 0=valid, 3=bad
    int muonStations(int subdet, int hitType) const ;

    int muonStationsWithValidHits() const ;
    int muonStationsWithBadHits() const ;
    int muonStationsWithAnyHits() const ;
    int dtStationsWithValidHits() const ;
    int dtStationsWithBadHits() const ;
    int dtStationsWithAnyHits() const ;
    int cscStationsWithValidHits() const ;
    int cscStationsWithBadHits() const ;
    int cscStationsWithAnyHits() const ;
    int rpcStationsWithValidHits() const ;
    int rpcStationsWithBadHits() const ;
    int rpcStationsWithAnyHits() const ;

    ///  hitType=-1(all), 0=valid, 3=bad; 0 = no stations at all
    int innermostMuonStationWithHits(int hitType) const ;
    int innermostMuonStationWithValidHits() const ;
    int innermostMuonStationWithBadHits() const ;
    int innermostMuonStationWithAnyHits() const ;

    ///  hitType=-1(all), 0=valid, 3=bad; 0 = no stations at all
    int outermostMuonStationWithHits(int hitType) const ;
    int outermostMuonStationWithValidHits() const ;
    int outermostMuonStationWithBadHits() const ;
    int outermostMuonStationWithAnyHits() const ;

    int numberOfDTStationsWithRPhiView() const ;
    int numberOfDTStationsWithRZView() const ;
    int numberOfDTStationsWithBothViews() const ;
  private:

 
    // 1 bit to distinguish tracker and muon subsystems
    const static unsigned short SubDetectorOffset = 10; 
    const static unsigned short SubDetectorMask = 0x1;

    // 3 bits to identify the tracker/muon detector substructure
    const static unsigned short SubstrOffset = 7; 
    const static unsigned short SubstrMask = 0x7;

    // 4 bits to identify the layer/disk/wheel within the substructure
    const static unsigned short LayerOffset = 3; 
    const static unsigned short LayerMask = 0xF;

    // 1 bit to identify the side in double-sided detectors
    const static unsigned short SideOffset = 2;
    const static unsigned short SideMask = 0x1;

    // 2 bits for hit type
    const static unsigned short HitTypeOffset = 0;
    const static unsigned short HitTypeMask = 0x3;

    // full hit pattern information is packed in PatternSize 32 bit words
    uint32_t hitPattern_[ PatternSize ]; 

    // set pattern for position-th hit
    void setHitPattern(int position, uint32_t pattern);

    // set pattern for i-th hit passing a reference
    void set(const TrackingRecHitRef & ref, unsigned int i) { set(* ref, i); }

    // detector side for tracker modules (mono/stereo)
    static uint32_t isStereo (DetId);

    // encoder for pattern
    uint32_t encode(const TrackingRecHit &,unsigned int);
  };

  // inline function

  inline bool HitPattern::pixelHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == PixelSubdetector::PixelBarrel || 
	substructure == PixelSubdetector::PixelEndcap) return true; 
    return false;
  }
  
  inline bool HitPattern::pixelBarrelHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == PixelSubdetector::PixelBarrel) return true; 
    return false;
  }
  
  inline bool HitPattern::pixelEndcapHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == PixelSubdetector::PixelEndcap) return true; 
    return false;
  }
  
  inline bool HitPattern::stripHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == StripSubdetector::TIB ||
	substructure == StripSubdetector::TID ||
	substructure == StripSubdetector::TOB ||
	substructure == StripSubdetector::TEC) return true; 
    return false;
  }
  
  inline bool HitPattern::stripTIBHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == StripSubdetector::TIB) return true; 
    return false;
  }
  
  inline bool HitPattern::stripTIDHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == StripSubdetector::TID) return true; 
    return false;
  }
  
  inline bool HitPattern::stripTOBHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == StripSubdetector::TOB) return true; 
    return false;
  }
  
  inline bool HitPattern::stripTECHitFilter(uint32_t pattern) { 
    if  unlikely(!trackerHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == StripSubdetector::TEC) return true; 
    return false;
  }
  
  inline bool HitPattern::muonDTHitFilter(uint32_t pattern) { 
    if  unlikely(!muonHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == (uint32_t) MuonSubdetId::DT) return true; 
    return false;
  }
  
  inline bool HitPattern::muonCSCHitFilter(uint32_t pattern) { 
    if  unlikely(!muonHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == (uint32_t) MuonSubdetId::CSC) return true; 
    return false;
  }

  inline bool HitPattern::muonRPCHitFilter(uint32_t pattern) { 
    if  unlikely(!muonHitFilter(pattern)) return false;
    uint32_t substructure = getSubStructure(pattern);
    if (substructure == (uint32_t) MuonSubdetId::RPC) return true; 
    return false;
  }
  
  
  inline bool HitPattern::trackerHitFilter(uint32_t pattern) {
    if  unlikely(pattern == 0) return false;
    if (((pattern>>SubDetectorOffset) & SubDetectorMask) == 1) return true;
    return false;
  }
  
  inline bool HitPattern::muonHitFilter(uint32_t pattern) {
    if  unlikely(pattern == 0) return false;
    if (((pattern>>SubDetectorOffset) & SubDetectorMask) == 0) return true; 
    return false;
  }


  inline uint32_t HitPattern::getSubStructure(uint32_t pattern) {
    if  unlikely(pattern == 0) return 999999;
    return ((pattern >> SubstrOffset) & SubstrMask);
  }
  
  
  inline uint32_t HitPattern::getLayer(uint32_t pattern) {
    if  unlikely(pattern == 0) return 999999;
    return ((pattern>>LayerOffset) & LayerMask);
  }
  
  inline uint32_t HitPattern::getSubSubStructure(uint32_t pattern) {
    if  unlikely(pattern == 0) return 999999;
    return ((pattern>>LayerOffset) & LayerMask);
  }
  
  
  inline uint32_t HitPattern::getSide (uint32_t pattern)  {
    if  unlikely(pattern == 0) return 999999;
    return (pattern >> SideOffset) & SideMask;
  }
  
  inline uint32_t HitPattern::getHitType( uint32_t pattern ) {
    if  unlikely(pattern == 0) return 999999;
    return ((pattern>>HitTypeOffset) & HitTypeMask);
  }
  
  inline uint32_t HitPattern::getMuonStation(uint32_t pattern) {
    return (getSubSubStructure(pattern)>>2) + 1;
  }
  
  inline uint32_t HitPattern::getDTSuperLayer(uint32_t pattern) {
    return (getSubSubStructure(pattern) & 3);
  }
  
  inline uint32_t HitPattern::getCSCRing(uint32_t pattern) {
    return (getSubSubStructure(pattern) & 3) + 1;
  }
  
  inline uint32_t HitPattern::getRPCLayer(uint32_t pattern) {
    uint32_t sss = getSubSubStructure(pattern), stat = sss >> 2;
    if likely(stat <= 1) return ((sss >> 1) & 1) + 1;
    return 0;
  }
  
  inline uint32_t HitPattern::getRPCregion(uint32_t pattern) {
    return getSubSubStructure(pattern) & 1;
  }
  
  
  inline bool  HitPattern::validHitFilter(uint32_t pattern) {
    if (getHitType(pattern) == 0) return true; 
    return false;
  }
  
  inline bool  HitPattern::type_1_HitFilter(uint32_t pattern) {
    if (getHitType(pattern) == 1) return true; 
    return false;
  }
  
  inline bool  HitPattern::type_2_HitFilter(uint32_t pattern) {
    if (getHitType(pattern) == 2) return true; 
    return false;
  }
  
  inline bool  HitPattern::type_3_HitFilter(uint32_t pattern) {
    if (getHitType(pattern) == 3) return true; 
    return false;
  }
  

  // count methods

// valid

inline int HitPattern::numberOfValidHits() const {
  return countHits(validHitFilter);
}

inline int HitPattern::numberOfValidTrackerHits() const {
  return countTypedHits(validHitFilter, trackerHitFilter);
}

inline int HitPattern::numberOfValidMuonHits() const {
  return countTypedHits(validHitFilter, muonHitFilter);
}

inline int HitPattern::numberOfValidPixelHits() const {
  return countTypedHits(validHitFilter, pixelHitFilter);
}

inline int HitPattern::numberOfValidPixelBarrelHits() const {
  return countTypedHits(validHitFilter, pixelBarrelHitFilter);
}

inline int HitPattern::numberOfValidPixelEndcapHits() const {
  return countTypedHits(validHitFilter, pixelEndcapHitFilter);
}

inline int HitPattern::numberOfValidStripHits() const {
  return countTypedHits(validHitFilter, stripHitFilter);
}

inline int HitPattern::numberOfValidStripTIBHits() const {
  return countTypedHits(validHitFilter, stripTIBHitFilter);
}

inline int HitPattern::numberOfValidStripTIDHits() const {
  return countTypedHits(validHitFilter, stripTIDHitFilter);
}

inline int HitPattern::numberOfValidStripTOBHits() const {
  return countTypedHits(validHitFilter, stripTOBHitFilter);
}

inline int HitPattern::numberOfValidStripTECHits() const {
  return countTypedHits(validHitFilter, stripTECHitFilter);
}

inline int HitPattern::numberOfValidMuonDTHits() const {
  return countTypedHits(validHitFilter, muonDTHitFilter);
}

inline int HitPattern::numberOfValidMuonCSCHits() const {
  return countTypedHits(validHitFilter, muonCSCHitFilter);
}

inline int HitPattern::numberOfValidMuonRPCHits() const {
  return countTypedHits(validHitFilter, muonRPCHitFilter);
}

// lost
inline int HitPattern::numberOfLostHits() const {
  return countHits(type_1_HitFilter);
}

inline int HitPattern::numberOfLostTrackerHits() const {
  return countTypedHits(type_1_HitFilter, trackerHitFilter);
}

inline int HitPattern::numberOfLostMuonHits() const {
  return countTypedHits(type_1_HitFilter, muonHitFilter);
}

inline int HitPattern::numberOfLostPixelHits() const {
  return countTypedHits(type_1_HitFilter, pixelHitFilter);
}

inline int HitPattern::numberOfLostPixelBarrelHits() const {
  return countTypedHits(type_1_HitFilter, pixelBarrelHitFilter);
}

inline int HitPattern::numberOfLostPixelEndcapHits() const {
  return countTypedHits(type_1_HitFilter, pixelEndcapHitFilter);
}

inline int HitPattern::numberOfLostStripHits() const {
  return countTypedHits(type_1_HitFilter, stripHitFilter);
}

inline int HitPattern::numberOfLostStripTIBHits() const {
  return countTypedHits(type_1_HitFilter, stripTIBHitFilter);
}

inline int HitPattern::numberOfLostStripTIDHits() const {
  return countTypedHits(type_1_HitFilter, stripTIDHitFilter);
}

inline int HitPattern::numberOfLostStripTOBHits() const {
  return countTypedHits(type_1_HitFilter, stripTOBHitFilter);
}

inline int HitPattern::numberOfLostStripTECHits() const {
  return countTypedHits(type_1_HitFilter, stripTECHitFilter);
}

inline int HitPattern::numberOfLostMuonDTHits() const {
  return countTypedHits(type_1_HitFilter, muonDTHitFilter);
}

inline int HitPattern::numberOfLostMuonCSCHits() const {
  return countTypedHits(type_1_HitFilter, muonCSCHitFilter);
}

inline int HitPattern::numberOfLostMuonRPCHits() const {
  return countTypedHits(type_1_HitFilter, muonRPCHitFilter);
}


// bad
inline int HitPattern::numberOfBadHits() const {
  return countHits(type_3_HitFilter);
}

inline int HitPattern::numberOfBadMuonHits() const {
  return countTypedHits(type_2_HitFilter, muonHitFilter);
}

inline int HitPattern::numberOfBadMuonDTHits() const {
  return countTypedHits(type_2_HitFilter, muonDTHitFilter);
}

inline int HitPattern::numberOfBadMuonCSCHits() const {
  return countTypedHits(type_2_HitFilter, muonCSCHitFilter);
}

inline int HitPattern::numberOfBadMuonRPCHits() const {
  return countTypedHits(type_2_HitFilter, muonRPCHitFilter);
}


// inactive
inline int HitPattern::numberOfInactiveHits() const {
  return countHits(type_2_HitFilter);
}

inline int HitPattern::numberOfInactiveTrackerHits() const {
  return countTypedHits(type_2_HitFilter, trackerHitFilter);
}






  
  inline int HitPattern::trackerLayersWithMeasurement() const {
    return pixelLayersWithMeasurement() + 
      stripLayersWithMeasurement();
  }
  
  inline int HitPattern::pixelLayersWithMeasurement() const {
    return pixelBarrelLayersWithMeasurement() +
      pixelEndcapLayersWithMeasurement();
  }
  
  inline int HitPattern::stripLayersWithMeasurement() const {
    return stripTIBLayersWithMeasurement() + 
      stripTIDLayersWithMeasurement() +
      stripTOBLayersWithMeasurement() + 
      stripTECLayersWithMeasurement();
  }
  

  inline int HitPattern::trackerLayersWithoutMeasurement() const {
    return pixelLayersWithoutMeasurement() + 
      stripLayersWithoutMeasurement();
  }
  
  inline int HitPattern::pixelLayersWithoutMeasurement() const {
    return pixelBarrelLayersWithoutMeasurement() +
      pixelEndcapLayersWithoutMeasurement();
  }
  
  inline int HitPattern::stripLayersWithoutMeasurement() const {
    return stripTIBLayersWithoutMeasurement() + 
      stripTIDLayersWithoutMeasurement() +
      stripTOBLayersWithoutMeasurement() + 
      stripTECLayersWithoutMeasurement();
  }


  inline int HitPattern::trackerLayersTotallyOffOrBad() const {
    return pixelLayersTotallyOffOrBad() + 
      stripLayersTotallyOffOrBad();
  }
  
  inline int HitPattern::pixelLayersTotallyOffOrBad() const {
    return pixelBarrelLayersTotallyOffOrBad() +
      pixelEndcapLayersTotallyOffOrBad();
  }
  
  inline int HitPattern::stripLayersTotallyOffOrBad() const {
    return stripTIBLayersTotallyOffOrBad() + 
      stripTIDLayersTotallyOffOrBad() +
      stripTOBLayersTotallyOffOrBad() + 
      stripTECLayersTotallyOffOrBad();
  }
  
  inline int HitPattern::trackerLayersNull() const {
    return pixelLayersNull() + 
      stripLayersNull();
  }
  
  inline int HitPattern::pixelLayersNull() const {
    return pixelBarrelLayersNull() +
      pixelEndcapLayersNull();
  }
  
  inline int HitPattern::stripLayersNull() const {
    return stripTIBLayersNull() + 
      stripTIDLayersNull() +
      stripTOBLayersNull() + 
      stripTECLayersNull();
  }
  

  inline int HitPattern::muonStationsWithValidHits() const { return muonStations(0, 0); }
  inline int HitPattern::muonStationsWithBadHits()   const { return muonStations(0, 3); }
  inline int HitPattern::muonStationsWithAnyHits()   const { return muonStations(0,-1); }
  inline int HitPattern::dtStationsWithValidHits()   const { return muonStations(1, 0); }
  inline int HitPattern::dtStationsWithBadHits()     const { return muonStations(1, 3); }
  inline int HitPattern::dtStationsWithAnyHits()     const { return muonStations(1,-1); }
  inline int HitPattern::cscStationsWithValidHits()  const { return muonStations(2, 0); }
  inline int HitPattern::cscStationsWithBadHits()    const { return muonStations(2, 3); }
  inline int HitPattern::cscStationsWithAnyHits()    const { return muonStations(2,-1); }
  inline int HitPattern::rpcStationsWithValidHits()  const { return muonStations(3, 0); }
  inline int HitPattern::rpcStationsWithBadHits()    const { return muonStations(3, 3); }
  inline int HitPattern::rpcStationsWithAnyHits()    const { return muonStations(3,-1); }
  
  inline int HitPattern::innermostMuonStationWithValidHits() const { return innermostMuonStationWithHits(0);  }
  inline int HitPattern::innermostMuonStationWithBadHits()   const { return innermostMuonStationWithHits(3);  }
  inline int HitPattern::innermostMuonStationWithAnyHits()   const { return innermostMuonStationWithHits(-1); }
  inline int HitPattern::outermostMuonStationWithValidHits() const { return outermostMuonStationWithHits(0);  }
  inline int HitPattern::outermostMuonStationWithBadHits()   const { return outermostMuonStationWithHits(3);  }
  inline int HitPattern::outermostMuonStationWithAnyHits()   const { return outermostMuonStationWithHits(-1); }

#ifndef CMS_NOCXX11 // cint....

  template<int N=reco::HitPattern::MaxHits>
  struct PatternSet {
    static constexpr int MaxHits=N;
    unsigned char hit[N];
    unsigned char nhit;
  
    unsigned char const * begin() const { return hit;}
    unsigned char const * end() const { return hit+nhit;}
    unsigned char  * begin()  { return hit;}
    unsigned char  * end()  { return hit+nhit;}
    int size() const { return nhit;}
    unsigned char operator[](int i) const{ return hit[i];}
    
    PatternSet(): nhit(0){}
    PatternSet(reco::HitPattern const & hp) {
      fill(hp);
    }
    
    void fill(reco::HitPattern const & hp) {
      int lhit=0;
      auto unpack =[&lhit,this](uint32_t pattern) -> bool {
	unsigned char p = 255&(pattern>>3);
	hit[lhit++]= p;
	
	// bouble sort
	if (lhit>1)
	  for (auto h=hit+lhit-1; h!=hit; --h) {
	    if ( (*(h-1)) <= p) break; // { (*h)=p;break;}
	    (*h)=*(h-1);  *(h-1)=p;
	}
	return lhit<MaxHits;
      };
      hp.call(reco::HitPattern::validHitFilter,unpack);
      nhit=lhit;
    }
  };

  template<int N>
  inline PatternSet<N> commonHits(PatternSet<N> const & p1, PatternSet<N> const & p2) {
    PatternSet<N> comm;
    comm.nhit = std::set_intersection(p1.begin(),p1.end(),p2.begin(),p2.end(),comm.begin())-comm.begin();
    return comm;
}
#endif // gcc11

} // namespace reco

#endif
