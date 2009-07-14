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
//      |mu = 0      DT  = 1            layer                             hit type = 0-3
//      |mu = 0      CSC = 2            layer                             hit type = 0-3
//      |mu = 0      RPC = 3            layer                             hit type = 0-3
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
#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

namespace reco {
  class HitPattern {
  public:
    enum { MONO = 1, STEREO = 2 };

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

    // print the pattern of the position-th hit
    void printHitPattern (int position, std::ostream &stream) const;
    void print (std::ostream &stream = std::cout) const;

    // set the pattern of the i-th hit
    void set(const TrackingRecHit &, unsigned int i); 

    // get the pattern of the position-th hit
    uint32_t getHitPattern(int position) const; 

    bool trackerHitFilter(uint32_t pattern) const; // tracker hit
    bool muonHitFilter(uint32_t pattern) const;    // muon hit

    uint32_t getSubStructure(uint32_t pattern) const;  // sub-structure
    bool pixelHitFilter(uint32_t pattern) const;       // pixel
    bool pixelBarrelHitFilter(uint32_t pattern) const; // pixel barrel
    bool pixelEndcapHitFilter(uint32_t pattern) const; // pixel endcap
    bool stripHitFilter(uint32_t pattern) const;       // strip 
    bool stripTIBHitFilter(uint32_t pattern) const;    // strip TIB
    bool stripTIDHitFilter(uint32_t pattern) const;    // strip TID
    bool stripTOBHitFilter(uint32_t pattern) const;    // strip TOB
    bool stripTECHitFilter(uint32_t pattern) const;    // strip TEC
    bool muonDTHitFilter(uint32_t pattern) const;      // muon DT
    bool muonCSCHitFilter(uint32_t pattern) const;     // muon CSC
    bool muonRPCHitFilter(uint32_t pattern) const;     // muon RPC

    uint32_t getLayer(uint32_t pattern) const; // sub-sub-structure

    uint32_t getHitType(uint32_t pattern) const;   // hit type
    bool validHitFilter(uint32_t pattern) const;   // hit type 0 = valid
    bool type_1_HitFilter(uint32_t pattern) const; // hit type 1
    bool type_2_HitFilter(uint32_t pattern) const; // hit type 2
    bool type_3_HitFilter(uint32_t pattern) const; // hit type 3

    static uint32_t getSide (uint32_t pattern);		// mono (0) or stereo (1)

    bool hasValidHitInFirstPixelBarrel() const; // has valid hit in PXB layer 1

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

  private:

    // number of 32 bit integers to store the full pattern
    const static unsigned short PatternSize = 25;

    // number of bits used for each hit
    const static unsigned short HitSize = 11;    
 
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
  };
} 

#endif
