#ifndef TrackReco_HitPattern_h
#define TrackReco_HitPattern_h

//
// File: DataFormats/TrackReco/interface/HitPattern.h
//
// Marcel Vos, INFN Pisa
// v1.10 2007/05/08 bellan
// Zongru Wan, Kansas State University
//
// Hit pattern is the summary information of the hits associated to track in
// AOD.  When RecHits are no longer available, the compact hit pattern should
// allow basic track selection based on the hits in various subdetectors.  The
// hits of a track are saved in unit32_t hitPattern_[25], initialized as
// 0x00000000, ..., 0x00000000.  Set one hit with 10 bits
//
//      +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//      |tk/mu|  sub-structure  |   sub-sub-structure   |  hit type |
//      +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//  ... |  9  |  8     7     6  |  5     4     3     2  |  1     0  | bit
//
//      tk = 1      PXB = 1            layer = 1-3         hit type = 0-3
//      tk = 1      PXF = 2            disk  = 1-2         hit type = 0-3
//      tk = 1      TIB = 3            layer = 1-4         hit type = 0-3
//      tk = 1      TID = 4            wheel = 1-3         hit type = 0-3
//      tk = 1      TOB = 5            layer = 1-6         hit type = 0-3
//      tk = 1      TEC = 6            wheel = 1-9         hit type = 0-3
//      mu = 0      DT  = 1            layer               hit type = 0-3
//      mu = 0      CSC = 2            layer               hit type = 0-3
//      mu = 0      RPC = 3            layer               hit type = 0-3
//
//      hit type, see DataFormats/TrackingRecHit/interface/TrackingRecHit.h
//      valid    = valid hit                                     = 0
//      missing  = detector is good, but no rec hit found        = 1
//      inactive = detector is off, so there was no hope         = 2
//      bad      = there were many bad strips within the ellipse = 3
//
// The maximum number of hits = 32*25/10 = 80.  It had been shown by Zongru
// using a 100 GeV muon sample with 5000 events uniform in eta and phi, the 
// average (maximum) number of tracker hits is 13 (17) and the average 
// (maximum) number of muon detector hits is about 26 (50).  If the number of 
// hits of a track is larger than 80 then the extra hits are ignored by hit 
// pattern.  The static hit pattern array might be improved to a dynamic one
// in the future.
//
// Given a track, here is an example usage of hit pattern
//
//      // loop over the hits of the track
//      const reco::HitPattern& p = track->hitPattern();
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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

namespace reco {
  class HitPattern {
  public:

    // default constructor
    HitPattern();

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

  private:

    // number of 32 bit integers to store the full pattern
    const static unsigned short PatternSize = 25;

    // number of bits used for each hit
    const static unsigned short HitSize = 10;    
 
    // 1 bit to distinguish tracker and muon subsystems
    const static unsigned short SubDetectorOffset = 9; 
    const static unsigned short SubDetectorMask = 0x1;

    // 3 bits to identify the tracker/muon detector substructure
    const static unsigned short SubstrOffset = 6; 
    const static unsigned short SubstrMask = 0x7;

    // 4 bits to identify the layer/disk/wheel within the substructure
    const static unsigned short LayerOffset = 2; 
    const static unsigned short LayerMask = 0xF;

    // 2 bits for hit type
    const static unsigned short HitTypeOffset = 0;
    const static unsigned short HitTypeMask = 0x3;

    // full hit pattern information is packed in PatternSize 32 bit words
    uint32_t hitPattern_[ PatternSize ]; 

    // set pattern for position-th hit
    void setHitPattern(int position, uint32_t pattern);

    // set pattern for i-th hit passing a reference
    void set(const TrackingRecHitRef & ref, unsigned int i) { set(* ref, i); }
  };
} 

#endif
