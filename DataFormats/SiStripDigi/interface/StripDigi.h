#ifndef TRACKINGOBJECTS_STRIPDIGI_H
#define TRACKINGOBJECTS_STRIPDIGI_H

/**
 * A digi in the strip detectors. It has low- and high-level methods to access
 * - side
 * - strip number
 * - time of flight
 * - adc counts
 */

class StripDigi {
public:

  typedef unsigned int PackedDigiType;
  typedef unsigned int ChannelType;

  StripDigi() : theData(0) {}
  
  StripDigi( int strip, int adc);
  
  StripDigi( int packed_value) : theData(packed_value) {}

  // Access to digi information
  int side() const    {return (theData >> thePacking.side_shift) & thePacking.side_mask;}
  int strip() const   {return (theData >> thePacking.strip_shift) & thePacking.strip_mask;}
  int time() const    {return (theData >> thePacking.time_shift) & thePacking.time_mask;}
  int adc() const     {return (theData >> thePacking.adc_shift) & thePacking.adc_mask;}
  int channel() const {return (side() << thePacking.strip_width) + strip();}

  PackedDigiType packedData() const {return theData;}

private:
  PackedDigiType theData;
  
  class Packing {
  public:

    // Constructor: pre-computes masks and shifts from field widths
    Packing(const int side_w, const int strip_w, 
	    const int time_w, const int adc_w);

    // public data:
    int adc_shift;
    int time_shift;
    int side_shift;
    int strip_shift;

    PackedDigiType adc_mask;
    PackedDigiType time_mask;
    PackedDigiType side_mask;
    PackedDigiType strip_mask;
    
    int strip_width;
    int adc_width;
    
    int max_strip;
    int max_adc;
  };
  static Packing   thePacking;
};  

// Comparison operators
inline bool operator<( const StripDigi& one, const StripDigi& other) {
  return one.channel() < other.channel();
}

#include<iostream>
// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const StripDigi& digi) {
  return o << digi.channel();
}

#endif
