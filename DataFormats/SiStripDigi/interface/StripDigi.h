#ifndef DATAFORMATS_SISTRIPDIGI_STRIPDIGI_H
#define DATAFORMATS_SISTRIPDIGI_STRIPDIGI_H

/**
 * A digi in the strip detectors. It has  methods to access
 * - strip number
 * - adc counts
 */

class StripDigi {
public:

  typedef unsigned int ChannelType;

  StripDigi() : strip_(0), adc_(0) {}

  StripDigi( int strip, int adc) : strip_(strip), adc_(adc) {}
    StripDigi( short strip, short adc) : strip_(strip), adc_(adc) {}


  // Access to digi information
  int strip() const   {return strip_;}
  int adc() const     {return adc_;}
  int channel() const {return strip();}


private:
  short strip_;
  short adc_;
};

// Comparison operators
inline bool operator<( const StripDigi& one, const StripDigi& other) {
  return one.channel() < other.channel();
}

#endif
