#ifndef CondFormats_SiPixelObjects_SiPixelDbItem_h
#define CondFormats_SiPixelObjects_SiPixelDbItem_h

//----------------------------------------------------------------------------
//! \class SiPixelPedestals
//! \brief Event Setup object which holds DB information for all pixels.
//!
//! \description Event Setup object which holds DB information for all pixels.
//! DB info for a single pixel is held in SiPixelDbItem, which contains
//! pedestal, noise, gain and status bits packed into an 32-bit wide unsigned
//! int.  The bit allocation is the following:
//! bits [31:24] - status   (0 if good, bits TBD set if not good)
//! bits [23:16] - gain     (upper 4 bits integer part, 4 bits fractional part)
//! bits  [15:8] - pedestal (upper 6 bits integer part, 2 bits fractional part)
//! bits   [7:0] - noise    (upper 6 bits integer part, 2 bits fractional part)
//-----------------------------------------------------------------------------

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class SiPixelDbItem {
  typedef uint32_t PackedPixDbType;

public:
  SiPixelDbItem() : packedVal_(0) { set(2, 0, 1.0, 0); }  // TO DO: is noise==2 in shifted rep or not???
  ~SiPixelDbItem() {}
  inline short noise() { return (packedVal_ >> packing_.noise_shift) & packing_.noise_mask; }
  inline short pedestal() { return (packedVal_ >> packing_.pedestal_shift) & packing_.pedestal_mask; }
  inline float gain() { return (packedVal_ >> packing_.gain_shift) & packing_.gain_mask; }
  inline char status() { return (packedVal_ >> packing_.status_shift) & packing_.status_mask; }
  inline PackedPixDbType packedValue() { return packedVal_; }

  inline void setPackedVal(PackedPixDbType p) { packedVal_ = p; }

  // The following setters are not inline since they are more complicated:
  void setNoise(short n);
  void setPedestal(short p);
  void setGain(float g);
  void setStatus(char s);
  void set(short noise, short pedestal, float gain, char status);

private:
  PackedPixDbType packedVal_;

  //! Pack the pixel information to use less memory

  class Packing {
  public:
    //--- Constructor: pre-computes masks and shifts from field widths
    Packing(int noise_w, int pedestal_w, int gain_w, int status_w);

    //--- Public data:
    int status_shift;
    int gain_shift;
    int noise_shift;
    int pedestal_shift;
    PackedPixDbType status_mask;
    PackedPixDbType gain_mask;
    PackedPixDbType noise_mask;
    PackedPixDbType pedestal_mask;
    int noise_width;
    int pedestal_width;
    int status_width;
  };
  static Packing packing_;

  COND_SERIALIZABLE;
};

#endif
