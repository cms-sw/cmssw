#include "CondFormats/SiPixelObjects/interface/SiPixelDbItem.h"

#include <iostream>
#include <algorithm>


//! The logic of individual setters:
//! First, we put the new value into newXXX, which is a 32-bit word
//! which has the new value at the bits where it should go, and 0's 
//! everywhere else.  To make it, we:
//! - assign the value
//! - AND with a mask -- so all others are zeroes
//! - shift up by noise_shift bits
//! 
//! Next, we prepare the oldValue with a whole where newXXX needs to go.
//!  ~(mask << shift) has 1's everywhere except where the new value will go,
//! so AND-ing packedVal_ with it creates a `whole' for the new value.
//!
//! Finally, the new value is an OR of the two (since the wholes have 0's)
//!
//! Example:
//!   00 03 c3 02 -- old, new one must blow c3 away, so we AND with ff ff 00 ff,
//!   and get 00 03 00 02.
//!   The new number is now positioned to be ff ff a2 ff, and then we AND these
//!   two and get 00 03 a2 02.  We have replaced c3 with a2.


void SiPixelDbItem::setNoise (short n)
{
  PackedPixDbType newNoise = (n & packing_.noise_mask) << packing_.noise_shift;
  PackedPixDbType oldValue = packedVal_ & ( ~(packing_.noise_mask << packing_.noise_shift) );
  packedVal_ = oldValue | newNoise;
}

void SiPixelDbItem::setPedestal (short p)
{
  PackedPixDbType newPedestal = (p & packing_.pedestal_mask) << packing_.pedestal_shift;
  PackedPixDbType oldValue = packedVal_ & ( ~(packing_.pedestal_mask << packing_.pedestal_shift) );
  packedVal_ = oldValue | newPedestal;
}

void SiPixelDbItem::setGain (float g)
{
  // Convert gain into a shifted-integer
  int mult_factor = 1 << packing_.gain_shift;   // TO DO: check
  unsigned short gInIntRep = int( g * mult_factor );

  PackedPixDbType newGain = (gInIntRep & packing_.gain_mask) << packing_.gain_shift;
  PackedPixDbType oldValue = packedVal_ & ( ~(packing_.gain_mask << packing_.gain_shift) );
  packedVal_ = oldValue | newGain;
}

void SiPixelDbItem::setStatus (char s)
{
  PackedPixDbType newStatus = (s & packing_.status_mask) << packing_.status_shift;
  PackedPixDbType oldValue = packedVal_ & ( ~(packing_.status_mask << packing_.status_shift) );
  packedVal_ = oldValue | newStatus;
}



//! A fast version which sets all in one go.
void SiPixelDbItem::set( short noise, short pedestal, float gain, char status) 
{
  // Convert gain into a shifted-integer
  int mult_factor = 1 << packing_.gain_shift;   // TO DO: check usage here
  unsigned short gInIntRep = int( gain * mult_factor );

  packedVal_ = 
    (noise     << packing_.noise_shift)     | 
    (pedestal  << packing_.pedestal_shift)  | 
    (gInIntRep << packing_.gain_shift)      |
    (status    << packing_.status_shift);
}




SiPixelDbItem::Packing::Packing(int noise_w, int pedestal_w, 
				int gain_w, int status_w)
  : noise_width(noise_w), pedestal_width(pedestal_w), status_width(status_w) 
{
  // Constructor: pre-computes masks and shifts from field widths
  // Order of fields (from right to left) is
  // noise, pedestal, gain, status count.
  
  if ( noise_w+pedestal_w+gain_w+status_w != 32) {
    std::cout << std::endl << "Error in SiPixelDbItem::Packing constructor:" 
	      << "sum of field widths != 32" << std::endl;
    // TO DO: throw an exception?
  }

  // Fields are counted from right to left!
  
  noise_shift     = 0;
  pedestal_shift  = noise_shift + noise_w;
  gain_shift    = pedestal_shift + pedestal_w;
  status_shift     = gain_shift + gain_w;

  // Ensure the complement of the correct 
  // number of bits:
  PackedPixDbType zero32 = 0;  // 32-bit wide

  noise_mask     = ~(~zero32 << noise_w);
  pedestal_mask  = ~(~zero32 << pedestal_w);
  gain_mask    = ~(~zero32 << gain_w);
  status_mask     = ~(~zero32 << status_w);
}

//  Initialize the packing (order is: noise, pedestal, gain, status)
SiPixelDbItem::Packing SiPixelDbItem::packing_( 8, 8, 8, 8);  // TO DO: TBD


