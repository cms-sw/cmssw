#include "DataFormats/SiStripDigi/interface/StripDigi.h"

#include <iostream>
#include <algorithm>

StripDigi::StripDigi( int strip, int adc) {

  // This check is for the maximal strip number that can be packed
  // in a StripDigi. The actual number of strips in a detector
  // may be smaller!
  if ( strip < 0 || strip > thePacking.max_strip) {
    std::cout << "StripDigi constructor: strip out of packing range" << std::endl;
  }

  // Set adc to max_adc in case of overflow
  adc = (adc > thePacking.max_adc) ? thePacking.max_adc : std::max(adc,0);

  theData = (strip << thePacking.strip_shift) | (adc << thePacking.adc_shift);
}

StripDigi::Packing::Packing(const int side_w, const int strip_w, 
			    const int time_w, const int adc_w) :
  strip_width(strip_w), adc_width(adc_w) {

  // Constructor: pre-computes masks and shifts from field widths
  // Order of fields (from right to left) is
  // side, strip, time, adc count.

  if ( side_w+strip_w+time_w+adc_w != 32) {
    std::cout << std::endl << "Warning in StripDigi::Packing constructor:" 
	 << "sum of field widths != 32" << std::endl;
  }

  // Fields are counted from right to left!

  side_shift    = 0;
  strip_shift   = side_shift + side_w;
  time_shift    = strip_shift + strip_w;
  adc_shift     = time_shift + time_w ;
  
  side_mask    = ~(~0 << side_w);
  strip_mask   = ~(~0 << strip_w);
  time_mask    = ~(~0 << time_w);
  adc_mask     = ~(~0 << adc_w);

  max_strip = strip_mask;


  // removed for compatibility
  // N.B. Default value of adcBits should really be 8, but is left equal
  // to 12 for reasons of backwards compatibility.
  //   static SimpleConfigurable<int> adcBits(8, "StripDigi:zeroSuppressedAdcBits");
  
  //   if (adcBits > adc_w || adcBits < 1) {
  //     std::cout<<"StripDigi WARNING: Number of bits in ADC can't exceed "<<adc_w<<std::endl;
  //     adcBits = adc_w;
  //   }
  
  //   max_adc = ~(~0 << adcBits);
  
  max_adc = adc_mask;
}

/*
// Extract from CMSIM manual (version Thu Jul 31 16:38:50 MET DST 1997)
// --------------------------------------------------------------------
// DIGI format for silicon and MSGC

// The DIGI format is the same for both silicon and MSGC strip detectors. 
// Again the information takes one word per
// fired strip. As an example listed below is DIGI definition for double 
// sided silicon - it is the same for all Si and MSGC
// detector types: 

 
//  :DETD  :TRAK  :SWDD    4    #. no. of digitization elements 
//   #. name     no. bits
//     :SIDE       1            #. 0 = normal plane, 1 = stereo plane (if any)
//     :STRP      10            #. strip (0..1023)
//     :TIME       9            #. time (ns)
//     :ADC       12            #. charge (ADC)

// One bit is used to identify the side of a double sided detector, 
// 10 bits for the strip number on wafer, 9 bits for time and
// 12 bits for ADC. 
*/

// Initialization of static data members - DEFINES DIGI PACKING !

StripDigi::Packing StripDigi::thePacking( 1, 10, 9,12); // side, strip, time, adc
