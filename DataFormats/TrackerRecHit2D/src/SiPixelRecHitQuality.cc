#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

#include <iostream>

SiPixelRecHitQuality::Packing::Packing()
{
  // Constructor: pre-computes masks and shifts from field widths
  // Order of fields (from right to left) is
  // noise, pedestal, gain, status count.
  probX_width    = 8;
  probY_width    = 8;
  cotAlpha_width = 4;
  cotBeta_width  = 4;
  qBin_width     = 3;
  edge_width     = 1;
  bad_width      = 1;
  twoROC_width   = 1;
  spare_width    = 2;
  
  if ( probX_width + probY_width + cotAlpha_width + cotBeta_width +
       qBin_width  + edge_width  + bad_width      + twoROC_width  +
       spare_width
       != 32 ) {
    std::cout << std::endl << "Error in SiPixelRecHitQuality::Packing constructor:" 
	      << "sum of field widths != 32" << std::endl;
    // &&& throw an exception?
  }

  probX_units    = 1.25;
  probY_units    = 1.25;
  probX_1_over_log_units = 1.0 / log( probX_units );
  probY_1_over_log_units = 1.0 / log( probY_units );

  cotAlpha_units = 1.0/16;
  cotBeta_units  = 1.0/16;

  
  // Fields are counted from right to left!
  probX_shift     = 0;
  probY_shift     = probX_shift + probX_width;
  cotAlpha_shift  = probY_shift + probY_width; 
  cotBeta_shift   = cotAlpha_shift + cotAlpha_width; 
  qBin_shift      = cotBeta_shift + cotBeta_width; 
  edge_shift      = qBin_shift + qBin_width; 
  bad_shift       = edge_shift + edge_width; 
  twoROC_shift    = bad_shift + bad_width;
  
  // Ensure the complement of the correct 
  // number of bits:
  QualWordType zero32 = 0;  // 32-bit wide set of 0's
  
  probX_mask     = ~(~zero32 << probX_width);
  probY_mask     = ~(~zero32 << probY_width);
  cotAlpha_mask  = ~(~zero32 << cotAlpha_width);
  cotBeta_mask   = ~(~zero32 << cotBeta_width);
  qBin_mask      = ~(~zero32 << qBin_width);
  edge_mask      = ~(~zero32 << edge_width);
  bad_mask       = ~(~zero32 << bad_width);
  twoROC_mask    = ~(~zero32 << twoROC_width);
}

//  Initialize the packing format singleton
SiPixelRecHitQuality::Packing SiPixelRecHitQuality::thePacking;


