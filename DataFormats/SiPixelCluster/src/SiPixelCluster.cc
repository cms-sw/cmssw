#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

//---------------------------------------------------------------------------
//!  \class SiPixelCluster
//!  \brief Pixel cluster -- collection of pixels with ADC counts
//!
//!  Class to contain and store all the topological information of pixel clusters:
//!  charge, global size, size and the barycenter in x and y
//!  local directions. It builds a vector of SiPixel (which is
//!  an inner class) and a container of channels. 
//!
//!  March 2007: Edge pixel methods moved to RectangularPixelTopology (V.Chiochia)
//! 
//!  \author Petar Maksimovic, JHU
//---------------------------------------------------------------------------


SiPixelCluster::SiPixelCluster( const SiPixelCluster::PixelPos& pix, float adc) :
  detId_(0),     // &&& To be fixed ?? 
  // The center of pixel with row # N is at N+0.5 in the meas. frame!
  theSumX( (pix.row()+0.5) * adc), 
  theSumY( (pix.col()+0.5) * adc),
  theCharge( adc),
  theMinPixelRow( pix.row()),
  theMaxPixelRow( pix.row()),
  theMinPixelCol( pix.col()),
  theMaxPixelCol( pix.col())
{
  // First pixel in this cluster.
  thePixels.push_back( Pixel( pix.row()+0.5, pix.col()+0.5, adc));
}

void SiPixelCluster::add( const SiPixelCluster::PixelPos& pix, float adc) {

  // The center of pixel with row # N is at N+0.5 in the meas. frame!
  theSumX += (pix.row()+0.5) * adc; 
  theSumY += (pix.col()+0.5) * adc; 
  theCharge += adc;

  thePixels.push_back( Pixel( (pix.row()+0.5), (pix.col()+0.5),adc ) );

  if (pix.row() < theMinPixelRow) theMinPixelRow = pix.row();
  if (pix.row() > theMaxPixelRow) theMaxPixelRow = pix.row();
  if (pix.col() < theMinPixelCol) theMinPixelCol = pix.col();
  if (pix.col() > theMaxPixelCol) theMaxPixelCol = pix.col();
}

