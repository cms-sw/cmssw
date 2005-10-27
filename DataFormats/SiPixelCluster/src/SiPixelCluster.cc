#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

//---------------------------------------------------------------------------
//!  \class SiPixelCluster
//!  \brief Pixel cluster -- collection of pixels with ADC counts + misc info.
//!
//!  Class to contain and store all the topological information of pixel clusters:
//!  charge, global size, size and the barycenter in x and y
//!  local directions. It builds a vector of SiPixel (which is
//!  an inner class) and a container of channels. 
//!
//!  Mostly ported from ORCA's class PixelReco::Cluster.
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

bool SiPixelCluster::edgeHitX() const {
  bool t1 = (theMinPixelRow == 0);
  bool t2 = false; /// !buffer.inside(theMaxSiPixelRow+1,0);
  return (t1||t2);
}
bool SiPixelCluster::edgeHitY() const {
  bool t1 = (theMinPixelCol == 0);
  bool t2 = false; /// !buffer.inside(0,theMaxSiPixelCol+1);
  return (t1||t2);
}

// &&& Do we need this?
// SiPixelCluster::ChannelContainer SiPixelCluster::channels() const 
// {
//   ChannelContainer result; 
//   result.reserve( pixels().size());
//   for (vector<SiPixel>::const_iterator i=pixels().begin();
//        i != pixels().end(); i++) {
//     result.push_back( PixelDigi::pixelToChannel( int(i->x), int(i->y)) );
//   }
//   return result;
// }
