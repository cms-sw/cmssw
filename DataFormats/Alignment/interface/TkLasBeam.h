#ifndef DataFormats_Alignment_TkLasBeam_h
#define DataFormats_Alignment_TkLasBeam_h

#include <vector>
#include <cmath>

#include "DataFormats/Alignment/interface/SiStripLaserRecHit2D.h"

/// \class TkLasBeam
/// a collection of tracker laser hits (SiStripLaserRecHit2D) originating from a single laser beam.
/// documentation in TkLasTrackBasedInterface TWiki
class TkLasBeam {

public:

  typedef std::vector<SiStripLaserRecHit2D>::const_iterator const_iterator;

  TkLasBeam() {}

  TkLasBeam( unsigned int aBeamId ) { beamId = aBeamId; }

  virtual ~TkLasBeam() {} // virtual destructor to work as base class
  
  /// return the full beam identifier
  unsigned int getBeamId( void ) const { return beamId; }

  /// access the collection of hits
  const std::vector<SiStripLaserRecHit2D>& getData( void ) const { return data; }

  /// access iterator to the collection of hits
  std::vector<SiStripLaserRecHit2D>::const_iterator begin( void ) const { return data.begin(); }

  /// access iterator to the collection of hits
  std::vector<SiStripLaserRecHit2D>::const_iterator end( void ) const { return data.end(); }

  /// insert a hit in the data vector
  void push_back( const SiStripLaserRecHit2D& aHit ) { data.push_back( aHit ); }

  /// returns the beam number (10^1 digit of beamId)
  unsigned int getBeamNumber( void ) const { return beamId%100/10; }
  
  /// true if this is a TEC internal beam (from 10^2 digit of beamId). side parameter: -1 = ask if TEC-, 1 = TEC+, 0 = any tec, don't care
  bool isTecInternal( int side = 0 ) const;
 
  /// true if this is an AT beam (from 10^2 digit of beamId)
  bool isAlignmentTube( void ) const { return ( beamId%1000/100 ) == 2; }

  /// true if this beam hits TEC R6 (last digit of beamId)
  bool isRing6( void ) const { return (beamId%10) == 1; }


private:

  unsigned int beamId;
  std::vector<SiStripLaserRecHit2D> data;

};


// To get the typedef for the collection:
#include "DataFormats/Alignment/interface/TkLasBeamCollectionFwd.h"


#endif
