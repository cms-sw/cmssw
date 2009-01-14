
#ifndef __TKLASBEAM_H
#define __TKLASBEAM_H

#include <vector>
#include "Test/TestHolder/interface/SiStripLaserRecHit2D.h"


///
///
///
class TkLasBeam : public std::vector<SiStripLaserRecHit2D> {

public:
  TkLasBeam() { }
  int getBeamId( void ) const { return beamId; }
  void setBeamId( int aBeamId ) { beamId = aBeamId; }

private:
  int beamId;

};


typedef std::vector<TkLasBeam> TkLasBeamCollection;



#endif
