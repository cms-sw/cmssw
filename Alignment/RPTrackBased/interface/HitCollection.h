/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Alignment_RPTrackBased_HitCollection
#define Alignment_RPTrackBased_HitCollection

#include <vector>

class TotemRPRecHit;

struct Hit {
  unsigned int id;    ///< detector decimal id
  double position;    ///< position in mm
  double sigma;       ///< uncertainty in mm

  Hit(unsigned int i=0, double p=0., double s=0.) : id(i), position(p), sigma(s) {}
  Hit(const TotemRPRecHit&);
};

typedef std::vector<Hit> HitCollection;

#endif

