
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

struct MaxHit_struct
{
  int iphihitm;
  int ietahitm;
  int depthhit;
  float hitenergy;
  float dr;
  GlobalPoint posMax;
  MaxHit_struct():iphihitm(0),ietahitm(0),
                  depthhit(0),hitenergy(-100),dr(0){}
} ;
