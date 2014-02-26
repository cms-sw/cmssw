#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
using namespace reco;

VoronoiBackground::VoronoiBackground()
  : 
pt_preeq(0),
pt_posteq(0),
mt_preeq(0),
mt_posteq(0),
voronoi_area(0)
{
}

VoronoiBackground::VoronoiBackground(double pt0, double pt1, double mt0, double mt1, double v) 
   :
   pt_preeq(pt0),
   pt_posteq(pt1),
   mt_preeq(mt0),
   mt_posteq(mt1),
   voronoi_area(v)
{
}

VoronoiBackground::~VoronoiBackground()
{
}


