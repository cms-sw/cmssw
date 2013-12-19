#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
using namespace reco;

VoronoiBackground::VoronoiBackground()
  : 
pt_preeq(0),
pt_posteq(0),
pt_corrected(0),
mt_preeq(0),
mt_posteq(0),
mt_corrected(0),
voronoi_area(0)
{
}

VoronoiBackground::VoronoiBackground(double pt0, double pt1, double pt2, double mt0, double mt1, double mt2, double v) 
   :
   pt_preeq(pt0),
   pt_posteq(pt1),
   pt_corrected(pt2),
   mt_preeq(mt0),
   mt_posteq(mt1),
   mt_corrected(mt2),
   voronoi_area(v)
{
}

VoronoiBackground::~VoronoiBackground()
{
}


