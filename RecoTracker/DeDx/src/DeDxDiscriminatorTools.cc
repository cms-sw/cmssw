
#include "RecoTracker/DeDx/interface/DeDxDiscriminatorTools.h"

namespace DeDxDiscriminatorTools {

using namespace std;

bool IsSpanningOver2APV(unsigned int FirstStrip, unsigned int ClusterSize)
{  
   if(FirstStrip==0                                ) return true;
   if(FirstStrip==128                              ) return true;
   if(FirstStrip==256                              ) return true;
   if(FirstStrip==384                              ) return true;
   if(FirstStrip==512                              ) return true;
   if(FirstStrip==640                              ) return true;

   if(FirstStrip<=127 && FirstStrip+ClusterSize>127) return true;
   if(FirstStrip<=255 && FirstStrip+ClusterSize>255) return true;
   if(FirstStrip<=383 && FirstStrip+ClusterSize>383) return true;
   if(FirstStrip<=511 && FirstStrip+ClusterSize>511) return true;
   if(FirstStrip<=639 && FirstStrip+ClusterSize>639) return true;
   
   if(FirstStrip+ClusterSize==127                  ) return true;
   if(FirstStrip+ClusterSize==255                  ) return true;
   if(FirstStrip+ClusterSize==383                  ) return true;
   if(FirstStrip+ClusterSize==511                  ) return true;
   if(FirstStrip+ClusterSize==639                  ) return true;
   if(FirstStrip+ClusterSize==767                  ) return true;
   
   return false;
}


bool IsSaturatingStrip(const vector<uint8_t>& Ampls)
{
   for(unsigned int i=0;i<Ampls.size();i++){
      if(Ampls[i]>=254)return true;
   }return false;
}



double charge(const vector<uint8_t>& Ampls)
{
   double charge = 0;
   for(unsigned int a=0;a<Ampls.size();a++){charge+=Ampls[a];}
   return charge;
}


double path(double cosine, double thickness)
{
   return (10.0*thickness)/fabs(cosine);
}


bool IsFarFromBorder(TrajectoryStateOnSurface trajState, const GeomDetUnit* it)
{
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
     std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
     return false;
  }

  LocalPoint  HitLocalPos   = trajState.localPosition();
  LocalError  HitLocalError = trajState.localError().positionError() ;

  const BoundPlane plane = it->surface();
  const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  const RectangularPlaneBounds* rectangularBounds( dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));

  double DistFromBorder = 1.0;
  //double HalfWidth      = it->surface().bounds().width()  /2.0;
  double HalfLength     = it->surface().bounds().length() /2.0;

  if(trapezoidalBounds)
  {
      std::array<const float, 4> const & parameters = (*trapezoidalBounds).parameters();
     HalfLength     = parameters[3];
     //double t       = (HalfLength + HitLocalPos.y()) / (2*HalfLength) ;
     //HalfWidth      = parameters[0] + (parameters[1]-parameters[0]) * t;
  }else if(rectangularBounds){
     //HalfWidth      = it->surface().bounds().width()  /2.0;
     HalfLength     = it->surface().bounds().length() /2.0;
  }else{return false;}

//  if (fabs(HitLocalPos.x())+HitLocalError.xx() >= (HalfWidth  - DistFromBorder) ) return false;//Don't think is really necessary
  if (fabs(HitLocalPos.y())+HitLocalError.yy() >= (HalfLength - DistFromBorder) ) return false;

  return true;
}

}
