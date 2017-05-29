#ifndef PPSTRACKERHIT_H
#define PPSTRACKERHIT_H
#include <vector>
#include "TObject.h"
class PPSTrackerHit: public TObject {
public:
       PPSTrackerHit();
       PPSTrackerHit(double x,double y) {X=x;Y=y;};
       virtual ~PPSTrackerHit() {};
       void set_Hit(double x, double y) {X=x;Y=y;};

       double X;
       double Y;
ClassDef(PPSTrackerHit,1);
};
#endif
