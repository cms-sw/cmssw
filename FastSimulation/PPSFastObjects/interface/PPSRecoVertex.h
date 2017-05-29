#ifndef PPSRECOVERTEX_H
#define PPSRECOVERTEX_H
#include "FastSimulation/PPSFastObjects/interface/PPSBaseVertex.h"
#include "TObject.h"

class PPSRecoVertex: public PPSBaseVertex {
public:
       PPSRecoVertex():PPSBaseVertex(){};
       virtual ~PPSRecoVertex(){};
       void AddGolden(double,double,double,int, int);
       void SetGolden(int i);
       PPSRecoVertex GetGoldenVertices();

ClassDef(PPSRecoVertex,1);
};
#endif
