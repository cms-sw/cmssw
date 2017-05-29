#ifndef PPSBASETRACK_H
#define PPSBASETRACK_H
#include "TLorentzVector.h"
#include "TObject.h"
class TVector3;

class PPSBaseTrack: public TObject {
      public:
             PPSBaseTrack();
             PPSBaseTrack(const TLorentzVector& _p,double _t, double _xi);
             virtual ~PPSBaseTrack() {};

             double t;
             double xi;
             double pT;
             double momentum;
             double eta;
             double phi;
             double theta;
             double E;
             double Px;
             double Py;
             double Pz;
ClassDef(PPSBaseTrack,2);
};
#endif
