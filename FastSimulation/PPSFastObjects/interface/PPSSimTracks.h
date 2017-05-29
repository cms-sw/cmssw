#ifndef PPSSIMTRACKS_H
#define PPSSIMTRACKS_H
#include "FastSimulation/PPSFastObjects/interface/PPSBaseTrack.h"
#include "FastSimulation/PPSFastObjects/interface/PPSTrackerHit.h"
#include "FastSimulation/PPSFastObjects/interface/PPSToFHit.h"
#include <vector>
#include "TObject.h"

class PPSSimTrack:public PPSBaseTrack {
      public:
             PPSSimTrack(){};
             PPSSimTrack(const PPSBaseTrack& trk):PPSBaseTrack(trk){};
             PPSSimTrack(const TLorentzVector& trk, double t, double xi):PPSBaseTrack(trk,t,xi){};
             void set_XatTCL4(double x) {XatTCL4=x;};
             void set_XatTCL5(double x) {XatTCL5=x;};
             void set_HitDet1(double x,double y) {Det1.set_Hit(x,y);};
             void set_HitDet2(double x,double y) {Det2.set_Hit(x,y);};
             void set_HitToF(int cellid, double tof, double x,double y) {sToF.set_Hit(x,y);ToF.set_Hit(cellid,tof,x,y);};

             ~PPSSimTrack(){};
      private:
              double XatTCL4;
              double XatTCL5;
              PPSTrackerHit Det1;
              PPSTrackerHit Det2;
              PPSTrackerHit sToF;// at SIM level, save the precise hit position, not the cell 
              PPSToFHit     ToF;
      public:
ClassDef(PPSSimTrack,1);
};
typedef std::vector<PPSSimTrack> PPSSimTracks;
#endif
