#ifndef PPSRecoData_h
#define PPSRecoData_h
#include <vector>
#include "FastSimulation/PPSFastObjects/interface/PPSRecoTracks.h"
#include "FastSimulation/PPSFastObjects/interface/PPSBaseData.h"
#include "TLorentzVector.h"
#include "TObject.h"

class PPSRecoData: public PPSBaseData {
public:
      PPSRecoData();
      virtual ~PPSRecoData(){};

      int AddTrack(const PPSRecoTrack& trk){Tracks.push_back(trk);return Tracks.size()-1;};
      int AddTrack(const TLorentzVector& trk,double t, double xi) {Tracks.push_back(PPSRecoTrack(trk,t,xi));return Tracks.size()-1;}
      virtual void clear() {Tracks.clear();PPSBaseData::clear();};
      PPSRecoTrack& get_Track(int idx) {return Tracks.at(idx);};
      PPSRecoTrack& get_Track()        {return Tracks.back();};


public:
      PPSRecoTracks  Tracks;

ClassDef(PPSRecoData,1);
};
#endif
