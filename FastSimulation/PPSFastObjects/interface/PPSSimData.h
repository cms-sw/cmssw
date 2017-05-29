#ifndef PPSSimData_h
#define PPSSimData_h
#include <vector>
#include "FastSimulation/PPSFastObjects/interface/PPSBaseData.h"
#include "FastSimulation/PPSFastObjects/interface/PPSSimTracks.h"
#include "TObject.h"

class PPSSimData: public PPSBaseData {
public:
      PPSSimData();
      virtual ~PPSSimData() {};


      int AddTrack(const PPSSimTrack& trk){Tracks.push_back(trk); return Tracks.size()-1;};
      int AddTrack(const TLorentzVector& trk,double t, double xi){Tracks.push_back(PPSSimTrack(trk,t,xi));return Tracks.size()-1;};
      PPSSimTrack& get_Track(int idx) {return Tracks.at(idx);};
      PPSSimTrack& get_Track()        {return Tracks.back();};

      void clear() {Tracks.clear();PPSBaseData::clear();};

public:
      PPSSimTracks        Tracks;
ClassDef(PPSSimData,1);
};
#endif
