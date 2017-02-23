#ifndef PPSTRACKERHITS_H
#define PPSTRACKERHITS_H
#include "FastSimulation/PPSFastObjects/interface/PPSTrackerHit.h"
#include <vector>
#include "TObject.h"

class PPSTrackerHits: public std::vector<PPSTrackerHit> {
public:
       PPSTrackerHits();
       virtual ~PPSTrackerHits() {};
       void AddHit(const PPSTrackerHit& hit) {this->push_back(hit);};
       int NHits() {int nhits=this->size();return nhits;};

ClassDef(PPSTrackerHits,1);
};
#endif
