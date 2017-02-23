#ifndef PPSTOFHITS_H
#define PPSTOFHITS_H
#include "FastSimulation/PPSFastObjects/interface/PPSToFHit.h"
#include <vector>
class PPSToFHits:public std::vector<PPSToFHit> {
public:
       PPSToFHits();
       virtual ~PPSToFHits(){};
       void AddHit(const PPSToFHit& hit) {this->push_back(hit);};
       int NHits() {return this->size();};

ClassDef(PPSToFHits,1);
};
#endif
