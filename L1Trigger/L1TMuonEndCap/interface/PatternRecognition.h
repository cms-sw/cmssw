///// Takes in a std::vector<PhiMemoryImage> Merged which is the merged image of the BXs per zone 
///// and outputs a vector of Integers containing strip,rank,layer and straightness

#ifndef ADD_PatternRecognition
#define ADD_PatternRecognition


#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"
#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"

PatternOutput DetectPatterns(ZonesOutput Eout);

std::vector<PatternOutput> Patterns(std::vector<ZonesOutput> Zones);

void PrintQuality (QualityOutput out);

#endif
