#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

using namespace l1t;

HGCalTriggerSums::HGCalTriggerSums(const LorentzVector& p4, int pt, int eta, int phi, int qual, uint32_t detid)
    : L1Candidate(p4, pt, eta, phi, qual), detid_(detid) {}

HGCalTriggerSums::~HGCalTriggerSums() {}
