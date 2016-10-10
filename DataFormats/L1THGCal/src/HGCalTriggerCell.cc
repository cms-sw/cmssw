#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

using namespace l1t;

HGCalTriggerCell::
HGCalTriggerCell( const LorentzVector& p4,
        int pt,
        int eta,
        int phi,
        int qual)
: L1Candidate(p4, pt, eta, phi, qual)
{
}

HGCalTriggerCell::
~HGCalTriggerCell() 
{
}
