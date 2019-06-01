#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

using namespace l1t;

HGCalTriggerCell::HGCalTriggerCell(const LorentzVector& p4, int pt, int eta, int phi, int qual, uint32_t detid)
    : L1Candidate(p4, pt, eta, phi, qual), detid_(detid), position_(), mipPt_(0) {}

HGCalTriggerCell::~HGCalTriggerCell() {}

void HGCalTriggerCell::setPt(double pt) {
  this->setP4(math::PtEtaPhiMLorentzVector(pt, this->eta(), this->phi(), this->mass()));
}
