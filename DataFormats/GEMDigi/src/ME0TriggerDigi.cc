#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

ME0TriggerDigi::ME0TriggerDigi(const int ichamberid,
                               const int iquality,
                               const int iphiposition,
                               const int ipartition,
                               const int ideltaphi,
                               const int ibend,
                               const int ibx)
    : chamberid_(ichamberid),
      quality_(iquality),
      phiposition_(iphiposition),
      partition_(ipartition),
      deltaphi_(ideltaphi),
      bend_(ibend),
      bx_(ibx) {}

ME0TriggerDigi::ME0TriggerDigi() {
  clear();  // set contents to zero
}

void ME0TriggerDigi::clear() {
  chamberid_ = 0;
  quality_ = 0;
  phiposition_ = 0;
  partition_ = 0;
  deltaphi_ = 0;
  bend_ = 0;
  bx_ = 0;
}

bool ME0TriggerDigi::operator==(const ME0TriggerDigi& rhs) const {
  return ((chamberid_ == rhs.chamberid_) && (quality_ == rhs.quality_) && (phiposition_ == rhs.phiposition_) &&
          (partition_ == rhs.partition_) && (deltaphi_ == rhs.deltaphi_) && (bend_ == rhs.bend_) && (bx_ == rhs.bx_));
}

std::ostream& operator<<(std::ostream& o, const ME0TriggerDigi& digi) {
  return o << "ME0 chamber id #" << digi.getChamberid() << " Partition = " << digi.getPartition()
           << ": Quality = " << digi.getQuality() << " Phiposition = " << digi.getPhiposition()
           << " Strip = " << digi.getStrip() << " deltaPhi = " << digi.getDeltaphi()
           << " Bend = " << ((digi.getBend() == 0) ? 'L' : 'R') << "\n"
           << " BX = " << digi.getBX() << "\n";
}
