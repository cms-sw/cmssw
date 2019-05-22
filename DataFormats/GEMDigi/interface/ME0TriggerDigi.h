#ifndef GEMDigi_ME0TriggerDigi_h
#define GEMDigi_ME0TriggerDigi_h

/**\class ME0TriggerDigi
 *
 * Digi for local ME0 trigger system
 * ME0 data format: 32bits in total
 * eta position:  4bits, 0-15
 * phi positoin:  10bits, halfstrip resolution
 * deltaPhi: 9bits,  phi difference between layer1 and layer6
 * bend: 1bits, left(+)=0 or right(-)=1
 * quality: 4bits, quality0 means invalid 
 * chamber id: 1bits, 2 chamber in CTP7
 * 30 bits used now and 2 bits reserved for future
 *
 * Other properties for MC
 * bx: center at BX8, similar to CSC
 * discussion: https://indico.cern.ch/event/780696/
 * first version of ME0 trigger is built from offline ME0 segment
 * added ME0 reference in trigger digi temporarily
 * \author Sven Dildick (TAMU), Tao Huang (TAMU)
 */

#include <cstdint>
#include <iosfwd>
#include "DataFormats/GEMRecHit/interface/ME0Segment.h"

class ME0TriggerDigi {
public:
  /// Constructors
  ME0TriggerDigi(const int chamberid,
                 const int quality,
                 const int phiposition,
                 const int partition,
                 const int deltaphi,
                 const int bend,
                 const int bx);

  /// default
  ME0TriggerDigi();

  /// clear this Trigger
  void clear();

  ///Comparison
  bool operator==(const ME0TriggerDigi &) const;
  bool operator!=(const ME0TriggerDigi &rhs) const { return !(this->operator==(rhs)); }

  /// return chamber number in one CTP7
  int getChamberid() const { return chamberid_; }

  /// return the Quality
  int getQuality() const { return quality_; }

  /// return the key strip
  int getStrip() const { return strip_; }

  /// return the phi position, resolution: half strip level
  int getPhiposition() const { return phiposition_; }

  /// return the key "partition"
  int getPartition() const { return partition_; }

  /// return bending angle
  int getDeltaphi() const { return deltaphi_; }

  /// return bend
  int getBend() const { return bend_; }

  /// return BX
  int getBX() const { return bx_; }

  /// is valid?
  bool isValid() const { return quality_ != 0; }

  /// Set track number.
  void setChamberid(const uint16_t number) { chamberid_ = number; }

  /// set quality code
  void setQuality(unsigned int q) { quality_ = q; }

  /// set strip
  void setStrip(unsigned int s) { strip_ = s; }

  /// set phi position
  void setPhiposition(unsigned int phi) { phiposition_ = phi; }

  /// set partition
  void setPartition(unsigned int p) { partition_ = p; }

  /// set bending angle
  void setDeltaphi(unsigned int dphi) { deltaphi_ = dphi; }

  /// set bend
  void setBend(unsigned int b) { bend_ = b; }

  /// set bx
  void setBX(unsigned int b) { bx_ = b; }

  /*
  /// return ME0 segment 
  const ME0Segment& getME0Segment () const {return segment_;}

  /// set ME0 segment 
  void setME0Segment(const ME0Segment &seg) {segment_ = seg;}
  */

private:
  uint16_t chamberid_;
  uint16_t quality_;
  uint16_t strip_;
  uint16_t phiposition_;
  uint16_t partition_;
  uint16_t deltaphi_;
  uint16_t bend_;
  uint16_t bx_;

private:
  ME0Segment segment_;
};

std::ostream &operator<<(std::ostream &o, const ME0TriggerDigi &digi);

#endif
