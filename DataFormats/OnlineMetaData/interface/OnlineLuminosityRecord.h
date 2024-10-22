#ifndef DATAFORMATS_ONLINEMETADATA_ONLINELUMINOSITYRECORD_H
#define DATAFORMATS_ONLINEMETADATA_ONLINELUMINOSITYRECORD_H

//---------------------------------------------------------------------------
//!  \class OnlineLuminosityRecord
//!  \brief Class to contain the online luminosity from soft FED 1022
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <cstdint>
#include <ostream>

#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

class OnlineLuminosityRecord {
public:
  OnlineLuminosityRecord();
  OnlineLuminosityRecord(const online::Luminosity_v1&);
  virtual ~OnlineLuminosityRecord();

  /// Return the time when the lumi was recorded
  const edm::Timestamp& timestamp() const { return timestamp_; }

  /// Return the lumi-section number
  uint16_t lumiSection() const { return lumiSection_; }

  /// Return the lumi-nibble number
  uint16_t lumiNibble() const { return lumiNibble_; }

  /// Return the luminosity for the current nibble
  float instLumi() const { return instLumi_; }

  /// Return the average pileup for th current nibble
  float avgPileUp() const { return avgPileUp_; }

private:
  edm::Timestamp timestamp_;
  float instLumi_;
  float avgPileUp_;
  uint16_t lumiSection_;
  uint16_t lumiNibble_;
};

/// Pretty-print operator for OnlineLuminosityRecord
std::ostream& operator<<(std::ostream&, const OnlineLuminosityRecord&);

#endif  // DATAFORMATS_ONLINEMETADATA_ONLINELUMINOSITYRECORD_H
