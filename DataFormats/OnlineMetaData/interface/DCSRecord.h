#ifndef DATAFORMATS_ONLINEMETADATA_DCSRECORD_H
#define DATAFORMATS_ONLINEMETADATA_DCSRECORD_H

//---------------------------------------------------------------------------
//!  \class DCSRecord
//!  \brief Class to contain DCS information from soft FED 1022
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <array>
#include <bitset>
#include <cstdint>
#include <ostream>
#include <string>

#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

class DCSRecord {
public:
  enum Partition {
    EBp,
    EBm,
    EEp,
    EEm,
    HBHEa,
    HBHEb,
    HBHEc,
    HF,
    HO,
    RPC,
    DT0,
    DTp,
    DTm,
    CSCp,
    CSCm,
    CASTOR,
    ZDC,
    TIBTID,
    TOB,
    TECp,
    TECm,
    BPIX,
    FPIX,
    ESp,
    ESm,
    Last
  };

  DCSRecord();
  explicit DCSRecord(const online::DCS_v1&);
  virtual ~DCSRecord();

  /// Return the time of the last change
  const edm::Timestamp& timestamp() const { return timestamp_; }

  /// Get the names of all high-voltage partitions
  typedef std::array<std::string, Last> ParitionNames;
  const ParitionNames& paritionNames() const { return partitionNames_; }

  /// Return the name of the high voltage of the given parition
  const std::string& partitionName(const uint8_t partitionNumber) const { return partitionNames_.at(partitionNumber); }

  /// Return true if the high voltage of the given parition is ready
  bool highVoltageReady(const uint8_t partitionNumber) const { return highVoltageReady_.test(partitionNumber); }

  /// Return the current of the CMS magnet in A
  float magnetCurrent() const { return magnetCurrent_; }

private:
  edm::Timestamp timestamp_;
  std::bitset<Partition::Last> highVoltageReady_;
  float magnetCurrent_;
  static const ParitionNames partitionNames_;
};

/// Pretty-print operator for DCSRecord
std::ostream& operator<<(std::ostream&, const DCSRecord&);

#endif  // DATAFORMATS_ONLINEMETADATA_DCSRECORD_H
