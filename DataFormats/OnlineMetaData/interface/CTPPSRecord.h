#ifndef DATAFORMATS_ONLINEMETADATA_CTPPSRECORD_H
#define DATAFORMATS_ONLINEMETADATA_CTPPSRECORD_H

//---------------------------------------------------------------------------
//!  \class CTPPSRecord
//!  \brief Class to contain CTPPS information from soft FED 1022
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

class CTPPSRecord {
public:
  enum RomanPot {
    RP_45_210_FR_BT,
    RP_45_210_FR_HR,
    RP_45_210_FR_TP,
    RP_45_220_C1,
    RP_45_220_FR_BT,
    RP_45_220_FR_HR,
    RP_45_220_FR_TP,
    RP_45_220_NR_BT,
    RP_45_220_NR_HR,
    RP_45_220_NR_TP,
    RP_56_210_FR_BT,
    RP_56_210_FR_HR,
    RP_56_210_FR_TP,
    RP_56_220_C1,
    RP_56_220_FR_BT,
    RP_56_220_FR_HR,
    RP_56_220_FR_TP,
    RP_56_220_NR_BT,
    RP_56_220_NR_HR,
    RP_56_220_NR_TP,
    Last
  };

  enum Status { unused, bad, warning, ok };

  CTPPSRecord();
  explicit CTPPSRecord(const online::CTPPS_v1&);
  virtual ~CTPPSRecord();

  /// Return the time of the last change
  const edm::Timestamp& timestamp() const { return timestamp_; }

  /// Get the names of all roman pots
  typedef std::array<std::string, Last> RomanPotNames;
  const RomanPotNames& romanPotNames() const { return romanPotNames_; }

  /// Return the name of the roman pot
  const std::string& romanPotName(const uint8_t rp) const { return romanPotNames_.at(rp); }

  /// Return the status of the given roman pot
  Status status(const uint8_t rp) const { return Status((status_ >> (rp * 2)) & 0x3); }

  /// Return the status as string
  const std::string& statusName(const uint8_t rp) const { return statusNames_.at(status(rp)); }

private:
  edm::Timestamp timestamp_;
  uint64_t status_;
  static const std::array<std::string, 4> statusNames_;
  static const RomanPotNames romanPotNames_;
};

/// Pretty-print operator for CTPPSRecord
std::ostream& operator<<(std::ostream&, const CTPPSRecord&);

#endif  // DATAFORMATS_ONLINEMETADATA_CTPPSRECORD_H
