#ifndef DATAFORMATS_ONLINEMETADATA_DCSRECORD_H
#define DATAFORMATS_ONLINEMETADATA_DCSRECORD_H

//---------------------------------------------------------------------------
//!  \class DCSRecord
//!  \brief Class to contain DCS information from soft FED 1022
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------


#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"
#include "DataFormats/Provenance/interface/Timestamp.h"


class DCSRecord
{
public:

  enum Partition {
    EBp,EBm,EEp,EEm,HBHEa,HBHEb,HBHEc,HF,HO,
    RPC,DT0,DTp,DTm,CSCp,CSCm,CASTOR,ZDC,
    TIBTID,TOB,TECp,TECm,BPIX,FPIX,ESp,ESm,
    Last
  };

  DCSRecord();
  DCSRecord(const onlineMetaData::DCS_v1&);
  virtual ~DCSRecord();

  // Return the time of the last change
  edm::Timestamp getTimestamp() const { return timestamp_; }

  // Get the names of all high-voltage partitions
  const std::vector<std::string>& getParitionNames() const { return partitionNames_; }

  // Return the bit field indicating which parition is ready
  uint32_t getHighVoltageReady() const { return highVoltageReady_; }

  // Return the name of the high voltage of the given parition
  const std::string& getPartitionName(uint8_t partitionNumber) const { return partitionNames_.at(partitionNumber); }

  // Return true if the high voltage of the given parition is ready
  bool highVoltageReady(uint8_t partitionNumber) const { return (highVoltageReady_ & (1 << partitionNumber)); }

  // Return the current of the CMS magnet in A
  float getMagnetCurrent() const { return magnetCurrent_; }


private:

  edm::Timestamp timestamp_;
  uint32_t highVoltageReady_;
  float magnetCurrent_;

  std::vector<std::string> partitionNames_ = {
    "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO",
    "RPC","DT0","DTp","DTm","CSCp","CSCm","CASTOR","ZDC",
    "TIBTID","TOB","TECp","TECm","BPIX","FPIX","ESp","ESm"
  };

};

/// Pretty-print operator for DCSRecord
std::ostream& operator<<(std::ostream&, const DCSRecord&);

#endif // DATAFORMATS_ONLINEMETADATA_DCSRECORD_H
