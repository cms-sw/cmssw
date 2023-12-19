#ifndef L1ScoutingRawData_SDSRawDataCollection_h
#define L1ScoutingRawData_SDSRawDataCollection_h

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/Common/interface/traits.h"

/** 
  *
  * This collection holds the raw data for all the 
  * scouting data sources. It is a collection of FEDRawData
  *
  */

class SDSRawDataCollection : public edm::DoNotRecordParents {
public:
  SDSRawDataCollection();

  // retrive data for the scouting source at sourceId
  const FEDRawData& FEDData(int sourceId) const;

  // retrive data for the scouting source at sourceId
  FEDRawData& FEDData(int sourceId);

  SDSRawDataCollection(const SDSRawDataCollection&);

  void swap(SDSRawDataCollection& other) { data_.swap(other.data_); }

private:
  std::vector<FEDRawData> data_;  // vector of raw data
};

inline void swap(SDSRawDataCollection& a, SDSRawDataCollection& b) { a.swap(b); }

#endif  // L1ScoutingRawData_SDSRawDataCollection_h