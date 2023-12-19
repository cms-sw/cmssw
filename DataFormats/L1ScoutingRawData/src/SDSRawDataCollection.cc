#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"

SDSRawDataCollection::SDSRawDataCollection() : data_(SDSNumbering::lastSDSId() + 1) {}

SDSRawDataCollection::SDSRawDataCollection(const SDSRawDataCollection& in) : data_(in.data_) {}

const FEDRawData& SDSRawDataCollection::FEDData(int sourceId) const { return data_[sourceId]; }

FEDRawData& SDSRawDataCollection::FEDData(int sourceId) { return data_[sourceId]; }