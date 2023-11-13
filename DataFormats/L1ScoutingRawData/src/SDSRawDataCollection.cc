#include <DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h>
#include <DataFormats/L1ScoutingRawData/interface/SDSNumbering.h>

SRDCollection::SRDCollection() : data_(SDSNumbering::lastSDSId() + 1) {}

SRDCollection::SRDCollection(const SRDCollection& in) : data_(in.data_) {}

SRDCollection::~SRDCollection() {}

const FEDRawData& SRDCollection::FEDData(int sourceId) const { return data_[sourceId]; }

FEDRawData& SRDCollection::FEDData(int sourceId) { return data_[sourceId]; }