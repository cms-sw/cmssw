#include <DataFormats/L1Scouting/interface/SDSRawDataCollection.h>
#include <DataFormats/L1Scouting/interface/SDSNumbering.h>

SRDCollection::SRDCollection() : data_(SDSNumbering::lastSDSId() + 1) {}

SRDCollection::SRDCollection(const SRDCollection& in) : data_(in.data_) {}

SRDCollection::~SRDCollection() {}

const FEDRawData& SRDCollection::FEDData(int sourceId) const { return data_[sourceId]; }

FEDRawData& SRDCollection::FEDData(int sourceId) { return data_[sourceId]; }