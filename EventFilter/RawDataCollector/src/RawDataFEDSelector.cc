//
// Original Author:  Marco ZANETTI
//         Created:  Mon Jan 28 18:22:13 CET 2008

#include "EventFilter/RawDataCollector/interface/RawDataFEDSelector.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <cstdio>
#include <cstring>

using namespace std;
using namespace edm;

std::unique_ptr<FEDRawDataCollection> RawDataFEDSelector::select(const Handle<FEDRawDataCollection>& rawData) {
  auto selectedRawData = std::make_unique<FEDRawDataCollection>();

  // if vector of FED indexes is defined, loop over it
  if (!fedList.empty()) {
    vector<int>::const_iterator it = fedList.begin();
    vector<int>::const_iterator itEnd = fedList.end();
    for (; it != itEnd; ++it) {
      const FEDRawData& fedData = rawData->FEDData(*it);
      size_t size = fedData.size();

      FEDRawData& fedDataProd = selectedRawData->FEDData(*it);
      fedDataProd.resize(size);

      memcpy(fedDataProd.data(), fedData.data(), size);
    }
  }

  // if vector of FED indexes is NOT defined, loop over it FED range
  else {
    // FED range is <0,0> (i.e. neither the list nor the rage are defined) copy the entire payload
    if (fedRange.second == 0)
      setRange(pair<int, int>(0, FEDNumbering::lastFEDId()));

    for (int i = fedRange.first; i <= fedRange.second; ++i) {
      const FEDRawData& fedData = rawData->FEDData(i);
      size_t size = fedData.size();

      FEDRawData& fedDataProd = selectedRawData->FEDData(i);
      fedDataProd.resize(size);

      memcpy(fedDataProd.data(), fedData.data(), size);
    }
  }

  return selectedRawData;
}

std::unique_ptr<FEDRawDataCollection> RawDataFEDSelector::select(const Handle<FEDRawDataCollection>& rawData,
                                                                 const pair<int, int>& range) {
  setRange(range);
  return select(rawData);
}

std::unique_ptr<FEDRawDataCollection> RawDataFEDSelector::select(const Handle<FEDRawDataCollection>& rawData,
                                                                 const vector<int>& list) {
  setRange(list);
  return select(rawData);
}
