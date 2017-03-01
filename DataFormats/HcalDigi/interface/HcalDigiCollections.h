#ifndef DIGIHCAL_HCALDIGICOLLECTION_H
#define DIGIHCAL_HCALDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalCalibDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalHistogramDigi.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalDigi/interface/CastorTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HOTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"

#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"

typedef edm::SortedCollection<HBHEDataFrame> HBHEDigiCollection;
typedef edm::SortedCollection<HODataFrame> HODigiCollection;
typedef edm::SortedCollection<HFDataFrame> HFDigiCollection;
typedef edm::SortedCollection<HcalCalibDataFrame> HcalCalibDigiCollection;
typedef edm::SortedCollection<HcalTriggerPrimitiveDigi> HcalTrigPrimDigiCollection;
typedef edm::SortedCollection<HcalHistogramDigi> HcalHistogramDigiCollection;
typedef edm::SortedCollection<ZDCDataFrame> ZDCDigiCollection;
typedef edm::SortedCollection<CastorDataFrame> CastorDigiCollection;
typedef edm::SortedCollection<CastorTriggerPrimitiveDigi> CastorTrigPrimDigiCollection;
typedef edm::SortedCollection<HOTriggerPrimitiveDigi> HOTrigPrimDigiCollection;
typedef edm::SortedCollection<HcalTTPDigi> HcalTTPDigiCollection;

#include "DataFormats/Common/interface/DataFrameContainer.h"

template <class Digi>
class HcalDataFrameContainer : public edm::DataFrameContainer {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  static const size_type MAXSAMPLES = 10;
  HcalDataFrameContainer(int nsamples_per_digi=MAXSAMPLES, int isubdet=0) : 
    edm::DataFrameContainer(nsamples_per_digi*Digi::WORDS_PER_SAMPLE+Digi::HEADER_WORDS+Digi::FLAG_WORDS, isubdet) { }
  void swap(DataFrameContainer& other) {this->DataFrameContainer::swap(other);}

  //helpful accessors
  using edm::DataFrameContainer::push_back;
  Digi backDataFrame() { return Digi(this->back()); }
  int samples() const { return int((stride()-Digi::HEADER_WORDS-Digi::FLAG_WORDS)/Digi::WORDS_PER_SAMPLE); }
  void addDataFrame(DetId detid, const uint16_t* data) { push_back(detid.rawId(),data); }
  void push_back(const Digi& digi){ push_back(digi.id(), digi.begin()); }
};

typedef HcalDataFrameContainer<QIE10DataFrame> QIE10DigiCollection;
typedef HcalDataFrameContainer<QIE11DataFrame> QIE11DigiCollection;


#endif
