#ifndef DIGIECAL_ECALDIGICOLLECTION_H
#define DIGIECAL_ECALDIGICOLLECTION_H

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalTrigPrimCompactColl.h"
#include "DataFormats/EcalDigi/interface/EcalPseudoStripInputDigi.h"
#include "DataFormats/EcalDigi/interface/EBSrFlag.h"
#include "DataFormats/EcalDigi/interface/EESrFlag.h"
#include "DataFormats/EcalDigi/interface/EcalPnDiodeDigi.h"
#include "DataFormats/EcalDigi/interface/EcalMatacqDigi.h"
#include "DataFormats/Common/interface/SortedCollection.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

class EcalDigiCollectionPh2 : public edm::DataFrameContainer {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  static const size_type MAXSAMPLES = ecalPh2::sampleSize;
  explicit EcalDigiCollectionPh2(size_type istride = MAXSAMPLES, int isubdet = 0)
      : edm::DataFrameContainer(istride, isubdet) {}
  void swap(DataFrameContainer& other) { this->DataFrameContainer::swap(other); }
};

class EBDigiCollectionPh2 : public EcalDigiCollectionPh2 {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  typedef EcalDataFrame_Ph2 Digi;
  typedef Digi::key_type DetId;

  EBDigiCollectionPh2(size_type istride = MAXSAMPLES) : EcalDigiCollectionPh2(istride, EcalBarrel) {}
  void swap(EBDigiCollectionPh2& other) { this->EcalDigiCollectionPh2::swap(other); }
  void push_back(const Digi& digi) { DataFrameContainer::push_back(digi.id(), digi.frame().begin()); }
  void push_back(id_type iid) { DataFrameContainer::push_back(iid); }
  void push_back(id_type iid, data_type const* idata) { DataFrameContainer::push_back(iid, idata); }
};

class EcalDigiCollection : public edm::DataFrameContainer {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  static const size_type MAXSAMPLES = 10;
  explicit EcalDigiCollection(size_type istride = MAXSAMPLES, int isubdet = 0)
      : edm::DataFrameContainer(istride, isubdet) {}
  void swap(DataFrameContainer& other) { this->DataFrameContainer::swap(other); }
};

// make edm (and ecal client) happy
class EBDigiCollection : public EcalDigiCollection {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  typedef EBDataFrame Digi;
  typedef Digi::key_type DetId;

  EBDigiCollection(size_type istride = MAXSAMPLES) : EcalDigiCollection(istride, EcalBarrel) {}
  void swap(EBDigiCollection& other) { this->EcalDigiCollection::swap(other); }
  void push_back(const Digi& digi) { DataFrameContainer::push_back(digi.id(), digi.frame().begin()); }
  void push_back(id_type iid) { DataFrameContainer::push_back(iid); }
  void push_back(id_type iid, data_type const* idata) { DataFrameContainer::push_back(iid, idata); }
};

class EEDigiCollection : public EcalDigiCollection {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  typedef EEDataFrame Digi;
  typedef Digi::key_type DetId;

  EEDigiCollection(size_type istride = MAXSAMPLES) : EcalDigiCollection(istride, EcalEndcap) {}
  void swap(EEDigiCollection& other) { this->EcalDigiCollection::swap(other); }
  void push_back(const Digi& digi) { edm::DataFrameContainer::push_back(digi.id(), digi.frame().begin()); }
  void push_back(id_type iid) { DataFrameContainer::push_back(iid); }
  void push_back(id_type iid, data_type const* idata) { DataFrameContainer::push_back(iid, idata); }
};

class ESDigiCollection : public EcalDigiCollection {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  typedef ESDataFrame Digi;
  typedef Digi::key_type DetId;

  static const size_type NSAMPLE = ESDataFrame::MAXSAMPLES;
  ESDigiCollection(size_type istride = NSAMPLE) : EcalDigiCollection(istride, EcalPreshower) {}
  void swap(ESDigiCollection& other) { this->EcalDigiCollection::swap(other); }

  void push_back(unsigned int i) { DataFrameContainer::push_back(i); }

  void push_back(const Digi& digi) {
    uint16_t esdata[NSAMPLE];
    for (unsigned int i(0); i != NSAMPLE; ++i) {
      static const int offset(65536);  // for int16 to uint16
      const int16_t dshort(digi[i].raw());
      const int dint((int)dshort +  // add offset for uint16 conversion
                     ((int16_t)0 > dshort ? offset : (int)0));
      esdata[i] = dint;
    }
    EcalDigiCollection::push_back(digi.id()(), esdata);
  }
};

// Free swap functions
inline void swap(EcalDigiCollection& lhs, EcalDigiCollection& rhs) { lhs.swap(rhs); }

inline void swap(EBDigiCollection& lhs, EBDigiCollection& rhs) { lhs.swap(rhs); }

inline void swap(EEDigiCollection& lhs, EEDigiCollection& rhs) { lhs.swap(rhs); }

inline void swap(ESDigiCollection& lhs, ESDigiCollection& rhs) { lhs.swap(rhs); }

inline void swap(EcalDigiCollectionPh2& lhs, EcalDigiCollectionPh2& rhs) { lhs.swap(rhs); }

inline void swap(EBDigiCollectionPh2& lhs, EBDigiCollectionPh2& rhs) { lhs.swap(rhs); }

typedef edm::SortedCollection<EcalTimeDigi> EcalTimeDigiCollection;
typedef edm::SortedCollection<EcalTriggerPrimitiveDigi> EcalTrigPrimDigiCollection;
typedef edm::SortedCollection<EcalEBTriggerPrimitiveDigi> EcalEBTrigPrimDigiCollection;
typedef edm::SortedCollection<EcalEBPhase2TriggerPrimitiveDigi> EcalEBPhase2TrigPrimDigiCollection;

typedef edm::SortedCollection<EcalPseudoStripInputDigi> EcalPSInputDigiCollection;
typedef edm::SortedCollection<EBSrFlag> EBSrFlagCollection;
typedef edm::SortedCollection<EESrFlag> EESrFlagCollection;
typedef edm::SortedCollection<EcalPnDiodeDigi> EcalPnDiodeDigiCollection;
typedef edm::SortedCollection<EcalMatacqDigi> EcalMatacqDigiCollection;

#endif
