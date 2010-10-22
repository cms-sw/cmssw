#ifndef DIGIECAL_ECALDIGICOLLECTION_H
#define DIGIECAL_ECALDIGICOLLECTION_H

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
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

class EcalDigiCollection : public edm::DataFrameContainer {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  static const size_type MAXSAMPLES = 10;
  explicit EcalDigiCollection(size_type istride=MAXSAMPLES, int isubdet=0)  : 
    edm::DataFrameContainer(istride, isubdet){}
  void swap(DataFrameContainer& other) {this->DataFrameContainer::swap(other);}
};

// make edm (and ecal client) happy
class EBDigiCollection : public  EcalDigiCollection {
public:
  typedef edm::DataFrameContainer::size_type size_type;
  typedef EBDataFrame Digi;
  typedef Digi::key_type DetId;

  EBDigiCollection(size_type istride=MAXSAMPLES) : 
    EcalDigiCollection(istride, EcalBarrel){}
  void swap(EBDigiCollection& other) {this->EcalDigiCollection::swap(other);}
};

class EEDigiCollection : public  EcalDigiCollection {
public:  
  typedef edm::DataFrameContainer::size_type size_type;
  typedef EEDataFrame Digi;
  typedef Digi::key_type DetId;

  EEDigiCollection(size_type istride=MAXSAMPLES) : 
    EcalDigiCollection(istride, EcalEndcap){}
  void swap(EEDigiCollection& other) {this->EcalDigiCollection::swap(other);}
};

// Free swap functions
inline
void swap(EcalDigiCollection& lhs, EcalDigiCollection& rhs) {
  lhs.swap(rhs);
}

inline
void swap(EBDigiCollection& lhs, EBDigiCollection& rhs) {
  lhs.swap(rhs);
}

inline
void swap(EEDigiCollection& lhs, EEDigiCollection& rhs) {
  lhs.swap(rhs);
}

//typedef  EcalDigiCollection EBDigiCollection;
//typedef  EcalDigiCollection EEDigiCollection;

typedef edm::SortedCollection<ESDataFrame> ESDigiCollection;

///Collection of ECAL trigger primitives. Note that there is also a compact
///form of this collection, used in RECO data files: see EcalTrigPrimCompactColl.
typedef edm::SortedCollection<EcalTriggerPrimitiveDigi> EcalTrigPrimDigiCollection;

typedef edm::SortedCollection<EcalPseudoStripInputDigi> EcalPSInputDigiCollection;
typedef edm::SortedCollection<EBSrFlag> EBSrFlagCollection;
typedef edm::SortedCollection<EESrFlag> EESrFlagCollection;
typedef edm::SortedCollection<EcalPnDiodeDigi> EcalPnDiodeDigiCollection;
typedef edm::SortedCollection<EcalMatacqDigi> EcalMatacqDigiCollection;

#endif
