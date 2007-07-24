#ifndef DIGIECAL_ECALDIGICOLLECTION_H
#define DIGIECAL_ECALDIGICOLLECTION_H

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
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
  static const int MAXSAMPLES = 10;
  explicit EcalDigiCollection(int isubdet)  : 
    edm::DataFrameContainer(MAXSAMPLES,isubdet){}
};

// make edm (and ecal client) happy
class EBDigiCollection : public  EcalDigiCollection {
public:  
  EcalDigiCollection() : EcalDigiCollection(EcalBarrel){}
};

class EBDigiCollection : public  EcalDigiCollection {
public:  
  EcalDigiCollection() : EcalDigiCollection(EcalEndcap){}
};



//typedef  EcalDigiCollection EBDigiCollection;
//typedef  EcalDigiCollection EEDigiCollection;

typedef edm::SortedCollection<ESDataFrame> ESDigiCollection;
typedef edm::SortedCollection<EcalTriggerPrimitiveDigi> EcalTrigPrimDigiCollection;
typedef edm::SortedCollection<EBSrFlag> EBSrFlagCollection;
typedef edm::SortedCollection<EESrFlag> EESrFlagCollection;
typedef edm::SortedCollection<EcalPnDiodeDigi> EcalPnDiodeDigiCollection;
typedef edm::SortedCollection<EcalMatacqDigi> EcalMatacqDigiCollection;

#endif
