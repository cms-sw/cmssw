#ifndef DIGIFTL_MTDDIGICOLLECTION_H
#define DIGIFTL_MTDDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"

#include "DataFormats/FTLDigi/interface/BTLDigi.h"
#include "DataFormats/FTLDigi/interface/ETLDigi.h"

typedef edm::SortedCollection<btldigi::BTLDigi> BTLDigiContentCollection;
typedef edm::SortedCollection<etldigi::ETLDigi> ETLDigiContentCollection;

#endif
