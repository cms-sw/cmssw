#include "EventFilter/SiStripRawToDigi/interface/SiStripDetSetVectorFiller.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

template class sistrip::DetSetVectorFiller<SiStripRawDigi,false>;
template class sistrip::DetSetVectorFiller<SiStripDigi,true>;

