#include "EventFilter/EcalRawToDigi/interface/DCCDataBlockPrototype.h"


DCCDataBlockPrototype::DCCDataBlockPrototype ( DCCDataUnpacker  * unp, EcalElectronicsMapper * mapper, DCCEventBlock * event, bool unpackInternalData) 
: unpacker_(unp), error_(false), mapper_(mapper), event_(event), unpackInternalData_(unpackInternalData), sync_(false)
{}
