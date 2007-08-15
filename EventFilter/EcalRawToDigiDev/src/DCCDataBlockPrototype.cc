#include "EventFilter/EcalRawToDigiDev/interface/DCCDataBlockPrototype.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"


DCCDataBlockPrototype::DCCDataBlockPrototype ( DCCDataUnpacker  * unp, EcalElectronicsMapper * mapper, DCCEventBlock * event, bool unpackInternalData) 
: unpacker_(unp), error_(false), mapper_(mapper), event_(event), unpackInternalData_(unpackInternalData), sync_(false)
{}
