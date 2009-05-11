#ifndef DataRecord_PixelDCSRcds_h
#define DataRecord_PixelDCSRcds_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class CaenChannel;

template <class> class PixelDCSObject;

struct PixelCaenChannelIsOnRcd:
  public edm::eventsetup::EventSetupRecordImplementation<PixelCaenChannelIsOnRcd>
{
  typedef PixelDCSObject<bool> Object;
};

struct PixelCaenChannelIMonRcd:
  public edm::eventsetup::EventSetupRecordImplementation<PixelCaenChannelIMonRcd>
{
  typedef PixelDCSObject<float> Object;
};

struct PixelCaenChannelRcd:
  public edm::eventsetup::EventSetupRecordImplementation<PixelCaenChannelRcd>
{
  typedef PixelDCSObject<CaenChannel> Object;
};

#endif
