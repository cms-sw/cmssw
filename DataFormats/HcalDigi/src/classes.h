#include <vector>
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalHistogramDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/HcalDigi/interface/HcalLaserDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"
#include "DataFormats/HcalDigi/interface/HcalUMNioDigi.h"
#include "DataFormats/Common/interface/Wrapper.h"

// dummy structs to ensure backward compatibility
struct HcalUpgradeDataFrame {
  typedef HcalDetId key_type;
};
struct HcalUpgradeQIESample {};
typedef edm::SortedCollection<HcalUpgradeDataFrame> HBHEUpgradeDigiCollection;
typedef edm::SortedCollection<HcalUpgradeDataFrame> HFUpgradeDigiCollection;
