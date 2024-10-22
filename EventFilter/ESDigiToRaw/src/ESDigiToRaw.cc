#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"

#include "EventFilter/ESDigiToRaw/interface/ESDigiToRaw.h"
#include "EventFilter/ESDigiToRaw/src/ESDataFormatterV1_1.h"
#include "EventFilter/ESDigiToRaw/src/ESDataFormatterV4.h"

using namespace std;
using namespace edm;

ESDigiToRaw::ESDigiToRaw(const edm::ParameterSet& ps)
    : ESDataFormatter_(nullptr),
      label_(ps.getParameter<string>("Label")),
      instanceName_(ps.getParameter<string>("InstanceES")),
      ESDigiToken_(consumes<ESDigiCollection>(edm::InputTag(label_, instanceName_))),
      lookup_(ps.getUntrackedParameter<FileInPath>("LookupTable")),
      debug_(ps.getUntrackedParameter<bool>("debugMode", false)),
      formatMajor_(ps.getUntrackedParameter<int>("formatMajor", 4)),
      formatMinor_(ps.getUntrackedParameter<int>("formatMinor", 1)) {
  if (formatMajor_ == 4)
    ESDataFormatter_ = new ESDataFormatterV4(ps);
  else
    ESDataFormatter_ = new ESDataFormatterV1_1(ps);

  produces<FEDRawDataCollection>();
  // initialize look-up table
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 40; ++k)
        for (int m = 0; m < 40; m++)
          fedId_[i][j][k][m] = -1;

  // read in look-up table
  int nLines, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  ifstream file;
  file.open(lookup_.fullPath().c_str());
  if (file.is_open()) {
    file >> nLines;
    for (int i = 0; i < nLines; ++i) {
      file >> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;
      fedId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = fed;
    }
  } else {
    cout << "[ESDigiToRaw] Look up table file can not be found in " << lookup_.fullPath().c_str() << endl;
  }

  file.close();
}

ESDigiToRaw::~ESDigiToRaw() {
  if (ESDataFormatter_)
    delete ESDataFormatter_;
}

void ESDigiToRaw::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const {
  ESDataFormatter::Meta_Data meta_data;
  meta_data.lv1 = ev.id().event();
  meta_data.run_number = ev.id().run();
  meta_data.orbit_number = meta_data.lv1 / LHC_BX_RANGE;
  meta_data.bx = (meta_data.lv1 % LHC_BX_RANGE);

  meta_data.kchip_ec = (meta_data.lv1 % KCHIP_EC_RANGE);
  meta_data.kchip_bc = (meta_data.lv1 % KCHIP_BC_RANGE);

  edm::Handle<ESDigiCollection> digis;
  ev.getByToken(ESDigiToken_, digis);

  int ifed;
  ESDataFormatter::Digis Digis;
  Digis.clear();

  for (ESDigiCollection::const_iterator it = digis->begin(); it != digis->end(); ++it) {
    const ESDataFrame& df = *it;
    const ESDetId& detId = it->id();

    ifed = fedId_[(3 - detId.zside()) / 2 - 1][detId.plane() - 1][detId.six() - 1][detId.siy() - 1];
    if (ifed < 0)
      continue;

    Digis[ifed].push_back(df);
  }

  auto productRawData = std::make_unique<FEDRawDataCollection>();

  ESDataFormatter::Digis::const_iterator itfed;
  for (itfed = Digis.begin(); itfed != Digis.end(); ++itfed) {
    int fId = (*itfed).first;

    FEDRawData& fedRawData = productRawData->FEDData(fId);
    ESDataFormatter_->DigiToRaw(fId, Digis, fedRawData, meta_data);

    if (debug_)
      cout << "FED : " << fId << " Data size : " << fedRawData.size() << " (Bytes)" << endl;
  }

  ev.put(std::move(productRawData));

  return;
}
