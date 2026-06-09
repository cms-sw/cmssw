#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/MTDRawToDigi/plugins/BTLReadoutMapESProducer.h"
#include "EventFilter/MTDRawToDigi/interface/BTLElectronicsMapping.h"

// ------------------------------------------------------------
//
BTLReadoutMapESProducer::BTLReadoutMapESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this, &BTLReadoutMapESProducer::produce);

  geomToken_ = cc.consumes();
  topoToken_ = cc.consumes();
}

// -----------------------------------------------------------------------------
//
BTLReadoutMapESProducer::~BTLReadoutMapESProducer() {}

// ------------------------------------------------------------
//
std::unique_ptr<BTLReadoutMap> BTLReadoutMapESProducer::produce(const BTLReadoutMapRcd& iRecord) {
  const auto& geom = iRecord.get(geomToken_);
  const auto& topo = iRecord.get(topoToken_);
  auto btlCrysLayout = MTDTopologyMode::crysLayoutFromTopoMode(topo.getMTDTopologyMode());

  // -- Readout map defined only for CrysLayout::v4 - If not v4, returns empty map
  if (static_cast<int>(btlCrysLayout) < static_cast<int>(BTLDetId::CrysLayout::v4)) {
    edm::LogError("BTLReadoutMapESProducer")
        << "BTL electronics mapping not available for BTL crystal layout " << static_cast<int>(btlCrysLayout)
        << ", use layout 7 (v4) or later!" << std::endl;
    return std::make_unique<BTLReadoutMap>();
    ;
  }

  // -- Initialize mapping helper
  BTLElectronicsMapping btlElMapping = BTLElectronicsMapping();

  auto readoutMap = std::make_unique<BTLReadoutMap>();

  // -- Loop over geometry
  for (const auto& det : geom.detUnits()) {
    BTLDetId modId = BTLDetId(det->geographicalId().rawId());  // this is the sensor module ID

    // -- Select BTL only
    if (modId.mtdSubDetector() != MTDDetId::BTL)
      continue;

    // -- Loop over crystals in a sensor module
    for (uint iCrystal = 0; iCrystal < BTLDetId::kCrystalsPerModuleV2; iCrystal++) {
      BTLDetId btlId(modId.mtdSide(), modId.mtdRR(), modId.runit(), modId.dmodule(), modId.smodule(), iCrystal);

      int sl = btlElMapping.slink(btlId);
      int hs = btlElMapping.hslink(btlId);
      int el = btlElMapping.elink(btlId);

      BTLElectronicsIdPair elecIds;
      int ch[2];

      // -- Loop over two sides of one crystal
      for (int side = 0; side < 2; ++side) {
        ch[side] = btlElMapping.TOFHIRCh(btlId, side);

        // Safety checks
        if (sl < 0 || hs < 0 || (el < 0 || el > 23) || (ch[side] < 0 || ch[side] > 31)) {
          throw cms::Exception("BTLReadoutMapESProducer")
              << "Invalid electronics mapping for DetId " << btlId.rawId() << " (slink=" << sl << ", hs=" << hs
              << ", elink=" << el << ", channel=" << ch << ")";
        }
      }

      elecIds.minus = BTLElectronicsId(
          static_cast<uint16_t>(sl), static_cast<uint8_t>(hs), static_cast<uint8_t>(el), static_cast<uint8_t>(ch[0]));
      elecIds.plus = BTLElectronicsId(
          static_cast<uint16_t>(sl), static_cast<uint8_t>(hs), static_cast<uint8_t>(el), static_cast<uint8_t>(ch[1]));

      readoutMap->add(btlId, elecIds);
    }
  }

  return readoutMap;
}

// ------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_MODULE(BTLReadoutMapESProducer);
