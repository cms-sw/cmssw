#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
// #include "Fireworks/Calo/interface/FWHeatmapProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"

#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

template <typename T>
class FWL1THGCalProxyTemplate : public FWSimpleProxyBuilderTemplate<T> {
protected:
  void setItem(const FWEventItem *iItem) override {
    FWProxyBuilderBase::setItem(iItem);
    if (iItem) {
      iItem->getConfig()->keepEntries(true);
      iItem->getConfig()->assertParam("Layer", 0L, 0L, 52L);
      iItem->getConfig()->assertParam("EnergyCutOff", 0.5, 0.2, 5.0);
      iItem->getConfig()->assertParam("Heatmap", true);
      iItem->getConfig()->assertParam("Z+", true);
      iItem->getConfig()->assertParam("Z-", true);
    }
  }

  std::unordered_set<unsigned> getCellsFromTriggerCell(const unsigned trigger_cell_id) const {
    constexpr unsigned hSc_triggercell_size_ = 2;
    // constexpr unsigned hSc_module_size_ = 12; // in TC units (144 TC / panel = 36 e-links)

    DetId trigger_cell_det_id(trigger_cell_id);
    unsigned det = trigger_cell_det_id.det();

    std::unordered_set<unsigned> cell_det_ids;

    // Scintillator
    if (det == DetId::HGCalHSc) {
      HGCScintillatorDetId trigger_cell_sc_id(trigger_cell_id);
      int ieta0 = (trigger_cell_sc_id.ietaAbs() - 1) * hSc_triggercell_size_ + 1;
      int iphi0 = (trigger_cell_sc_id.iphi() - 1) * hSc_triggercell_size_ + 1;
      for (int ietaAbs = ieta0; ietaAbs < ieta0 + (int)hSc_triggercell_size_; ietaAbs++) {
        int ieta = ietaAbs * trigger_cell_sc_id.zside();
        for (int iphi = iphi0; iphi < iphi0 + (int)hSc_triggercell_size_; iphi++) {
          unsigned cell_id = HGCScintillatorDetId(trigger_cell_sc_id.type(), trigger_cell_sc_id.layer(), ieta, iphi);
#if 0
        if (validCellId(DetId::HGCalHSc, cell_id))
#endif
          cell_det_ids.emplace(cell_id);
        }
      }
    }
    // Silicon
    else {
      HGCalTriggerDetId trigger_cell_trig_id(trigger_cell_id);
      unsigned subdet = trigger_cell_trig_id.subdet();
      if (subdet == HGCalTriggerSubdetector::HGCalEETrigger || subdet == HGCalTriggerSubdetector::HGCalHSiTrigger) {
        DetId::Detector cell_det =
            (subdet == HGCalTriggerSubdetector::HGCalEETrigger ? DetId::HGCalEE : DetId::HGCalHSi);
        int layer = trigger_cell_trig_id.layer();
        int zside = trigger_cell_trig_id.zside();
        int type = trigger_cell_trig_id.type();
        int waferu = trigger_cell_trig_id.waferU();
        int waferv = trigger_cell_trig_id.waferV();
        std::vector<int> cellus = trigger_cell_trig_id.cellU();
        std::vector<int> cellvs = trigger_cell_trig_id.cellV();
        for (unsigned ic = 0; ic < cellus.size(); ic++) {
          HGCSiliconDetId cell_det_id(cell_det, zside, type, layer, waferu, waferv, cellus[ic], cellvs[ic]);
          cell_det_ids.emplace(cell_det_id.rawId());
        }
      }
    }
    return cell_det_ids;
  }
};
