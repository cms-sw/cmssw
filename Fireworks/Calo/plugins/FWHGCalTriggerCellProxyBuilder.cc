#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
// #include "Fireworks/Calo/interface/FWHeatmapProxyBuilderTemplate.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
// #include "L1Trigger/L1THGCal/plugins/geometries/HGCalTriggerGeometryV9Imp2.cc"

#include "TEveBoxSet.h"

class FWHGCalTriggerCellProxyBuilder : public FWSimpleProxyBuilderTemplate<l1t::HGCalTriggerCell>
{
public:
    FWHGCalTriggerCellProxyBuilder(void) {}
    ~FWHGCalTriggerCellProxyBuilder(void) override {}

    REGISTER_PROXYBUILDER_METHODS();

private:
    FWHGCalTriggerCellProxyBuilder(const FWHGCalTriggerCellProxyBuilder &) = delete;                  // stop default
    const FWHGCalTriggerCellProxyBuilder &operator=(const FWHGCalTriggerCellProxyBuilder &) = delete; // stop default

    void build(const l1t::HGCalTriggerCell &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *) override;

    //https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1THGCal/plugins/geometries/HGCalTriggerGeometryV9Imp2.cc#L191
    std::unordered_set<unsigned> getCellsFromTriggerCell(const unsigned trigger_cell_id) const;

    // bool validCellId(unsigned subdet, unsigned cell_id) const;
};

/*
bool 
FWHGCalTriggerCellProxyBuilder::validCellId(unsigned subdet, unsigned cell_id) const {
  bool is_valid = false;
  switch (subdet) {
    case DetId::HGCalEE:
      is_valid = eeTopology().valid(cell_id);
      break;
    case DetId::HGCalHSi:
      is_valid = hsiTopology().valid(cell_id);
      break;
    case DetId::HGCalHSc:
      is_valid = hscTopology().valid(cell_id);
      break;
    default:
      is_valid = false;
      break;
  }
  return is_valid;
}
*/

std::unordered_set<unsigned> 
FWHGCalTriggerCellProxyBuilder::getCellsFromTriggerCell(const unsigned trigger_cell_id) const {
  
  constexpr unsigned hSc_triggercell_size_ = 2;
  constexpr unsigned hSc_module_size_ = 12;  // in TC units (144 TC / panel = 36 e-links)
  
  
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
      DetId::Detector cell_det = (subdet == HGCalTriggerSubdetector::HGCalEETrigger ? DetId::HGCalEE : DetId::HGCalHSi);
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

void FWHGCalTriggerCellProxyBuilder::build(const l1t::HGCalTriggerCell &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *)
{
    bool h_hex(false);
    TEveBoxSet *hex_boxset = new TEveBoxSet();
    hex_boxset->UseSingleColor();
    hex_boxset->SetPickable(true);
    hex_boxset->Reset(TEveBoxSet::kBT_Hex, true, 64);
    hex_boxset->SetAntiFlick(true);

    bool h_box(false);
    TEveBoxSet *boxset = new TEveBoxSet();
    boxset->UseSingleColor();
    boxset->SetPickable(true);
    boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
    boxset->SetAntiFlick(true);

    std::unordered_set<unsigned> cells = getCellsFromTriggerCell(iData.detId());

    for (std::unordered_set<unsigned>::const_iterator it = cells.begin(), itEnd = cells.end();
         it != itEnd; ++it)
    {
        const float *corners = item()->getGeom()->getCorners(*it);
        const float *parameters = item()->getGeom()->getParameters(*it);
        const float *shapes = item()->getGeom()->getShapePars(*it);

        if (corners == nullptr || parameters == nullptr || shapes == nullptr){
            continue;
        }

        const int total_points = parameters[0];
        const bool isScintillator = (total_points == 4);

        // Scintillator
        if (isScintillator)
        {
            const int total_vertices = 3 * total_points;

            std::vector<float> pnts(24);
            for (int i = 0; i < total_points; ++i)
            {
                pnts[i * 3 + 0] = corners[i * 3];
                pnts[i * 3 + 1] = corners[i * 3 + 1];
                pnts[i * 3 + 2] = corners[i * 3 + 2];

                pnts[(i * 3 + 0) + total_vertices] = corners[i * 3];
                pnts[(i * 3 + 1) + total_vertices] = corners[i * 3 + 1];
                pnts[(i * 3 + 2) + total_vertices] = corners[i * 3 + 2] + shapes[3];
            }
            boxset->AddBox(&pnts[0]);

            h_box = true;
        }
        // Silicon
        else
        {
            const int offset = 9;

            float centerX = (corners[6] + corners[6 + offset]) / 2;
            float centerY = (corners[7] + corners[7 + offset]) / 2;
            float radius = fabs(corners[6] - corners[6 + offset]) / 2;
            hex_boxset->AddHex(TEveVector(centerX, centerY, corners[2]),
                               radius, 90.0, shapes[3]);

            h_hex = true;
        }
    }

    if (h_hex)
    {
        hex_boxset->RefitPlex();

        hex_boxset->CSCTakeAnyParentAsMaster();
        hex_boxset->CSCApplyMainColorToMatchingChildren();
        hex_boxset->CSCApplyMainTransparencyToMatchingChildren();
        hex_boxset->SetMainColor(item()->modelInfo(iIndex).displayProperties().color());
        hex_boxset->SetMainTransparency(item()->defaultDisplayProperties().transparency());
        oItemHolder.AddElement(hex_boxset);
    }

    if (h_box)
    {
        boxset->RefitPlex();

        boxset->CSCTakeAnyParentAsMaster();
        boxset->CSCApplyMainColorToMatchingChildren();
        boxset->CSCApplyMainTransparencyToMatchingChildren();
        boxset->SetMainColor(item()->modelInfo(iIndex).displayProperties().color());
        boxset->SetMainTransparency(item()->defaultDisplayProperties().transparency());
        oItemHolder.AddElement(boxset);
    }
}

REGISTER_FWPROXYBUILDER(FWHGCalTriggerCellProxyBuilder, l1t::HGCalTriggerCell, "HGCal Trigger Cell", FWViewType::kAll3DBits);
