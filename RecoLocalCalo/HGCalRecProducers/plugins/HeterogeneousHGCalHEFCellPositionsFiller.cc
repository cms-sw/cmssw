#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEFCellPositionsFiller.h"

HeterogeneousHGCalHEFCellPositionsFiller::HeterogeneousHGCalHEFCellPositionsFiller(const edm::ParameterSet& ps) {
  geometryToken_ = setWhatProduced(this).consumesFrom<HGCalGeometry, IdealGeometryRecord>(
      edm::ESInputTag{"", "HGCalHESiliconSensitive"});
  posmap_ = new hgcal_conditions::positions::HGCalPositionsMapping();
}

HeterogeneousHGCalHEFCellPositionsFiller::~HeterogeneousHGCalHEFCellPositionsFiller() { delete posmap_; }

//the geometry is not required if the layer offset is hardcoded (potential speed-up)
void HeterogeneousHGCalHEFCellPositionsFiller::set_conditions_() {
  //fill the CPU position structure from the geometry
  posmap_->zLayer.clear();
  posmap_->nCellsLayer.clear();
  posmap_->nCellsWaferUChunk.clear();
  posmap_->nCellsHexagon.clear();
  posmap_->detid.clear();

  int nlayers = ddd_->lastLayer(true) - ddd_->firstLayer() + 1;
  int upper_estimate_wafer_number_1D = 2 * nlayers * (ddd_->waferMax() - ddd_->waferMin());
  int upper_estimate_wafer_number_2D = upper_estimate_wafer_number_1D * (ddd_->waferMax() - ddd_->waferMin());
  int upper_estimate_cell_number = upper_estimate_wafer_number_2D * 3 * 12 * 12;
  posmap_->zLayer.resize(nlayers * 2);
  posmap_->nCellsLayer.reserve(nlayers * 2);
  posmap_->nCellsWaferUChunk.reserve(upper_estimate_wafer_number_1D);
  posmap_->nCellsHexagon.reserve(upper_estimate_wafer_number_2D);
  posmap_->detid.reserve(upper_estimate_cell_number);
  //set position-related variables
  posmap_->waferSize = static_cast<float>(params_->waferSize_);
  posmap_->sensorSeparation = static_cast<float>(params_->sensorSeparation_);
  posmap_->firstLayer = ddd_->firstLayer();
  assert(posmap_->firstLayer == 1);  //otherwise the loop over the layers has to be changed
  posmap_->lastLayer = ddd_->lastLayer(true);
  posmap_->waferMin = ddd_->waferMin();
  posmap_->waferMax = ddd_->waferMax();

  unsigned sumCellsLayer, sumCellsWaferUChunk;

  //store detids following a geometry ordering
  for (int ilayer = 1; ilayer <= posmap_->lastLayer; ++ilayer) {
    sumCellsLayer = 0;
    posmap_->zLayer[ilayer - 1] = static_cast<float>(ddd_->waferZ(ilayer, true));            //originally a double
    posmap_->zLayer[ilayer - 1 + nlayers] = static_cast<float>(ddd_->waferZ(ilayer, true));  //originally a double

    for (int iwaferU = posmap_->waferMin; iwaferU < posmap_->waferMax; ++iwaferU) {
      sumCellsWaferUChunk = 0;

      for (int iwaferV = posmap_->waferMin; iwaferV < posmap_->waferMax; ++iwaferV) {
        //0: fine; 1: coarseThin; 2: coarseThick (as defined in DataFormats/ForwardDetId/interface/HGCSiliconDetId.h)
        int type_ = ddd_->waferType(ilayer, iwaferU, iwaferV, false);

        int nCellsHexSide = ddd_->numberCellsHexagon(ilayer, iwaferU, iwaferV, false);
        int nCellsHexTotal = ddd_->numberCellsHexagon(ilayer, iwaferU, iwaferV, true);
        sumCellsLayer += nCellsHexTotal;
        sumCellsWaferUChunk += nCellsHexTotal;
        posmap_->nCellsHexagon.push_back(nCellsHexTotal);

        //left side of wafer
        for (int cellUmax = nCellsHexSide, icellV = 0; cellUmax < 2 * nCellsHexSide and icellV < nCellsHexSide;
             ++cellUmax, ++icellV) {
          for (int icellU = 0; icellU <= cellUmax; ++icellU) {
            HGCSiliconDetId detid_(DetId::HGCalHSi, 1, type_, ilayer, iwaferU, iwaferV, icellU, icellV);
            posmap_->detid.push_back(detid_.rawId());
          }
        }
        //right side of wafer
        for (int cellUmin = 1, icellV = nCellsHexSide; cellUmin <= nCellsHexSide and icellV < 2 * nCellsHexSide;
             ++cellUmin, ++icellV) {
          for (int icellU = cellUmin; icellU < 2 * nCellsHexSide; ++icellU) {
            HGCSiliconDetId detid_(DetId::HGCalHSi, 1, type_, ilayer, iwaferU, iwaferV, icellU, icellV);
            posmap_->detid.push_back(detid_.rawId());
          }
        }
      }
      posmap_->nCellsWaferUChunk.push_back(sumCellsWaferUChunk);
    }
    posmap_->nCellsLayer.push_back(sumCellsLayer);
  }
}

std::unique_ptr<HeterogeneousHGCalHEFCellPositionsConditions> HeterogeneousHGCalHEFCellPositionsFiller::produce(
    const HeterogeneousHGCalHEFCellPositionsConditionsRecord& iRecord) {
  auto geom = iRecord.getTransientHandle(geometryToken_);
  ddd_ = &(geom->topology().dddConstants());
  params_ = ddd_->getParameter();

  set_conditions_();

  std::unique_ptr<HeterogeneousHGCalHEFCellPositionsConditions> up =
      std::make_unique<HeterogeneousHGCalHEFCellPositionsConditions>(posmap_);
  return up;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
DEFINE_FWK_EVENTSETUP_MODULE(HeterogeneousHGCalHEFCellPositionsFiller);
