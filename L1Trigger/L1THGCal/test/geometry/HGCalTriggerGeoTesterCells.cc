#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeoTesterBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

class HGCalTriggerGeoTesterCells : public HGCalTriggerGeoTesterBase {
public:
  HGCalTriggerGeoTesterCells(const edm::ParameterSet& conf);
  ~HGCalTriggerGeoTesterCells() override {};
  void initialize(TTree*, const edm::ParameterSet&) final;
  void check(const HGCalTriggerGeoTesterEventSetup& es) final;
  void fill(const HGCalTriggerGeoTesterEventSetup& es) final;

private:
  void clear() final;
  bool validCellIdFromPosition(unsigned cell_id, const HGCalTriggerGeoTesterEventSetup& es) const;

  unsigned id_ = 0;
  unsigned errorbits_ = 0;
  int valid_ = 0;
  int zside_ = 0;
  int subdet_ = 0;
  int layer_ = 0;
  int u_ = 0;
  int v_ = 0;
  int ieta_ = 0;
  int iphi_ = 0;
  int waferU_ = 0;
  int waferV_ = 0;
  int type_ = 0;
  unsigned tcid_ = 0;
  unsigned modid_ = 0;
  float x_ = 0.;
  float y_ = 0.;
  float z_ = 0.;
  float eta_ = 0.;
  float phi_ = 0.;
  int corners_n_ = 0;
  std::vector<float> corners_x_;
  std::vector<float> corners_y_;
  std::vector<float> corners_z_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeoTesterFactory, HGCalTriggerGeoTesterCells, "HGCalTriggerGeoTesterCells");

HGCalTriggerGeoTesterCells::HGCalTriggerGeoTesterCells(const edm::ParameterSet& conf)
    : HGCalTriggerGeoTesterBase(conf) {}

void HGCalTriggerGeoTesterCells::initialize(TTree* tree, const edm::ParameterSet& conf) {
  tree_ = tree;

  tree_->Branch("valid", &valid_, "valid/I");
  tree_->Branch("id", &id_, "id/i");
  tree_->Branch("errorbits", &errorbits_, "errorbits/i");
  tree_->Branch("zside", &zside_, "zside/I");
  tree_->Branch("subdet", &subdet_, "subdet/I");
  tree_->Branch("layer", &layer_, "layer/I");
  tree_->Branch("u", &u_, "u/I");
  tree_->Branch("v", &v_, "v/I");
  tree_->Branch("ieta", &ieta_, "ieta/I");
  tree_->Branch("iphi", &iphi_, "iphi/I");
  tree_->Branch("waferU", &waferU_, "waferU/I");
  tree_->Branch("waferV", &waferV_, "waferV/I");
  tree_->Branch("type", &type_, "type/i");
  tree_->Branch("tcid", &tcid_, "tcid/i");
  tree_->Branch("modid", &modid_, "modid/I");
  tree_->Branch("x", &x_, "x/F");
  tree_->Branch("y", &y_, "y/F");
  tree_->Branch("z", &z_, "z/F");
  tree_->Branch("eta", &eta_, "eta/F");
  tree_->Branch("phi", &phi_, "phi/F");
  tree_->Branch("corners_n", &corners_n_, "corners_n/I");
  tree_->Branch("corners_x", &corners_x_);
  tree_->Branch("corners_y", &corners_y_);
  tree_->Branch("corners_z", &corners_z_);
}

void HGCalTriggerGeoTesterCells::fill(const HGCalTriggerGeoTesterEventSetup& es) {
  clear();
  // Loop over cells
  edm::LogPrint("GeoTester") << "Filling cells tree";
  // EE
  for (const auto& id : es.geometry->eeGeometry()->getValidDetIds()) {
    valid_ = es.geometry->eeTopology().valid(id);
    id_ = id.rawId();
    const auto error_itr = errors_.detids().find(id);
    if (error_itr != errors_.detids().end()) {
      for (const auto& error : error_itr->second) {
        errorbits_ |= (0x1 << error);
      }
    }
    HGCSiliconDetId detid(id);
    zside_ = detid.zside();
    subdet_ = detid.subdet();
    layer_ = detid.layer();
    u_ = detid.cellU();
    v_ = detid.cellV();
    ieta_ = -999;
    iphi_ = -999;
    waferU_ = detid.waferU();
    waferV_ = detid.waferV();
    type_ = detid.type();
    tcid_ = es.geometry->getTriggerCellFromCell(id);
    modid_ = es.geometry->getModuleFromTriggerCell(tcid_);
    //
    GlobalPoint center = es.geometry->eeGeometry()->getPosition(id);
    x_ = center.x();
    y_ = center.y();
    z_ = center.z();
    eta_ = center.eta();
    phi_ = center.phi();
    //
    std::vector<GlobalPoint> corners = es.geometry->eeGeometry()->getCorners(id);
    corners_n_ = corners.size();
    corners_x_.reserve(corners_n_);
    corners_y_.reserve(corners_n_);
    corners_z_.reserve(corners_n_);
    for (const auto& corner : corners) {
      corners_x_.emplace_back(corner.x());
      corners_y_.emplace_back(corner.y());
      corners_z_.emplace_back(corner.z());
    }
    tree_->Fill();
    clear();
  }
  for (const auto& id : es.geometry->hsiGeometry()->getValidDetIds()) {
    valid_ = es.geometry->hsiTopology().valid(id);
    id_ = id.rawId();
    const auto error_itr = errors_.detids().find(id);
    if (error_itr != errors_.detids().end()) {
      for (const auto& error : error_itr->second) {
        errorbits_ |= (0x1 << error);
      }
    }
    HGCSiliconDetId detid(id);
    zside_ = detid.zside();
    subdet_ = detid.subdet();
    layer_ = detid.layer();
    u_ = detid.cellU();
    v_ = detid.cellV();
    ieta_ = -999;
    iphi_ = -999;
    waferU_ = detid.waferU();
    waferV_ = detid.waferV();
    type_ = detid.type();
    tcid_ = es.geometry->getTriggerCellFromCell(id_);
    modid_ = es.geometry->getModuleFromTriggerCell(tcid_);
    //
    GlobalPoint center = es.geometry->hsiGeometry()->getPosition(id);
    x_ = center.x();
    y_ = center.y();
    z_ = center.z();
    eta_ = center.eta();
    phi_ = center.phi();
    //
    std::vector<GlobalPoint> corners = es.geometry->hsiGeometry()->getCorners(id);
    corners_n_ = corners.size();
    corners_x_.reserve(corners_n_);
    corners_y_.reserve(corners_n_);
    corners_z_.reserve(corners_n_);
    for (const auto& corner : corners) {
      corners_x_.emplace_back(corner.x());
      corners_y_.emplace_back(corner.y());
      corners_z_.emplace_back(corner.z());
    }
    tree_->Fill();
    clear();
  }

  if (es.geometry->isWithNoseGeometry()) {
    for (const auto& id : es.geometry->noseGeometry()->getValidDetIds()) {
      valid_ = es.geometry->noseTopology().valid(id);
      id_ = id.rawId();
      const auto error_itr = errors_.detids().find(id);
      if (error_itr != errors_.detids().end()) {
        for (const auto& error : error_itr->second) {
          errorbits_ |= (0x1 << error);
        }
      }
      HFNoseDetId detid(id);
      zside_ = detid.zside();
      subdet_ = detid.subdet();
      layer_ = detid.layer();
      u_ = detid.cellU();
      v_ = detid.cellV();
      ieta_ = -999;
      iphi_ = -999;
      waferU_ = detid.waferU();
      waferV_ = detid.waferV();
      type_ = detid.type();
      tcid_ = es.geometry->getTriggerCellFromCell(id_);
      modid_ = es.geometry->getModuleFromTriggerCell(tcid_);
      //
      GlobalPoint center = es.geometry->noseGeometry()->getPosition(id);
      x_ = center.x();
      y_ = center.y();
      z_ = center.z();
      eta_ = center.eta();
      phi_ = center.phi();
      //
      std::vector<GlobalPoint> corners = es.geometry->noseGeometry()->getCorners(id);
      corners_n_ = corners.size();
      corners_x_.reserve(corners_n_);
      corners_y_.reserve(corners_n_);
      corners_z_.reserve(corners_n_);
      for (const auto& corner : corners) {
        corners_x_.emplace_back(corner.x());
        corners_y_.emplace_back(corner.y());
        corners_z_.emplace_back(corner.z());
      }
      tree_->Fill();
      clear();
    }
  }
  for (const auto& id : es.geometry->hscGeometry()->getValidDetIds()) {
    valid_ = es.geometry->hscTopology().valid(id);
    id_ = id.rawId();
    const auto error_itr = errors_.detids().find(id);
    if (error_itr != errors_.detids().end()) {
      for (const auto& error : error_itr->second) {
        errorbits_ |= (0x1 << error);
      }
    }
    HGCScintillatorDetId detid(id);
    zside_ = detid.zside();
    subdet_ = detid.subdet();
    layer_ = detid.layer();
    u_ = -999;
    v_ = -999;
    ieta_ = detid.ieta();
    iphi_ = detid.iphi();
    waferU_ = -999;
    waferV_ = -999;
    type_ = detid.type();
    tcid_ = es.geometry->getTriggerCellFromCell(id_);
    modid_ = es.geometry->getModuleFromTriggerCell(tcid_);
    //
    GlobalPoint center = es.geometry->hscGeometry()->getPosition(id);
    x_ = center.x();
    y_ = center.y();
    z_ = center.z();
    eta_ = center.eta();
    phi_ = center.phi();
    //
    std::vector<GlobalPoint> corners = es.geometry->hscGeometry()->getCorners(id);
    corners_n_ = corners.size();
    corners_x_.reserve(corners_n_);
    corners_y_.reserve(corners_n_);
    corners_z_.reserve(corners_n_);
    for (const auto& corner : corners) {
      corners_x_.emplace_back(corner.x());
      corners_y_.emplace_back(corner.y());
      corners_z_.emplace_back(corner.z());
    }
    tree_->Fill();
    clear();
  }
}

void HGCalTriggerGeoTesterCells::check(const HGCalTriggerGeoTesterEventSetup& es) {
  edm::LogPrint("GeoTester") << "Checking cells";
  for (const auto& id : es.geometry->eeGeometry()->getValidDetIds()) {
    if (!es.geometry->eeTopology().valid(id)) {
      errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
    }
    if (!validCellIdFromPosition(id, es)) {
      errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
    }
  }

  for (const auto& id : es.geometry->hsiGeometry()->getValidDetIds()) {
    if (!es.geometry->hsiTopology().valid(id)) {
      errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
    }
    if (!validCellIdFromPosition(id, es)) {
      errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
    }
  }

  for (const auto& id : es.geometry->hscGeometry()->getValidDetIds()) {
    if (!es.geometry->hscTopology().valid(id)) {
      errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
    }
    if (!validCellIdFromPosition(id, es)) {
      errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
    }
  }

  if (es.geometry->isWithNoseGeometry()) {
    for (const auto& id : es.geometry->noseGeometry()->getValidDetIds()) {
      if (!es.geometry->noseTopology().valid(id)) {
        errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
      }
      if (!validCellIdFromPosition(id, es)) {
        errors_.fill(HGCalTriggerGeoTesterErrors::CellValidity, id);
      }
    }
  }
}

bool HGCalTriggerGeoTesterCells::validCellIdFromPosition(unsigned cell_id,
                                                         const HGCalTriggerGeoTesterEventSetup& es) const {
  float threshold = 0.1;
  float cell_z = 0.;
  unsigned subdet = DetId(cell_id).det();
  switch (subdet) {
    case DetId::HGCalEE:
      cell_z = GlobalPoint(es.geometry->eeGeometry()->getPosition(cell_id).basicVector()).z();
      break;
    case DetId::HGCalHSi:
      cell_z = GlobalPoint(es.geometry->hsiGeometry()->getPosition(cell_id).basicVector()).z();
      break;
    case DetId::HGCalHSc:
      cell_z = GlobalPoint(es.geometry->hscGeometry()->getPosition(cell_id).basicVector()).z();
      break;
    case DetId::Forward:
      cell_z = GlobalPoint(es.geometry->noseGeometry()->getPosition(cell_id).basicVector()).z();
      break;
    default:
      break;
  }
  bool is_valid = (std::abs(cell_z) > threshold);
  return is_valid;
}

void HGCalTriggerGeoTesterCells::clear() {
  id_ = 0;
  valid_ = 0;
  errorbits_ = 0;
  zside_ = 0;
  subdet_ = 0;
  layer_ = 0;
  u_ = 0;
  v_ = 0;
  ieta_ = 0;
  iphi_ = 0;
  waferU_ = 0;
  waferV_ = 0;
  type_ = 0;
  tcid_ = 0;
  modid_ = 0;
  x_ = 0.;
  y_ = 0.;
  z_ = 0.;
  eta_ = 0.;
  phi_ = 0.;
  corners_n_ = 0;
  corners_x_.clear();
  corners_y_.clear();
  corners_z_.clear();
}
