#include <iostream>
#include <string>
#include <vector>

#include "TTree.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"

#include <cstdlib>

namespace {
  template <typename T>
  struct array_deleter {
    void operator()(T* arr) { delete[] arr; }
  };
}  // namespace

class HGCalTriggerGeomTesterV9Imp3 : public edm::stream::EDAnalyzer<> {
public:
  explicit HGCalTriggerGeomTesterV9Imp3(const edm::ParameterSet&);
  ~HGCalTriggerGeomTesterV9Imp3() override = default;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void fillTriggerGeometry();
  bool checkMappingConsistency();
  void setTreeModuleSize(const size_t n);
  void setTreeModuleCellSize(const size_t n);
  void setTreeTriggerCellSize(const size_t n);
  void setTreeCellCornerSize(const size_t n);
  void setTreeTriggerCellNeighborSize(const size_t n);

  const edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  bool no_trigger_;
  bool no_neighbors_;
  TTree* treeModules_;
  TTree* treeTriggerCells_;
  TTree* treeCells_;
  TTree* treeCellsBH_;
  TTree* treeCellsNose_;
  TTree* treeModuleErrors_;
  // tree variables
  int moduleId_ = 0;
  int moduleSide_ = 0;
  int moduleSubdet_ = 0;
  int moduleLayer_ = 0;
  int moduleIEta_ = 0;
  int moduleIPhi_ = 0;
  int module_ = 0;
  float moduleX_ = 0;
  float moduleY_ = 0;
  float moduleZ_ = 0;
  float moduleEta_ = 0;
  float modulePhi_ = 0;
  int moduleTC_N_ = 0;
  int moduleLinks_ = 0;
  std::shared_ptr<int> moduleTC_id_;
  std::shared_ptr<int> moduleTC_zside_;
  std::shared_ptr<int> moduleTC_subdet_;
  std::shared_ptr<int> moduleTC_layer_;
  std::shared_ptr<int> moduleTC_waferU_;
  std::shared_ptr<int> moduleTC_waferV_;
  std::shared_ptr<int> moduleTC_cellU_;
  std::shared_ptr<int> moduleTC_cellV_;
  std::shared_ptr<int> moduleTC_ieta_;
  std::shared_ptr<int> moduleTC_iphi_;
  std::shared_ptr<float> moduleTC_x_;
  std::shared_ptr<float> moduleTC_y_;
  std::shared_ptr<float> moduleTC_z_;
  std::shared_ptr<float> moduleTC_eta_;
  std::shared_ptr<float> moduleTC_phi_;
  int moduleCell_N_ = 0;
  std::shared_ptr<int> moduleCell_id_;
  std::shared_ptr<int> moduleCell_zside_;
  std::shared_ptr<int> moduleCell_subdet_;
  std::shared_ptr<int> moduleCell_layer_;
  std::shared_ptr<int> moduleCell_waferU_;
  std::shared_ptr<int> moduleCell_waferV_;
  std::shared_ptr<int> moduleCell_cellU_;
  std::shared_ptr<int> moduleCell_cellV_;
  std::shared_ptr<int> moduleCell_ieta_;
  std::shared_ptr<int> moduleCell_iphi_;
  std::shared_ptr<float> moduleCell_x_;
  std::shared_ptr<float> moduleCell_y_;
  std::shared_ptr<float> moduleCell_z_;
  std::shared_ptr<float> moduleCell_eta_;
  std::shared_ptr<float> moduleCell_phi_;
  int triggerCellId_ = 0;
  int triggerCellSide_ = 0;
  int triggerCellSubdet_ = 0;
  int triggerCellLayer_ = 0;
  int triggerCellWaferU_ = 0;
  int triggerCellWaferV_ = 0;
  int triggerCellU_ = 0;
  int triggerCellV_ = 0;
  int triggerCellIEta_ = 0;
  int triggerCellIPhi_ = 0;
  float triggerCellX_ = 0;
  float triggerCellY_ = 0;
  float triggerCellZ_ = 0;
  float triggerCellEta_ = 0;
  float triggerCellPhi_ = 0;
  int triggerCellNeighbor_N_ = 0;
  std::shared_ptr<int> triggerCellNeighbor_id_;
  std::shared_ptr<int> triggerCellNeighbor_zside_;
  std::shared_ptr<int> triggerCellNeighbor_subdet_;
  std::shared_ptr<int> triggerCellNeighbor_layer_;
  std::shared_ptr<int> triggerCellNeighbor_waferU_;
  std::shared_ptr<int> triggerCellNeighbor_waferV_;
  std::shared_ptr<int> triggerCellNeighbor_cellU_;
  std::shared_ptr<int> triggerCellNeighbor_cellV_;
  std::shared_ptr<int> triggerCellNeighbor_cellIEta_;
  std::shared_ptr<int> triggerCellNeighbor_cellIPhi_;
  std::shared_ptr<float> triggerCellNeighbor_distance_;
  int triggerCellCell_N_ = 0;
  std::shared_ptr<int> triggerCellCell_id_;
  std::shared_ptr<int> triggerCellCell_zside_;
  std::shared_ptr<int> triggerCellCell_subdet_;
  std::shared_ptr<int> triggerCellCell_layer_;
  std::shared_ptr<int> triggerCellCell_waferU_;
  std::shared_ptr<int> triggerCellCell_waferV_;
  std::shared_ptr<int> triggerCellCell_cellU_;
  std::shared_ptr<int> triggerCellCell_cellV_;
  std::shared_ptr<int> triggerCellCell_ieta_;
  std::shared_ptr<int> triggerCellCell_iphi_;
  std::shared_ptr<float> triggerCellCell_x_;
  std::shared_ptr<float> triggerCellCell_y_;
  std::shared_ptr<float> triggerCellCell_z_;
  std::shared_ptr<float> triggerCellCell_eta_;
  std::shared_ptr<float> triggerCellCell_phi_;
  int cellId_ = 0;
  int cellSide_ = 0;
  int cellSubdet_ = 0;
  int cellLayer_ = 0;
  int cellWaferU_ = 0;
  int cellWaferV_ = 0;
  int cellWaferType_ = 0;
  int cellWaferRow_ = 0;
  int cellWaferColumn_ = 0;
  int cellU_ = 0;
  int cellV_ = 0;
  float cellX_ = 0;
  float cellY_ = 0;
  float cellZ_ = 0;
  float cellEta_ = 0;
  float cellPhi_ = 0;
  int cellCornersN_ = 0;
  std::shared_ptr<float> cellCornersX_;
  std::shared_ptr<float> cellCornersY_;
  std::shared_ptr<float> cellCornersZ_;
  int cellBHId_ = 0;
  int cellBHType_ = 0;
  int cellBHSide_ = 0;
  int cellBHSubdet_ = 0;
  int cellBHLayer_ = 0;
  int cellBHIEta_ = 0;
  int cellBHIPhi_ = 0;
  float cellBHEta_ = 0;
  float cellBHPhi_ = 0;
  float cellBHX_ = 0;
  float cellBHY_ = 0;
  float cellBHZ_ = 0;
  float cellBHX1_ = 0;
  float cellBHY1_ = 0;
  float cellBHX2_ = 0;
  float cellBHY2_ = 0;
  float cellBHX3_ = 0;
  float cellBHY3_ = 0;
  float cellBHX4_ = 0;
  float cellBHY4_ = 0;
  //
  int moduleErrorSubdet_ = 0;
  int moduleErrorLayer_ = 0;
  int moduleErrorWaferU_ = 0;
  int moduleErrorWaferV_ = 0;

private:
  typedef std::unordered_map<uint32_t, std::unordered_set<uint32_t>> trigger_map_set;
};

/*****************************************************************/
HGCalTriggerGeomTesterV9Imp3::HGCalTriggerGeomTesterV9Imp3(const edm::ParameterSet& conf)
    : triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()),
      no_trigger_(false),
      no_neighbors_(true)
/*****************************************************************/
{
  edm::Service<TFileService> fs;
  // initialize output trees
  treeModules_ = fs->make<TTree>("TreeModules", "Tree of all HGC modules");
  treeModules_->Branch("id", &moduleId_, "id/I");
  treeModules_->Branch("zside", &moduleSide_, "zside/I");
  treeModules_->Branch("subdet", &moduleSubdet_, "subdet/I");
  treeModules_->Branch("layer", &moduleLayer_, "layer/I");
  treeModules_->Branch("ieta", &moduleIEta_, "ieta/I");
  treeModules_->Branch("iphi", &moduleIPhi_, "iphi/I");
  treeModules_->Branch("module", &module_, "module/I");
  treeModules_->Branch("links", &moduleLinks_, "links/I");
  treeModules_->Branch("x", &moduleX_, "x/F");
  treeModules_->Branch("y", &moduleY_, "y/F");
  treeModules_->Branch("z", &moduleZ_, "z/F");
  treeModules_->Branch("eta", &moduleEta_, "eta/F");
  treeModules_->Branch("phi", &modulePhi_, "phi/F");
  treeModules_->Branch("tc_n", &moduleTC_N_, "tc_n/I");
  moduleTC_id_.reset(new int[1], array_deleter<int>());
  moduleTC_zside_.reset(new int[1], array_deleter<int>());
  moduleTC_subdet_.reset(new int[1], array_deleter<int>());
  moduleTC_layer_.reset(new int[1], array_deleter<int>());
  moduleTC_waferU_.reset(new int[1], array_deleter<int>());
  moduleTC_waferV_.reset(new int[1], array_deleter<int>());
  moduleTC_cellU_.reset(new int[1], array_deleter<int>());
  moduleTC_cellV_.reset(new int[1], array_deleter<int>());
  moduleTC_x_.reset(new float[1], array_deleter<float>());
  moduleTC_y_.reset(new float[1], array_deleter<float>());
  moduleTC_z_.reset(new float[1], array_deleter<float>());
  moduleTC_eta_.reset(new float[1], array_deleter<float>());
  moduleTC_phi_.reset(new float[1], array_deleter<float>());
  treeModules_->Branch("tc_id", moduleTC_id_.get(), "tc_id[tc_n]/I");
  treeModules_->Branch("tc_zside", moduleTC_zside_.get(), "tc_zside[tc_n]/I");
  treeModules_->Branch("tc_subdet", moduleTC_subdet_.get(), "tc_subdet[tc_n]/I");
  treeModules_->Branch("tc_layer", moduleTC_layer_.get(), "tc_layer[tc_n]/I");
  treeModules_->Branch("tc_waferu", moduleTC_waferU_.get(), "tc_waferu[tc_n]/I");
  treeModules_->Branch("tc_waferv", moduleTC_waferV_.get(), "tc_waferv[tc_n]/I");
  treeModules_->Branch("tc_cellu", moduleTC_cellU_.get(), "tc_cellu[tc_n]/I");
  treeModules_->Branch("tc_cellv", moduleTC_cellV_.get(), "tc_cellv[tc_n]/I");
  treeModules_->Branch("tc_ieta", moduleTC_ieta_.get(), "tc_ieta[tc_n]/I");
  treeModules_->Branch("tc_iphi", moduleTC_iphi_.get(), "tc_iphi[tc_n]/I");
  treeModules_->Branch("tc_x", moduleTC_x_.get(), "tc_x[tc_n]/F");
  treeModules_->Branch("tc_y", moduleTC_y_.get(), "tc_y[tc_n]/F");
  treeModules_->Branch("tc_z", moduleTC_z_.get(), "tc_z[tc_n]/F");
  treeModules_->Branch("tc_eta", moduleTC_eta_.get(), "tc_eta[tc_n]/F");
  treeModules_->Branch("tc_phi", moduleTC_phi_.get(), "tc_phi[tc_n]/F");
  treeModules_->Branch("c_n", &moduleCell_N_, "c_n/I");
  moduleCell_id_.reset(new int[1], array_deleter<int>());
  moduleCell_zside_.reset(new int[1], array_deleter<int>());
  moduleCell_subdet_.reset(new int[1], array_deleter<int>());
  moduleCell_layer_.reset(new int[1], array_deleter<int>());
  moduleCell_waferU_.reset(new int[1], array_deleter<int>());
  moduleCell_waferV_.reset(new int[1], array_deleter<int>());
  moduleCell_cellU_.reset(new int[1], array_deleter<int>());
  moduleCell_cellV_.reset(new int[1], array_deleter<int>());
  moduleCell_ieta_.reset(new int[1], array_deleter<int>());
  moduleCell_iphi_.reset(new int[1], array_deleter<int>());
  moduleCell_x_.reset(new float[1], array_deleter<float>());
  moduleCell_y_.reset(new float[1], array_deleter<float>());
  moduleCell_z_.reset(new float[1], array_deleter<float>());
  moduleCell_eta_.reset(new float[1], array_deleter<float>());
  moduleCell_phi_.reset(new float[1], array_deleter<float>());
  treeModules_->Branch("c_id", moduleCell_id_.get(), "c_id[c_n]/I");
  treeModules_->Branch("c_zside", moduleCell_zside_.get(), "c_zside[c_n]/I");
  treeModules_->Branch("c_subdet", moduleCell_subdet_.get(), "c_subdet[c_n]/I");
  treeModules_->Branch("c_layer", moduleCell_layer_.get(), "c_layer[c_n]/I");
  treeModules_->Branch("c_waferu", moduleCell_waferU_.get(), "c_waferu[c_n]/I");
  treeModules_->Branch("c_waferv", moduleCell_waferV_.get(), "c_waferv[c_n]/I");
  treeModules_->Branch("c_cellu", moduleCell_cellU_.get(), "c_cellu[c_n]/I");
  treeModules_->Branch("c_cellv", moduleCell_cellV_.get(), "c_cellv[c_n]/I");
  treeModules_->Branch("c_ieta", moduleCell_ieta_.get(), "c_ieta[c_n]/I");
  treeModules_->Branch("c_iphi", moduleCell_iphi_.get(), "c_iphi[c_n]/I");
  treeModules_->Branch("c_x", moduleCell_x_.get(), "c_x[c_n]/F");
  treeModules_->Branch("c_y", moduleCell_y_.get(), "c_y[c_n]/F");
  treeModules_->Branch("c_z", moduleCell_z_.get(), "c_z[c_n]/F");
  treeModules_->Branch("c_eta", moduleCell_eta_.get(), "c_eta[c_n]/F");
  treeModules_->Branch("c_phi", moduleCell_phi_.get(), "c_phi[c_n]/F");
  //
  treeTriggerCells_ = fs->make<TTree>("TreeTriggerCells", "Tree of all HGC trigger cells");
  treeTriggerCells_->Branch("id", &triggerCellId_, "id/I");
  treeTriggerCells_->Branch("zside", &triggerCellSide_, "zside/I");
  treeTriggerCells_->Branch("subdet", &triggerCellSubdet_, "subdet/I");
  treeTriggerCells_->Branch("layer", &triggerCellLayer_, "layer/I");
  treeTriggerCells_->Branch("waferu", &triggerCellWaferU_, "waferu/I");
  treeTriggerCells_->Branch("waferv", &triggerCellWaferV_, "waferv/I");
  treeTriggerCells_->Branch("triggercellu", &triggerCellU_, "triggercellu/I");
  treeTriggerCells_->Branch("triggercellv", &triggerCellV_, "triggercellv/I");
  treeTriggerCells_->Branch("triggercellieta", &triggerCellIEta_, "triggercellieta/I");
  treeTriggerCells_->Branch("triggercelliphi", &triggerCellIPhi_, "triggercelliphi/I");
  treeTriggerCells_->Branch("x", &triggerCellX_, "x/F");
  treeTriggerCells_->Branch("y", &triggerCellY_, "y/F");
  treeTriggerCells_->Branch("z", &triggerCellZ_, "z/F");
  treeTriggerCells_->Branch("eta", &triggerCellEta_, "eta/F");
  treeTriggerCells_->Branch("phi", &triggerCellPhi_, "phi/F");
  treeTriggerCells_->Branch("neighbor_n", &triggerCellNeighbor_N_, "neighbor_n/I");
  triggerCellNeighbor_id_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_zside_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_subdet_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_layer_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_waferU_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_waferV_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_cellU_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_cellV_.reset(new int[1], array_deleter<int>());
  triggerCellNeighbor_distance_.reset(new float[1], array_deleter<float>());
  treeTriggerCells_->Branch("neighbor_id", triggerCellNeighbor_id_.get(), "neighbor_id[neighbor_n]/I");
  treeTriggerCells_->Branch("neighbor_zside", triggerCellNeighbor_zside_.get(), "neighbor_zside[neighbor_n]/I");
  treeTriggerCells_->Branch("neighbor_subdet", triggerCellNeighbor_subdet_.get(), "neighbor_subdet[neighbor_n]/I");
  treeTriggerCells_->Branch("neighbor_layer", triggerCellNeighbor_layer_.get(), "neighbor_layer[neighbor_n]/I");
  treeTriggerCells_->Branch("neighbor_waferu", triggerCellNeighbor_waferU_.get(), "neighbor_waferu[neighbor_n]/I");
  treeTriggerCells_->Branch("neighbor_waferv", triggerCellNeighbor_waferV_.get(), "neighbor_waferv[neighbor_n]/I");
  treeTriggerCells_->Branch("neighbor_cellu", triggerCellNeighbor_cellU_.get(), "neighbor_cellu[neighbor_n]/I");
  treeTriggerCells_->Branch("neighbor_cellv", triggerCellNeighbor_cellV_.get(), "neighbor_cellv[neighbor_n]/I");
  treeTriggerCells_->Branch(
      "neighbor_distance", triggerCellNeighbor_distance_.get(), "neighbor_distance[neighbor_n]/F");
  treeTriggerCells_->Branch("c_n", &triggerCellCell_N_, "c_n/I");
  triggerCellCell_id_.reset(new int[1], array_deleter<int>());
  triggerCellCell_zside_.reset(new int[1], array_deleter<int>());
  triggerCellCell_subdet_.reset(new int[1], array_deleter<int>());
  triggerCellCell_layer_.reset(new int[1], array_deleter<int>());
  triggerCellCell_waferU_.reset(new int[1], array_deleter<int>());
  triggerCellCell_waferV_.reset(new int[1], array_deleter<int>());
  triggerCellCell_cellU_.reset(new int[1], array_deleter<int>());
  triggerCellCell_cellV_.reset(new int[1], array_deleter<int>());
  triggerCellCell_ieta_.reset(new int[1], array_deleter<int>());
  triggerCellCell_iphi_.reset(new int[1], array_deleter<int>());
  triggerCellCell_x_.reset(new float[1], array_deleter<float>());
  triggerCellCell_y_.reset(new float[1], array_deleter<float>());
  triggerCellCell_z_.reset(new float[1], array_deleter<float>());
  triggerCellCell_eta_.reset(new float[1], array_deleter<float>());
  triggerCellCell_phi_.reset(new float[1], array_deleter<float>());
  treeTriggerCells_->Branch("c_id", triggerCellCell_id_.get(), "c_id[c_n]/I");
  treeTriggerCells_->Branch("c_zside", triggerCellCell_zside_.get(), "c_zside[c_n]/I");
  treeTriggerCells_->Branch("c_subdet", triggerCellCell_subdet_.get(), "c_subdet[c_n]/I");
  treeTriggerCells_->Branch("c_layer", triggerCellCell_layer_.get(), "c_layer[c_n]/I");
  treeTriggerCells_->Branch("c_waferu", triggerCellCell_waferU_.get(), "c_waferu[c_n]/I");
  treeTriggerCells_->Branch("c_waferv", triggerCellCell_waferV_.get(), "c_waferv[c_n]/I");
  treeTriggerCells_->Branch("c_cellu", triggerCellCell_cellU_.get(), "c_cellu[c_n]/I");
  treeTriggerCells_->Branch("c_cellv", triggerCellCell_cellV_.get(), "c_cellv[c_n]/I");
  treeTriggerCells_->Branch("c_ieta", triggerCellCell_ieta_.get(), "c_cell[c_n]/I");
  treeTriggerCells_->Branch("c_iphi", triggerCellCell_iphi_.get(), "c_cell[c_n]/I");
  treeTriggerCells_->Branch("c_x", triggerCellCell_x_.get(), "c_x[c_n]/F");
  treeTriggerCells_->Branch("c_y", triggerCellCell_y_.get(), "c_y[c_n]/F");
  treeTriggerCells_->Branch("c_z", triggerCellCell_z_.get(), "c_z[c_n]/F");
  treeTriggerCells_->Branch("c_eta", triggerCellCell_eta_.get(), "c_eta[c_n]/F");
  treeTriggerCells_->Branch("c_phi", triggerCellCell_phi_.get(), "c_phi[c_n]/F");
  //
  treeCells_ = fs->make<TTree>("TreeCells", "Tree of all HGC cells");
  treeCells_->Branch("id", &cellId_, "id/I");
  treeCells_->Branch("zside", &cellSide_, "zside/I");
  treeCells_->Branch("subdet", &cellSubdet_, "subdet/I");
  treeCells_->Branch("layer", &cellLayer_, "layer/I");
  treeCells_->Branch("waferu", &cellWaferU_, "waferu/I");
  treeCells_->Branch("waferv", &cellWaferV_, "waferv/I");
  treeCells_->Branch("wafertype", &cellWaferType_, "wafertype/I");
  treeCells_->Branch("waferrow", &cellWaferRow_, "waferrow/I");
  treeCells_->Branch("wafercolumn", &cellWaferColumn_, "wafercolumn/I");
  treeCells_->Branch("cellu", &cellU_, "cellu/I");
  treeCells_->Branch("cellv", &cellV_, "cellv/I");
  treeCells_->Branch("x", &cellX_, "x/F");
  treeCells_->Branch("y", &cellY_, "y/F");
  treeCells_->Branch("z", &cellZ_, "z/F");
  treeCells_->Branch("eta", &cellEta_, "eta/F");
  treeCells_->Branch("phi", &cellPhi_, "phi/F");
  treeCells_->Branch("corner_n", &cellCornersN_, "corner_n/I");
  treeCells_->Branch("corner_x", cellCornersX_.get(), "corner_x[corner_n]/F");
  treeCells_->Branch("corner_y", cellCornersY_.get(), "corner_y[corner_n]/F");
  treeCells_->Branch("corner_z", cellCornersZ_.get(), "corner_z[corner_n]/F");
  //
  treeCellsBH_ = fs->make<TTree>("TreeCellsBH", "Tree of all BH cells");
  treeCellsBH_->Branch("id", &cellBHId_, "id/I");
  treeCellsBH_->Branch("type", &cellBHType_, "type/I");
  treeCellsBH_->Branch("zside", &cellBHSide_, "zside/I");
  treeCellsBH_->Branch("subdet", &cellBHSubdet_, "subdet/I");
  treeCellsBH_->Branch("layer", &cellBHLayer_, "layer/I");
  treeCellsBH_->Branch("ieta", &cellBHIEta_, "ieta/I");
  treeCellsBH_->Branch("iphi", &cellBHIPhi_, "iphi/I");
  treeCellsBH_->Branch("eta", &cellBHEta_, "eta/F");
  treeCellsBH_->Branch("phi", &cellBHPhi_, "phi/F");
  treeCellsBH_->Branch("x", &cellBHX_, "x/F");
  treeCellsBH_->Branch("y", &cellBHY_, "y/F");
  treeCellsBH_->Branch("z", &cellBHZ_, "z/F");
  treeCellsBH_->Branch("x1", &cellBHX1_, "x1/F");
  treeCellsBH_->Branch("y1", &cellBHY1_, "y1/F");
  treeCellsBH_->Branch("x2", &cellBHX2_, "x2/F");
  treeCellsBH_->Branch("y2", &cellBHY2_, "y2/F");
  treeCellsBH_->Branch("x3", &cellBHX3_, "x3/F");
  treeCellsBH_->Branch("y3", &cellBHY3_, "y3/F");
  treeCellsBH_->Branch("x4", &cellBHX4_, "x4/F");
  treeCellsBH_->Branch("y4", &cellBHY4_, "y4/F");
  //
  treeCellsNose_ = fs->make<TTree>("TreeCellsNose", "Tree of all HGCnose cells");
  treeCellsNose_->Branch("id", &cellId_, "id/I");
  treeCellsNose_->Branch("zside", &cellSide_, "zside/I");
  treeCellsNose_->Branch("subdet", &cellSubdet_, "subdet/I");
  treeCellsNose_->Branch("layer", &cellLayer_, "layer/I");
  treeCellsNose_->Branch("waferu", &cellWaferU_, "waferu/I");
  treeCellsNose_->Branch("waferv", &cellWaferV_, "waferv/I");
  treeCellsNose_->Branch("wafertype", &cellWaferType_, "wafertype/I");
  treeCellsNose_->Branch("waferrow", &cellWaferRow_, "waferrow/I");
  treeCellsNose_->Branch("wafercolumn", &cellWaferColumn_, "wafercolumn/I");
  treeCellsNose_->Branch("cellu", &cellU_, "cellu/I");
  treeCellsNose_->Branch("cellv", &cellV_, "cellv/I");
  treeCellsNose_->Branch("x", &cellX_, "x/F");
  treeCellsNose_->Branch("y", &cellY_, "y/F");
  treeCellsNose_->Branch("z", &cellZ_, "z/F");
  treeCellsNose_->Branch("eta", &cellEta_, "eta/F");
  treeCellsNose_->Branch("phi", &cellPhi_, "phi/F");
  treeCellsNose_->Branch("corner_n", &cellCornersN_, "corner_n/I");
  treeCellsNose_->Branch("corner_x", cellCornersX_.get(), "corner_x[corner_n]/F");
  treeCellsNose_->Branch("corner_y", cellCornersY_.get(), "corner_y[corner_n]/F");
  treeCellsNose_->Branch("corner_z", cellCornersZ_.get(), "corner_z[corner_n]/F");
  //
  treeModuleErrors_ = fs->make<TTree>("TreeModuleErrors", "Tree of module mapping errors");
  treeModuleErrors_->Branch("subdet", &moduleErrorSubdet_, "subdet/I");
  treeModuleErrors_->Branch("layer", &moduleErrorLayer_, "layer/I");
  treeModuleErrors_->Branch("waferu", &moduleErrorWaferU_, "waferu/I");
  treeModuleErrors_->Branch("waferv", &moduleErrorWaferV_, "waferv/I");
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es)
/*****************************************************************/
{
  triggerGeometry_ = es.getHandle(triggerGeomToken_);

  no_trigger_ = !checkMappingConsistency();

  fillTriggerGeometry();
}

bool HGCalTriggerGeomTesterV9Imp3::checkMappingConsistency() {
  try {
    // Set of (subdet,layer,waferU,waferV) with module mapping errors
    std::set<std::tuple<unsigned, unsigned, int, int>> module_errors;
    trigger_map_set modules_to_triggercells;
    trigger_map_set modules_to_cells;
    trigger_map_set triggercells_to_cells;
    // EE
    for (const auto& id : triggerGeometry_->eeGeometry()->getValidDetIds()) {
      HGCSiliconDetId detid(id);
      if (!triggerGeometry_->eeTopology().valid(id))
        continue;
      // fill trigger cells
      uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
      auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
      // fill modules
      uint32_t module = 0;
      try {
        module = triggerGeometry_->getModuleFromCell(id);
        triggerGeometry_->getLinksInModule(module);
      } catch (const std::exception& e) {
        module_errors.emplace(std::make_tuple(HGCalTriggerModuleDetId(module).triggerSubdetId(),
                                              HGCalTriggerModuleDetId(module).layer(),
                                              HGCalTriggerModuleDetId(module).moduleU(),
                                              HGCalTriggerModuleDetId(module).moduleV()));
        continue;
      }
      itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
    }

    // HSi
    for (const auto& id : triggerGeometry_->hsiGeometry()->getValidDetIds()) {
      if (!triggerGeometry_->hsiTopology().valid(id))
        continue;
      // fill trigger cells
      uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
      auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
      // fill modules
      uint32_t module = 0;
      try {
        module = triggerGeometry_->getModuleFromCell(id);
        triggerGeometry_->getLinksInModule(module);
      } catch (const std::exception& e) {
        module_errors.emplace(std::make_tuple(HGCalTriggerModuleDetId(module).triggerSubdetId(),
                                              HGCalTriggerModuleDetId(module).layer(),
                                              HGCalTriggerModuleDetId(module).moduleU(),
                                              HGCalTriggerModuleDetId(module).moduleV()));
        continue;
      }
      itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
    }

    // HSc
    for (const auto& id : triggerGeometry_->hscGeometry()->getValidDetIds()) {
      // fill trigger cells
      unsigned layer = HGCScintillatorDetId(id).layer();
      if (HGCScintillatorDetId(id).type() != triggerGeometry_->hscTopology().dddConstants().getTypeTrap(layer)) {
        std::cout << "Sci cell type = " << HGCScintillatorDetId(id).type()
                  << " != " << triggerGeometry_->hscTopology().dddConstants().getTypeTrap(layer) << "\n";
      }
      uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
      auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
      // fill modules
      uint32_t module = triggerGeometry_->getModuleFromCell(id);
      if (module != 0) {
        itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
      }
    }

    // NOSE
    if (triggerGeometry_->isWithNoseGeometry()) {
      for (const auto& id : triggerGeometry_->noseGeometry()->getValidDetIds()) {
        if (!triggerGeometry_->noseTopology().valid(id))
          continue;
        // fill trigger cells
        uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
        auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
        // fill modules
        uint32_t module = triggerGeometry_->getModuleFromCell(id);
        if (module != 0) {
          itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
          itr_insert.first->second.emplace(id);
        }
      }
    }

    if (module_errors.size() > 0) {
      for (const auto& module : module_errors) {
        moduleErrorSubdet_ = std::get<0>(module);
        moduleErrorLayer_ = std::get<1>(module);
        moduleErrorWaferU_ = std::get<2>(module);
        moduleErrorWaferV_ = std::get<3>(module);
        treeModuleErrors_->Fill();
      }
      throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: Found  module mapping problems. Check the produced "
                                             "tree to see the list of problematic wafers";
    }

    edm::LogPrint("TriggerCellCheck") << "Checking cell -> trigger cell -> cell consistency";
    // Loop over trigger cells
    for (const auto& triggercell_cells : triggercells_to_cells) {
      DetId id(triggercell_cells.first);

      // fill modules
      uint32_t module = triggerGeometry_->getModuleFromTriggerCell(id);
      if (module != 0) {
        auto itr_insert = modules_to_triggercells.emplace(module, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
      }

      // Check consistency of cells included in trigger cell
      HGCalTriggerGeometryBase::geom_set cells_geom = triggerGeometry_->getCellsFromTriggerCell(id);
      const auto& cells = triggercell_cells.second;
      for (auto cell : cells) {
        if (cells_geom.find(cell) == cells_geom.end()) {
          if (id.det() == DetId::HGCalHSc) {
            edm::LogProblem("BadTriggerCell")
                << "Error: \n Cell " << cell << "(" << HGCScintillatorDetId(cell)
                << ")\n has not been found in \n trigger cell " << HGCScintillatorDetId(id);
            std::stringstream output;
            output << " Available cells are:\n";
            for (auto cell_geom : cells_geom)
              output << "     " << HGCScintillatorDetId(cell_geom) << "\n";
            edm::LogProblem("BadTriggerCell") << output.str();
          } else if (HFNoseTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
            edm::LogProblem("BadTriggerCell")
                << "Error: \n Cell " << cell << "(" << HFNoseDetId(cell) << ")\n has not been found in \n trigger cell "
                << HFNoseTriggerDetId(triggercell_cells.first);
            std::stringstream output;
            output << " Available cells are:\n";
            for (auto cell_geom : cells_geom)
              output << "     " << HFNoseDetId(cell_geom) << "\n";
            edm::LogProblem("BadTriggerCell") << output.str();
          } else if (HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalEETrigger ||
                     HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalHSiTrigger) {
            edm::LogProblem("BadTriggerCell")
                << "Error: \n Cell " << cell << "(" << HGCSiliconDetId(cell)
                << ")\n has not been found in \n trigger cell " << HGCalTriggerDetId(triggercell_cells.first);
            std::stringstream output;
            output << " Available cells are:\n";
            for (auto cell_geom : cells_geom)
              output << "     " << HGCSiliconDetId(cell_geom) << "\n";
            edm::LogProblem("BadTriggerCell") << output.str();
          } else {
            edm::LogProblem("BadTriggerCell")
                << "Unknown detector type " << id.det() << " " << id.subdetId() << " " << id.rawId() << "\n";
            edm::LogProblem("BadTriggerCell") << " cell " << std::hex << cell << std::dec << " "
                                              << "\n";
            edm::LogProblem("BadTriggerCell")
                << "Cell ID " << HGCSiliconDetId(cell) << " or " << HFNoseDetId(cell) << "\n";
          }
          throw cms::Exception("BadGeometry")
              << "HGCalTriggerGeometry: Found inconsistency in cell <-> trigger cell mapping";
        }
      }
    }
    edm::LogPrint("ModuleCheck") << "Checking trigger cell -> module -> trigger cell consistency";
    // Loop over modules
    for (const auto& module_triggercells : modules_to_triggercells) {
      HGCalTriggerModuleDetId id(module_triggercells.first);
      // Check consistency of trigger cells included in module
      HGCalTriggerGeometryBase::geom_set triggercells_geom = triggerGeometry_->getTriggerCellsFromModule(id);
      const auto& triggercells = module_triggercells.second;
      for (auto cell : triggercells) {
        if (triggercells_geom.find(cell) == triggercells_geom.end()) {
          if (id.triggerSubdetId() == HGCalTriggerSubdetector::HGCalHScTrigger) {
            HGCScintillatorDetId cellid(cell);
            edm::LogProblem("BadModule") << "Error: \n Trigger cell " << cell << "(" << cellid
                                         << ")\n has not been found in \n module " << HGCalTriggerModuleDetId(id);
            std::stringstream output;
            output << " Available trigger cells are:\n";
            for (auto cell_geom : triggercells_geom) {
              output << "     " << HGCScintillatorDetId(cell_geom) << "\n";
            }
            edm::LogProblem("BadModule") << output.str();
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: Found inconsistency in trigger cell <->  module mapping";
          } else if (id.triggerSubdetId() == HGCalTriggerSubdetector::HFNoseTrigger) {
            HFNoseTriggerDetId cellid(cell);
            edm::LogProblem("BadModule") << "Error : \n Trigger cell " << cell << "(" << cellid
                                         << ")\n has not been found in \n module " << HGCalTriggerModuleDetId(id);
            std::stringstream output;
            output << " Available trigger cells are:\n";
            for (auto cell_geom : triggercells_geom) {
              output << "     " << HFNoseTriggerDetId(cell_geom) << "\n";
            }
            edm::LogProblem("BadModule") << output.str();
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: Found inconsistency in trigger cell <->  module mapping";
          } else {
            HGCalTriggerDetId cellid(cell);
            edm::LogProblem("BadModule") << "Error : \n Trigger cell " << cell << "(" << cellid
                                         << ")\n has not been found in \n module " << HGCalTriggerModuleDetId(id);
            std::stringstream output;
            output << " Available trigger cells are:\n";
            for (auto cell_geom : triggercells_geom) {
              output << "     " << HGCalTriggerDetId(cell_geom) << "\n";
            }
            edm::LogProblem("BadModule") << output.str();
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: Found inconsistency in trigger cell <->  module mapping";
          }
        }
      }
    }
    edm::LogPrint("ModuleCheck") << "Checking cell -> module -> cell consistency";
    for (const auto& module_cells : modules_to_cells) {
      HGCalTriggerModuleDetId id(module_cells.first);
      // Check consistency of cells included in module
      HGCalTriggerGeometryBase::geom_set cells_geom = triggerGeometry_->getCellsFromModule(id);
      const auto& cells = module_cells.second;
      for (auto cell : cells) {
        if (cells_geom.find(cell) == cells_geom.end()) {
          if (id.triggerSubdetId() == HGCalTriggerSubdetector::HGCalHScTrigger) {
            edm::LogProblem("BadModule") << "Error: \n Cell " << cell << "(" << HGCScintillatorDetId(cell)
                                         << ")\n has not been found in \n module " << HGCalTriggerModuleDetId(id);
          } else if (id.triggerSubdetId() == HGCalTriggerSubdetector::HFNoseTrigger) {
            edm::LogProblem("BadModule") << "Error: \n Cell " << cell << "(" << HFNoseDetId(cell)
                                         << ")\n has not been found in \n module " << HGCalTriggerModuleDetId(id);
          } else {
            edm::LogProblem("BadModule") << "Error: \n Cell " << cell << "(" << HGCSiliconDetId(cell)
                                         << ")\n has not been found in \n module " << HGCalTriggerModuleDetId(id);
          }
          std::stringstream output;
          output << " Available cells are:\n";
          for (auto cell_geom : cells_geom) {
            output << cell_geom << " ";
          }
          edm::LogProblem("BadModule") << output.str();
          throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: Found inconsistency in cell <-> module mapping";
        }
      }
    }

    // Filling Stage 1 FPGA -> modules

    edm::LogPrint("ModuleCheck") << "Checking module -> stage-1 -> module consistency";
    trigger_map_set stage1_to_modules;
    for (const auto& module_tc : modules_to_triggercells) {
      HGCalTriggerModuleDetId id(module_tc.first);
      HGCalTriggerGeometryBase::geom_set lpgbts = triggerGeometry_->getLpgbtsFromModule(id);
      if (lpgbts.size() == 0)
        continue;  //Module is not connected to an lpGBT and therefore not to a Stage 1 FPGA
      uint32_t stage1 = 0;
      for (const auto& lpgbt : lpgbts) {
        uint32_t stage1_tmp = triggerGeometry_->getStage1FpgaFromLpgbt(lpgbt);
        if (stage1 != 0 && stage1_tmp != stage1) {
          throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: Module " << HGCalTriggerModuleDetId(id)
                                              << " is split is split into more than one Stage-1 FPGA";
        }
        stage1 = stage1_tmp;
      }
      auto itr_insert = stage1_to_modules.emplace(stage1, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
    }
    // checking S1 -> module consistency

    for (const auto& stage1_modules : stage1_to_modules) {
      HGCalTriggerBackendDetId stage1(stage1_modules.first);
      HGCalTriggerGeometryBase::geom_set modules_geom;
      // Check consistency of modules going to Stage-1 FPGA
      std::vector<unsigned> lpgbts = triggerGeometry_->getLpgbtsFromStage1Fpga(stage1);
      for (const auto& lpgbt : lpgbts) {
        HGCalTriggerGeometryBase::geom_set modules = triggerGeometry_->getModulesFromLpgbt(lpgbt);
        modules_geom.insert(modules.begin(), modules.end());
      }
      const auto& modules = stage1_modules.second;
      for (auto module : modules) {
        if (modules_geom.find(module) == modules_geom.end()) {
          edm::LogProblem("BadStage1") << "Error: \n Module " << module << "(" << HGCalTriggerModuleDetId(module)
                                       << ")\n has not been found in \n stage-1 " << HGCalTriggerBackendDetId(stage1);
          std::stringstream output;
          output << "   Available modules are:\n";
          for (auto module_geom : modules_geom) {
            output << module_geom << " ";
          }
          output << "   Connected lpgbts are:\n";
          for (auto lpgbt : lpgbts) {
            output << lpgbt << " ";
          }
          edm::LogProblem("BadStage1") << output.str();
          throw cms::Exception("BadGeometry")
              << "HGCalTriggerGeometry: Found inconsistency in Stage1 <-> module mapping";
        }
      }
    }

    // Filling Stage 2 FPGA -> Stage 1 FPGA

    edm::LogPrint("ModuleCheck") << "Checking Stage 1 -> Stage 2 -> Stage 1 consistency";
    trigger_map_set stage2_to_stage1;
    for (const auto& stage1 : stage1_to_modules) {
      HGCalTriggerBackendDetId id(stage1.first);
      HGCalTriggerGeometryBase::geom_set stage2FPGAs = triggerGeometry_->getStage2FpgasFromStage1Fpga(id);
      for (const auto& stage2 : stage2FPGAs) {
        auto itr_insert = stage2_to_stage1.emplace(stage2, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
      }
    }
    // checking S1 -> S2 consistency

    for (const auto& stage2_modules : stage2_to_stage1) {
      HGCalTriggerBackendDetId stage2(stage2_modules.first);

      // Check consistency of Stage-1 FPGA going to Stage 2 FPGA
      HGCalTriggerGeometryBase::geom_set stage1FPGAs = triggerGeometry_->getStage1FpgasFromStage2Fpga(stage2);

      const auto& stage1fpgas = stage2_modules.second;

      for (auto stage1fpga : stage1fpgas) {
        if (stage1FPGAs.find(stage1fpga) == stage1FPGAs.end()) {
          edm::LogProblem("BadStage2") << "Error: \n Stage-1 FPGA " << stage1fpga << "("
                                       << HGCalTriggerBackendDetId(stage1fpga)
                                       << ")\n has not been found in \n Stage-2 " << HGCalTriggerBackendDetId(stage2);
          std::stringstream output;
          output << "\n   Available Stage-1 FPGAs are:\n";
          for (auto stage1FPGA : stage1FPGAs) {
            output << HGCalTriggerBackendDetId(stage1FPGA) << "\n";
          }
          edm::LogProblem("BadStage2") << output.str();
          throw cms::Exception("BadGeometry")
              << "HGCalTriggerGeometry: Found inconsistency in Stage2 <-> Stage1 mapping";
        }
      }
    }

  } catch (const cms::Exception& e) {
    edm::LogWarning("HGCalTriggerGeometryTester")
        << "Problem with the trigger geometry detected. Only the basic cells tree will be filled\n";
    edm::LogWarning("HGCalTriggerGeometryTester") << e.message() << "\n";
    return false;
  }
  return true;
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::fillTriggerGeometry()
/*****************************************************************/
{
  trigger_map_set modules;
  trigger_map_set trigger_cells;

  // Loop over cells
  edm::LogPrint("TreeFilling") << "Filling cells tree";
  // EE
  std::cout << "Filling EE geometry\n";
  for (const auto& id : triggerGeometry_->eeGeometry()->getValidDetIds()) {
    HGCSiliconDetId detid(id);
    if (!triggerGeometry_->eeTopology().valid(id))
      continue;
    cellId_ = detid.rawId();
    cellSide_ = detid.zside();
    cellSubdet_ = detid.subdet();
    cellLayer_ = detid.layer();
    cellWaferU_ = detid.waferU();
    cellWaferV_ = detid.waferV();
    cellU_ = detid.cellU();
    cellV_ = detid.cellV();
    int type1 = detid.type();
    int type2 = triggerGeometry_->eeTopology().dddConstants().getTypeHex(cellLayer_, cellWaferU_, cellWaferV_);
    if (type1 != type2) {
      std::cout << "Found incompatible wafer types:\n  " << detid << "\n";
    }
    //
    GlobalPoint center = triggerGeometry_->eeGeometry()->getPosition(id);
    cellX_ = center.x();
    cellY_ = center.y();
    cellZ_ = center.z();
    cellEta_ = center.eta();
    cellPhi_ = center.phi();
    std::vector<GlobalPoint> corners = triggerGeometry_->eeGeometry()->getCorners(id);
    cellCornersN_ = corners.size();
    setTreeCellCornerSize(cellCornersN_);
    for (unsigned i = 0; i < corners.size(); i++) {
      cellCornersX_.get()[i] = corners[i].x();
      cellCornersY_.get()[i] = corners[i].y();
      cellCornersZ_.get()[i] = corners[i].z();
    }
    treeCells_->Fill();
    // fill trigger cells
    if (!no_trigger_) {
      uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
      // Skip trigger cells in module 0
      // uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
      //if (HGCalDetId(module).wafer() == 0)
      //  continue;
      auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
    }
  }
  std::cout << "Filling HSi geometry\n";
  for (const auto& id : triggerGeometry_->hsiGeometry()->getValidDetIds()) {
    HGCSiliconDetId detid(id);
    if (!triggerGeometry_->hsiTopology().valid(id))
      continue;
    cellId_ = detid.rawId();
    cellSide_ = detid.zside();
    cellSubdet_ = detid.subdet();
    cellLayer_ = detid.layer();
    cellWaferU_ = detid.waferU();
    cellWaferV_ = detid.waferV();
    cellU_ = detid.cellU();
    cellV_ = detid.cellV();
    int type1 = detid.type();
    int type2 = triggerGeometry_->hsiTopology().dddConstants().getTypeHex(cellLayer_, cellWaferU_, cellWaferV_);
    if (type1 != type2) {
      std::cout << "Found incompatible wafer types:\n  " << detid << "\n";
    }
    //
    GlobalPoint center = triggerGeometry_->hsiGeometry()->getPosition(id);
    cellX_ = center.x();
    cellY_ = center.y();
    cellZ_ = center.z();
    cellEta_ = center.eta();
    cellPhi_ = center.phi();
    std::vector<GlobalPoint> corners = triggerGeometry_->hsiGeometry()->getCorners(id);
    cellCornersN_ = corners.size();
    setTreeCellCornerSize(cellCornersN_);
    for (unsigned i = 0; i < corners.size(); i++) {
      cellCornersX_.get()[i] = corners[i].x();
      cellCornersY_.get()[i] = corners[i].y();
      cellCornersZ_.get()[i] = corners[i].z();
    }
    treeCells_->Fill();
    // fill trigger cells
    if (!no_trigger_) {
      uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
      // Skip trigger cells in module 0
      // uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
      //if (HGCalDetId(module).wafer() == 0)
      //  continue;
      auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
    }
  }
  std::cout << "Filling HSc geometry\n";
  for (const auto& id : triggerGeometry_->hscGeometry()->getValidDetIds()) {
    HGCScintillatorDetId cellid(id);
    cellBHId_ = cellid.rawId();
    cellBHType_ = cellid.type();
    cellBHSide_ = cellid.zside();
    cellBHSubdet_ = cellid.subdet();
    cellBHLayer_ = cellid.layer();
    cellBHIEta_ = cellid.ieta();
    cellBHIPhi_ = cellid.iphi();
    //
    GlobalPoint center = triggerGeometry_->hscGeometry()->getPosition(id);
    cellBHEta_ = center.eta();
    cellBHPhi_ = center.phi();
    cellBHX_ = center.x();
    cellBHY_ = center.y();
    cellBHZ_ = center.z();
    auto corners = triggerGeometry_->hscGeometry()->getCorners(id);
    if (corners.size() >= 4) {
      cellBHX1_ = corners[0].x();
      cellBHY1_ = corners[0].y();
      cellBHX2_ = corners[1].x();
      cellBHY2_ = corners[1].y();
      cellBHX3_ = corners[2].x();
      cellBHY3_ = corners[2].y();
      cellBHX4_ = corners[3].x();
      cellBHY4_ = corners[3].y();
    }
    treeCellsBH_->Fill();
    // fill trigger cells
    if (!no_trigger_) {
      uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
      auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
    }
  }

  if (triggerGeometry_->isWithNoseGeometry()) {
    // NOSE
    std::cout << "Filling NOSE geometry\n";
    for (const auto& id : triggerGeometry_->noseGeometry()->getValidDetIds()) {
      HFNoseDetId detid(id);
      cellId_ = detid.rawId();
      cellSide_ = detid.zside();
      cellSubdet_ = detid.subdet();
      cellLayer_ = detid.layer();
      cellWaferU_ = detid.waferU();
      cellWaferV_ = detid.waferV();
      cellU_ = detid.cellU();
      cellV_ = detid.cellV();
      int type1 = detid.type();
      int type2 = triggerGeometry_->noseTopology().dddConstants().getTypeHex(cellLayer_, cellWaferU_, cellWaferV_);
      if (type1 != type2) {
        std::cout << "Found incompatible wafer types:\n  " << detid << "\n";
      }
      GlobalPoint center = triggerGeometry_->noseGeometry()->getPosition(id);
      cellX_ = center.x();
      cellY_ = center.y();
      cellZ_ = center.z();
      cellEta_ = center.eta();
      cellPhi_ = center.phi();
      std::vector<GlobalPoint> corners = triggerGeometry_->noseGeometry()->getCorners(id);
      cellCornersN_ = corners.size();
      setTreeCellCornerSize(cellCornersN_);
      for (unsigned i = 0; i < corners.size(); i++) {
        cellCornersX_.get()[i] = corners[i].x();
        cellCornersY_.get()[i] = corners[i].y();
        cellCornersZ_.get()[i] = corners[i].z();
      }
      treeCellsNose_->Fill();
      // fill trigger cells
      if (!no_trigger_) {
        uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
        // Skip trigger cells in module 0
        // uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
        //if (HGCalDetId(module).wafer() == 0)
        //  continue;
        auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
      }
    }
  }

  // if problem detected in the trigger geometry, don't produce trigger trees
  if (no_trigger_)
    return;

  // Loop over trigger cells
  edm::LogPrint("TreeFilling") << "Filling trigger cells tree";
  for (const auto& triggercell_cells : trigger_cells) {
    DetId id(triggercell_cells.first);
    GlobalPoint position = triggerGeometry_->getTriggerCellPosition(triggercell_cells.first);
    triggerCellId_ = id.rawId();
    if (id.det() == DetId::HGCalHSc) {
      HGCScintillatorDetId id_sc(triggercell_cells.first);
      triggerCellSide_ = id_sc.zside();
      triggerCellSubdet_ = id_sc.subdet();
      triggerCellLayer_ = id_sc.layer();
      triggerCellIEta_ = id_sc.ietaAbs();
      triggerCellIPhi_ = id_sc.iphi();
      triggerCellWaferU_ = 0;
      triggerCellWaferV_ = 0;
      triggerCellU_ = 0;
      triggerCellV_ = 0;
    } else if (HFNoseTriggerDetId(triggercell_cells.first).det() == DetId::HGCalTrigger &&
               HFNoseTriggerDetId(triggercell_cells.first).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
      HFNoseTriggerDetId id_nose_trig(triggercell_cells.first);
      triggerCellSide_ = id_nose_trig.zside();
      triggerCellSubdet_ = id_nose_trig.subdet();
      triggerCellLayer_ = id_nose_trig.layer();
      triggerCellIEta_ = 0;
      triggerCellIPhi_ = 0;
      triggerCellWaferU_ = id_nose_trig.waferU();
      triggerCellWaferV_ = id_nose_trig.waferV();
      triggerCellU_ = id_nose_trig.triggerCellU();
      triggerCellV_ = id_nose_trig.triggerCellV();
    } else {
      HGCalTriggerDetId id_si_trig(triggercell_cells.first);
      triggerCellSide_ = id_si_trig.zside();
      triggerCellSubdet_ = id_si_trig.subdet();
      triggerCellLayer_ = id_si_trig.layer();
      triggerCellIEta_ = 0;
      triggerCellIPhi_ = 0;
      triggerCellWaferU_ = id_si_trig.waferU();
      triggerCellWaferV_ = id_si_trig.waferV();
      triggerCellU_ = id_si_trig.triggerCellU();
      triggerCellV_ = id_si_trig.triggerCellV();
    }
    triggerCellX_ = position.x();
    triggerCellY_ = position.y();
    triggerCellZ_ = position.z();
    triggerCellEta_ = position.eta();
    triggerCellPhi_ = position.phi();
    triggerCellCell_N_ = triggercell_cells.second.size();
    //
    setTreeTriggerCellSize(triggerCellCell_N_);
    size_t ic = 0;
    for (const auto& c : triggercell_cells.second) {
      if (id.det() == DetId::HGCalHSc) {
        HGCScintillatorDetId cId(c);
        GlobalPoint cell_position = triggerGeometry_->hscGeometry()->getPosition(cId);
        triggerCellCell_id_.get()[ic] = c;
        triggerCellCell_zside_.get()[ic] = cId.zside();
        triggerCellCell_subdet_.get()[ic] = cId.subdetId();
        triggerCellCell_layer_.get()[ic] = cId.layer();
        triggerCellCell_waferU_.get()[ic] = 0;
        triggerCellCell_waferV_.get()[ic] = 0;
        triggerCellCell_cellU_.get()[ic] = 0;
        triggerCellCell_cellV_.get()[ic] = 0;
        triggerCellCell_ieta_.get()[ic] = cId.ietaAbs();
        triggerCellCell_iphi_.get()[ic] = cId.iphi();
        triggerCellCell_x_.get()[ic] = cell_position.x();
        triggerCellCell_y_.get()[ic] = cell_position.y();
        triggerCellCell_z_.get()[ic] = cell_position.z();
        triggerCellCell_eta_.get()[ic] = cell_position.eta();
        triggerCellCell_phi_.get()[ic] = cell_position.phi();
      } else if (HFNoseTriggerDetId(triggercell_cells.first).det() == DetId::HGCalTrigger &&
                 HFNoseTriggerDetId(triggercell_cells.first).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
        HFNoseDetId cId(c);
        GlobalPoint cell_position = triggerGeometry_->noseGeometry()->getPosition(cId);
        triggerCellCell_id_.get()[ic] = c;
        triggerCellCell_zside_.get()[ic] = cId.zside();
        triggerCellCell_subdet_.get()[ic] = cId.subdet();
        triggerCellCell_layer_.get()[ic] = cId.layer();
        triggerCellCell_waferU_.get()[ic] = cId.waferU();
        triggerCellCell_waferV_.get()[ic] = cId.waferV();
        triggerCellCell_cellU_.get()[ic] = cId.cellU();
        triggerCellCell_cellV_.get()[ic] = cId.cellV();
        triggerCellCell_ieta_.get()[ic] = 0;
        triggerCellCell_iphi_.get()[ic] = 0;
        triggerCellCell_x_.get()[ic] = cell_position.x();
        triggerCellCell_y_.get()[ic] = cell_position.y();
        triggerCellCell_z_.get()[ic] = cell_position.z();
        triggerCellCell_eta_.get()[ic] = cell_position.eta();
        triggerCellCell_phi_.get()[ic] = cell_position.phi();
      } else {
        HGCSiliconDetId cId(c);
        GlobalPoint cell_position = (cId.det() == DetId::HGCalEE ? triggerGeometry_->eeGeometry()->getPosition(cId)
                                                                 : triggerGeometry_->hsiGeometry()->getPosition(cId));
        triggerCellCell_id_.get()[ic] = c;
        triggerCellCell_zside_.get()[ic] = cId.zside();
        triggerCellCell_subdet_.get()[ic] = cId.subdet();
        triggerCellCell_layer_.get()[ic] = cId.layer();
        triggerCellCell_waferU_.get()[ic] = cId.waferU();
        triggerCellCell_waferV_.get()[ic] = cId.waferV();
        triggerCellCell_cellU_.get()[ic] = cId.cellU();
        triggerCellCell_cellV_.get()[ic] = cId.cellV();
        triggerCellCell_ieta_.get()[ic] = 0;
        triggerCellCell_iphi_.get()[ic] = 0;
        triggerCellCell_x_.get()[ic] = cell_position.x();
        triggerCellCell_y_.get()[ic] = cell_position.y();
        triggerCellCell_z_.get()[ic] = cell_position.z();
        triggerCellCell_eta_.get()[ic] = cell_position.eta();
        triggerCellCell_phi_.get()[ic] = cell_position.phi();
      }
      ic++;
    }

    treeTriggerCells_->Fill();

    // fill modules
    uint32_t module = triggerGeometry_->getModuleFromTriggerCell(id);
    auto itr_insert = modules.emplace(module, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(id);
  }

  // Loop over modules
  edm::LogPrint("TreeFilling") << "Filling modules tree";

  for (const auto& module_triggercells : modules) {
    HGCalTriggerModuleDetId id(module_triggercells.first);
    GlobalPoint position = triggerGeometry_->getModulePosition(id);
    moduleId_ = id.rawId();
    moduleX_ = position.x();
    moduleY_ = position.y();
    moduleZ_ = position.z();
    moduleEta_ = position.eta();
    modulePhi_ = position.phi();

    moduleSide_ = id.zside();
    moduleSubdet_ = id.triggerSubdetId();
    moduleLayer_ = id.layer();
    module_ = 0;
    if (moduleSubdet_ == HGCalTriggerSubdetector::HGCalHScTrigger) {
      moduleIEta_ = id.eta();
      moduleIPhi_ = id.phi();
    } else {
      moduleIEta_ = 0;
      moduleIPhi_ = 0;
    }
    moduleTC_N_ = module_triggercells.second.size();
    if (triggerGeometry_->disconnectedModule(id)) {
      moduleLinks_ = 0;
    } else {
      moduleLinks_ = triggerGeometry_->getLinksInModule(id);
    }
    //
    setTreeModuleSize(moduleTC_N_);
    size_t itc = 0;
    for (const auto& tc : module_triggercells.second) {
      moduleTC_id_.get()[itc] = tc;
      if (moduleSubdet_ == HGCalTriggerSubdetector::HGCalHScTrigger) {
        HGCScintillatorDetId tcId(tc);
        moduleTC_zside_.get()[itc] = tcId.zside();
        moduleTC_subdet_.get()[itc] = tcId.subdet();
        moduleTC_layer_.get()[itc] = tcId.layer();
        moduleTC_waferU_.get()[itc] = 0;
        moduleTC_waferV_.get()[itc] = 0;
        moduleTC_cellU_.get()[itc] = 0;
        moduleTC_cellV_.get()[itc] = 0;
        moduleTC_ieta_.get()[itc] = tcId.ietaAbs();
        moduleTC_iphi_.get()[itc] = tcId.iphi();
      } else if (moduleSubdet_ == HGCalTriggerSubdetector::HFNoseTrigger) {
        HFNoseTriggerDetId tcId(tc);
        moduleTC_zside_.get()[itc] = tcId.zside();
        moduleTC_subdet_.get()[itc] = tcId.subdet();
        moduleTC_layer_.get()[itc] = tcId.layer();
        moduleTC_waferU_.get()[itc] = tcId.waferU();
        moduleTC_waferV_.get()[itc] = tcId.waferV();
        moduleTC_cellU_.get()[itc] = tcId.triggerCellU();
        moduleTC_cellV_.get()[itc] = tcId.triggerCellV();
        moduleTC_ieta_.get()[itc] = 0;
        moduleTC_iphi_.get()[itc] = 0;
      } else {
        HGCalTriggerDetId tcId(tc);
        moduleTC_zside_.get()[itc] = tcId.zside();
        moduleTC_subdet_.get()[itc] = tcId.subdet();
        moduleTC_layer_.get()[itc] = tcId.layer();
        moduleTC_waferU_.get()[itc] = tcId.waferU();
        moduleTC_waferV_.get()[itc] = tcId.waferV();
        moduleTC_cellU_.get()[itc] = tcId.triggerCellU();
        moduleTC_cellV_.get()[itc] = tcId.triggerCellV();
        moduleTC_ieta_.get()[itc] = 0;
        moduleTC_iphi_.get()[itc] = 0;
      }
      GlobalPoint position = triggerGeometry_->getTriggerCellPosition(tc);
      moduleTC_x_.get()[itc] = position.x();
      moduleTC_y_.get()[itc] = position.y();
      moduleTC_z_.get()[itc] = position.z();
      moduleTC_eta_.get()[itc] = position.eta();
      moduleTC_phi_.get()[itc] = position.phi();
      itc++;
    }
    auto cells_in_module = triggerGeometry_->getCellsFromModule(id);
    moduleCell_N_ = cells_in_module.size();
    //
    setTreeModuleCellSize(moduleCell_N_);
    size_t ic = 0;
    for (const auto& c : cells_in_module) {
      if (moduleSubdet_ == HGCalTriggerSubdetector::HGCalHScTrigger) {
        HGCScintillatorDetId cId(c);
        GlobalPoint cell_position = triggerGeometry_->hscGeometry()->getPosition(cId);
        moduleCell_id_.get()[ic] = c;
        moduleCell_zside_.get()[ic] = cId.zside();
        moduleCell_subdet_.get()[ic] = cId.subdetId();
        moduleCell_layer_.get()[ic] = cId.layer();
        moduleCell_waferU_.get()[ic] = 0;
        moduleCell_waferV_.get()[ic] = 0;
        moduleCell_cellU_.get()[ic] = 0;
        moduleCell_cellV_.get()[ic] = 0;
        moduleCell_ieta_.get()[ic] = cId.ietaAbs();
        moduleCell_iphi_.get()[ic] = cId.iphi();
        moduleCell_x_.get()[ic] = cell_position.x();
        moduleCell_y_.get()[ic] = cell_position.y();
        moduleCell_z_.get()[ic] = cell_position.z();
        moduleCell_eta_.get()[ic] = cell_position.eta();
        moduleCell_phi_.get()[ic] = cell_position.phi();
      } else if (moduleSubdet_ == HGCalTriggerSubdetector::HFNoseTrigger) {
        HFNoseDetId cId(c);
        const GlobalPoint position = triggerGeometry_->noseGeometry()->getPosition(c);
        moduleCell_id_.get()[ic] = c;
        moduleCell_zside_.get()[ic] = cId.zside();
        moduleCell_subdet_.get()[ic] = cId.subdetId();
        moduleCell_layer_.get()[ic] = cId.layer();
        moduleCell_waferU_.get()[ic] = cId.waferU();
        moduleCell_waferV_.get()[ic] = cId.waferV();
        moduleCell_cellU_.get()[ic] = cId.cellU();
        moduleCell_cellV_.get()[ic] = cId.cellV();
        moduleCell_ieta_.get()[ic] = 0;
        moduleCell_iphi_.get()[ic] = 0;
        moduleCell_x_.get()[ic] = position.x();
        moduleCell_y_.get()[ic] = position.y();
        moduleCell_z_.get()[ic] = position.z();
        moduleCell_eta_.get()[ic] = position.eta();
        moduleCell_phi_.get()[ic] = position.phi();
      } else {
        HGCSiliconDetId cId(c);
        const GlobalPoint position = (cId.det() == DetId::HGCalEE ? triggerGeometry_->eeGeometry()->getPosition(cId)
                                                                  : triggerGeometry_->hsiGeometry()->getPosition(cId));
        moduleCell_id_.get()[ic] = c;
        moduleCell_zside_.get()[ic] = cId.zside();
        moduleCell_subdet_.get()[ic] = cId.subdetId();
        moduleCell_layer_.get()[ic] = cId.layer();
        moduleCell_waferU_.get()[ic] = cId.waferU();
        moduleCell_waferV_.get()[ic] = cId.waferV();
        moduleCell_cellU_.get()[ic] = cId.cellU();
        moduleCell_cellV_.get()[ic] = cId.cellV();
        moduleCell_ieta_.get()[ic] = 0;
        moduleCell_iphi_.get()[ic] = 0;
        moduleCell_x_.get()[ic] = position.x();
        moduleCell_y_.get()[ic] = position.y();
        moduleCell_z_.get()[ic] = position.z();
        moduleCell_eta_.get()[ic] = position.eta();
        moduleCell_phi_.get()[ic] = position.phi();
      }
      ic++;
    }
    treeModules_->Fill();
  }
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::analyze(const edm::Event& e, const edm::EventSetup& es)
/*****************************************************************/
{}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::setTreeModuleSize(const size_t n)
/*****************************************************************/
{
  moduleTC_id_.reset(new int[n], array_deleter<int>());
  moduleTC_zside_.reset(new int[n], array_deleter<int>());
  moduleTC_subdet_.reset(new int[n], array_deleter<int>());
  moduleTC_layer_.reset(new int[n], array_deleter<int>());
  moduleTC_waferU_.reset(new int[n], array_deleter<int>());
  moduleTC_waferV_.reset(new int[n], array_deleter<int>());
  moduleTC_cellU_.reset(new int[n], array_deleter<int>());
  moduleTC_cellV_.reset(new int[n], array_deleter<int>());
  moduleTC_ieta_.reset(new int[n], array_deleter<int>());
  moduleTC_iphi_.reset(new int[n], array_deleter<int>());
  moduleTC_x_.reset(new float[n], array_deleter<float>());
  moduleTC_y_.reset(new float[n], array_deleter<float>());
  moduleTC_z_.reset(new float[n], array_deleter<float>());
  moduleTC_eta_.reset(new float[n], array_deleter<float>());
  moduleTC_phi_.reset(new float[n], array_deleter<float>());

  treeModules_->GetBranch("tc_id")->SetAddress(moduleTC_id_.get());
  treeModules_->GetBranch("tc_zside")->SetAddress(moduleTC_zside_.get());
  treeModules_->GetBranch("tc_subdet")->SetAddress(moduleTC_subdet_.get());
  treeModules_->GetBranch("tc_layer")->SetAddress(moduleTC_layer_.get());
  treeModules_->GetBranch("tc_waferu")->SetAddress(moduleTC_waferU_.get());
  treeModules_->GetBranch("tc_waferv")->SetAddress(moduleTC_waferV_.get());
  treeModules_->GetBranch("tc_cellu")->SetAddress(moduleTC_cellU_.get());
  treeModules_->GetBranch("tc_cellv")->SetAddress(moduleTC_cellV_.get());
  treeModules_->GetBranch("tc_ieta")->SetAddress(moduleTC_ieta_.get());
  treeModules_->GetBranch("tc_iphi")->SetAddress(moduleTC_iphi_.get());
  treeModules_->GetBranch("tc_x")->SetAddress(moduleTC_x_.get());
  treeModules_->GetBranch("tc_y")->SetAddress(moduleTC_y_.get());
  treeModules_->GetBranch("tc_z")->SetAddress(moduleTC_z_.get());
  treeModules_->GetBranch("tc_eta")->SetAddress(moduleTC_eta_.get());
  treeModules_->GetBranch("tc_phi")->SetAddress(moduleTC_phi_.get());
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::setTreeModuleCellSize(const size_t n)
/*****************************************************************/
{
  moduleCell_id_.reset(new int[n], array_deleter<int>());
  moduleCell_zside_.reset(new int[n], array_deleter<int>());
  moduleCell_subdet_.reset(new int[n], array_deleter<int>());
  moduleCell_layer_.reset(new int[n], array_deleter<int>());
  moduleCell_waferU_.reset(new int[n], array_deleter<int>());
  moduleCell_waferV_.reset(new int[n], array_deleter<int>());
  moduleCell_cellU_.reset(new int[n], array_deleter<int>());
  moduleCell_cellV_.reset(new int[n], array_deleter<int>());
  moduleCell_x_.reset(new float[n], array_deleter<float>());
  moduleCell_y_.reset(new float[n], array_deleter<float>());
  moduleCell_z_.reset(new float[n], array_deleter<float>());
  moduleCell_eta_.reset(new float[n], array_deleter<float>());
  moduleCell_phi_.reset(new float[n], array_deleter<float>());

  treeModules_->GetBranch("c_id")->SetAddress(moduleCell_id_.get());
  treeModules_->GetBranch("c_zside")->SetAddress(moduleCell_zside_.get());
  treeModules_->GetBranch("c_subdet")->SetAddress(moduleCell_subdet_.get());
  treeModules_->GetBranch("c_layer")->SetAddress(moduleCell_layer_.get());
  treeModules_->GetBranch("c_waferu")->SetAddress(moduleCell_waferU_.get());
  treeModules_->GetBranch("c_waferv")->SetAddress(moduleCell_waferV_.get());
  treeModules_->GetBranch("c_cellu")->SetAddress(moduleCell_cellU_.get());
  treeModules_->GetBranch("c_cellv")->SetAddress(moduleCell_cellV_.get());
  treeModules_->GetBranch("c_x")->SetAddress(moduleCell_x_.get());
  treeModules_->GetBranch("c_y")->SetAddress(moduleCell_y_.get());
  treeModules_->GetBranch("c_z")->SetAddress(moduleCell_z_.get());
  treeModules_->GetBranch("c_eta")->SetAddress(moduleCell_eta_.get());
  treeModules_->GetBranch("c_phi")->SetAddress(moduleCell_phi_.get());
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::setTreeTriggerCellSize(const size_t n)
/*****************************************************************/
{
  triggerCellCell_id_.reset(new int[n], array_deleter<int>());
  triggerCellCell_zside_.reset(new int[n], array_deleter<int>());
  triggerCellCell_subdet_.reset(new int[n], array_deleter<int>());
  triggerCellCell_layer_.reset(new int[n], array_deleter<int>());
  triggerCellCell_waferU_.reset(new int[n], array_deleter<int>());
  triggerCellCell_waferV_.reset(new int[n], array_deleter<int>());
  triggerCellCell_cellU_.reset(new int[n], array_deleter<int>());
  triggerCellCell_cellV_.reset(new int[n], array_deleter<int>());
  triggerCellCell_ieta_.reset(new int[n], array_deleter<int>());
  triggerCellCell_iphi_.reset(new int[n], array_deleter<int>());
  triggerCellCell_x_.reset(new float[n], array_deleter<float>());
  triggerCellCell_y_.reset(new float[n], array_deleter<float>());
  triggerCellCell_z_.reset(new float[n], array_deleter<float>());
  triggerCellCell_eta_.reset(new float[n], array_deleter<float>());
  triggerCellCell_phi_.reset(new float[n], array_deleter<float>());

  treeTriggerCells_->GetBranch("c_id")->SetAddress(triggerCellCell_id_.get());
  treeTriggerCells_->GetBranch("c_zside")->SetAddress(triggerCellCell_zside_.get());
  treeTriggerCells_->GetBranch("c_subdet")->SetAddress(triggerCellCell_subdet_.get());
  treeTriggerCells_->GetBranch("c_layer")->SetAddress(triggerCellCell_layer_.get());
  treeTriggerCells_->GetBranch("c_waferu")->SetAddress(triggerCellCell_waferU_.get());
  treeTriggerCells_->GetBranch("c_waferv")->SetAddress(triggerCellCell_waferV_.get());
  treeTriggerCells_->GetBranch("c_cellu")->SetAddress(triggerCellCell_cellU_.get());
  treeTriggerCells_->GetBranch("c_cellv")->SetAddress(triggerCellCell_cellV_.get());
  treeTriggerCells_->GetBranch("c_ieta")->SetAddress(triggerCellCell_ieta_.get());
  treeTriggerCells_->GetBranch("c_iphi")->SetAddress(triggerCellCell_iphi_.get());
  treeTriggerCells_->GetBranch("c_x")->SetAddress(triggerCellCell_x_.get());
  treeTriggerCells_->GetBranch("c_y")->SetAddress(triggerCellCell_y_.get());
  treeTriggerCells_->GetBranch("c_z")->SetAddress(triggerCellCell_z_.get());
  treeTriggerCells_->GetBranch("c_eta")->SetAddress(triggerCellCell_eta_.get());
  treeTriggerCells_->GetBranch("c_phi")->SetAddress(triggerCellCell_phi_.get());
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::setTreeCellCornerSize(const size_t n)
/*****************************************************************/
{
  cellCornersX_.reset(new float[n], array_deleter<float>());
  cellCornersY_.reset(new float[n], array_deleter<float>());
  cellCornersZ_.reset(new float[n], array_deleter<float>());

  treeCells_->GetBranch("corner_x")->SetAddress(cellCornersX_.get());
  treeCells_->GetBranch("corner_y")->SetAddress(cellCornersY_.get());
  treeCells_->GetBranch("corner_z")->SetAddress(cellCornersZ_.get());
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9Imp3::setTreeTriggerCellNeighborSize(const size_t n)
/*****************************************************************/
{
  triggerCellNeighbor_id_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_zside_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_subdet_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_layer_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_waferU_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_waferV_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_cellU_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_cellV_.reset(new int[n], array_deleter<int>());
  triggerCellNeighbor_distance_.reset(new float[n], array_deleter<float>());
  treeTriggerCells_->GetBranch("neighbor_id")->SetAddress(triggerCellNeighbor_id_.get());
  treeTriggerCells_->GetBranch("neighbor_zside")->SetAddress(triggerCellNeighbor_zside_.get());
  treeTriggerCells_->GetBranch("neighbor_subdet")->SetAddress(triggerCellNeighbor_subdet_.get());
  treeTriggerCells_->GetBranch("neighbor_layer")->SetAddress(triggerCellNeighbor_layer_.get());
  treeTriggerCells_->GetBranch("neighbor_waferu")->SetAddress(triggerCellNeighbor_waferU_.get());
  treeTriggerCells_->GetBranch("neighbor_waferv")->SetAddress(triggerCellNeighbor_waferV_.get());
  treeTriggerCells_->GetBranch("neighbor_cellu")->SetAddress(triggerCellNeighbor_cellU_.get());
  treeTriggerCells_->GetBranch("neighbor_cellv")->SetAddress(triggerCellNeighbor_cellV_.get());
  treeTriggerCells_->GetBranch("neighbor_distance")->SetAddress(triggerCellNeighbor_distance_.get());
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerGeomTesterV9Imp3);
