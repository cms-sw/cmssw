
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class HGCalTriggerNtupleHGCTriggerCells : public HGCalTriggerNtupleBase
{

  public:
    HGCalTriggerNtupleHGCTriggerCells(const edm::ParameterSet& conf);
    ~HGCalTriggerNtupleHGCTriggerCells(){};
    virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
    virtual void fill(const edm::Event& e, const edm::EventSetup& es) override final;

  private:
    virtual void clear() override final;


    edm::EDGetToken trigger_cells_token_;

    int tc_n_ ;
    std::vector<uint32_t> tc_id_;
    std::vector<int> tc_subdet_;
    std::vector<int> tc_side_;
    std::vector<int> tc_layer_;
    std::vector<int> tc_wafer_;
    std::vector<int> tc_wafertype_ ;
    std::vector<int> tc_cell_;
    std::vector<uint32_t> tc_data_;
    std::vector<float> tc_energy_;
    std::vector<float> tc_eta_;
    std::vector<float> tc_phi_;
    std::vector<float> tc_z_;

};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    HGCalTriggerNtupleHGCTriggerCells,
    "HGCalTriggerNtupleHGCTriggerCells" );


HGCalTriggerNtupleHGCTriggerCells::
HGCalTriggerNtupleHGCTriggerCells(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleHGCTriggerCells::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  trigger_cells_token_ = collector.consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("TriggerCells"));

  tree.Branch("tc_n", &tc_n_, "tc_n/I");
  tree.Branch("tc_id", &tc_id_);
  tree.Branch("tc_subdet", &tc_subdet_);
  tree.Branch("tc_zside", &tc_side_);
  tree.Branch("tc_layer", &tc_layer_);
  tree.Branch("tc_wafer", &tc_wafer_);
  tree.Branch("tc_wafertype", &tc_wafertype_);
  tree.Branch("tc_cell", &tc_cell_);    
  tree.Branch("tc_data", &tc_data_);
  tree.Branch("tc_energy", &tc_energy_);
  tree.Branch("tc_eta", &tc_eta_);
  tree.Branch("tc_phi", &tc_phi_);
  tree.Branch("tc_z", &tc_z_);

}

void
HGCalTriggerNtupleHGCTriggerCells::
fill(const edm::Event& e, const edm::EventSetup& es)
{

  // retrieve trigger cells
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigger_cells_h;
  e.getByToken(trigger_cells_token_, trigger_cells_h);
  const l1t::HGCalTriggerCellBxCollection& trigger_cells = *trigger_cells_h;

  // retrieve geometry
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
  es.get<IdealGeometryRecord>().get(geometry);

  clear();
  for(auto tc_itr=trigger_cells.begin(0); tc_itr!=trigger_cells.end(0); tc_itr++)
  {
    if(tc_itr->hwPt()>0)
    {
      tc_n_++;
      // hardware data
      HGCalDetId id(tc_itr->detId());
      tc_id_.emplace_back(tc_itr->detId());
      tc_subdet_.emplace_back(id.subdetId());
      tc_side_.emplace_back(id.zside());
      tc_layer_.emplace_back(id.layer());
      tc_wafer_.emplace_back(id.wafer());
      tc_wafertype_.emplace_back(id.waferType());
      tc_cell_.emplace_back(id.cell());
      tc_data_.emplace_back(tc_itr->hwPt());
      // physical values 
      const auto& position = geometry->getTriggerCellPosition(tc_itr->detId());
      tc_energy_.emplace_back(tc_itr->energy());
      tc_eta_.emplace_back(tc_itr->eta());
      tc_phi_.emplace_back(tc_itr->phi());
      tc_z_.emplace_back(position.z());
    }
  }
}


void
HGCalTriggerNtupleHGCTriggerCells::
clear()
{
  tc_n_ = 0;
  tc_id_.clear();
  tc_subdet_.clear();
  tc_side_.clear();
  tc_layer_.clear();
  tc_wafer_.clear();
  tc_wafertype_.clear();
  tc_cell_.clear();
  tc_data_.clear();
  tc_energy_.clear();
  tc_eta_.clear();
  tc_phi_.clear();
  tc_z_.clear();
}




