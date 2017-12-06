
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class HGCalTriggerNtupleHGCPanels : public HGCalTriggerNtupleBase
{

  public:
    HGCalTriggerNtupleHGCPanels(const edm::ParameterSet& conf);
    ~HGCalTriggerNtupleHGCPanels(){};   
    virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
    virtual void fill(const edm::Event& e, const edm::EventSetup& es) override final;

  private:
    virtual void clear() override final;


    edm::EDGetToken trigger_cells_token_;
    edm::ESHandle<HGCalTriggerGeometryBase> geometry_;

    int panel_n_ ;
    std::vector<uint32_t> panel_id_;
    std::vector<int> panel_zside_;
    std::vector<int> panel_layer_;
    std::vector<int> panel_sector_;
    std::vector<int> panel_number_;
    // std::vector<uint32_t> panel_mod_n_;
    // std::vector<std::vector<uint32_t> > panel_mod_id_;
    // std::vector<std::vector<float> > panel_mod_mipPt_;
    // std::vector<uint32_t> panel_third_n_;
    // std::vector<std::vector<uint32_t> > panel_third_id_;
    // std::vector<std::vector<float> > panel_third_mipPt_;
    std::vector<unsigned> panel_tc_n_;
    std::vector<std::vector<uint32_t> > panel_tc_id_;
    std::vector<std::vector<uint32_t> > panel_tc_mod_;
    std::vector<std::vector<uint32_t> > panel_tc_third_;
    std::vector<std::vector<uint32_t> > panel_tc_cell_;
    std::vector<std::vector<float> > panel_tc_mipPt_;
    std::vector<std::vector<float> > panel_tc_pt_;

};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    HGCalTriggerNtupleHGCPanels,
    "HGCalTriggerNtupleHGCPanels" );


HGCalTriggerNtupleHGCPanels::
HGCalTriggerNtupleHGCPanels(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleHGCPanels::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  trigger_cells_token_ = collector.consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("TriggerCells"));

  tree.Branch("panel_n", &panel_n_, "panel_n/I");  
  tree.Branch("panel_id", &panel_id_);           
  tree.Branch("panel_zside", &panel_zside_);
  tree.Branch("panel_layer", &panel_layer_);
  tree.Branch("panel_sector", &panel_sector_);
  tree.Branch("panel_number", &panel_number_);
  // tree.Branch("panel_mod_n", &panel_mod_n_);         
  // tree.Branch("panel_mod_id", &panel_mod_id_); 
  // tree.Branch("panel_mod_mipPt", &panel_mod_mipPt_); 
  // tree.Branch("panel_third_n", &panel_third_n_);         
  // tree.Branch("panel_third_id", &panel_third_id_); 
  // tree.Branch("panel_third_mipPt", &panel_third_mipPt_); 
  tree.Branch("panel_tc_n", &panel_tc_n_);         
  tree.Branch("panel_tc_id", &panel_tc_id_); 
  tree.Branch("panel_tc_mod", &panel_tc_mod_); 
  tree.Branch("panel_tc_third", &panel_tc_third_); 
  tree.Branch("panel_tc_cell", &panel_tc_cell_); 
  tree.Branch("panel_tc_mipPt", &panel_tc_mipPt_); 
  tree.Branch("panel_tc_pt", &panel_tc_pt_); 

}

void
HGCalTriggerNtupleHGCPanels::
fill(const edm::Event& e, const edm::EventSetup& es)
{

  // retrieve trigger cells
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigger_cells_h;
  e.getByToken(trigger_cells_token_, trigger_cells_h);
  const l1t::HGCalTriggerCellBxCollection& trigger_cells = *trigger_cells_h;

  // retrieve geometry
  es.get<CaloGeometryRecord>().get(geometry_);

  clear();

  // Regroup trigger cells by panel
  std::unordered_map< uint32_t,vector<l1t::HGCalTriggerCellBxCollection::const_iterator> > panelids_tcs;
  for(auto tc_itr=trigger_cells.begin(0); tc_itr!=trigger_cells.end(0); tc_itr++)
  {
    if(tc_itr->hwPt()>0 && tc_itr->subdetId()!=ForwardSubdetector::HGCHEB)
    {
      HGCalDetId id(tc_itr->detId());
      HGCalDetId panelid(geometry_->getModuleFromTriggerCell(id));
      panelids_tcs[panelid].push_back(tc_itr);
    }
  }
  unsigned panel_offset = 0;
  unsigned panel_mask = 0x1F;
  unsigned sector_offset = 5;
  unsigned sector_mask = 0x7;
  unsigned third_offset = 4;
  unsigned third_mask = 0x3;
  unsigned cell_mask = 0xF;

  for (const auto& panelid_tcs : panelids_tcs)
  {
    panel_n_++;
    HGCalDetId panelid(panelid_tcs.first);
    int panel_sector = (panelid.wafer()>>sector_offset) & sector_mask ;
    int panel_number = (panelid.wafer()>>panel_offset) & panel_mask ;
    const auto& tcs = panelid_tcs.second;
    panel_id_.emplace_back(panelid);
    panel_zside_.emplace_back(panelid.zside());
    panel_layer_.emplace_back(geometry_->triggerLayer(panelid));
    panel_sector_.emplace_back(panel_sector);
    panel_number_.emplace_back(panel_number);
    panel_tc_n_.emplace_back(tcs.size());
    panel_tc_id_.emplace_back();
    panel_tc_mod_.emplace_back();
    panel_tc_third_.emplace_back();
    panel_tc_cell_.emplace_back();
    panel_tc_mipPt_.emplace_back();
    panel_tc_pt_.emplace_back();

    std::unordered_map<uint32_t, float> modules_mipPt;
    std::unordered_map<uint32_t, float> thirds_mipPt;
    for (const auto& tc : tcs)
    {
      panel_tc_id_.back().push_back(tc->detId());
      panel_tc_mipPt_.back().push_back(tc->mipPt());
      panel_tc_pt_.back().push_back(tc->pt());
      HGCalDetId tc_detid(tc->detId());
      unsigned module_id = tc_detid.wafer();
      unsigned third_id = (tc_detid.cell()>>third_offset) & third_mask;
      unsigned cell_id = tc_detid.cell() & cell_mask;
      panel_tc_mod_.back().push_back(module_id);
      panel_tc_third_.back().push_back(third_id);
      panel_tc_cell_.back().push_back(cell_id);
      modules_mipPt[module_id] += tc->mipPt();
      thirds_mipPt[third_id] += tc->mipPt();
    }
    // panel_mod_n_.emplace_back(modules_mipPt.size());
    // panel_mod_id_.emplace_back();
    // panel_mod_mipPt_.emplace_back();
    // for(const auto& id_mipPt : modules_mipPt)
    // {
      // panel_mod_id_.back().push_back(id_mipPt.first);
      // panel_mod_mipPt_.back().push_back(id_mipPt.second);
    // }
    // panel_third_n_.emplace_back(thirds_mipPt.size());
    // panel_third_id_.emplace_back();
    // panel_third_mipPt_.emplace_back();
    // for(const auto& id_mipPt : thirds_mipPt)
    // {
      // panel_third_id_.back().push_back(id_mipPt.first);
      // panel_third_mipPt_.back().push_back(id_mipPt.second);
    // }
  }
}


void
HGCalTriggerNtupleHGCPanels::
clear()
{
  panel_n_ = 0;
  panel_id_.clear();
  panel_zside_.clear();
  panel_layer_.clear();
  panel_sector_.clear();
  panel_number_.clear();
  // panel_mod_n_.clear();
  // panel_mod_id_.clear();
  // panel_mod_mipPt_.clear();
  // panel_third_n_.clear();
  // panel_third_id_.clear();
  // panel_third_mipPt_.clear();
  panel_tc_n_.clear();
  panel_tc_id_.clear();
  panel_tc_mod_.clear();
  panel_tc_third_.clear();
  panel_tc_cell_.clear();
  panel_tc_mipPt_.clear();
  panel_tc_pt_.clear();
}




