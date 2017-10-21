
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
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
    void simhits(const edm::Event& e, std::unordered_map<uint32_t, double>& simhits_ee, std::unordered_map<uint32_t, double>& simhits_fh, std::unordered_map<uint32_t, double>& simhits_bh);
    virtual void clear() override final;


    edm::EDGetToken trigger_cells_token_;
    edm::EDGetToken simhits_ee_token_, simhits_fh_token_, simhits_bh_token_;
    bool fill_simenergy_;
    edm::ESHandle<HGCalTriggerGeometryBase> geometry_;


    int tc_n_ ;
    std::vector<uint32_t> tc_id_;
    std::vector<int> tc_subdet_;
    std::vector<int> tc_side_;
    std::vector<int> tc_layer_;
    std::vector<int> tc_wafer_;
    std::vector<int> tc_wafertype_ ;
    std::vector<int> tc_cell_;
    std::vector<uint32_t> tc_data_;
    std::vector<float> tc_mipPt_;
    std::vector<float> tc_energy_;
    std::vector<float> tc_simenergy_;
    std::vector<float> tc_eta_;
    std::vector<float> tc_phi_;
    std::vector<float> tc_pt_;
    std::vector<float> tc_z_;

};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    HGCalTriggerNtupleHGCTriggerCells,
    "HGCalTriggerNtupleHGCTriggerCells" );


HGCalTriggerNtupleHGCTriggerCells::
HGCalTriggerNtupleHGCTriggerCells(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
  fill_simenergy_ = conf.getParameter<bool>("FillSimEnergy");
}

void
HGCalTriggerNtupleHGCTriggerCells::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  trigger_cells_token_ = collector.consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("TriggerCells"));

  if (fill_simenergy_) 
  {
    simhits_ee_token_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("eeSimHits"));
    simhits_fh_token_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("fhSimHits"));
    simhits_bh_token_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("bhSimHits"));
  }

  tree.Branch("tc_n", &tc_n_, "tc_n/I");
  tree.Branch("tc_id", &tc_id_);
  tree.Branch("tc_subdet", &tc_subdet_);
  tree.Branch("tc_zside", &tc_side_);
  tree.Branch("tc_layer", &tc_layer_);
  tree.Branch("tc_wafer", &tc_wafer_);
  tree.Branch("tc_wafertype", &tc_wafertype_);
  tree.Branch("tc_cell", &tc_cell_);    
  tree.Branch("tc_data", &tc_data_);
  tree.Branch("tc_mipPt", &tc_mipPt_);
  tree.Branch("tc_energy", &tc_energy_);
  if(fill_simenergy_) tree.Branch("tc_simenergy", &tc_simenergy_);
  tree.Branch("tc_eta", &tc_eta_);
  tree.Branch("tc_phi", &tc_phi_);
  tree.Branch("tc_pt", &tc_pt_);
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
  es.get<CaloGeometryRecord>().get(geometry_);

  // sim hit association
  std::unordered_map<uint32_t, double> simhits_ee;
  std::unordered_map<uint32_t, double> simhits_fh;  
  std::unordered_map<uint32_t, double> simhits_bh;  
  if(fill_simenergy_) simhits(e, simhits_ee, simhits_fh, simhits_bh);

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
      tc_mipPt_.emplace_back(tc_itr->mipPt());
      // physical values 
      tc_energy_.emplace_back(tc_itr->energy());
      tc_eta_.emplace_back(tc_itr->eta());
      tc_phi_.emplace_back(tc_itr->phi());
      tc_pt_.emplace_back(tc_itr->pt());
      tc_z_.emplace_back(tc_itr->position().z());

      if(fill_simenergy_)
      {
        double energy = 0;
        int subdet = id.subdetId();
        // search for simhit for all the cells inside the trigger cell
        for(uint32_t c_id : geometry_->getCellsFromTriggerCell(id))
        {
          switch(subdet)
          {
            case ForwardSubdetector::HGCEE:
              {
                auto itr = simhits_ee.find(c_id);
                if(itr!=simhits_ee.end()) energy += itr->second;
                break;
              }
            case ForwardSubdetector::HGCHEF:
              {
                auto itr = simhits_fh.find(c_id);
                if(itr!=simhits_fh.end()) energy += itr->second;
                break;
              }
            case ForwardSubdetector::HGCHEB:
              {
                auto itr = simhits_bh.find(c_id);
                if(itr!=simhits_bh.end()) energy += itr->second;
                break;
              }
            default:
              break;
          }
        }
        tc_simenergy_.emplace_back(energy); 
      }
    }
  }
}


void
HGCalTriggerNtupleHGCTriggerCells::
simhits(const edm::Event& e, std::unordered_map<uint32_t, double>& simhits_ee, std::unordered_map<uint32_t, double>& simhits_fh, std::unordered_map<uint32_t, double>& simhits_bh)
{
      edm::Handle<edm::PCaloHitContainer> ee_simhits_h;
      e.getByToken(simhits_ee_token_,ee_simhits_h);
      const edm::PCaloHitContainer& ee_simhits = *ee_simhits_h;
      edm::Handle<edm::PCaloHitContainer> fh_simhits_h;
      e.getByToken(simhits_fh_token_,fh_simhits_h);
      const edm::PCaloHitContainer& fh_simhits = *fh_simhits_h;
      edm::Handle<edm::PCaloHitContainer> bh_simhits_h;
      e.getByToken(simhits_bh_token_,bh_simhits_h);
      const edm::PCaloHitContainer& bh_simhits = *bh_simhits_h;
      
      //EE
      int layer=0,cell=0, sec=0, subsec=0, zp=0,subdet=0;
      ForwardSubdetector mysubdet;
      for( const auto& simhit : ee_simhits )
      { 
        HGCalTestNumbering::unpackHexagonIndex(simhit.id(), subdet, zp, layer, sec, subsec, cell); 
        mysubdet = (ForwardSubdetector)subdet;
        std::pair<int,int> recoLayerCell = geometry_->eeTopology().dddConstants().simToReco(cell,layer,sec,geometry_->eeTopology().detectorType());
        cell  = recoLayerCell.first;
        layer = recoLayerCell.second;
        if (layer<0 || cell<0) continue;
        auto itr_insert = simhits_ee.emplace(HGCalDetId(mysubdet,zp,layer,subsec,sec,cell), 0.);
        itr_insert.first->second += simhit.energy();
      }

      //  FH
      layer=0; cell=0; sec=0; subsec=0; zp=0; subdet=0;
      for( const auto& simhit : fh_simhits ) 
      { 
        HGCalTestNumbering::unpackHexagonIndex(simhit.id(), subdet, zp, layer, sec, subsec, cell); 
        mysubdet = (ForwardSubdetector)(subdet);
        std::pair<int,int> recoLayerCell = geometry_->fhTopology().dddConstants().simToReco(cell,layer,sec,geometry_->fhTopology().detectorType());
        cell  = recoLayerCell.first;
        layer = recoLayerCell.second;
        if (layer<0 || cell<0) continue;
        auto itr_insert = simhits_fh.emplace(HGCalDetId(mysubdet,zp,layer,subsec,sec,cell), 0.);
        itr_insert.first->second += simhit.energy();
      }      

      //  BH
      for( const auto& simhit : bh_simhits ) 
      { 
        HcalDetId id = HcalHitRelabeller::relabel(simhit.id(), geometry_->bhTopology().dddConstants());
        if (id.subdetId()!=HcalEndcap) continue;
        auto itr_insert = simhits_bh.emplace(id, 0.);
        itr_insert.first->second += simhit.energy();
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
  tc_mipPt_.clear();
  tc_energy_.clear();
  tc_simenergy_.clear();
  tc_eta_.clear();
  tc_phi_.clear();
  tc_pt_.clear();
  tc_z_.clear();
}




