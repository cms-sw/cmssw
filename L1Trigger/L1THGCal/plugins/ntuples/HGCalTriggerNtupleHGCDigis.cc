#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

class HGCalTriggerNtupleHGCDigis : public HGCalTriggerNtupleBase
{

    public:
        HGCalTriggerNtupleHGCDigis(const edm::ParameterSet& conf);
        ~HGCalTriggerNtupleHGCDigis(){};
        virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
        virtual void fill(const edm::Event& e, const edm::EventSetup& es) override final;

    private:
        virtual void clear() override final;

        edm::EDGetToken ee_token_, fh_token_;
        bool is_Simhit_comp_;
        edm::EDGetToken SimHits_inputee_, SimHits_inputfh_;

        int hgcdigi_n_ ;
        std::vector<int> hgcdigi_id_;
        std::vector<int> hgcdigi_subdet_;
        std::vector<int> hgcdigi_side_;
        std::vector<int> hgcdigi_layer_;
        std::vector<int> hgcdigi_wafer_;
        std::vector<int> hgcdigi_wafertype_ ;
        std::vector<int> hgcdigi_cell_;
        std::vector<float> hgcdigi_cell_eta_;
        std::vector<float> hgcdigi_cell_phi_;
        std::vector<float> hgcdigi_cell_z_;
        std::vector<uint32_t> hgcdigi_data_;
        std::vector<int> hgcdigi_isadc_;
        std::vector<int> hgcdigi_istot_;
        std::vector<float> hgcdigi_cell_simhit_energy_;

        edm::ESHandle<HGCalGeometry> geom_ee, geom_fh;
        edm::ESHandle<HGCalTopology> topo_ee, topo_fh;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
        HGCalTriggerNtupleHGCDigis,
        "HGCalTriggerNtupleHGCDigis" );


HGCalTriggerNtupleHGCDigis::
HGCalTriggerNtupleHGCDigis(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{

}

void
HGCalTriggerNtupleHGCDigis::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{

    ee_token_ = collector.consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("HGCDigisEE")); 
    fh_token_ = collector.consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("HGCDigisFH"));
    SimHits_inputee_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("eeSimHits"));
    SimHits_inputfh_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("fhSimHits"));
    is_Simhit_comp_ = conf.getParameter<bool>("isSimhitComp");

    tree.Branch("hgcdigi_n", &hgcdigi_n_, "hgcdigi_n/I");
    tree.Branch("hgcdigi_id", &hgcdigi_id_);
    tree.Branch("hgcdigi_subdet", &hgcdigi_subdet_);
    tree.Branch("hgcdigi_zside", &hgcdigi_side_);
    tree.Branch("hgcdigi_layer", &hgcdigi_layer_);
    tree.Branch("hgcdigi_wafer", &hgcdigi_wafer_);
    tree.Branch("hgcdigi_wafertype", &hgcdigi_wafertype_);
    tree.Branch("hgcdigi_cell", &hgcdigi_cell_);    
    tree.Branch("hgcdigi_cell_eta", &hgcdigi_cell_eta_);    
    tree.Branch("hgcdigi_cell_phi", &hgcdigi_cell_phi_);    
    tree.Branch("hgcdigi_cell_z", &hgcdigi_cell_z_);    
    tree.Branch("hgcdigi_data", &hgcdigi_data_);
    tree.Branch("hgcdigi_isadc", &hgcdigi_isadc_);
    tree.Branch("hgcdigi_istot", &hgcdigi_istot_);
    if (is_Simhit_comp_) tree.Branch("hgcdigi_cell_simhit_energy", &hgcdigi_cell_simhit_energy_);
}

void
HGCalTriggerNtupleHGCDigis::
fill(const edm::Event& e, const edm::EventSetup& es)
{
 
    es.get<IdealGeometryRecord>().get("HGCalEESensitive", geom_ee);
    es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive", geom_fh);
    es.get<IdealGeometryRecord>().get("HGCalEESensitive",topo_ee);
    es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",topo_fh);
    edm::Handle<HGCEEDigiCollection> ee_digis_h;
    e.getByToken(ee_token_, ee_digis_h);
    const HGCEEDigiCollection& ee_digis = *ee_digis_h;
    edm::Handle<HGCHEDigiCollection> fh_digis_h;
    e.getByToken(fh_token_, fh_digis_h);
    const HGCHEDigiCollection& fh_digis = *fh_digis_h;

    // retrieve simhit collections
    std::unordered_map<uint32_t, double> simhits_ee;
    std::unordered_map<uint32_t, double> simhits_fh;
    
    // sim hit association
    if (is_Simhit_comp_) {
      edm::Handle<edm::PCaloHitContainer> ee_simhits_h;
      e.getByToken(SimHits_inputee_,ee_simhits_h);
      const edm::PCaloHitContainer& ee_simhits = *ee_simhits_h;
      edm::Handle<edm::PCaloHitContainer> fh_simhits_h;
      e.getByToken(SimHits_inputfh_,fh_simhits_h);
      const edm::PCaloHitContainer& fh_simhits = *fh_simhits_h;
      
      //EE
      HGCalDetId digiid, simid;
      int layer=0,cell=0, sec=0, subsec=0, zp=0,subdet=0;
      ForwardSubdetector mysubdet;
      HGCalDetId recoDetId ;
      
      for( const auto& simhit : ee_simhits ) { 
        simid = (HGCalDetId)simhit.id();
        HGCalTestNumbering::unpackHexagonIndex(simid, subdet, zp, layer, sec, subsec, cell); 
        mysubdet = (ForwardSubdetector)(subdet);
        std::pair<int,int> recoLayerCell = topo_ee->dddConstants().simToReco(cell,layer,sec,topo_ee->detectorType());
        cell  = recoLayerCell.first;
        layer = recoLayerCell.second;
        if (layer<0 || cell<0) {
          continue;
        }
        recoDetId = HGCalDetId(mysubdet,zp,layer,subsec,sec,cell);
        auto itr_insert = simhits_ee.emplace(recoDetId, 0.);
        itr_insert.first->second += simhit.energy();
      }

      //  FH
      layer=0;
      cell=0;
      sec=0;
      subsec=0;
      zp=0;
      subdet=0;
      
      for( const auto& simhit : fh_simhits ) { 
        simid = (HGCalDetId) simhit.id();
        HGCalTestNumbering::unpackHexagonIndex(simid, subdet, zp, layer, sec, subsec, cell); 
        mysubdet = (ForwardSubdetector)(subdet);
        std::pair<int,int> recoLayerCell = topo_fh->dddConstants().simToReco(cell,layer,sec,topo_fh->detectorType());
        cell  = recoLayerCell.first;
        layer = recoLayerCell.second;
        if (layer<0 || cell<0) {
          continue;
        }
        recoDetId = HGCalDetId(mysubdet,zp,layer,subsec,sec,cell);
        auto itr_insert = simhits_fh.emplace(recoDetId, 0.);
        itr_insert.first->second += simhit.energy();
      }      
    }
    //
    
    clear();
    hgcdigi_n_ = ee_digis.size() + fh_digis.size();
    hgcdigi_id_.reserve(hgcdigi_n_);
    hgcdigi_subdet_.reserve(hgcdigi_n_);
    hgcdigi_side_.reserve(hgcdigi_n_);
    hgcdigi_layer_.reserve(hgcdigi_n_);
    hgcdigi_wafer_.reserve(hgcdigi_n_);
    hgcdigi_wafertype_.reserve(hgcdigi_n_);
    hgcdigi_cell_.reserve(hgcdigi_n_);
    hgcdigi_cell_eta_.reserve(hgcdigi_n_);
    hgcdigi_cell_phi_.reserve(hgcdigi_n_);
    hgcdigi_cell_z_.reserve(hgcdigi_n_);
    hgcdigi_data_.reserve(hgcdigi_n_);
    hgcdigi_isadc_.reserve(hgcdigi_n_);
    hgcdigi_istot_.reserve(hgcdigi_n_);
    hgcdigi_cell_simhit_energy_.reserve(hgcdigi_n_);
    for(const auto& digi : ee_digis)
      {
        const HGCalDetId id(digi.id());
        hgcdigi_id_.emplace_back(id.rawId());
        hgcdigi_subdet_.emplace_back(ForwardSubdetector::HGCEE);
        hgcdigi_side_.emplace_back(id.zside());
        hgcdigi_layer_.emplace_back(id.layer());
        hgcdigi_wafer_.emplace_back(id.wafer());
        hgcdigi_wafertype_.emplace_back(id.waferType());
        hgcdigi_cell_.emplace_back(id.cell());
        hgcdigi_cell_eta_.emplace_back(geom_ee->getPosition(id.rawId()).eta());
        hgcdigi_cell_phi_.emplace_back(geom_ee->getPosition(id.rawId()).phi());
        hgcdigi_cell_z_.emplace_back(geom_ee->getPosition(id.rawId()).z());
        hgcdigi_data_.emplace_back(digi[2].data()); 
        int is_tot=0,is_adc=0;
        if (digi[2].mode()) is_tot=1;
        else is_adc =1;
        hgcdigi_isadc_.emplace_back(is_adc);
        hgcdigi_istot_.emplace_back(is_tot);
        if (is_Simhit_comp_) {
          double hit_energy=0;
          auto itr = simhits_ee.find(id);
          if(itr!=simhits_ee.end())hit_energy = itr->second;
          hgcdigi_cell_simhit_energy_.emplace_back(hit_energy); 
        }
      }
    
    for(const auto& digi : fh_digis)
      {
        const HGCalDetId id(digi.id());
        hgcdigi_id_.emplace_back(id.rawId());
        hgcdigi_subdet_.emplace_back(ForwardSubdetector::HGCHEF);
        hgcdigi_side_.emplace_back(id.zside());
        hgcdigi_layer_.emplace_back(id.layer());
        hgcdigi_wafer_.emplace_back(id.wafer());
        hgcdigi_wafertype_.emplace_back(id.waferType());
        hgcdigi_cell_.emplace_back(id.cell());
        hgcdigi_cell_eta_.emplace_back(geom_fh->getPosition(id.rawId()).eta());
        hgcdigi_cell_phi_.emplace_back(geom_fh->getPosition(id.rawId()).phi());
        hgcdigi_cell_z_.emplace_back(geom_fh->getPosition(id.rawId()).z());
        hgcdigi_data_.emplace_back(digi[2].data()); 
        int is_tot=0,is_adc=0;
        if (digi[2].mode()) is_tot=1;
        else is_adc =1;
        hgcdigi_istot_.emplace_back(is_tot);
        hgcdigi_isadc_.emplace_back(is_adc);
        if (is_Simhit_comp_) {
          double hit_energy=0;
          auto itr = simhits_fh.find(id);
          if(itr!=simhits_fh.end())hit_energy = itr->second;
          hgcdigi_cell_simhit_energy_.emplace_back(hit_energy); 
        }
      }
}


void
HGCalTriggerNtupleHGCDigis::
clear()
{
    hgcdigi_n_ = 0;
    hgcdigi_id_.clear();
    hgcdigi_subdet_.clear();
    hgcdigi_side_.clear();
    hgcdigi_layer_.clear();
    hgcdigi_wafer_.clear();
    hgcdigi_wafertype_.clear();
    hgcdigi_cell_.clear();
    hgcdigi_cell_eta_.clear();
    hgcdigi_cell_phi_.clear();
    hgcdigi_cell_z_.clear();
    hgcdigi_data_.clear();
    hgcdigi_istot_.clear();
    hgcdigi_isadc_.clear();
    if  (is_Simhit_comp_) hgcdigi_cell_simhit_energy_.clear();
}




