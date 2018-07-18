#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class HGCalTriggerNtupleHGCTowers : public HGCalTriggerNtupleBase
{

  public:
    HGCalTriggerNtupleHGCTowers(const edm::ParameterSet& conf);
    ~HGCalTriggerNtupleHGCTowers() override{};
    void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
    void fill(const edm::Event& e, const edm::EventSetup& es) final;

  private:
    void clear() final;

    edm::EDGetToken towers_token_;

    int tower_n_ ;
    std::vector<float> tower_pt_;
    std::vector<float> tower_energy_;
    std::vector<float> tower_eta_;
    std::vector<float> tower_phi_;
    std::vector<float> tower_etEm_;
    std::vector<float> tower_etHad_;
    std::vector<int> tower_iEta_;
    std::vector<int> tower_iPhi_;

};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    HGCalTriggerNtupleHGCTowers,
    "HGCalTriggerNtupleHGCTowers" );


HGCalTriggerNtupleHGCTowers::
HGCalTriggerNtupleHGCTowers(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleHGCTowers::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  towers_token_ = collector.consumes<l1t::HGCalTowerBxCollection>(conf.getParameter<edm::InputTag>("Towers"));

  tree.Branch("tower_n", &tower_n_, "tower_n/I");
  tree.Branch("tower_pt", &tower_pt_);
  tree.Branch("tower_energy", &tower_energy_);
  tree.Branch("tower_eta", &tower_eta_);
  tree.Branch("tower_phi", &tower_phi_);
  tree.Branch("tower_etEm", &tower_etEm_);
  tree.Branch("tower_etHad", &tower_etHad_);
  tree.Branch("tower_iEta", &tower_iEta_);
  tree.Branch("tower_iPhi", &tower_iPhi_);

}



void
HGCalTriggerNtupleHGCTowers::
fill(const edm::Event& e, const edm::EventSetup& es)
{

  // retrieve towers
  edm::Handle<l1t::HGCalTowerBxCollection> towers_h;
  e.getByToken(towers_token_, towers_h);
  const l1t::HGCalTowerBxCollection& towers = *towers_h;

  // retrieve geometry
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
  es.get<CaloGeometryRecord>().get(geometry);

  clear();
  for(auto tower_itr=towers.begin(0); tower_itr!=towers.end(0); tower_itr++)
  {
    tower_n_++;
    // physical values
    tower_pt_.emplace_back(tower_itr->pt());
    tower_energy_.emplace_back(tower_itr->energy());
    tower_eta_.emplace_back(tower_itr->eta());
    tower_phi_.emplace_back(tower_itr->phi());
    tower_etEm_.emplace_back(tower_itr->etEm());
    tower_etHad_.emplace_back(tower_itr->etHad());

    tower_iEta_.emplace_back(tower_itr->id().iEta());
    tower_iPhi_.emplace_back(tower_itr->id().iPhi());
  }
}


void
HGCalTriggerNtupleHGCTowers::
clear()
{
  tower_n_ = 0;
  tower_pt_.clear();
  tower_energy_.clear();
  tower_eta_.clear();
  tower_phi_.clear();
  tower_etEm_.clear();
  tower_etHad_.clear();
  tower_iEta_.clear();
  tower_iPhi_.clear();
}
