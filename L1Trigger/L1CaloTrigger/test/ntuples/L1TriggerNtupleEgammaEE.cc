#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class L1TriggerNtupleEgammaEE : public HGCalTriggerNtupleBase
{

  public:
    L1TriggerNtupleEgammaEE(const edm::ParameterSet& conf);
    ~L1TriggerNtupleEgammaEE() override{};
    void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
    void fill(const edm::Event& e, const edm::EventSetup& es) final;

  private:
    void clear() final;
    // HGCalTriggerTools triggerTools_;

    edm::EDGetToken egamma_ee_token_;

    int egammaEE_n_ ;
    std::vector<float> egammaEE_pt_;
    std::vector<float> egammaEE_energy_;
    std::vector<float> egammaEE_eta_;
    std::vector<float> egammaEE_phi_;
    std::vector<float> egammaEE_hwQual_;

};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    L1TriggerNtupleEgammaEE,
    "L1TriggerNtupleEgammaEE" );


L1TriggerNtupleEgammaEE::
L1TriggerNtupleEgammaEE(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
L1TriggerNtupleEgammaEE::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  egamma_ee_token_ = collector.consumes<l1t::EGammaBxCollection>(conf.getParameter<edm::InputTag>("EgammaEE"));

  tree.Branch("egammaEE_n",     &egammaEE_n_, "egammaEE_n/I");
  tree.Branch("egammaEE_pt",     &egammaEE_pt_);
  tree.Branch("egammaEE_energy", &egammaEE_energy_);
  tree.Branch("egammaEE_eta",    &egammaEE_eta_);
  tree.Branch("egammaEE_phi",    &egammaEE_phi_);
  tree.Branch("egammaEE_hwQual", &egammaEE_hwQual_);

}



void
L1TriggerNtupleEgammaEE::
fill(const edm::Event& e, const edm::EventSetup& es)
{

  // retrieve towers
  edm::Handle<l1t::EGammaBxCollection> egamma_ee_h;
  e.getByToken(egamma_ee_token_, egamma_ee_h);
  const l1t::EGammaBxCollection& egamma_ee_collection = *egamma_ee_h;


  // triggerTools_.eventSetup(es);

  clear();
  for(auto egee_itr=egamma_ee_collection.begin(0); egee_itr!=egamma_ee_collection.end(0); egee_itr++) {
    egammaEE_n_++;
    // physical values
    egammaEE_pt_.emplace_back(egee_itr->pt());
    egammaEE_energy_.emplace_back(egee_itr->energy());
    egammaEE_eta_.emplace_back(egee_itr->eta());
    egammaEE_phi_.emplace_back(egee_itr->phi());
    egammaEE_hwQual_.emplace_back(egee_itr->hwQual());
  }
}


void
L1TriggerNtupleEgammaEE::
clear()
{
  egammaEE_n_ = 0;
  egammaEE_pt_.clear();
  egammaEE_energy_.clear();
  egammaEE_eta_.clear();
  egammaEE_phi_.clear();
  egammaEE_hwQual_.clear();
}
