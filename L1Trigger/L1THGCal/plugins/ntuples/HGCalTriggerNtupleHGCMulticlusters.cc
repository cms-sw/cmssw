#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerClusterIdentificationBase.h"



class HGCalTriggerNtupleHGCMulticlusters : public HGCalTriggerNtupleBase
{

  public:
    HGCalTriggerNtupleHGCMulticlusters(const edm::ParameterSet& conf);
    ~HGCalTriggerNtupleHGCMulticlusters() override{};
    void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
    void fill(const edm::Event& e, const edm::EventSetup& es) final;

  private:
    void clear() final;

    edm::EDGetToken multiclusters_token_;

    std::unique_ptr<HGCalTriggerClusterIdentificationBase> id_;

    int cl3d_n_ ;
    std::vector<uint32_t> cl3d_id_;
    std::vector<float> cl3d_pt_;
    std::vector<float> cl3d_energy_;
    std::vector<float> cl3d_eta_;
    std::vector<float> cl3d_phi_;
    std::vector<int> cl3d_clusters_n_;
    std::vector<std::vector<uint32_t>> cl3d_clusters_id_;
    // cluster shower shapes
    std::vector<int> cl3d_showerlength_;
    std::vector<int> cl3d_coreshowerlength_;
    std::vector<int> cl3d_firstlayer_;
    std::vector<int> cl3d_maxlayer_;
    std::vector<float> cl3d_seetot_;
    std::vector<float> cl3d_seemax_;
    std::vector<float> cl3d_spptot_;
    std::vector<float> cl3d_sppmax_;
    std::vector<float> cl3d_szz_;
    std::vector<float> cl3d_srrtot_;
    std::vector<float> cl3d_srrmax_;
    std::vector<float> cl3d_srrmean_;
    std::vector<float> cl3d_emaxe_;
    std::vector<float> cl3d_bdteg_;
    std::vector<int> cl3d_quality_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    HGCalTriggerNtupleHGCMulticlusters,
    "HGCalTriggerNtupleHGCMulticlusters" );


HGCalTriggerNtupleHGCMulticlusters::
HGCalTriggerNtupleHGCMulticlusters(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleHGCMulticlusters::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  multiclusters_token_ = collector.consumes<l1t::HGCalMulticlusterBxCollection>(conf.getParameter<edm::InputTag>("Multiclusters"));
  id_.reset( HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT") );
  id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification")); 

  tree.Branch("cl3d_n", &cl3d_n_, "cl3d_n/I");
  tree.Branch("cl3d_id", &cl3d_id_);
  tree.Branch("cl3d_pt", &cl3d_pt_);
  tree.Branch("cl3d_energy", &cl3d_energy_);
  tree.Branch("cl3d_eta", &cl3d_eta_);
  tree.Branch("cl3d_phi", &cl3d_phi_);
  tree.Branch("cl3d_clusters_n", &cl3d_clusters_n_);
  tree.Branch("cl3d_clusters_id", &cl3d_clusters_id_);
  tree.Branch("cl3d_showerlength", &cl3d_showerlength_);
  tree.Branch("cl3d_coreshowerlength", &cl3d_coreshowerlength_);
  tree.Branch("cl3d_firstlayer", &cl3d_firstlayer_);
  tree.Branch("cl3d_maxlayer", &cl3d_maxlayer_);
  tree.Branch("cl3d_seetot", &cl3d_seetot_);
  tree.Branch("cl3d_seemax", &cl3d_seemax_);
  tree.Branch("cl3d_spptot", &cl3d_spptot_);
  tree.Branch("cl3d_sppmax", &cl3d_sppmax_);
  tree.Branch("cl3d_szz", &cl3d_szz_);
  tree.Branch("cl3d_srrtot", &cl3d_srrtot_);
  tree.Branch("cl3d_srrmax", &cl3d_srrmax_);
  tree.Branch("cl3d_srrmean", &cl3d_srrmean_);
  tree.Branch("cl3d_emaxe", &cl3d_emaxe_);
  tree.Branch("cl3d_bdteg", &cl3d_bdteg_);
  tree.Branch("cl3d_quality", &cl3d_quality_);

}

void
HGCalTriggerNtupleHGCMulticlusters::
fill(const edm::Event& e, const edm::EventSetup& es)
{

  // retrieve clusters 3D
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters_h;
  e.getByToken(multiclusters_token_, multiclusters_h);
  const l1t::HGCalMulticlusterBxCollection& multiclusters = *multiclusters_h;

  // retrieve geometry
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
  es.get<CaloGeometryRecord>().get(geometry);

  clear();
  for(auto cl3d_itr=multiclusters.begin(0); cl3d_itr!=multiclusters.end(0); cl3d_itr++)
  {
    cl3d_n_++;
    cl3d_id_.emplace_back(cl3d_itr->detId());
    // physical values 
    cl3d_pt_.emplace_back(cl3d_itr->pt());
    cl3d_energy_.emplace_back(cl3d_itr->energy());
    cl3d_eta_.emplace_back(cl3d_itr->eta());
    cl3d_phi_.emplace_back(cl3d_itr->phi());
    cl3d_clusters_n_.emplace_back(cl3d_itr->constituents().size());
    cl3d_showerlength_.emplace_back(cl3d_itr->showerLength());
    cl3d_coreshowerlength_.emplace_back(cl3d_itr->coreShowerLength());
    cl3d_firstlayer_.emplace_back(cl3d_itr->firstLayer());
    cl3d_maxlayer_.emplace_back(cl3d_itr->maxLayer());
    cl3d_seetot_.emplace_back(cl3d_itr->sigmaEtaEtaTot());
    cl3d_seemax_.emplace_back(cl3d_itr->sigmaEtaEtaMax());
    cl3d_spptot_.emplace_back(cl3d_itr->sigmaPhiPhiTot());
    cl3d_sppmax_.emplace_back(cl3d_itr->sigmaPhiPhiMax());
    cl3d_szz_.emplace_back(cl3d_itr->sigmaZZ());
    cl3d_srrtot_.emplace_back(cl3d_itr->sigmaRRTot());
    cl3d_srrmax_.emplace_back(cl3d_itr->sigmaRRMax());
    cl3d_srrmean_.emplace_back(cl3d_itr->sigmaRRMean());
    cl3d_emaxe_.emplace_back(cl3d_itr->eMax()/cl3d_itr->energy());
    cl3d_bdteg_.emplace_back(id_->value(*cl3d_itr));
    cl3d_quality_.emplace_back(cl3d_itr->hwQual());

    // Retrieve indices of trigger cells inside cluster
    cl3d_clusters_id_.emplace_back(cl3d_itr->constituents().size());
    std::transform(cl3d_itr->constituents_begin(), cl3d_itr->constituents_end(),
        cl3d_clusters_id_.back().begin(), [](const std::pair<uint32_t,edm::Ptr<l1t::HGCalCluster>>& id_cl){return id_cl.second->detId();}
        );
  }
}


void
HGCalTriggerNtupleHGCMulticlusters::
clear()
{
  cl3d_n_ = 0;
  cl3d_id_.clear();
  cl3d_pt_.clear();
  cl3d_energy_.clear();
  cl3d_eta_.clear();
  cl3d_phi_.clear();
  cl3d_clusters_n_.clear();
  cl3d_clusters_id_.clear();
  cl3d_showerlength_.clear();
  cl3d_coreshowerlength_.clear();
  cl3d_firstlayer_.clear();
  cl3d_maxlayer_.clear();
  cl3d_seetot_.clear();
  cl3d_seemax_.clear();
  cl3d_spptot_.clear();
  cl3d_sppmax_.clear();
  cl3d_szz_.clear();
  cl3d_srrtot_.clear();
  cl3d_srrmax_.clear();
  cl3d_srrmean_.clear();
  cl3d_emaxe_.clear();
  cl3d_bdteg_.clear();
  cl3d_quality_.clear();
}




