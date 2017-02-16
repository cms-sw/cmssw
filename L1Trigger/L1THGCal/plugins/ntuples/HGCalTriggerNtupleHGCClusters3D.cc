
#include "DataFormats/L1THGCal/interface/HGCalCluster3D.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class HGCalTriggerNtupleHGCClusters3D : public HGCalTriggerNtupleBase
{

  public:
    HGCalTriggerNtupleHGCClusters3D(const edm::ParameterSet& conf);
    ~HGCalTriggerNtupleHGCClusters3D(){};
    virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
    virtual void fill(const edm::Event& e, const edm::EventSetup& es) override final;

  private:
    virtual void clear() override final;


    edm::EDGetToken clusters3D_token_;

    int cl3d_n_ ;
    std::vector<float> cl3d_pt_;
    std::vector<float> cl3d_energy_;
    std::vector<float> cl3d_eta_;
    std::vector<float> cl3d_phi_;

};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    HGCalTriggerNtupleHGCClusters3D,
    "HGCalTriggerNtupleHGCClusters3D" );


HGCalTriggerNtupleHGCClusters3D::
HGCalTriggerNtupleHGCClusters3D(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleHGCClusters3D::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  clusters3D_token_ = collector.consumes<l1t::HGCalCluster3DBxCollection>(conf.getParameter<edm::InputTag>("Clusters3D"));

  tree.Branch("cl3d_n", &cl3d_n_, "cl3d_n/I");
  tree.Branch("cl3d_pt", &cl3d_pt_);
  tree.Branch("cl3d_energy", &cl3d_energy_);
  tree.Branch("cl3d_eta", &cl3d_eta_);
  tree.Branch("cl3d_phi", &cl3d_phi_);

}

void
HGCalTriggerNtupleHGCClusters3D::
fill(const edm::Event& e, const edm::EventSetup& es)
{

  // retrieve clusters 3D
  edm::Handle<l1t::HGCalCluster3DBxCollection> clusters3D_h;
  e.getByToken(clusters3D_token_, clusters3D_h);
  const l1t::HGCalCluster3DBxCollection& clusters3D = *clusters3D_h;

  // retrieve geometry
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
  es.get<IdealGeometryRecord>().get(geometry);

  clear();
  for(auto cl3d_itr=clusters3D.begin(0); cl3d_itr!=clusters3D.end(0); cl3d_itr++)
  {
    cl3d_n_++;
    // physical values 
    cl3d_pt_.emplace_back(cl3d_itr->pt());
    cl3d_energy_.emplace_back(cl3d_itr->energy());
    cl3d_eta_.emplace_back(cl3d_itr->eta());
    cl3d_phi_.emplace_back(cl3d_itr->phi());
  }
}


void
HGCalTriggerNtupleHGCClusters3D::
clear()
{
  cl3d_n_ = 0;
  cl3d_pt_.clear();
  cl3d_energy_.clear();
  cl3d_eta_.clear();
  cl3d_phi_.clear();
}




