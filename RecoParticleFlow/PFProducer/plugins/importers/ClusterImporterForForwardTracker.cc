#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"

class ClusterImporterForForwardTracker : public BlockElementImporterBase {
public:
  ClusterImporterForForwardTracker(const edm::ParameterSet& conf,
		    edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _src(sumes.consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("source"))) {}
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  edm::EDGetTokenT<reco::PFClusterCollection> _src;
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  ClusterImporterForForwardTracker, 
		  "ClusterImporterForForwardTracker");

void ClusterImporterForForwardTracker::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  edm::Handle<reco::PFClusterCollection> clusters;
  e.getByToken(_src,clusters);
  auto cbegin = clusters->cbegin();
  auto cend   = clusters->cend(); 
  for( auto clus = cbegin; clus != cend; ++clus ) {
    reco::PFBlockElement::Type type = reco::PFBlockElement::NONE;  
    reco::PFClusterRef cref(clusters,std::distance(cbegin,clus));
    switch( clus->layer() ) {
    case PFLayer::PS1:
      type = reco::PFBlockElement::PS1;
      break;
    case PFLayer::PS2:
      type = reco::PFBlockElement::PS2;
      break;
    case PFLayer::ECAL_BARREL:
    case PFLayer::ECAL_ENDCAP:
    case PFLayer::HF_EM:
      type = reco::PFBlockElement::ECAL;
      break;
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_ENDCAP:
    case PFLayer::HF_HAD:
      type = reco::PFBlockElement::HCAL;
      break;
    case PFLayer::HCAL_BARREL2:
      type = reco::PFBlockElement::HO;
      break;
    default:
      throw cms::Exception("InvalidPFLayer")
	<< "Layer given, " << clus->layer() << " is not a valid PFLayer!";
    }

    //Outside tracker call it HF
    if (type == reco::PFBlockElement::ECAL && 
	clus->layer()== PFLayer::HF_EM && abs(clus->position().Eta())>3.8)
            type = reco::PFBlockElement::HFEM;
    if (type == reco::PFBlockElement::HCAL && 
	clus->layer()== PFLayer::HF_HAD && abs(clus->position().Eta())>3.8)
            type = reco::PFBlockElement::HFHAD;




    reco::PFBlockElement* cptr = new reco::PFBlockElementCluster(cref,type);
    elems.emplace_back(cptr);
  }
}
