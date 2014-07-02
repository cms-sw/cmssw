#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"

// NOTE! This should come *after* and importers that bring in super clusters
// of their own (like electron seeds or photons)
// otherwise ECAL <-> ECAL linking will not work correctly

class ECALClusterImporter : public BlockElementImporterBase {
public:
  ECALClusterImporter(const edm::ParameterSet& conf,
		      edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _src(sumes.consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("source"))),
    _assoc(sumes.consumes<edm::ValueMap<reco::CaloClusterPtr> >(conf.getParameter<edm::InputTag>("BCtoPFCMap"))) {}
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  edm::EDGetTokenT<reco::PFClusterCollection> _src;
  edm::EDGetTokenT<edm::ValueMap<reco::CaloClusterPtr> > _assoc;
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  ECALClusterImporter, 
		  "ECALClusterImporter");

void ECALClusterImporter::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  BlockElementImporterBase::ElementList ecals;
  edm::Handle<reco::PFClusterCollection> clusters;
  edm::Handle<edm::ValueMap<reco::CaloClusterPtr> > assoc;
  e.getByToken(_src,clusters);
  e.getByToken(_assoc,assoc);
  auto bclus = clusters->cbegin();
  auto eclus = clusters->cend();
  // get all the SCs in the element list
  auto sc_end = std::partition(elems.begin(),elems.end(),
			       [](const ElementList::value_type& o){
				 return o->type() == reco::PFBlockElement::SC;
			       });   
  ecals.reserve(clusters->size());
  for( auto clus = bclus; clus != eclus; ++clus  ) {    
    reco::PFClusterRef tempref(clusters, std::distance(bclus,clus));
    reco::PFBlockElementCluster* newelem = 
      new reco::PFBlockElementCluster(tempref,reco::PFBlockElement::ECAL);
    for( auto scelem = elems.begin(); scelem != sc_end; ++scelem ) {
      const reco::PFBlockElementSuperCluster* elem_as_sc =
	static_cast<const reco::PFBlockElementSuperCluster*>(scelem->get());
      const reco::SuperClusterRef& this_sc = elem_as_sc->superClusterRef();
      const bool in_sc = ( elem_as_sc->fromPFSuperCluster() ?
			   // use association map if from PFSC
			   ClusterClusterMapping::overlap(tempref,
							  *this_sc,
							  *assoc) :
			   // match by overlapping rechit otherwise
			   ClusterClusterMapping::overlap(*tempref,
							  *this_sc) );
      if( in_sc ) {	
	newelem->setSuperClusterRef(this_sc);
	break;
      }
    }
    ecals.emplace_back(newelem);
  }
  elems.reserve(elems.size()+ecals.size());
  for( auto& ecal : ecals ) {
    elems.emplace_back(ecal.release());
  }
}
