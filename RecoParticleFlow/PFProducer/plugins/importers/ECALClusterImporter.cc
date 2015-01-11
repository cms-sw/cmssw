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

template<reco::PFBlockElement::Type the_type> 
class ECALClusterImporter : public BlockElementImporterBase {
public:
  ECALClusterImporter(const edm::ParameterSet& conf,
		      edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _src(sumes.consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("source"))) {}
     
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  edm::EDGetTokenT<reco::PFClusterCollection> _src;
};



template<reco::PFBlockElement::Type the_type> 
void ECALClusterImporter<the_type>::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  BlockElementImporterBase::ElementList ecals;
  edm::Handle<reco::PFClusterCollection> clusters;
  edm::Handle<edm::ValueMap<reco::CaloClusterPtr> > assoc;
  e.getByToken(_src,clusters);
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
      new reco::PFBlockElementCluster(tempref,the_type);
    for( auto scelem = elems.begin(); scelem != sc_end; ++scelem ) {
      const reco::PFBlockElementSuperCluster* elem_as_sc =
	static_cast<const reco::PFBlockElementSuperCluster*>(scelem->get());
      const reco::SuperClusterRef& this_sc = elem_as_sc->superClusterRef();
      const bool in_sc = ClusterClusterMapping::overlap(*tempref,
							*this_sc);
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

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  ECALClusterImporter<reco::PFBlockElement::ECAL>, 
		  "ECALClusterImporter");

#include "RecoEcal/EgammaClusterAlgos/interface/HGCALShowerBasedEmIdentification.h" 
class HGCECALClusterImporter : public BlockElementImporterBase {
public:
  HGCECALClusterImporter(const edm::ParameterSet& conf,
		      edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _src(sumes.consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("source"))),
    _emPreID( new HGCALShowerBasedEmIdentification(true) ){}
     
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  edm::EDGetTokenT<reco::PFClusterCollection> _src;
  std::unique_ptr<HGCALShowerBasedEmIdentification> _emPreID;
};

void HGCECALClusterImporter::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  BlockElementImporterBase::ElementList ecals;
  edm::Handle<reco::PFClusterCollection> clusters;
  edm::Handle<edm::ValueMap<reco::CaloClusterPtr> > assoc;
  e.getByToken(_src,clusters);
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
    reco::PFBlockElement::Type the_type = reco::PFBlockElement::NONE;
    switch( tempref->layer() ) {
    case PFLayer::HGC_ECAL:
      the_type = reco::PFBlockElement::HGC_ECAL;
      break;
    case PFLayer::HGC_HCALF:
      the_type = reco::PFBlockElement::HGC_HCALF;
      break;
    case PFLayer::HGC_HCALB:
      the_type = reco::PFBlockElement::HGC_HCALB;
      break;
    default:
      throw cms::Exception("BadInput") 
	<< "HGC importer expected HGC clusters!";
    }
    /*
    if( tempref->energy() > 10.0 && 
	the_type == reco::PFBlockElement::HGC_ECAL && 
	!_emPreID->isEm(*tempref) ) {    
      the_type = reco::PFBlockElement::HGC_HCALF;
    }
    */
    
    reco::PFBlockElementCluster* newelem = 
      new reco::PFBlockElementCluster(tempref,the_type);
    for( auto scelem = elems.begin(); scelem != sc_end; ++scelem ) {
      const reco::PFBlockElementSuperCluster* elem_as_sc =
	static_cast<const reco::PFBlockElementSuperCluster*>(scelem->get());
      const reco::SuperClusterRef& this_sc = elem_as_sc->superClusterRef();
      const bool in_sc = ClusterClusterMapping::overlap(*tempref,
							*this_sc);
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

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  HGCECALClusterImporter, 
		  "HGCECALClusterImporter");
