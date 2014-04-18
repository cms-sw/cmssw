#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"

#include <unordered_map>

class SuperClusterImporter : public BlockElementImporterBase {
public:  
  SuperClusterImporter(const edm::ParameterSet&,edm::ConsumesCollector&);
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  edm::EDGetTokenT<reco::SuperClusterCollection> _srcEB,_srcEE;  
  bool _superClustersArePF;
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  SuperClusterImporter, 
		  "SuperClusterImporter");

SuperClusterImporter::SuperClusterImporter(const edm::ParameterSet& conf,
				   edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _srcEB(sumes.consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("source_eb"))),
    _srcEE(sumes.consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("source_ee"))),
    _superClustersArePF(conf.getParameter<bool>("superClustersArePF")) {  
}

void SuperClusterImporter::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  typedef BlockElementImporterBase::ElementList::value_type ElementType;  
  edm::Handle<reco::SuperClusterCollection> eb_scs;
  e.getByToken(_srcEB,eb_scs);
  edm::Handle<reco::SuperClusterCollection> ee_scs;
  e.getByToken(_srcEE,ee_scs);
  elems.reserve(elems.size()+eb_scs->size()+ee_scs->size());
  // setup our elements so that all the SCs are grouped together
  auto SCs_end = std::partition(elems.begin(),elems.end(),
				[](const ElementType& a){
				  return a->type() == reco::PFBlockElement::SC;
				});  
  // add eb superclusters
  auto bsc = eb_scs->cbegin();
  auto esc = eb_scs->cend();
  reco::PFBlockElementSuperCluster* scbe = NULL;
  reco::SuperClusterRef scref;
  for( auto sc = bsc; sc != esc; ++sc ) {
    scref = reco::SuperClusterRef(eb_scs,std::distance(bsc,sc));
    PFBlockElementSCEqual myEqual(scref);
    auto sc_elem = std::find_if(elems.begin(),SCs_end,myEqual);
    if( sc_elem == SCs_end ) {	
      scbe = new reco::PFBlockElementSuperCluster(scref);      
      //scbe->setFromPFSuperCluster(_superClustersArePF);
      SCs_end = elems.insert(SCs_end,ElementType(scbe));
      ++SCs_end; // point to element *after* the new one
    }    
  }// loop on eb superclusters
  // add ee superclusters
  bsc = ee_scs->cbegin();
  esc = ee_scs->cend();
  for( auto sc = bsc; sc != esc; ++sc ) {
    scref = reco::SuperClusterRef(ee_scs,std::distance(bsc,sc));
    PFBlockElementSCEqual myEqual(scref);
    auto sc_elem = std::find_if(elems.begin(),SCs_end,myEqual);
    if( sc_elem == SCs_end ) {	
      scbe = new reco::PFBlockElementSuperCluster(scref);  
      //scbe->setFromPFSuperCluster(_superClustersArePF);
      SCs_end = elems.insert(SCs_end,ElementType(scbe));
      ++SCs_end; // point to element *after* the new one
    }    
  }// loop on ee superclusters
  elems.shrink_to_fit();
}
