#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"

#include <unordered_map>

class EGPhotonImporter : public BlockElementImporterBase {
public:
  enum SelectionChoices {SeparateDetectorIso,CombinedDetectorIso};

  EGPhotonImporter(const edm::ParameterSet&,edm::ConsumesCollector&);
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  edm::EDGetTokenT<reco::PhotonCollection> _src;
  const std::unordered_map<std::string,SelectionChoices> _selectionTypes;
  SelectionChoices _selectionChoice;
  std::unique_ptr<const PhotonSelectorAlgo> _selector;
  bool _superClustersArePF;
  
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  EGPhotonImporter, 
		  "EGPhotonImporter");

EGPhotonImporter::EGPhotonImporter(const edm::ParameterSet& conf,
				   edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _src(sumes.consumes<reco::PhotonCollection>(conf.getParameter<edm::InputTag>("source"))),
    _selectionTypes({ {"SeparateDetectorIso",EGPhotonImporter::SeparateDetectorIso},
	  {"CombinedDetectorIso",EGPhotonImporter::CombinedDetectorIso} }),
    _superClustersArePF(conf.getParameter<bool>("superClustersArePF")) {
  const std::string& selChoice = 
    conf.getParameter<std::string>("SelectionChoice");
  _selectionChoice = _selectionTypes.at(selChoice);
  const edm::ParameterSet& selDef = 
    conf.getParameterSet("SelectionDefinition");
  const float minEt = selDef.getParameter<double>("minEt");
  const float trackIso_const = selDef.getParameter<double>("trackIsoConstTerm");
  const float trackIso_slope = selDef.getParameter<double>("trackIsoSlopeTerm");
  const float ecalIso_const = selDef.getParameter<double>("ecalIsoConstTerm");
  const float ecalIso_slope = selDef.getParameter<double>("ecalIsoSlopeTerm");
  const float hcalIso_const = selDef.getParameter<double>("hcalIsoConstTerm");
  const float hcalIso_slope = selDef.getParameter<double>("hcalIsoSlopeTerm");
  const float hoe = selDef.getParameter<double>("HoverE");
  const float loose_hoe = selDef.getParameter<double>("LooseHoverE");
  const float combIso = selDef.getParameter<double>("combIsoConstTerm");
  _selector.reset(new PhotonSelectorAlgo((float)_selectionChoice,
					 minEt,
					 trackIso_const, trackIso_slope,
					 ecalIso_const, ecalIso_slope,
					 hcalIso_const, hcalIso_slope,
					 hoe,
					 combIso,
					 loose_hoe));
}

void EGPhotonImporter::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  typedef BlockElementImporterBase::ElementList::value_type ElementType;  
  edm::Handle<reco::PhotonCollection> photons;
  e.getByToken(_src,photons);
  elems.reserve(elems.size()+photons->size());
  // setup our elements so that all the SCs are grouped together
  auto SCs_end = std::partition(elems.begin(),elems.end(),
				[](const ElementType& a){
				  return a->type() == reco::PFBlockElement::SC;
				});  
  //now add the photons
  auto bphoton = photons->cbegin();
  auto ephoton = photons->cend();
  reco::PFBlockElementSuperCluster* scbe = nullptr;
  reco::PhotonRef phoref;
  for( auto photon = bphoton; photon != ephoton; ++photon ) {
    if( _selector->passPhotonSelection(*photon) ) {
      phoref = reco::PhotonRef(photons,std::distance(bphoton,photon));
      const reco::SuperClusterRef& scref = photon->superCluster();
      PFBlockElementSCEqual myEqual(scref);
      auto sc_elem = std::find_if(elems.begin(),SCs_end,myEqual);
      if( sc_elem != SCs_end ) {
	scbe = static_cast<reco::PFBlockElementSuperCluster*>(sc_elem->get());
	scbe->setFromPhoton(true);
	scbe->setPhotonRef(phoref);
	scbe->setTrackIso(photon->trkSumPtHollowConeDR04());
	scbe->setEcalIso(photon->ecalRecHitSumEtConeDR04());
	scbe->setHcalIso(photon->hcalTowerSumEtConeDR04());
	scbe->setHoE(photon->hadronicOverEm());
      } else {
	scbe = new reco::PFBlockElementSuperCluster(scref);
	scbe->setFromPhoton(true);
	scbe->setFromPFSuperCluster(_superClustersArePF);
	scbe->setPhotonRef(phoref);
	scbe->setTrackIso(photon->trkSumPtHollowConeDR04());
	scbe->setEcalIso(photon->ecalRecHitSumEtConeDR04());
	scbe->setHcalIso(photon->hcalTowerSumEtConeDR04());
	scbe->setHoE(photon->hadronicOverEm());
	SCs_end = elems.insert(SCs_end,ElementType(scbe));
	++SCs_end; // point to element *after* the new one
      }
    }
  }// loop on photons
  elems.shrink_to_fit();
}
