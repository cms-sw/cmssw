#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"

class GSFTrackImporter : public BlockElementImporterBase {
public:
  GSFTrackImporter(const edm::ParameterSet& conf,
		    edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _src(sumes.consumes<reco::GsfPFRecTrackCollection>(conf.getParameter<edm::InputTag>("source"))),
    _isSecondary(conf.getParameter<bool>("gsfsAreSecondary")),
    _superClustersArePF(conf.getParameter<bool>("superClustersArePF")){}
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  edm::EDGetTokenT<reco::GsfPFRecTrackCollection> _src;
  const bool _isSecondary, _superClustersArePF;
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  GSFTrackImporter, 
		  "GSFTrackImporter");

void GSFTrackImporter::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  typedef BlockElementImporterBase::ElementList::value_type ElementType;  
  edm::Handle<reco::GsfPFRecTrackCollection> gsftracks;
  e.getByToken(_src,gsftracks);
  elems.reserve(elems.size() + gsftracks->size());
  // setup our elements so that all the SCs are grouped together
  auto SCs_end = std::partition(elems.begin(),elems.end(),
				[](const ElementType& a){
				  return a->type() == reco::PFBlockElement::SC;
				});
  size_t SCs_end_position = std::distance(elems.begin(),SCs_end);
  // insert gsf tracks and SCs, binding pre-existing SCs to ECAL-Driven GSF
  auto bgsf = gsftracks->cbegin();
  auto egsf = gsftracks->cend();
  for( auto gsftrack =  bgsf; gsftrack != egsf; ++gsftrack ) {
    reco::GsfPFRecTrackRef gsfref(gsftracks,std::distance(bgsf,gsftrack));
    const reco::GsfTrackRef& basegsfref = gsftrack->gsfTrackRef();    
    const auto& gsfextraref = basegsfref->extra();
    // import associated super clusters
    if( gsfextraref.isAvailable() && gsfextraref->seedRef().isAvailable()) {
      reco::ElectronSeedRef seedref = 
	gsfextraref->seedRef().castTo<reco::ElectronSeedRef>();
      if( seedref.isAvailable() && seedref->isEcalDriven() ) {
	reco::SuperClusterRef scref = 
	  seedref->caloCluster().castTo<reco::SuperClusterRef>();
	if( scref.isNonnull() ) {	  
	  PFBlockElementSCEqual myEqual(scref);
	  auto sc_elem = std::find_if(elems.begin(),SCs_end,myEqual);
	  if( sc_elem != SCs_end ) {
	    reco::PFBlockElementSuperCluster* scbe = 
	      static_cast<reco::PFBlockElementSuperCluster*>(sc_elem->get());
	    scbe->setFromGsfElectron(true);
	  } else {
	    reco::PFBlockElementSuperCluster* scbe = 
	      new reco::PFBlockElementSuperCluster(scref);
	    scbe->setFromGsfElectron(true);
	    scbe->setFromPFSuperCluster(_superClustersArePF);
	    SCs_end = elems.insert(SCs_end,ElementType(scbe));
	    ++SCs_end; // point to element *after* the new one
	  }
	}	
      }
    }// gsf extra ref?
    // cache the SC_end offset
    SCs_end_position = std::distance(elems.begin(),SCs_end);
    // get track momentum information
    const std::vector<reco::PFTrajectoryPoint>& PfGsfPoint
      = gsftrack->trajectoryPoints();
    unsigned int c_gsf=0;
    bool PassTracker = false;
    bool GetPout = false;
    unsigned int IndexPout = 0;
    for(auto itPfGsfPoint =  PfGsfPoint.begin();  
	itPfGsfPoint!= PfGsfPoint.end();++itPfGsfPoint) {      
      if (itPfGsfPoint->isValid()){
	int layGsfP = itPfGsfPoint->layer();
	if (layGsfP == -1) PassTracker = true;
	if (PassTracker && layGsfP > 0 && GetPout == false) {
	  IndexPout = c_gsf-1;
	  GetPout = true;
	}
	//const math::XYZTLorentzVector GsfMoment = itPfGsfPoint->momentum();
	++c_gsf;
      }
    }
    const math::XYZTLorentzVector& pin = PfGsfPoint[0].momentum();      
    const math::XYZTLorentzVector& pout = PfGsfPoint[IndexPout].momentum();
    reco::PFBlockElementGsfTrack* temp =
      new reco::PFBlockElementGsfTrack(gsfref,pin,pout);
    if( _isSecondary ) {
      temp->setTrackType(reco::PFBlockElement::T_FROM_GAMMACONV,true);
    }
    elems.emplace_back(temp);
    // import brems from this primary gsf
    for( const auto& brem : gsfref->PFRecBrem() ) {
      const unsigned TrajP = brem.indTrajPoint();
      if( TrajP != 99 ) {
	const double DP = brem.DeltaP();
	const double sDP = brem.SigmaDeltaP();
	elems.emplace_back(new reco::PFBlockElementBrem(gsfref,DP,sDP,TrajP));
      }
    }
    // protect against reallocations, create a fresh iterator
    SCs_end = elems.begin() + SCs_end_position;
  }// loop on gsf tracks
  elems.shrink_to_fit();
}
