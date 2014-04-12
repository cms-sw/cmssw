#ifndef RecoAlgos_PhotonSelector_h
#define RecoAlgos_PhotonSelector_h
/** \class PhotonSelector
 *
 * selects a subset of an photon collection. Also clones
 * all referenced objects
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: PhotonSelector.h,v 1.1 2009/03/04 13:11:28 llista Exp $
 *
 */

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"

namespace helper {
  struct PhotonCollectionStoreManager {
    typedef reco::PhotonCollection collection;
    PhotonCollectionStoreManager(const edm::Handle<reco::PhotonCollection>&) :
      selPhotons_( new reco::PhotonCollection ),
      selSuperClusters_( new reco::SuperClusterCollection ) {
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & evt ) {
      using namespace reco;
      PhotonRefProd rPhotons = evt.template getRefBeforePut<PhotonCollection>();      
      SuperClusterRefProd rSuperClusters = evt.template getRefBeforePut<SuperClusterCollection>();      
      size_t idx = 0;
      for( I i = begin; i != end; ++ i ) {
	const Photon & ele = * * i;
	selPhotons_->push_back( Photon( ele ) );
	selPhotons_->back().setSuperCluster( SuperClusterRef( rSuperClusters, idx ++ ) );
	selSuperClusters_->push_back( SuperCluster( * ( ele.superCluster() ) ) );
      }
    }
    edm::OrphanHandle<reco::PhotonCollection> put( edm::Event & evt ) {
      edm::OrphanHandle<reco::PhotonCollection> h = evt.put( selPhotons_ );
      evt.put( selSuperClusters_ );
      return h;
    }
    size_t size() const { return selPhotons_->size(); }
  private:
    std::auto_ptr<reco::PhotonCollection> selPhotons_;
    std::auto_ptr<reco::SuperClusterCollection> selSuperClusters_;
  };

  class PhotonSelectorBase : public edm::EDFilter {
  public:
    PhotonSelectorBase( const edm::ParameterSet & cfg ) {
      std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
      produces<reco::PhotonCollection>().setBranchAlias( alias + "Photons" );
      produces<reco::SuperClusterCollection>().setBranchAlias( alias + "SuperClusters" );
    }
   };

  template<>
  struct StoreManagerTrait<reco::PhotonCollection> {
    typedef PhotonCollectionStoreManager type;
    typedef PhotonSelectorBase base;
  };

}

#endif
