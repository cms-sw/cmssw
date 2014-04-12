#ifndef GsfElectronCoreProducer_h
#define GsfElectronCoreProducer_h

#include "GsfElectronCoreBaseProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class GsfElectronCoreProducer : public GsfElectronCoreBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronCoreProducer( const edm::ParameterSet & ) ;
    virtual ~GsfElectronCoreProducer() ;
    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

  private:

    edm::EDGetTokenT<reco::GsfElectronCoreCollection> edCoresTag_ ;
    edm::EDGetTokenT<reco::GsfElectronCoreCollection> pfCoresTag_ ;
//    edm::InputTag pfSuperClustersTag_ ;
//    edm::InputTag pfSuperClusterTrackMapTag_ ;

    edm::Handle<reco::GsfElectronCoreCollection> edCoresH_ ;
    edm::Handle<reco::GsfElectronCoreCollection> pfCoresH_ ;
//    edm::Handle<reco::SuperClusterCollection> pfClustersH_ ;
//    edm::Handle<edm::ValueMap<reco::SuperClusterRef> > pfClusterTracksH_ ;

    void produceTrackerDrivenCore( const reco::GsfTrackRef & gsfTrackRef, std::list<reco::GsfElectronCore *> & electrons ) ;

 } ;

#endif
