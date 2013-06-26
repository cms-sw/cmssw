
#include "GsfElectronCoreEcalDrivenProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <map>

using namespace reco ;

// void GsfElectronCoreEcalDrivenProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
//  {
//   edm::ParameterSetDescription desc ;
//   GsfElectronCoreBaseProducer::fillDescriptions(desc) ;
//   descriptions.add("produceEcalDrivenGsfElectronCores",desc) ;
//  }

GsfElectronCoreEcalDrivenProducer::GsfElectronCoreEcalDrivenProducer( const edm::ParameterSet & config )
 : GsfElectronCoreBaseProducer(config)
 {}

void GsfElectronCoreEcalDrivenProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  // base input
  GsfElectronCoreBaseProducer::initEvent(event,setup) ;

  // output
  std::auto_ptr<GsfElectronCoreCollection> electrons(new GsfElectronCoreCollection) ;

  // loop on ecal driven tracks
  if (useGsfPfRecTracks_)
   {
    const GsfPFRecTrackCollection * gsfPfRecTrackCollection = gsfPfRecTracksH_.product() ;
    GsfPFRecTrackCollection::const_iterator gsfPfRecTrack ;
    for ( gsfPfRecTrack=gsfPfRecTrackCollection->begin() ;
          gsfPfRecTrack!=gsfPfRecTrackCollection->end() ;
          ++gsfPfRecTrack )
     {
      const GsfTrackRef gsfTrackRef = gsfPfRecTrack->gsfTrackRef() ;
      produceEcalDrivenCore(gsfTrackRef,electrons.get()) ;
     }
   }
  else
   {
    const GsfTrackCollection * gsfTrackCollection = gsfTracksH_.product() ;
    for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i )
     {
      const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH_,i) ;
      produceEcalDrivenCore(gsfTrackRef,electrons.get()) ;
     }
   }

  event.put(electrons) ;
 }

void GsfElectronCoreEcalDrivenProducer::produceEcalDrivenCore( const GsfTrackRef & gsfTrackRef, GsfElectronCoreCollection * electrons )
 {
  GsfElectronCore * eleCore = new GsfElectronCore(gsfTrackRef) ;

  if (!eleCore->ecalDrivenSeed())
   { delete eleCore ; return ; }

  GsfElectronCoreBaseProducer::fillElectronCore(eleCore) ;

  edm::RefToBase<TrajectorySeed> seed = gsfTrackRef->extra()->seedRef() ;
  ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>() ;
  edm::RefToBase<CaloCluster> caloCluster = elseed->caloCluster() ;
  SuperClusterRef scRef = caloCluster.castTo<SuperClusterRef>() ;
  if (!scRef.isNull())
   {
    eleCore->setSuperCluster(scRef) ;
    electrons->push_back(*eleCore) ;
   }
  else
   { edm::LogWarning("GsfElectronCoreEcalDrivenProducer")<<"Seed CaloCluster is not a SuperCluster, unexpected..." ; }

  delete eleCore ;
 }

GsfElectronCoreEcalDrivenProducer::~GsfElectronCoreEcalDrivenProducer()
 {}

