#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronSeedValueMapsProducer.h"

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSeedValueMapsProducer::LowPtGsfElectronSeedValueMapsProducer( const edm::ParameterSet& conf ) :
  gsfElectrons_(consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("electrons"))),
  preIdsValueMap_(consumes< edm::ValueMap<reco::PreIdRef> >(conf.getParameter<edm::InputTag>("preIdsValueMap"))),
  names_(conf.getParameter< std::vector<std::string> >("ModelNames"))
{
  for ( const auto& name : names_ ) { produces< edm::ValueMap<float> >(name); }
}

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSeedValueMapsProducer::~LowPtGsfElectronSeedValueMapsProducer() {}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedValueMapsProducer::produce( edm::Event& event, const edm::EventSetup& setup ) {

  // Retrieve GsfElectrons from Event
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  event.getByToken(gsfElectrons_,gsfElectrons);
  if ( !gsfElectrons.isValid() ) { edm::LogError("Problem with gsfElectrons handle"); }

  // Retrieve PreIds from Event
  edm::Handle< edm::ValueMap<reco::PreIdRef> > preIdsValueMap;
  event.getByToken(preIdsValueMap_,preIdsValueMap);
  if ( !preIdsValueMap.isValid() ) { edm::LogError("Problem with preIdsValueMap handle"); }
  
  // Iterate through Electrons, extract BDT output, and store result in ValueMap for each model
  std::vector< std::vector<float> > output;
  for ( unsigned int iname = 0; iname < names_.size(); ++iname ) { 
    output.push_back( std::vector<float>(gsfElectrons->size(),-999.) );
  }
  for ( unsigned int iele = 0; iele < gsfElectrons->size(); iele++ ) {
    reco::GsfElectronRef ele(gsfElectrons,iele);
    if ( ele->core().isNonnull() && 
	 ele->core()->gsfTrack().isNonnull() && 
	 ele->core()->gsfTrack()->extra().isNonnull() && 
	 ele->core()->gsfTrack()->extra()->seedRef().isNonnull() ) {
      reco::ElectronSeedRef seed = ele->core()->gsfTrack()->extra()->seedRef().castTo<reco::ElectronSeedRef>();
      if ( seed.isNonnull() ) {
	const reco::PreIdRef preid = (*preIdsValueMap)[seed];
	if ( preid.isNonnull() ) {
	  for ( unsigned int iname = 0; iname < names_.size(); ++iname ) {
	    output[iname][iele] = preid->mva(iname);
	  }
	}
      }
    }
  }
  
  // Create and put ValueMap in Event
  for ( unsigned int iname = 0; iname < names_.size(); ++iname ) {
    auto ptr = std::make_unique< edm::ValueMap<float> >( edm::ValueMap<float>() );
    edm::ValueMap<float>::Filler filler(*ptr);
    filler.insert(gsfElectrons, output[iname].begin(), output[iname].end());
    filler.fill();
    event.put(std::move(ptr),names_[iname]);
  }

}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedValueMapsProducer::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electrons",edm::InputTag("lowPtGsfElectrons"));
  desc.add<edm::InputTag>("preIdsValueMap",edm::InputTag("lowPtGsfElectronSeeds"));
  desc.add< std::vector<std::string> >("ModelNames",std::vector<std::string>({"default"}));
  descriptions.add("lowPtGsfElectronValueMap",desc);
}
