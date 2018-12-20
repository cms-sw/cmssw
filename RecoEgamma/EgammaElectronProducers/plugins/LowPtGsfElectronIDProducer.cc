#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronIDProducer.h"

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronIDProducer::LowPtGsfElectronIDProducer( const edm::ParameterSet& conf,
							const lowptgsfeleid::HeavyObjectCache* ) :
  gsfElectrons_(consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("electrons"))),
  rho_(consumes<double>(conf.getParameter<edm::InputTag>("rho"))),
  names_(conf.getParameter< std::vector<std::string> >("ModelNames")),
  passThrough_(conf.getParameter<bool>("PassThrough")),
  minPtThreshold_(conf.getParameter<double>("MinPtThreshold")),
  maxPtThreshold_(conf.getParameter<double>("MaxPtThreshold"))
{
  for ( const auto& name : names_ ) {
    produces< edm::ValueMap<float> >(name);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronIDProducer::~LowPtGsfElectronIDProducer() {}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::produce( edm::Event& event, const edm::EventSetup& setup ) {

  // Pileup
  edm::Handle<double> rho;
  event.getByToken(rho_,rho);
  if ( !rho.isValid() ) { edm::LogError("Problem with rho handle"); }

  // Retrieve GsfElectrons from Event
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  event.getByToken(gsfElectrons_,gsfElectrons);
  if ( !gsfElectrons.isValid() ) { edm::LogError("Problem with gsfElectrons handle"); }

  // Iterate through Electrons, evaluate BDT, and store result
  std::vector< std::vector<float> > output;
  for ( unsigned int iname = 0; iname < names_.size(); ++iname ) { 
    output.push_back( std::vector<float>(gsfElectrons->size(),-999.) );
  }
  for ( unsigned int iele = 0; iele < gsfElectrons->size(); iele++ ) {
    reco::GsfElectronRef ele(gsfElectrons,iele);
    //if ( !passThrough_ && ( ele->pt() < minPtThreshold_ ) ) { continue; }
    for ( unsigned int iname = 0; iname < names_.size(); ++iname ) {
      output[iname][iele] = globalCache()->eval( names_[iname], ele, *rho );
    }
  }
  
  // Create and put ValueMap in Event
  for ( unsigned int iname = 0; iname < names_.size(); ++iname ) { 
    auto ptr = std::make_unique< edm::ValueMap<float> >( edm::ValueMap<float>() );
    edm::ValueMap<float>::Filler filler(*ptr);
    filler.insert(gsfElectrons, output[iname].begin(), output[iname].end());
    filler.fill();
    reco::GsfElectronRef ele(gsfElectrons,0);
    event.put(std::move(ptr),names_[iname]);
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electrons",edm::InputTag("lowPtGsfElectrons"));
  desc.add<edm::InputTag>("rho",edm::InputTag("fixedGridRhoFastjetAllTmp"));
  desc.add< std::vector<std::string> >("ModelNames",std::vector<std::string>({""}));
  desc.add< std::vector<std::string> >("ModelWeights",std::vector<std::string>({"RecoEgamma/ElectronIdentification/data/LowPtElectrons/mva_id.xml.gz"}));
  desc.add< std::vector<double> >("ModelThresholds",std::vector<double>({-1.}));
  desc.add<bool>("PassThrough",false);
  desc.add<double>("MinPtThreshold",0.5);
  desc.add<double>("MaxPtThreshold",15.);
  descriptions.add("lowPtGsfElectronID",desc);
}
