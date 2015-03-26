#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Common/interface/ValueMap.h"


class PatElectronEAIsoCorrectionProducer : public edm::EDProducer
{
  public:
    explicit PatElectronEAIsoCorrectionProducer( const edm::ParameterSet& iConfig );
    virtual ~PatElectronEAIsoCorrectionProducer() {};
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

  private:
    edm::EDGetTokenT< pat::ElectronCollection > patElectronsToken_;
    edm::EDGetTokenT< edm::ValueMap< double > > eaIsolatorToken_;
};

PatElectronEAIsoCorrectionProducer::PatElectronEAIsoCorrectionProducer( const edm::ParameterSet& iConfig )
: patElectronsToken_( consumes< pat::ElectronCollection >( iConfig.getParameter< edm::InputTag >( "patElectrons" ) ) )
, eaIsolatorToken_( consumes< edm::ValueMap< double > >( iConfig.getParameter< edm::InputTag >( "eaIsolator" ) ) )
{
  produces< pat::ElectronCollection >();
}

void PatElectronEAIsoCorrectionProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::Handle< pat::ElectronCollection > patElectrons;
  iEvent.getByToken( patElectronsToken_, patElectrons );
  edm::Handle<  edm::ValueMap< double > > eaIsolator;
  iEvent.getByToken( eaIsolatorToken_, eaIsolator );

  std::auto_ptr< pat::ElectronCollection > updatedPatElectrons( new pat::ElectronCollection );

  for ( size_t iElectron = 0; iElectron < patElectrons->size(); ++iElectron ) {
    pat::Electron* updatedPatElectron = patElectrons->at( iElectron ).clone();
    pat::ElectronRef electronRef( patElectrons, iElectron );
    updatedPatElectron->setIsolation( pat::User1Iso, ( *eaIsolator )[electronRef] ); // FIXME: hard-coded isolation key
    updatedPatElectrons->push_back( *updatedPatElectron );
  }

  iEvent.put( updatedPatElectrons );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatElectronEAIsoCorrectionProducer );
