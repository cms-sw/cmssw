#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"

class ElectronIsolatorFromEffectiveArea : public edm::EDFilter {

 public:
  typedef edm::ValueMap<double> CandDoubleMap;
  typedef ElectronEffectiveArea EEA;
  explicit ElectronIsolatorFromEffectiveArea(const edm::ParameterSet&);

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectronToken;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfElectronToken;
  edm::EDGetTokenT<double> rhoIsoToken;
  const EEA::ElectronEffectiveAreaType modeEEA;
  const EEA::ElectronEffectiveAreaTarget targetEEA;
  static std::map<std::string,EEA::ElectronEffectiveAreaType> EEA_type();
  static std::map<std::string,EEA::ElectronEffectiveAreaTarget> EEA_target();
};



ElectronIsolatorFromEffectiveArea::
ElectronIsolatorFromEffectiveArea(const edm::ParameterSet& config)
  : gsfElectronToken( consumes<reco::GsfElectronCollection>( config.getParameter<edm::InputTag>("gsfElectrons") ) )
  , pfElectronToken( consumes<reco::PFCandidateCollection>( config.getParameter<edm::InputTag>("pfElectrons") ) )
  , rhoIsoToken( consumes<double>( config.getParameter<edm::InputTag>("rhoIso") ) )
  , modeEEA( EEA_type()[ config.getParameter<std::string>("EffectiveAreaType") ] )
  , targetEEA( EEA_target()[ config.getParameter<std::string>("EffectiveAreaTarget") ] )
{  produces<CandDoubleMap>(); }



bool ElectronIsolatorFromEffectiveArea::
filter(edm::Event& event, const edm::EventSetup& )
{
  std::auto_ptr<CandDoubleMap> product(new CandDoubleMap());
  CandDoubleMap::Filler filler(*product);

  edm::Handle<double> rho;                               event.getByToken(rhoIsoToken, rho);
  edm::Handle<reco::GsfElectronCollection> gsfElectrons; event.getByToken(gsfElectronToken,gsfElectrons);
  edm::Handle<reco::PFCandidateCollection>  pfElectrons; event.getByToken( pfElectronToken, pfElectrons);
  std::vector<double> gsfCorrectionsEA,pfCorrectionsEA;

  if(gsfElectrons.isValid()) {
    for ( reco::GsfElectronCollection::const_iterator it = gsfElectrons->begin(); it != gsfElectrons->end(); ++it)
      gsfCorrectionsEA.push_back( (*rho) * EEA::GetElectronEffectiveArea( modeEEA, it->superCluster()->eta(), targetEEA ) );
    filler.insert(gsfElectrons, gsfCorrectionsEA.begin(), gsfCorrectionsEA.end() );
  }

  if(pfElectrons.isValid()) {
    for ( reco::PFCandidateCollection::const_iterator it = pfElectrons->begin(); it != pfElectrons->end(); ++it)
      pfCorrectionsEA.push_back( (*rho) * EEA::GetElectronEffectiveArea( modeEEA, it->gsfElectronRef()->superCluster()->eta(), targetEEA ) );
    filler.insert( pfElectrons,  pfCorrectionsEA.begin(),  pfCorrectionsEA.end() );
  }

  filler.fill();
  event.put(product);
  return true;
}




// These maps should really be static const members of interface/ElectronEffectiveArea.h
// Here mapping strings to only a subset of the enum
std::map<std::string,ElectronEffectiveArea::ElectronEffectiveAreaType>
ElectronIsolatorFromEffectiveArea::EEA_type()
{
  std::map<std::string,EEA::ElectronEffectiveAreaType> m;
  m["kEleGammaAndNeutralHadronIso03"] = EEA::kEleGammaAndNeutralHadronIso03;
  m["kEleGammaAndNeutralHadronIso04"] = EEA::kEleGammaAndNeutralHadronIso04;
  return m;
}

std::map<std::string,ElectronEffectiveArea::ElectronEffectiveAreaTarget>
ElectronIsolatorFromEffectiveArea::EEA_target()
{
  std::map<std::string,EEA::ElectronEffectiveAreaTarget > m;
  m["kEleEANoCorr"]     = EEA::kEleEANoCorr;
  m["kEleEAData2011"]   = EEA::kEleEAData2011;
  m["kEleEASummer11MC"] = EEA::kEleEASummer11MC;
  m["kEleEAFall11MC"]   = EEA::kEleEAFall11MC;
  m["kEleEAData2012"]   = EEA::kEleEAData2012;
  return m;
}

DEFINE_FWK_MODULE(ElectronIsolatorFromEffectiveArea);
