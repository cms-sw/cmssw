#include "EgammaAnalysis/ElectronTools/interface/VersionedGsfElectronSelector.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

VersionedGsfElectronSelector::
VersionedGsfElectronSelector( edm::ParameterSet const & parameters ):
  VersionedSelector<reco::GsfElectron>(parameters) {
  initialize(parameters);  
  retInternal_ = getBitTemplate();
}
  
void VersionedGsfElectronSelector::
initialize( const edm::ParameterSet& conf ) {
  if(initialized_) return;
  
  const std::vector<edm::ParameterSet>& cutflow =
    conf.getParameterSetVector("cutFlow");
  
  // this lets us keep track of cuts without knowing what they are :D
  for( const auto& cut : cutflow ) {
    const std::string& name = cut.getParameter<std::string>("cutName");
    const bool isIso = cut.getParameter<bool>("isIsolation");    
    const bool ignored = cut.getParameter<bool>("isIgnored");
    cuts_.emplace_back(CutApplicatorFactory::get()->create(name,cut));
    is_isolation_.push_back(isIso);
    push_back(name);
    set(name);
    if(ignored) ignoreCut(name);
  }  
  
  //have to loop again to set cut indices after all are filled
  for( const auto& cut: cutflow ) {
    const std::string& name = cut.getParameter<std::string>("cutName");
    cut_indices_.emplace_back(&bits_,name);
  }

  initialized_ = true;
}

bool VersionedGsfElectronSelector::
operator()(const reco::GsfElectron & electron,pat::strbitset & ret ) {
  if( !initialized_ ) {
    throw cms::Exception("CutNotInitialized")
      << "VersionedGsfElectronSelector not initialized!" << std::endl;
  }
  
  for( unsigned i = 0; i < cuts_.size(); ++i ) {
    const bool result = (*cuts_[i])(electron);
    if( result || ignoreCut(cut_indices_[i]) ) passCut(ret,cut_indices_[i]);
  }
  setIgnored(ret);

  return (bool)ret;
}
