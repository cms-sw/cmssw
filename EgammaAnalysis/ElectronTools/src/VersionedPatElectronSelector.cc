#include "EgammaAnalysis/ElectronTools/interface/VersionedPatElectronSelector.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

VersionedPatElectronSelector::
VersionedPatElectronSelector( edm::ParameterSet const & parameters ):
  VersionedSelector<pat::Electron>(parameters) {
  initialize(parameters);  
  retInternal_ = getBitTemplate();
}
  
void VersionedPatElectronSelector::
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

bool VersionedPatElectronSelector::
operator()(const pat::Electron & electron,pat::strbitset & ret ) {
  howfar_ = 0;
  bool failed = false;
  if( !initialized_ ) {
    throw cms::Exception("CutNotInitialized")
      << "VersionedPatElectronSelector not initialized!" << std::endl;
  }  
  for( unsigned i = 0; i < cuts_.size(); ++i ) {
    const bool result = (*cuts_[i])(electron);
    if( result || ignoreCut(cut_indices_[i]) ) {
      passCut(ret,cut_indices_[i]);
      if( !failed) ++howfar_;
    } else {
      failed = true;
    }
  }
  setIgnored(ret);
  return (bool)ret;
}

bool VersionedPatElectronSelector::
operator()(const pat::Electron & electron,
	   edm::EventBase const & e,
	   pat::strbitset & ret) {
  // setup isolation needs
  for( size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i ) {
    if( is_isolation_[i] ) {
      IsolationCutApplicatorBase* asIso = 
	static_cast<IsolationCutApplicatorBase*>(cuts_[i].get());
      asIso->setIsolationValuesFromEvent(e);
    }
  }
  return operator()(electron, ret);
}
