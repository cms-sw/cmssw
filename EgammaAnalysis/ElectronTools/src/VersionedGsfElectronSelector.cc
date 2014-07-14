#include "EgammaAnalysis/ElectronTools/interface/VersionedGsfElectronSelector.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"

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
    const bool needsContent = cut.getParameter<bool>("needsAdditionalProducts");    
    const bool ignored = cut.getParameter<bool>("isIgnored");
    cuts_.emplace_back(CutApplicatorFactory::get()->create(name,cut));    
    needs_event_content_.push_back(needsContent);
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
  howfar_ = 0;
  bool failed = false;
  if( !initialized_ ) {
    throw cms::Exception("CutNotInitialized")
      << "VersionedGsfElectronSelector not initialized!" << std::endl;
  }  
  for( unsigned i = 0; i < cuts_.size(); ++i ) {
    const bool result = (*cuts_[i])(electron);
    if( result || ignoreCut(cut_indices_[i]) ) {
      passCut(ret,cut_indices_[i]);
      if( !failed ) ++howfar_;
    } else {
      failed = true;
    }
  }
  setIgnored(ret);
  return (bool)ret;
}

bool VersionedGsfElectronSelector::
operator()(const reco::GsfElectron & electron,
	   edm::EventBase const & e,
	   pat::strbitset & ret) {
  // setup isolation needs
  for( size_t i = 0, cutssize = cuts_.size(); i < cutssize; ++i ) {
    if( needs_event_content_[i] ) {
      CutApplicatorWithEventContentBase* needsEvent = 
	static_cast<CutApplicatorWithEventContentBase*>(cuts_[i].get());
      needsEvent->getEventContent(e);      
    }
  }
  return operator()(electron, ret);
}
