#include "EgammaAnalysis/ElectronTools/interface/VersionedGsfElectronSelector.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"

VersionedGsfElectronSelector::
VersionedGsfElectronSelector( edm::ParameterSet const & parameters ):
  VersionedSelector<reco::GsfElectronRef>(parameters),
  initialized_(false) {
  initialize(parameters);  
  retInternal_ = getBitTemplate();
}
  
void VersionedGsfElectronSelector::
initialize( const edm::ParameterSet& conf ) {
  if(initialized_) {
    edm::LogWarning("VersionedGsfElectronSelector")
      << "ID was already initialized!";
    return;
  }
  
  const std::vector<edm::ParameterSet>& cutflow =
    conf.getParameterSetVector("cutFlow");
  
  if( cutflow.size() == 0 ) {
    throw cms::Exception("InvalidCutFlow")
      << "You have supplied a null/empty cutflow to VersionedIDSelector,"
      << " please add content to the cuflow and try again.";
  }

  // this lets us keep track of cuts without knowing what they are :D
  for( const auto& cut : cutflow ) {
   
    const std::string& name = cut.getParameter<std::string>("cutName");
    const bool needsContent = cut.getParameter<bool>("needsAdditionalProducts");     
    const bool ignored = cut.getParameter<bool>("isIgnored");
    auto* plugin = CutApplicatorFactory::get()->create(name,cut);
    if( plugin != nullptr ) {
      cuts_.emplace_back(plugin);    
    } else {
      throw cms::Exception("BadPluginName")
	<< "The requested cut: " << name << " is not available!";
    }
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
operator()(const reco::GsfElectronRef & electron,pat::strbitset & ret ) { 
  howfar_ = 0;
  bool failed = false;
  if( !initialized_ ) {
    throw cms::Exception("CutNotInitialized")
      << "VersionedGsfElectronSelector not initialized!" << std::endl;
  }  
  for( unsigned i = 0; i < cuts_.size(); ++i ) {
     reco::CandidateBaseRef temp(electron);
    const bool result = (*cuts_[i])(temp);
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
operator()(const reco::GsfElectronRef & electron,
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
