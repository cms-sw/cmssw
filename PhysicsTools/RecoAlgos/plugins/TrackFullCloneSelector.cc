#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/RecoAlgos/interface/TrackFullCloneSelectorBase.h"
#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"

namespace reco { 
  namespace modules {

      template <class ObjType> class StringCutObjectSelectorWithEvent {
          public:
              StringCutObjectSelectorWithEvent<ObjType>(const edm::ParameterSet &cfg) : sel_(cfg.template getParameter<std::string>("cut")) { }
              bool operator()( const ObjType & t, const edm::Event &evt ) const {
                  return sel_(t);
              }
          private:
              StringCutObjectSelector<ObjType> sel_;
      };

      typedef TrackFullCloneSelectorBase<
          reco::modules::StringCutObjectSelectorWithEvent<reco::Track>
          > TrackFullCloneSelector;

      DEFINE_FWK_MODULE(TrackFullCloneSelector);
  } }
