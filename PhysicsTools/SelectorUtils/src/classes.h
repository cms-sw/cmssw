#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "PhysicsTools/SelectorUtils/interface/strbitset.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"
#include "PhysicsTools/SelectorUtils/interface/ElectronVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/MuonVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PVSelector.h"
#include "PhysicsTools/SelectorUtils/interface/RunLumiSelector.h"
#include "PhysicsTools/SelectorUtils/interface/MakePyVIDClassBuilder.h"


namespace PhysicsTools_SelectorUtils {
  struct dictionary {

    pat::strbitset strbitset;
    edm::Wrapper<pat::strbitset> wstrbitset;
    std::vector< pat::strbitset> vstrbitset;
    edm::Wrapper< std::vector< pat::strbitset> > wvstrbitset;
    
  };

}
