#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "DataFormats/Common/interface/Wrapper.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "PhysicsTools/Utilities/interface/Lumi3DReWeighting.h"


namespace {
  struct dictionary {

    reweight::PoissonMeanShifter a;
    edm::LumiReWeighting b;
    edm::Lumi3DReWeighting c;

    
  };

}
