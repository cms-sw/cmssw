#include "DataFormats/Common/interface/Wrapper.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfoProduct.h"

namespace {
	namespace {
		edm::Wrapper<LHERunInfoProduct>	wcommon;
		edm::Wrapper<LHEEventProduct>	wevent;
	}
}
