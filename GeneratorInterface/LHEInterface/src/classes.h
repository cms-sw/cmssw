#include "DataFormats/Common/interface/Wrapper.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommonProduct.h"

namespace {
	namespace {
		edm::Wrapper<LHECommonProduct>	wcommon;
		edm::Wrapper<LHEEventProduct>	wevent;
	}
}
