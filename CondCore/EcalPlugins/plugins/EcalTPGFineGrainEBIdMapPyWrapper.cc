#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
//#include "CondTools/Ecal/interface/EcalTPGFineGrainEBMapXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
//#include "TROOT.h"
//#include "TH2F.h"
//#include "TCanvas.h"
//#include "TStyle.h"
//#include "TColor.h"
//#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>
//#include <algorithm>
//#include <numeric>
//#include <iterator>
//#include <boost/ref.hpp>
//#include <boost/bind.hpp>
//#include <boost/function.hpp>
//#include <boost/iterator/transform_iterator.hpp>
//
//#include <fstream>

namespace cond {
	template<>
	std::string PayLoadInspector<EcalTPGFineGrainEBIdMap>::summary() const {
		std::stringstream ss;

		EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMapItr it;
		const EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMap map= object().getMap();
		uint32_t ThresholdETLow, ThresholdETHigh, RatioLow, RatioHigh, LUT;
		ss<<std::endl;
		for (it=map.begin();it!=map.end();++it) {
			ss <<"FG "<<(*it).first<<std::endl;
			(*it).second.getValues(ThresholdETLow, ThresholdETHigh, RatioLow, RatioHigh, LUT);
			ss <<std::hex<<"0x"<<ThresholdETLow<<" 0x"<<ThresholdETHigh<<" 0x"<<RatioLow<<" 0x"<<RatioHigh<<" 0x"<<LUT<<std::endl;
		}
		return ss.str();
	}
}
PYTHON_WRAPPER(EcalTPGFineGrainEBIdMap,EcalTPGFineGrainEBIdMap);
