#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
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
	std::string PayLoadInspector<EcalTPGWeightIdMap>::summary() const {
		std::stringstream ss;

		ss<<std::endl;
		EcalTPGWeightIdMap::EcalTPGWeightMapItr it;
		uint32_t w0,w1,w2,w3,w4;
		const EcalTPGWeightIdMap::EcalTPGWeightMap map= object().getMap();
		for (it=map.begin();it!=map.end();++it) {
			ss <<"WEIGHT "<<(*it).first<<std::endl;
			(*it).second.getValues(w0,w1,w2,w3,w4);
			ss <<std::hex<<"0x"<<w0<<" 0x"<<w1<<" 0x"<<w2<<" 0x"<<w3<<" 0x"<<w4<<" "<<std::endl;
		}
		return ss.str();
	}
}

PYTHON_WRAPPER(EcalTPGWeightIdMap,EcalTPGWeightIdMap);
