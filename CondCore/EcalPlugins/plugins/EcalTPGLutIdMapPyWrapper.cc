#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
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
	std::string PayLoadInspector<EcalTPGLutIdMap>::summary() const {
		std::stringstream ss;
		EcalTPGLutIdMap::EcalTPGLutMapItr it;
		const EcalTPGLutIdMap::EcalTPGLutMap map=object().getMap();

		ss<<std::endl;
		for (it=map.begin();it!=map.end();++it) {
			ss <<"LUT "<<(*it).first<<std::endl;
			const unsigned int * lut=(*it).second.getLut();
			for (unsigned int i=0;i<1024;++i)  ss <<std::hex<<"0x"<<*lut++<<std::endl;
		}

		return ss.str();
	}
}
PYTHON_WRAPPER(EcalTPGLutIdMap,EcalTPGLutIdMap);
