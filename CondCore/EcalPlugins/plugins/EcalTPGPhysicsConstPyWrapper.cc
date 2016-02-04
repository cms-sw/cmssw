#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
//#include "CondTools/Ecal/interface/EcalTPGPhysicsConstXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <fstream>

#include "CondCore/EcalPlugins/plugins/EcalPyWrapperFunctions.h"

namespace cond {

   template<>
   std::string PayLoadInspector<EcalTPGPhysicsConst>::summary() const {
	std::stringstream ss;
	EcalTPGPhysicsConstMap valuesMap = object().getMap();

	EcalTPGPhysicsConstMapIterator iValue = valuesMap.begin();
	ss << "---Barrels: " << std::endl; 
		ss << "EtSat: " << (*iValue).second.EtSat << std::endl;
		ss << "ttf_threshold_Low: " << (*iValue).second.ttf_threshold_Low << std::endl;
		ss << "ttf_threshold_High: " << (*iValue).second.ttf_threshold_High << std::endl;
		ss << "FG_lowThreshold: " << (*iValue).second.FG_lowThreshold << std::endl;
		ss << "FG_highThreshold: " << (*iValue).second.FG_highThreshold << std::endl;
		ss << "FG_lowRatio: " << (*iValue).second.FG_lowRatio << std::endl;
		ss << "FG_highRatio: " << (*iValue).second.FG_highRatio << std::endl;
	
	++iValue;
	ss << "---Endcaps: " << std::endl; 
	    ss << "EtSat: " << (*iValue).second.EtSat << std::endl;
		ss << "ttf_threshold_Low: " << (*iValue).second.ttf_threshold_Low << std::endl;
		ss << "ttf_threshold_High: " << (*iValue).second.ttf_threshold_High << std::endl;
		ss << "FG_lowThreshold: " << (*iValue).second.FG_lowThreshold << std::endl;
		ss << "FG_highThreshold: " << (*iValue).second.FG_highThreshold << std::endl;
		ss << "FG_lowRatio: " << (*iValue).second.FG_lowRatio << std::endl;
		ss << "FG_highRatio: " << (*iValue).second.FG_highRatio << std::endl;
	return ss.str();
   }
}
PYTHON_WRAPPER(EcalTPGPhysicsConst,EcalTPGPhysicsConst);
