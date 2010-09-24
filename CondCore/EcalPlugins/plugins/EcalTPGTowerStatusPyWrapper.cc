#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
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
	std::string PayLoadInspector<EcalTPGTowerStatus>::summary() const {
		std::stringstream ss;
		int ieta = 0;
		int iphi = 0;

		const EcalTPGTowerStatusMap & badTTMap = object().getMap();
		EcalTPGTowerStatusMapIterator it;

		ss <<"Barrel and endcap masked Trigger Towers"<<std::endl;
		ss <<"RawId " << "     iphi " << "  ieta " << std::endl;
		ss <<""<< std::endl;

		for (it=badTTMap.begin();it!=badTTMap.end();++it) {

			// Print in the text file only the masked barrel and endcap TTs
			if ((*it).second != 0){
				EcalTrigTowerDetId  ttId((*it).first);
				ieta = ttId.ieta();
				iphi = ttId.iphi();
				ss <<""<< std::dec<<(*it).first << "  " << iphi << "     " << ieta << std::endl;
			}
		}
		return ss.str();
	}
}
PYTHON_WRAPPER(EcalTPGTowerStatus,EcalTPGTowerStatus);
