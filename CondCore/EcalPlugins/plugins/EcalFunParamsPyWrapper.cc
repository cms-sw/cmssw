#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondTools/Ecal/interface/EcalClusterEnergyCorrectionXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalClusterCrackCorrXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalClusterLocalContCorrXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalClusterEnergyCorrectionObjectSpecificXMLTranslator.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"

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

namespace cond {
  template<>
  std::string PayLoadInspector<EcalFunParams>::dump() const {
    std::cout << "EcalFunParamsPyWrapper dump " << std::endl;
    //    std::cout << " token " << object().token() << "\n";
    //    PayLoadInspector::dump();
    std::cout << " Collection size " << object().params().size() << "\n";
    //    for ( EcalFunctionParameters::const_iterator it = object().params().begin(); it != object().params().end(); ++it ) {
    //      std::cout << " " << *it;
    //    }
    //    std::cout << "\n";
    std::stringstream ss;
    EcalCondHeader header;
    if(object().params().size() == 56)
      ss << EcalClusterEnergyCorrectionXMLTranslator::dumpXML(header,object());
    else if(object().params().size() == 20)
      ss << EcalClusterCrackCorrXMLTranslator::dumpXML(header,object());
    else if(object().params().size() == 11 || object().params().size() == 24)
      ss << EcalClusterLocalContCorrXMLTranslator::dumpXML(header,object());
    else if(object().params().size() == 208)
      ss << EcalClusterEnergyCorrectionObjectSpecificXMLTranslator::dumpXML(header,object());
    else
      ss << " EcalFunParamsPyWrapper dump : unknown tag. Please send a mail to jean.fay@cern.ch";
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalFunParams>::summary() const {
    std::cout << "EcalFunParamsPyWrapper summary " << std::endl;
    std::stringstream ss;
    ss << "EcalFunParamsPyWrapper nb of parameters : " << object().params().size();

    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalFunParams>::plot(std::string const & filename,
						    std::string const &, 
						    std::vector<int> const&, 
						    std::vector<float> const& ) const {
    std::cout << "EcalFunParamsPyWrapper plot " << std::endl;
    return filename;
  }  // plot
}
PYTHON_WRAPPER(EcalFunParams,EcalFunParams);
