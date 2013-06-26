#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondTools/Ecal/interface/EcalTimeOffsetXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TPave.h"
#include "TPaveStats.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {
  template<>
  std::string PayLoadInspector<EcalTimeOffsetConstant>::dump() const {
    std::stringstream ss;
    EcalCondHeader header;
    ss << EcalTimeOffsetXMLTranslator::dumpXML(header,object());
    return ss.str();
  }
  
  template<>
  std::string PayLoadInspector<EcalTimeOffsetConstant>::summary() const {
    std::stringstream ss;
    ss <<" Barrel and endcap Time Offset" << std::endl;
    ss << " EB " << object().getEBValue()
       << " EE " << object().getEEValue() << std::endl;
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<EcalTimeOffsetConstant>::plot(std::string const & filename,
							     std::string const &, 
							     std::vector<int> const&, 
							     std::vector<float> const& ) const {
    return filename;
  }
}

PYTHON_WRAPPER(EcalTimeOffsetConstant,EcalTimeOffsetConstant);
