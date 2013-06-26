#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondTools/Ecal/interface/EcalTPGTowerStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

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
  std::string PayLoadInspector<EcalTPGTowerStatus>::dump() const {
    std::stringstream ss;
    EcalCondHeader h;
    ss << EcalTPGTowerStatusXMLTranslator::dumpXML(h,object());
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalTPGTowerStatus>::summary() const {
    std::stringstream ss;

    const EcalTPGTowerStatusMap &towerMap = object().getMap();
    EcalTPGTowerStatusMapIterator it;
    int NbMaskedTT = 0;
    for(it = towerMap.begin(); it != towerMap.end(); ++it)
      if((*it).second > 0) NbMaskedTT++;
    ss <<"Barrel masked Trigger Towers " << NbMaskedTT <<std::endl;
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalTPGTowerStatus>::plot(std::string const & filename,
							 std::string const &, 
							 std::vector<int> const&, 
							 std::vector<float> const& ) const {
    gStyle->SetPalette(1);

    TCanvas canvas("CC map","CC map",800, 400);

    TH2F* barrel = new TH2F("EB","EB TPG Tower Status", 72, 0, 72, 34, -17, 17);

    const EcalTPGTowerStatusMap &towerMap = object().getMap();
    std::cout << " tower map size " << towerMap.size() << std::endl;
    EcalTPGTowerStatusMapIterator it;
    for(it = towerMap.begin(); it != towerMap.end(); ++it) {
      if((*it).second > 0) {
	EcalTrigTowerDetId ttId((*it).first);
	int ieta = ttId.ieta();
	if(ieta < 0) ieta--;   // 1 to 17
	int iphi = ttId.iphi() - 1;  // 0 to 71
	barrel->Fill(iphi, ieta, (*it).second);
      }
    }
    TLine* l = new TLine(0., 0., 0., 0.);
    l->SetLineWidth(1);
    canvas.cd();
    barrel->SetStats(0);
    barrel->Draw("col");
    for(int i = 0; i <17; i++) {
      Double_t x = 4.+ (i * 4);
      l = new TLine(x, -17., x, 17.);
      l->Draw();
    }
    l = new TLine(0., 0., 72., 0.);
    l->Draw();

    canvas.SaveAs(filename.c_str());
    return filename;
  }  // plot
}
PYTHON_WRAPPER(EcalTPGTowerStatus,EcalTPGTowerStatus);
