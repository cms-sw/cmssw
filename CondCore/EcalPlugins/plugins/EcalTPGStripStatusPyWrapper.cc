#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondTools/Ecal/interface/EcalTPGStripStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

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
  std::string PayLoadInspector<EcalTPGStripStatus>::dump() const {
    std::stringstream ss;
    EcalCondHeader h;
    ss << EcalTPGStripStatusXMLTranslator::dumpXML(h,object());
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalTPGStripStatus>::summary() const {
    std::stringstream ss;
    const EcalTPGStripStatusMap &stripMap = object().getMap();
    std::cout << " tower map size " << stripMap.size() << std::endl;
    ss <<" Endcap : Number of masked Trigger Strips " << stripMap.size() << std::endl;
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalTPGStripStatus>::plot(std::string const & filename,
							 std::string const &, 
							 std::vector<int> const&, 
							 std::vector<float> const& ) const {
    gStyle->SetPalette(1);

    const Int_t NRGBs = 5;
    const Int_t NCont = 255;

    Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
    Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);
 
    TCanvas canvas("CC map","CC map",1200, 600);
    TPad** pad = new TPad*[2];
    float xmi[2] = {0.0, 0.5};
    float xma[2] = {0.5, 1.0};
    for (int obj = 0; obj < 2; obj++) {
      pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj),
			  xmi[obj], 0.0, xma[obj], 1.0);
      pad[obj]->Draw();
    }

    TH2F** hEEStrip = new TH2F*[2];
    for (int iz = 0; iz < 2; iz++) {
      int izz = iz;
      if(iz == 0) izz = -1;
      hEEStrip[iz] = new TH2F(Form("EEStrip_%i", iz),
			      Form("EE masked strips side %i", izz),101,1.,101.,101,1.,101.);
      hEEStrip[iz]->SetStats(0);
    }

    //    EcalMappingElectronics *mapping = new EcalMappingElectronics ;
    std::string mappingFile = "Geometry/EcalMapping/data/EEMap.txt";
   
    std::ifstream f(edm::FileInPath(mappingFile).fullPath().c_str());
    if (!f.good()) {
      std::cout << "EcalTPGStripStatusPyWrapper File not found" << std::endl;
      throw cms::Exception("FileNotFound");
    }

    int ix, iy, iz, CL;
    int dccid, towerid, pseudostrip_in_SC, xtal_in_pseudostrip;
    int tccid, tower, pseudostrip_in_TCC, pseudostrip_in_TT;
    uint32_t rawEE[108][28][5][5];
    int NbrawEE[108][28][5];
    for(int TCC = 0; TCC < 108; TCC++)
      for(int TT = 0; TT < 28; TT++)
	for(int ST = 0; ST < 5; ST++)
	  NbrawEE[TCC][TT][ST] = 0;
    while ( ! f.eof()) {
      f >> ix >> iy >> iz >> CL >> dccid >> towerid >> pseudostrip_in_SC >> xtal_in_pseudostrip 
	>> tccid >> tower >> pseudostrip_in_TCC >> pseudostrip_in_TT ;
      
      EEDetId detid(ix,iy,iz,EEDetId::XYMODE);
      uint32_t rawId = detid.denseIndex();
      if(tccid > 108 || tower > 28 || pseudostrip_in_TT > 5 || xtal_in_pseudostrip > 5)
	std::cout << " tccid " << tccid <<  " tower " << tower << " pseudostrip_in_TT "<< pseudostrip_in_TT
		  <<" xtal_in_pseudostrip " << xtal_in_pseudostrip << std::endl;
      else {
	rawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1][xtal_in_pseudostrip - 1] = rawId;
	NbrawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1]++;
      }
      /*
      //      if(ix%10 == 0 && iy%10 == 0) std::cout << " dcc tower ps_in_SC xtal_in_ps " << dccid << " " << towerid << " " << pseudostrip_in_SC << " " << xtal_in_pseudostrip << std::endl;
      EcalElectronicsId elecid(dccid,towerid, pseudostrip_in_SC, xtal_in_pseudostrip);
      //      if(ix%10 == 0 && iy%10 == 0) std::cout << " tcc tt ps_in_TT xtal_in_ps " << tccid << " " << tower << " " << pseudostrip_in_TT << " " << xtal_in_pseudostrip << std::endl;
      EcalTriggerElectronicsId triggerid(tccid, tower, pseudostrip_in_TT, xtal_in_pseudostrip);
      EcalMappingElement aElement;
      aElement.electronicsid = elecid.rawId();
      aElement.triggerid = triggerid.rawId();
      (*mapping).setValue(detid, aElement);
      */
    }

    f.close();

    ////////
    const EcalTPGStripStatusMap &stripMap = object().getMap();
    std::cout << " tower map size " << stripMap.size() << std::endl;
    double wei[2] = {0., 0.};
    EcalTPGStripStatusMapIterator itSt;
    for(itSt = stripMap.begin(); itSt != stripMap.end(); ++itSt) {
      if(itSt->second > 0) {
	// let's decode the ID
	int strip = itSt->first/8;
	int pseudostrip = strip & 0x7;
	strip /= 8;
	int tt = strip & 0x7F;
	strip /= 128;
	int tccid = strip & 0x7F;
	int NbXtalInStrip = NbrawEE[tccid - 1][tt - 1][pseudostrip - 1];
	if(NbXtalInStrip != 5) std::cout << " Strip TCC " << tccid << " TT " << tt << " ST " << pseudostrip
					 << " Nx Xtals " << NbXtalInStrip << std::endl;
	for(int Xtal = 0; Xtal < NbXtalInStrip; Xtal++) {
	  uint32_t rawId = rawEE[tccid - 1][tt - 1][pseudostrip - 1][Xtal];
	  //	std::cout << " rawid " << rawId << std::endl;
	  EEDetId detid = EEDetId::detIdFromDenseIndex(rawId);
	  float x = (float)detid.ix();
	  float y = (float)detid.iy();
	  int iz = detid.zside();
	  if(iz == -1) iz++;
	  if(Xtal == 0)   wei[iz] += 1.;
	  hEEStrip[iz]->Fill(x + 0.5, y + 0.5, wei[iz]);
	  //	std::cout << " x " << ix << " y " << iy << " z " << iz << std::endl;
	}
      }
    }

    TLine* l = new TLine(0., 0., 0., 0.);
    l->SetLineWidth(1);
    int ixSectorsEE[202] = {
      62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 56, 56, 46, 46, 44, 44, 43, 43, 42, 42, 
      41, 41, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 46, 46, 56, 56, 58, 58, 59, 59, 
      60, 60, 61, 61, 62, 62,  0,101,101, 98, 98, 96, 96, 93, 93, 88, 88, 86, 86, 81, 
      81, 76, 76, 66, 66, 61, 61, 41, 41, 36, 36, 26, 26, 21, 21, 16, 16, 14, 14,  9,
      9,  6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9, 14, 14, 16, 16, 21, 21, 26, 
      26, 36, 36, 41, 41, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 93, 93, 96, 
      96, 98, 98,101,101,  0, 62, 66, 66, 71, 71, 81, 81, 91, 91, 93,  0, 62, 66, 66, 
      91, 91, 98,  0, 58, 61, 61, 66, 66, 71, 71, 76, 76, 81, 81,  0, 51, 51,  0, 44, 
      41, 41, 36, 36, 31, 31, 26, 26, 21, 21,  0, 40, 36, 36, 11, 11,  4,  0, 40, 36, 
      36, 31, 31, 21, 21, 11, 11,  9,  0, 46, 46, 41, 41, 36, 36,  0, 56, 56, 61, 61, 66, 66};

    int iySectorsEE[202] = {
      51, 56, 56, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 61, 61, 60, 60, 59, 59, 58, 
      58, 56, 56, 46, 46, 44, 44, 43, 43, 42, 42, 41, 41, 40, 40, 41, 41, 42, 42, 43, 
      43, 44, 44, 46, 46, 51,  0, 51, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 
      93, 93, 96, 96, 98, 98,101,101, 98, 98, 96, 96, 93, 93, 88, 88, 86, 86, 81, 81, 
      76, 76, 66, 66, 61, 61, 41, 41, 36, 36, 26, 26, 21, 21, 16, 16, 14, 14,  9,  9, 
      6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9, 14, 14, 16, 16, 21, 21, 26, 26, 
      36, 36, 41, 41, 51,  0, 46, 46, 41, 41, 36, 36, 31, 31, 26, 26,  0, 51, 51, 56, 
      56, 61, 61,  0, 61, 61, 66, 66, 71, 71, 76, 76, 86, 86, 88,  0, 62,101,  0, 61, 
      61, 66, 66, 71, 71, 76, 76, 86, 86, 88,  0, 51, 51, 56, 56, 61, 61,  0, 46, 46, 
      41, 41, 36, 36, 31, 31, 26, 26,  0, 40, 31, 31, 16, 16,  6,  0, 40, 31, 31, 16, 16,  6};

    pad[0]->cd();
    hEEStrip[0]->Draw("col");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
	l->SetLineWidth(0.2);
      }
    }

    pad[1]->cd();
    hEEStrip[1]->Draw("col");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
	l->SetLineWidth(0.2);
      }
    }

    canvas.SaveAs(filename.c_str());
    return filename;
  }  // plot
}
PYTHON_WRAPPER(EcalTPGStripStatus,EcalTPGStripStatus);
