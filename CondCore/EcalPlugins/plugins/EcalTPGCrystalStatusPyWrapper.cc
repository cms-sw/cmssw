#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondTools/Ecal/interface/EcalTPGCrystalStatusXMLTranslator.h"
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

  namespace ecalcond {

    typedef EcalTPGCrystalStatus::Items  Items;
    typedef EcalTPGCrystalStatus::value_type  value_type;

    enum How { singleChannel, bySuperModule, all};

    int bad(Items const & cont) {
      return  std::count_if(cont.begin(),cont.end(),
			    boost::bind(std::greater<int>(),
					boost::bind(&value_type::getStatusCode,_1),0)
			    );
    }

    void extractBarrel(EcalTPGCrystalStatus const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] =  bad(cont.barrelItems());
    }
    
    void extractEndcap(EcalTPGCrystalStatus const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] = bad(cont.endcapItems());
    }
    void extractAll(EcalTPGCrystalStatus const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] = bad(cont.barrelItems())+bad(cont.endcapItems());
    }

    void extractSuperModules(EcalTPGCrystalStatus const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      // bho...
    }

    void extractSingleChannel(EcalTPGCrystalStatus const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      result.reserve(which.size());
      for (unsigned int i=0; i<which.size();i++) {
	result.push_back(cont[which[i]].getStatusCode());
      }
    }

	typedef boost::function<void(EcalTPGCrystalStatus const & cont, std::vector<int> const & which,  std::vector<float> & result)> CondExtractor;
  }  // namespace ecalcond

  template<>
  struct ExtractWhat<EcalTPGCrystalStatus> {

    ecalcond::How m_how;
    std::vector<int> m_which;

    ecalcond::How const & how() const { return m_how;}
    std::vector<int> const & which() const { return m_which;}
 
    void set_how(ecalcond::How i) {m_how=i;}
    void set_which(std::vector<int> & i) { m_which.swap(i);}
  };


  template<>
  class ValueExtractor<EcalTPGCrystalStatus>: public  BaseValueExtractor<EcalTPGCrystalStatus> {
  public:

    static ecalcond::CondExtractor & extractor(ecalcond::How how) {
      static  ecalcond::CondExtractor fun[3] = { 
	ecalcond::CondExtractor(ecalcond::extractSingleChannel),
	ecalcond::CondExtractor(ecalcond::extractSuperModules),
	ecalcond::CondExtractor(ecalcond::extractAll)
      };
      return fun[how];
    }

    typedef EcalTPGCrystalStatus Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
      : m_what(what)
    {
      // here one can make stuff really complicated... 
    }

    void compute(Class const & it){
      std::vector<float> res;
      extractor(m_what.how())(it,m_what.which(),res);
      swap(res);
    }

  private:
    What  m_what;  
  };


  template<>
  std::string PayLoadInspector<EcalTPGCrystalStatus>::dump() const {
    std::stringstream ss;
    EcalCondHeader h;
    ss << EcalTPGCrystalStatusXMLTranslator::dumpXML(h,object());
    return ss.str();
  }

  class EcalTPGCrystalStatusHelper: public EcalPyWrapperHelper<EcalTPGCrystalStatusCode>{
  public:
    //change me
    EcalTPGCrystalStatusHelper():EcalPyWrapperHelper<EcalObject>(1, STATUS, "-Errors total: "){}
  protected:

    //change me
    typedef EcalTPGCrystalStatusCode EcalObject;

    type_vValues getValues( const std::vector<EcalObject> & vItems)
      {
	//change me
	//unsigned int totalValues = 2; 

	type_vValues vValues(total_values);
			
	//change us
	vValues[0].first = "[0]StatusCode";

			
	vValues[0].second = .0;
	
	//get info:
	unsigned int shift = 0, mask = 1;
	unsigned int statusCode;
	for(std::vector<EcalObject>::const_iterator iItems = vItems.begin(); iItems != vItems.end(); ++iItems){
	  //change us
	  statusCode =  iItems->getStatusCode();
	  for (shift = 0; shift < total_values; ++shift){
	    mask = 1 << (shift);
	    //std::cout << "; statuscode: " << statusCode;
	    if (statusCode & mask){
	      vValues[shift].second += 1;
	    }
	  }
	}
	return vValues;
      }
  };

  template<>
  std::string PayLoadInspector<EcalTPGCrystalStatus>::summary() const {
    std::stringstream ss;
    //   EcalTPGCrystalStatusHelper helper;
    //   ss << helper.printBarrelsEndcaps(object().barrelItems(), object().endcapItems());
    const int kSides       = 2;
    const int kBarlRings   = EBDetId::MAX_IETA;
    const int kBarlWedges  = EBDetId::MAX_IPHI;
    const int kEndcWedgesX = EEDetId::IX_MAX;
    const int kEndcWedgesY = EEDetId::IY_MAX;

    int EB[2] = {0, 0}, EE[2] = {0, 0};
    for (int sign=0; sign < kSides; sign++) {
      int thesign = sign==1 ? 1:-1;

      for (int ieta=0; ieta<kBarlRings; ieta++) {
	for (int iphi=0; iphi<kBarlWedges; iphi++) {
	  EBDetId id((ieta+1)*thesign, iphi+1);
	  if(object()[id.rawId()].getStatusCode() > 0) EB[sign]++;
	}  // iphi
      }   // ieta

      for (int ix=0; ix<kEndcWedgesX; ix++) {
	for (int iy=0; iy<kEndcWedgesY; iy++) {
	  if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
	  EEDetId id(ix+1,iy+1,thesign);
	  if(object()[id.rawId()].getStatusCode() > 0) EE[sign]++;
	}  // iy
      }   // ix
    }    // side
    ss << " number of masked Crystals  EB- " << EB[0] << " EB+ " <<  EB[1]
       << "   EE- " << EE[0] << " EE+ " <<  EE[1] << std::endl;
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalTPGCrystalStatus>::plot(std::string const & filename,
							   std::string const &, 
							   std::vector<int> const&, 
							   std::vector<float> const& ) const {
    gStyle->SetPalette(1);
    const int TOTAL_PADS = 3;

    //    TCanvas canvas("CC map","CC map",700,800);
    Double_t w = 600;
    Double_t h = 600;
    TCanvas canvas("c", "c", w, h);
    canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));
    //    const float IMG_SIZE = 1.5;
    //    TCanvas canvas("CC map","CC map",800*IMG_SIZE, 200 * IMG_SIZE);//800, 1200

    float xmi[3] = {0.0, 0.0, 0.5};
    float xma[3] = {1.0, 0.5, 1.0};
    float ymi[3] = {0.5, 0.0, 0.0};
    float yma[3] = {1.0, 0.5, 0.5};


    TPad** pad = new TPad*[TOTAL_PADS];
    for (int obj = 0; obj < TOTAL_PADS; obj++) {
      pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj),
			  xmi[obj], ymi[obj], xma[obj], yma[obj]);
      pad[obj]->Draw();
    }

    const int kSides       = 2;
    const int kBarlRings   = EBDetId::MAX_IETA;
    const int kBarlWedges  = EBDetId::MAX_IPHI;
    const int kEndcWedgesX = EEDetId::IX_MAX;
    const int kEndcWedgesY = EEDetId::IY_MAX;

    TH2F* barrel = new TH2F("EB","EB TPG Crystal Status", 360,0,360, 170, -85,85);
    TH2F* endc_m = new TH2F("EEm","EE- TPG Crystal Status",100,1,101,100,1,101);
    TH2F* endc_p = new TH2F("EEp","EE+ TPG Crystal Status",100,1,101,100,1,101);

    for (int sign=0; sign < kSides; sign++) {
      int thesign = sign==1 ? 1:-1;

      for (int ieta=0; ieta<kBarlRings; ieta++) {
	for (int iphi=0; iphi<kBarlWedges; iphi++) {
	  EBDetId id((ieta+1)*thesign, iphi+1);
	  float y = -1 - ieta;
	  if(sign == 1) y = ieta;
	  barrel->Fill(iphi, y, object()[id.rawId()].getStatusCode());
	}  // iphi
      }   // ieta

      for (int ix=0; ix<kEndcWedgesX; ix++) {
	for (int iy=0; iy<kEndcWedgesY; iy++) {
	  if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
	  EEDetId id(ix+1,iy+1,thesign);
	  if (thesign==1) {
	    endc_p->Fill(ix+1,iy+1,object()[id.rawId()].getStatusCode());
	  }
	  else{ 
	    endc_m->Fill(ix+1,iy+1,object()[id.rawId()].getStatusCode());
	  }
	}  // iy
      }   // ix
    }    // side

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
    barrel->SetStats(0);
    barrel->Draw("colz");
    for(int i = 0; i <17; i++) {
      Double_t x = 20.+ (i *20);
      l = new TLine(x,-85.,x,86.);
      l->Draw();
    }
    l = new TLine(0.,0.,360.,0.);
    l->Draw();

    pad[1]->cd();
    endc_m->SetStats(0);
    endc_m->Draw("col");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
	l->SetLineWidth(0.2);
      }
    }

    pad[2]->cd();
    endc_p->SetStats(0);
    endc_p->Draw("col");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
      }
    }

    canvas.SaveAs(filename.c_str());
    return filename;
  }  // plot
}

PYTHON_WRAPPER(EcalTPGCrystalStatus,EcalTPGCrystalStatus);
