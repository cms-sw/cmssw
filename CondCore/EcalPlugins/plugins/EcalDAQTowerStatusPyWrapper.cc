#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondTools/Ecal/interface/EcalDAQTowerStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
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

    typedef EcalDAQTowerStatus::Items  Items;
    typedef EcalDAQTowerStatus::value_type  value_type;

    enum How { singleChannel, bySuperModule, all};

    int bad(Items const & cont) {
      return  std::count_if(cont.begin(),cont.end(),
			    boost::bind(std::greater<int>(),
					boost::bind(&value_type::getStatusCode,_1),0)
			    );
    }

    void extractBarrel(EcalDAQTowerStatus const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] =  bad(cont.barrelItems());
    }
    
    void extractEndcap(EcalDAQTowerStatus const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] = bad(cont.endcapItems());
    }
    void extractAll(EcalDAQTowerStatus const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] = bad(cont.barrelItems())+bad(cont.endcapItems());
    }

    void extractSuperModules(EcalDAQTowerStatus const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      // bho...
    }

    void extractSingleChannel(EcalDAQTowerStatus const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      result.reserve(which.size());
      for (unsigned int i=0; i<which.size();i++) {
	result.push_back(cont[which[i]].getStatusCode());
      }
    }

	typedef boost::function<void(EcalDAQTowerStatus const & cont, std::vector<int> const & which,  std::vector<float> & result)> CondExtractor;
  }  // namespace ecalcond

  template<>
  struct ExtractWhat<EcalDAQTowerStatus> {

    ecalcond::How m_how;
    std::vector<int> m_which;

    ecalcond::How const & how() const { return m_how;}
    std::vector<int> const & which() const { return m_which;}
 
    void set_how(ecalcond::How i) {m_how=i;}
    void set_which(std::vector<int> & i) { m_which.swap(i);}
  };


  template<>
  class ValueExtractor<EcalDAQTowerStatus>: public  BaseValueExtractor<EcalDAQTowerStatus> {
  public:

    static ecalcond::CondExtractor & extractor(ecalcond::How how) {
      static  ecalcond::CondExtractor fun[3] = { 
	ecalcond::CondExtractor(ecalcond::extractSingleChannel),
	ecalcond::CondExtractor(ecalcond::extractSuperModules),
	ecalcond::CondExtractor(ecalcond::extractAll)
      };
      return fun[how];
    }

    typedef EcalDAQTowerStatus Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
      : m_what(what)
    {
      // here one can make stuff really complicated...
    }

    void compute(Class const & it) override{
      std::vector<float> res;
      extractor(m_what.how())(it,m_what.which(),res);
      swap(res);
    }

  private:
    What  m_what;  
  };


  template<>
  std::string PayLoadInspector<EcalDAQTowerStatus>::dump() const {
    std::stringstream ss;
    EcalCondHeader h;
    ss << EcalDAQTowerStatusXMLTranslator::dumpXML(h,object());
    return ss.str();
  }

  	class EcalDAQTowerStatusHelper: public EcalPyWrapperHelper<EcalDAQStatusCode>{
	public:
		//change me
		EcalDAQTowerStatusHelper():EcalPyWrapperHelper<EcalObject>(1, STATUS){}
	protected:

		//change me
		typedef EcalDAQStatusCode EcalObject;

		type_vValues getValues( const std::vector<EcalObject> & vItems) override
		{
			type_vValues vValues(total_values);
			
			//change us
			vValues[0].first = "bit 0 -> towers excluded from the DAQ";
			
			vValues[0].second = .0;
			
			//get info:
			for(std::vector<EcalObject>::const_iterator iItems = vItems.begin(); iItems != vItems.end(); ++iItems){
				//change us
				vValues[0].second += iItems->getStatusCode();
			}
			return vValues;
		}
	};

   template<>
   std::string PayLoadInspector<EcalDAQTowerStatus>::summary() const {
	std::stringstream ss;
	EcalDAQTowerStatusHelper helper;
	ss << helper.printBarrelsEndcaps(object().barrelItems(), object().endcapItems());
	return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<EcalDAQTowerStatus>::plot(std::string const & filename,
							 std::string const &, 
							 std::vector<int> const&, 
							 std::vector<float> const& ) const {
    //    std::string fname = filename + ".txt";
    //    EcalDAQTowerStatusXMLTranslator::plot(fname, object());
    //    return fname;
    TCanvas canvas("CC map","CC map",800,800);
    TPad* padb = new TPad("padb","padb", 0., 0.55, 1., 1.);
    padb->Draw();
    TPad* padem = new TPad("padem","padem", 0., 0., 0.45, 0.45);
    padem->Draw();
    TPad* padep = new TPad("padep","padep", 0.55, 0., 1., 0.45);
    padep->Draw();

    TH2F* barrel = new TH2F("EB","EB Tower Status", 72, 0, 72, 34, -17, 17);
    TH2F* endc_p = new TH2F("EE+","EE+ Tower Status",22, 0, 22, 22, 0, 22);
    TH2F* endc_m = new TH2F("EE-","EE- Tower Status",22, 0, 22, 22, 0, 22);
    for(uint cellid = 0;
	cellid < EcalTrigTowerDetId::kEBTotalTowers;
	++cellid) {
      EcalTrigTowerDetId rawid = EcalTrigTowerDetId::detIdFromDenseIndex(cellid);
      if (object().find(rawid) == object().end()) continue;
      int ieta = rawid.ieta();
      if(ieta > 0) ieta--;   // 1 to 17
      int iphi = rawid.iphi() - 1;  // 0 to 71
      barrel->Fill(iphi, ieta, object()[rawid].getStatusCode());
    }
    for(uint cellid = 0;
	cellid < EcalTrigTowerDetId::kEETotalTowers;
	++cellid) {
      if(EcalScDetId::validHashIndex(cellid)) {
	EcalScDetId rawid = EcalScDetId::unhashIndex(cellid); 
	int ix = rawid.ix();  // 1 to 20
	int iy = rawid.iy();  // 1 to 20
	int side = rawid.zside();
	if(side == -1)
	  endc_m->Fill(ix, iy, object()[rawid].getStatusCode());
	else
	  endc_p->Fill(ix, iy, object()[rawid].getStatusCode());
      }
    }
    TLine* l = new TLine(0., 0., 0., 0.);
    l->SetLineWidth(1);
    padb->cd();
    barrel->SetStats(0);
    //    barrel->SetMaximum(14);
    //    barrel->SetMinimum(0);
    //    barrel->Draw("colz");
    barrel->Draw("col");
    for(int i = 0; i <17; i++) {
      Double_t x = 4.+ (i * 4);
      l = new TLine(x, -17., x, 17.);
      l->Draw();
    }
    l = new TLine(0., 0., 72., 0.);
    l->Draw();

    int ixSectorsEE[133] = {
       8,14,14,17,17,18,18,19,19,20,20,21,21,20,20,19,19,18,18,17,
      17,14,14, 8, 8, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3, 4,
       4, 5, 5, 8, 8, 8, 9, 9,10,10,12,12,13,13,12,12,10,10, 9, 9,
      10,10, 0,11,11, 0,10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 0,12,13,
      13,14,14,15,15,16,16,17,17, 0, 9, 8, 8, 3, 3, 1, 0,13,14,14,
      19,19,21, 0, 9, 8, 8, 7, 7, 5, 5, 3, 3, 2, 0,13,14,14,15,15,
      17,17,19,19,20, 0,14,14,13,13,12,12,0};
    int iySectorsEE[133] = {
       1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 8,14,14,17,17,18,18,19,19,
      20,20,21,21,20,20,19,19,18,18,17,17,14,14, 8, 8, 5, 5, 4, 4,
       3, 3, 2, 2, 1, 4, 4, 7, 7, 9, 9,10,10,12,12,13,13,12,12,10,
      10, 9, 0,13,21, 0,13,13,14,14,15,15,16,16,18,18,19, 0,13,13,
      14,14,15,15,16,16,18,18,19, 0,11,11,12,12,13,13, 0,11,11,12,
      12,13,13, 0,10,10, 9, 9, 8, 8, 7, 7, 6, 6, 0,10,10, 9, 9, 8,
       8, 7, 7, 6, 6, 0, 2, 4, 4, 7, 7, 9, 0} ;
    padem->cd();
    endc_m->SetStats(0);
    endc_m->Draw("col");
    for ( int i = 0; i < 132; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
      }
    }

    padep->cd();
    endc_p->SetStats(0);
    endc_p->Draw("col");
    for ( int i = 0; i < 132; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
      }
    }

    canvas.SaveAs(filename.c_str());
    return filename;
  }
}


namespace condPython {
  template<>
  void defineWhat<EcalDAQTowerStatus>() {
    using namespace boost::python;
    enum_<cond::ecalcond::How>("How")
      .value("singleChannel",cond::ecalcond::singleChannel)
      .value("bySuperModule",cond::ecalcond::bySuperModule) 
      .value("all",cond::ecalcond::all)
      ;

    typedef cond::ExtractWhat<EcalDAQTowerStatus> What;
    class_<What>("What",init<>())
      .def("set_how",&What::set_how)
      .def("set_which",&What::set_which)
      .def("how",&What::how, return_value_policy<copy_const_reference>())
      .def("which",&What::which, return_value_policy<copy_const_reference>())
      ;
  }
}

PYTHON_WRAPPER(EcalDAQTowerStatus,EcalDAQTowerStatus);
