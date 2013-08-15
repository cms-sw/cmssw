

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerXMLTranslator.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <string>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <algorithm>

namespace {
  struct Printer {
    Printer() : i(0){}
    void reset() { i=0;}
    void doB(float const & item) {
      ss << i <<":"<< item << "\n";
      i++;
    }
    void doE(float const & item) {
      ss << i <<":"<< item << "\n";
      i++;
    }
    int i;
    std::stringstream ss;
  };
}


namespace cond {

  // migrate to a common trait (when fully understood)
  namespace ecalcond {

    typedef EcalFloatCondObjectContainer Container;
    typedef Container::value_type  value_type;

    enum How { singleChannel, bySuperModule, barrel, endcap, all};


    void extractBarrel(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(cont.barrelItems().size());
      std::copy(cont.barrelItems().begin(),cont.barrelItems().end(),result.begin());
    }

    void extractEndcap(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(cont.endcapItems().size());
      std::copy(cont.endcapItems().begin(),cont.endcapItems().end(),result.begin());
    }

     void extractAll(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {

       // EcalCondObjectContainer does not have a real iterator ... 
      result.resize(cont.barrelItems().size()+ cont.endcapItems().size());
      
      std::copy(cont.barrelItems().begin(),cont.barrelItems().end(),result.begin());
      std::copy(cont.endcapItems().begin(),cont.endcapItems().end(),result.end());
    }
    
    void extractSuperModules(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      // bho...
    }

    void extractSingleChannel(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      result.reserve(which.size());
      for (unsigned int i=0; i<which.size();i++) {
	  result.push_back(cont[which[i]]);
      }
    }

    typedef boost::function<void(Container const & cont, std::vector<int> const & which,  std::vector<float> & result)> CondExtractor;

  } // ecalcond

  template<>
  struct ExtractWhat<ecalcond::Container> {
    
    ecalcond::How m_how;
    std::vector<int> m_which;
    
    ecalcond::How const & how() const { return m_how;}
    std::vector<int> const & which() const { return m_which;}
    
    void set_how(ecalcond::How i) {m_how=i;}
    void set_which(std::vector<int> & i) { m_which.swap(i);}
  };
  



  template<>
  class ValueExtractor<ecalcond::Container>: public  BaseValueExtractor<ecalcond::Container> {
  public:
    
    static ecalcond::CondExtractor & extractor(ecalcond::How how) {
      static  ecalcond::CondExtractor fun[5] = { 
	ecalcond::CondExtractor(ecalcond::extractSingleChannel),
	ecalcond::CondExtractor(ecalcond::extractSuperModules),
	ecalcond::CondExtractor(ecalcond::extractBarrel),
	ecalcond::CondExtractor(ecalcond::extractEndcap),
	ecalcond::CondExtractor(ecalcond::extractAll)
      };
      return fun[how];
    }
    
    
    typedef ecalcond::Container Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}
    
    ValueExtractor(){}
    ValueExtractor(What const & what)
      : m_what(what)
    {
      // here one can make stuff really complicated... 
      // ask to make average on selected channels...
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
  std::string
  PayLoadInspector<EcalFloatCondObjectContainer>::dump() const {

    std::stringstream ss;
    EcalCondHeader header;
    ss<<EcalFloatCondObjectContainerXMLTranslator::dumpXML(header,object());
    return ss.str();

 //    Printer p;
//     std::for_each(object().barrelItems().begin(),object().barrelItems().end(),boost::bind(&Printer::doB,boost::ref(p),_1));
//     p.ss <<"\n";
//     p.reset();
//     std::for_each(object().endcapItems().begin(),object().endcapItems().end(),boost::bind(&Printer::doE,boost::ref(p),_1));
//     p.ss << std::endl;
//     return p.ss.str();
  }
  
  template<>
  std::string PayLoadInspector<EcalFloatCondObjectContainer>::summary() const {

    std::stringstream ss;

    const int kSides       = 2;
    const int kBarlRings   = EBDetId::MAX_IETA;
    const int kBarlWedges  = EBDetId::MAX_IPHI;
    const int kEndcWedgesX = EEDetId::IX_MAX;
    const int kEndcWedgesY = EEDetId::IY_MAX;

    /// calculate mean and sigma 

    float mean_x_EB=0;
    float mean_xx_EB=0;
    int num_x_EB=0;

    float mean_x_EE=0;
    float mean_xx_EE=0;
    int num_x_EE=0;


    for (int sign=0; sign<kSides; sign++) {

       int thesign = sign==1 ? 1:-1;

       for (int ieta=0; ieta<kBarlRings; ieta++) {
         for (int iphi=0; iphi<kBarlWedges; iphi++) {
	   EBDetId id((ieta+1)*thesign, iphi+1);
	    float x= object()[id.rawId()];
	    num_x_EB++;
	    mean_x_EB=mean_x_EB+x;
	    mean_xx_EB=mean_xx_EB+x*x;
         }
       }

       for (int ix=0; ix<kEndcWedgesX; ix++) {
	 for (int iy=0; iy<kEndcWedgesY; iy++) {
	   if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
	   EEDetId id(ix+1,iy+1,thesign);
	    float x=object()[id.rawId()];
	    num_x_EE++;
	    mean_x_EE=mean_x_EE+x;
	    mean_xx_EE=mean_xx_EE+x*x;

	 }//iy
       }//ix


    }

    mean_x_EB=mean_x_EB/num_x_EB;
    mean_x_EE=mean_x_EE/num_x_EE;
    mean_xx_EB=mean_xx_EB/num_x_EB;
    mean_xx_EE=mean_xx_EE/num_x_EE;
    float rms_EB=(mean_xx_EB-mean_x_EB*mean_x_EB);
    float rms_EE=(mean_xx_EE-mean_x_EE*mean_x_EE);

    ss << "ECAL BARREL Mean: "<< mean_x_EB <<" RMS: "<<  rms_EB << " Nchan: "<< num_x_EB<< std::endl
       << "ECAL Endcap Mean: "<< mean_x_EE <<" RMS: "<<  rms_EE << " Nchan: "<< num_x_EE<< std::endl ;


    return ss.str();
  }
  


  template<>
  std::string PayLoadInspector<EcalFloatCondObjectContainer>::plot(std::string const & filename,
								   std::string const &, 
								   std::vector<int> const&, 
								   std::vector<float> const& ) const {

    gStyle->SetPalette(1);
    TCanvas canvas("CC map","CC map",840,280);
    //canvas.Divide(3,1);
 
    TPad pad1("p1","p1", 0.0, 0.0, 0.2, 1.0);
    TPad pad2("p2","p2", 0.2, 0.0, 0.8, 1.0);
    TPad pad3("p3","p3", 0.8, 0.0, 1.0, 1.0);
    pad1.Draw();
    pad2.Draw();
    pad3.Draw();

    TH2F barrelmap("EB","EB",360,1,360, 171, -85,86);
    TH2F endcmap_p("EE+","EE+",100,1,101,100,1,101);
    TH2F endcmap_m("EE-","EE-",100,1,101,100,1,101);

    const int kSides       = 2;
    const int kBarlRings   = EBDetId::MAX_IETA;
    const int kBarlWedges  = EBDetId::MAX_IPHI;
    const int kEndcWedgesX = EEDetId::IX_MAX;
    const int kEndcWedgesY = EEDetId::IY_MAX;

    /// there's a cleaner way to plot this map...
    for (int sign=0; sign<kSides; sign++) {

       int thesign = sign==1 ? 1:-1;

       for (int ieta=0; ieta<kBarlRings; ieta++) {
         for (int iphi=0; iphi<kBarlWedges; iphi++) {
	   EBDetId id((ieta+1)*thesign, iphi+1);
	   barrelmap.Fill(iphi+1,ieta*thesign + thesign, object()[id.rawId()]);
         }
       }

       for (int ix=0; ix<kEndcWedgesX; ix++) {
	 for (int iy=0; iy<kEndcWedgesY; iy++) {
	   if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
	   EEDetId id(ix+1,iy+1,thesign);
    
	   if (thesign==1) {
	     endcmap_p.Fill(ix+1,iy+1,object()[id.rawId()]);
	   }
	   else{ 
	     endcmap_m.Fill(ix+1,iy+1,object()[id.rawId()]);
	
	   }
	 }//iy
       }//ix



    }

    
    //canvas.cd(1);
    pad1.cd();
    endcmap_m.SetStats(0);
    endcmap_m.Draw("colz");
    //canvas.cd(2);
    pad2.cd();
    barrelmap.SetStats(0);
    barrelmap.Draw("colz");
    //canvas.cd(3);
    pad3.cd();
    endcmap_p.SetStats(0);
    endcmap_p.Draw("colz");

    canvas.SaveAs(filename.c_str());
    return filename;
  }
  
  
}

namespace condPython {
  template<>
  void defineWhat<cond::ecalcond::Container>() {
    using namespace boost::python;
    enum_<cond::ecalcond::How>("How")
      .value("singleChannel",cond::ecalcond::singleChannel)
      .value("bySuperModule",cond::ecalcond::bySuperModule) 
      .value("barrel",cond::ecalcond::barrel)
      .value("endcap",cond::ecalcond::endcap)
      .value("all",cond::ecalcond::all)
      ;
    
    typedef cond::ExtractWhat<cond::ecalcond::Container> What;
    class_<What>("What",init<>())
      .def("set_how",&What::set_how)
      .def("set_which",&What::set_which)
      .def("how",&What::how, return_value_policy<copy_const_reference>())
      .def("which",&What::which, return_value_policy<copy_const_reference>())
      ;
  }
}




PYTHON_WRAPPER(EcalFloatCondObjectContainer,EcalFloatCondObjectContainer);
