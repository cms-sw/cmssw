#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

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

namespace {
  struct Printer {
    void doit(EcalPedestal const & item) {
      for (int i=1; i<4; i++)
	ss << item.mean(i) << "," << item.rms(i) <<";";
      ss << " ";
    }
    std::stringstream ss;
  };
}

namespace cond {

  namespace ecalped {
    enum Quantity { mean_x12=1, mean_x6=2, mean_x3=3 };
    enum How { singleChannel, bySuperModule, all};

    float average(EcalPedestals const & peds, Quantity q) {
      return std::accumulate(
			     boost::make_transform_iterator(peds.barrelItems().begin(),bind(&EcalPedestal::mean,_1,q)),
			     boost::make_transform_iterator(peds.barrelItems().end(),bind(&EcalPedestal::mean,_1,q)),
			     0.)/float(peds.barrelItems().size());
    }

    void extractAverage(EcalPedestals const & peds, Quantity q, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] = average(peds,q);
    }
    
    void extractSuperModules(EcalPedestals const & peds, Quantity q, std::vector<int> const & which,  std::vector<float> & result) {
      // bho...
    }

    void extractSingleChannel(EcalPedestals const & peds, Quantity q, std::vector<int> const & which,  std::vector<float> & result) {
      for (int i=0; i<which.size();i++) {
	// absolutely arbitraty
	if (which[i]<  peds.barrelItems().size())
	  result.push_back( peds.barrelItems()[which[i]].mean(q));
      }
    }

	typedef boost::function<void(EcalPedestals const & peds, Quantity q, std::vector<int> const & which,  std::vector<float> & result)> PedExtractor;
  }

  template<>
  struct ExtractWhat<EcalPedestals> {

    ecalped::Quantity m_quantity;
    ecalped::How m_how;

    ecalped::Quantity const & quantity() const { return m_quantity;}
    ecalped::How const & how() const { return m_how;}
 
    void set_quantity( ecalped::Quantity i) { m_quantity=i;}
    void set_how(ecalped::How i) {m_how=i;}
  };


  template<>
  class ValueExtractor<EcalPedestals>: public  BaseValueExtractor<EcalPedestals> {
  public:

    static ecalped::PedExtractor & extractor(ecalped::How how) {
      static  ecalped::PedExtractor fun[3] = { 
	ecalped::PedExtractor(ecalped::extractSingleChannel),
	ecalped::PedExtractor(ecalped::extractSuperModules),
	ecalped::PedExtractor(ecalped::extractAverage)
              };
      return fun[how];
    }

    typedef EcalPedestals Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what, std::vector<int> const& which)
      : m_what(what), m_which(which)
    {
      // here one can make stuff really complicated... (select mean rms, 12,6,1)
      // ask to make average on selected channels...
    }

    void compute(Class const & it){
      std::vector<float> res;
      extractor(m_what.how())(it,m_what.quantity(),m_which,res);
      swap(res);
    }

  private:
    What  m_what;
    std::vector<int> m_which;
  };


  template<>
  std::string
  PayLoadInspector<EcalPedestals>::dump() const {
    Printer p;
    std::for_each(object->barrelItems().begin(),object->barrelItems().end(),boost::bind(&Printer::doit,boost::ref(p),_1));
    p.ss <<"\n";
    std::for_each(object->endcapItems().begin(),object->endcapItems().end(),boost::bind(&Printer::doit,boost::ref(p),_1));
    p.ss << std::endl;
    return p.ss.str();
  }

   template<>
   std::string PayLoadInspector<EcalPedestals>::summary() const {
     std::stringstream ss;
     ss << "sizes="
	<< object->barrelItems().size() <<","
	<< object->endcapItems().size() <<";";
     ss << std::endl;
     return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<EcalPedestals>::plot(std::string const & filename,
						   std::string const &, std::vector<int> const&, std::vector<float> const& ) const {
    std::string fname = filename + ".txt";
    std::ofstream f(fname.c_str());
    f << dump();
    return fname;
  }
  
}


namespace condPython {
  template<>
  void defineWhat<EcalPedestals>() {
    enum_<cond::ecalped::Quantity>("Quantity")
      .value("mean_x12",cond::ecalped::mean_x12)
      .value("mean_x6",  cond::ecalped::mean_x6)
      .value("mean_x3", cond::ecalped::mean_x3)
      ;
    enum_<cond::ecalped::How>("How")
      .value("singleChannel",cond::ecalped::singleChannel)
      .value("bySuperModule",cond::ecalped::bySuperModule) 
      .value("all",cond::ecalped::all)
      ;

    typedef cond::ExtractWhat<EcalPedestals> What;
    class_<What>("What",init<>())
      .def("set_quantity",&What::set_quantity)
      .def("set_how",&What::set_how)
      .def("quantity",&What::quantity, return_value_policy<copy_const_reference>())
      .def("how",&What::how, return_value_policy<copy_const_reference>())
      ;
  }
}



PYTHON_WRAPPER(EcalPedestals,EcalPedestals);
