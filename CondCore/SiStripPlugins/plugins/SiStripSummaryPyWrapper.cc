
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

  
namespace sistripsummary {
  enum Quantity { pippo=1 };
}

namespace cond {

  template<>
  struct ExtractWhat<SiStripSummary> {

    sistripsummary::Quantity m_quantity;
    sistripsummary::TrackerRegion m_trackerregion;

    sistripsummary::Quantity const & quantity() const { return m_quantity;}
    sistripsummary::TrackerRegion const & trackerregion() const { return m_trackerregion;}
 
    void set_quantity( sistripsummary::Quantity i) { m_quantity=i;}
    void set_trackerregion(sistripsummary::TrackerRegion i) {m_trackerregion=i;}
  };


  

  template<>
  class ValueExtractor<SiStripSummary>: public  BaseValueExtractor<SiStripSummary> {
  public:
    
    typedef SiStripSummary Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){};
    ValueExtractor(What const & what, std::vector<int> const& which)
      : m_what(what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
      std::vector<float> res;
      res.push_back(it.getSummaryObj(m_what.trackerregion(),std::string("Summary_NumberOfClusters_OffTrack@mean")));
      swap(res);
    }
  private:
    What  m_what;
  };


  template<>
  std::string
  PayLoadInspector<SiStripSummary>::dump() const {
    std::stringstream ss;
    ss << "I'm  PayLoadInspector<SiStripSummary>::dump() " ;
    std::vector<std::string>  listWhat= object->getUserDBContent();
    for(size_t i=0;i<listWhat.size();++i)
      ss << listWhat[i] << "\n";
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<SiStripSummary>::summary() const {
    std::stringstream ss;
    ss << "I'm  PayLoadInspector<SiStripSummary>::summary() \n" 
       << "Nr. of detector elements in SiStripSummary object is " << object->getRegistryVectorEnd()-object->getRegistryVectorBegin()
       << "Nr. of summary elements in SiStripSummary object is " << object->getDataVectorEnd()-object->getDataVectorBegin()
       << " RunNr= " << object->getRunNr()
       << " timeValue= " << object->getTimeValue();
    ss << "names of DBquantities ";
    std::vector<std::string>  listWhat= object->getUserDBContent();
    for(size_t i=0;i<listWhat.size();++i)
      ss << listWhat[i] << "\n";
    return ss.str(); 
  }
  

  template<>
  std::string PayLoadInspector<SiStripSummary>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}


namespace condPython {
  template<>
  void defineWhat<SiStripSummary>() {
    enum_<sistripsummary::Quantity>("Quantity")
      .value("Summary_NumberOfClusters_OffTrack@mean",sistripsummary::pippo)
      ;
    enum_<sistripsummary::TrackerRegion>("Trackerregion")
      .value("Tracker",sistripsummary::TRACKER)
      .value("TIB",sistripsummary::TIB)
      ;

    typedef cond::ExtractWhat<SiStripSummary> What;
    class_<What>("What",init<>())
      .def("set_quantity",&What::set_quantity)
      .def("set_how",&What::set_trackerregion)
      .def("quantity",&What::quantity, return_value_policy<copy_const_reference>())
      .def("how",&What::trackerregion, return_value_policy<copy_const_reference>())
      ;
  }
}




PYTHON_WRAPPER(SiStripSummary,SiStripSummary);
