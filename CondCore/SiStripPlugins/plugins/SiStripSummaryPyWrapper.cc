
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

  
namespace cond {

  template<>
  struct ExtractWhat<SiStripSummary> {

    std::string m_quantity;
    sistripsummary::TrackerRegion m_trackerregion;

    std::string const & quantity() const { return m_quantity;}
    sistripsummary::TrackerRegion const & trackerregion() const { return m_trackerregion;}
 
    void set_quantity( std::string i) { m_quantity=i;}
    void set_trackerregion(sistripsummary::TrackerRegion i) {m_trackerregion=i;}
  };


  

  template<>
  class ValueExtractor<SiStripSummary>: public  BaseValueExtractor<SiStripSummary> {
      public:

      typedef SiStripSummary Class;
      typedef ExtractWhat<Class> What;
      static What what() { return What();}

      ValueExtractor(){};
      ValueExtractor(What const & what)
      : m_what(what)
      {
          // here one can make stuff really complicated...
      }
      void compute(Class const & it){
          std::vector<std::string> vlistItems;
          std::vector<float> res;
          uint32_t detid=m_what.trackerregion();
          std::string::size_type oldloc=0; 
          std::string ListItems   =   m_what.quantity();
          std::string::size_type loc = ListItems.find( ",", oldloc );
          size_t count=1;
          while( loc != std::string::npos ) {
              vlistItems.push_back(ListItems.substr(oldloc,loc-oldloc));
              oldloc=loc+1;
              loc=ListItems.find( ",", oldloc );
              count++; 
          } 
          //there is a single item
          vlistItems.push_back(ListItems.substr(oldloc,loc-oldloc));
	  std::vector<float> vres=it.getSummaryObj(detid,vlistItems);
          res.insert(res.end(),vres.begin(),vres.end());
          //res.push_back(detid);
          swap(res);
      }
  private:
    What  m_what;
  };


  template<>
  std::string
  PayLoadInspector<SiStripSummary>::dump() const {
    std::stringstream ss;
    std::vector<std::string>  listWhat= object().getUserDBContent();
    for(size_t i=0;i<listWhat.size();++i)
      ss << listWhat[i] << "###";
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<SiStripSummary>::summary() const {
    std::stringstream ss;
    ss << "Nr.Det " << object().getRegistryVectorEnd()-object().getRegistryVectorBegin()
       << "\nNr.Quantities " << object().getUserDBContent().size()
       << "\nNr.values " << object().getDataVectorEnd()-object().getDataVectorBegin()
       << "\nRunNr= " << object().getRunNr()
       << "\ntimeValue= " << object().getTimeValue();
    //ss << "names of DBquantities ";
    //std::vector<std::string>  listWhat= object().getUserDBContent();
    //for(size_t i=0;i<listWhat.size();++i)
    // ss << listWhat[i] << "\n";
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
    using namespace boost::python;
    enum_<sistripsummary::TrackerRegion>("TrackerRegion")
      .value("Tracker",sistripsummary::TRACKER)
      .value("TIB",sistripsummary::TIB)
      .value("TID",sistripsummary::TID)
      .value("TOB",sistripsummary::TOB)
      .value("TEC",sistripsummary::TEC)
      .value("TIB_L1",sistripsummary::TIB_1)
      .value("TIB_L2",sistripsummary::TIB_2)
      .value("TIB_L3",sistripsummary::TIB_3)
      .value("TIB_L4",sistripsummary::TIB_4)
      .value("TOB_L1",sistripsummary::TOB_1)
      .value("TOB_L2",sistripsummary::TOB_2)
      .value("TOB_L3",sistripsummary::TOB_3)
      .value("TOB_L4",sistripsummary::TOB_4)
      .value("TOB_L5",sistripsummary::TOB_5)
      .value("TOB_L6",sistripsummary::TOB_6)
      .value("TIDM_D1",sistripsummary::TIDM_1)
      .value("TIDM_D2",sistripsummary::TIDM_2)
      .value("TIDM_D3",sistripsummary::TIDM_3)
      .value("TIDP_D1",sistripsummary::TIDP_1)
      .value("TIDP_D2",sistripsummary::TIDP_2)
      .value("TIDP_D3",sistripsummary::TIDP_3)
      .value("TECP_D1",sistripsummary::TECP_1)
      .value("TECP_D2",sistripsummary::TECP_2)
      .value("TECP_D3",sistripsummary::TECP_3)
      .value("TECP_D4",sistripsummary::TECP_4)
      .value("TECP_D5",sistripsummary::TECP_5)
      .value("TECP_D6",sistripsummary::TECP_6)
      .value("TECP_D7",sistripsummary::TECP_7)
      .value("TECP_D8",sistripsummary::TECP_8)
      .value("TECP_D9",sistripsummary::TECP_9)
      .value("TECM_D1",sistripsummary::TECM_1)
      .value("TECM_D2",sistripsummary::TECM_2)
      .value("TECM_D3",sistripsummary::TECM_3)
      .value("TECM_D4",sistripsummary::TECM_4)
      .value("TECM_D5",sistripsummary::TECM_5)
      .value("TECM_D6",sistripsummary::TECM_6)
      .value("TECM_D7",sistripsummary::TECM_7)
      .value("TECM_D8",sistripsummary::TECM_8)
      .value("TECM_D9",sistripsummary::TECM_9)
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
