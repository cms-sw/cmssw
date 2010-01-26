#include "CondFormats/RPCObjects/interface/RPCObCond.h"

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
#include <iostream>
#include <fstream>

namespace cond {

  namespace rpcobimon {
    enum How { detid, day, time, current};

    void extractDetId(RPCObImon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObImon::I_Item> const & imon = pl.ObImon_rpc;
      for(unsigned int i = 0; i < imon.size(); ++i ){
	if (which[0] == 0){
	  result.push_back(imon[i].dpid);
	}
	else{
	  if(imon[i].dpid == which[0])
	    result.push_back(imon[i].dpid);
	}
      }
    }
    
    void extractDay(RPCObImon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObImon::I_Item> const & imon = pl.ObImon_rpc;
      for(unsigned int i = 0; i < imon.size(); ++i ){
	if (which[0] == 0){
	  result.push_back(imon[i].day);
	}
	else{
	  if(imon[i].dpid == which[0])
	    result.push_back(imon[i].day);
	}
      }
    }

    void extractTime(RPCObImon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObImon::I_Item> const & imon = pl.ObImon_rpc;
      for(unsigned int i = 0; i < imon.size(); ++i ){
	if (which[0] == 0){
	  result.push_back(imon[i].time);
	}
	else{
	  if(imon[i].dpid == which[0])
	    result.push_back(imon[i].time);
	}
      }
    }

    void extractCurrent(RPCObImon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime,const float& endtime) {
      std::vector<RPCObImon::I_Item> const & imon = pl.ObImon_rpc;
      for(unsigned int i = 0; i < imon.size(); ++i ){
	if (which[0] == 0){
	  result.push_back(imon[i].value);
	}
	else{
	  if(imon[i].dpid == which[0])
	    result.push_back(imon[i].value);
	}
      }
    }

    typedef boost::function<void(RPCObImon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime,const float& endtime)> RPCObImonExtractor;
  }

  template<>
  struct ExtractWhat<RPCObImon> {

    rpcobimon::How m_how;
    std::vector<int> m_which;
    float m_starttime;
    float m_endtime;

    rpcobimon::How const & how() const { return m_how;}
    std::vector<int> const & which() const { return m_which;}
    float const & startTime() const {return m_starttime;}
    float const & endTime() const {return m_endtime;}

    void set_how(rpcobimon::How i) {m_how=i;}
    void set_which(std::vector<int> & i) { m_which.swap(i);}
    void set_starttime(float& i){m_starttime = i;}
    void set_endtime(float& i){m_endtime = i;}

  };


  template<>
  class ValueExtractor<RPCObImon>: public  BaseValueExtractor<RPCObImon> {
  public:

    static rpcobimon::RPCObImonExtractor & extractor(rpcobimon::How how) {
      static  rpcobimon::RPCObImonExtractor fun[4] = { 
	rpcobimon::RPCObImonExtractor(rpcobimon::extractDetId),
	rpcobimon::RPCObImonExtractor(rpcobimon::extractDay),
	rpcobimon::RPCObImonExtractor(rpcobimon::extractTime),
	rpcobimon::RPCObImonExtractor(rpcobimon::extractCurrent)
              };
      return fun[how];
    }

    typedef RPCObImon Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
      : m_what(what)
    {
      // here one can make stuff really complicated... (select mean rms, 12,6,1)
      // ask to make average on selected channels...
    }

    void compute(Class const & it){
      std::vector<float> res;
      extractor(m_what.how())(it,m_what.which(),res,m_what.startTime(),m_what.endTime());
      swap(res);
    }

  private:
    What  m_what;
    
  };


  template<>
  std::string
  PayLoadInspector<RPCObImon>::dump() const {return std::string();}

  template<>
  std::string PayLoadInspector<RPCObImon>::summary() const {
    std::stringstream ss;

    std::vector<RPCObImon::I_Item> const & imon = object().ObImon_rpc;

    for(unsigned int i = 0; i < imon.size(); ++i ){
      ss <<imon[i].dpid <<" "<<imon[i].value<<" "<<imon[i].time<<" "<<imon[i].day<<" ";
    }

    return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<RPCObImon>::plot(std::string const & filename,
						   std::string const &, std::vector<int> const&, std::vector<float> const& ) const {
    std::string fname = filename + ".txt";
    std::ofstream f(fname.c_str());
    f << dump();
    return fname;
  }
  
}


namespace condPython {
  template<>
  void defineWhat<RPCObImon>() {

    enum_<cond::rpcobimon::How>("How")
      .value("detid",cond::rpcobimon::detid)
      .value("day",cond::rpcobimon::day) 
      .value("time",cond::rpcobimon::time)
      .value("current",cond::rpcobimon::current)
      ;

    typedef cond::ExtractWhat<RPCObImon> What;
    class_<What>("What",init<>())
      .def("set_how",&What::set_how)
      .def("set_which",&What::set_which)
      .def("how",&What::how, return_value_policy<copy_const_reference>())
      .def("which",&What::which, return_value_policy<copy_const_reference>())
      .def("set_starttime",&What::set_starttime)
      .def("set_endtime",&What::set_endtime)
      .def("startTime",&What::startTime, return_value_policy<copy_const_reference>())
      .def("endTime",&What::endTime, return_value_policy<copy_const_reference>())
      ;
  }
}

PYTHON_WRAPPER(RPCObImon,RPCObImon);



