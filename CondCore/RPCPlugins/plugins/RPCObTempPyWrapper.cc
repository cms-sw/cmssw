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

  namespace rpcobtemp {
    enum How { detid, day, time, temp};

    void extractDetId(RPCObTemp const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObTemp::T_Item> const & imon = pl.ObTemp_rpc;
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
    
    void extractDay(RPCObTemp const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObTemp::T_Item> const & imon = pl.ObTemp_rpc;
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

    void extractTime(RPCObTemp const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObTemp::T_Item> const & imon = pl.ObTemp_rpc;
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

    void extractTemp(RPCObTemp const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime,const float& endtime) {
      std::vector<RPCObTemp::T_Item> const & imon = pl.ObTemp_rpc;
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

    typedef boost::function<void(RPCObTemp const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime,const float& endtime)> RPCObTempExtractor;
  }

  template<>
  struct ExtractWhat<RPCObTemp> {

    rpcobtemp::How m_how;
    std::vector<int> m_which;
    float m_starttime;
    float m_endtime;

    rpcobtemp::How const & how() const { return m_how;}
    std::vector<int> const & which() const { return m_which;}
    float const & startTime() const {return m_starttime;}
    float const & endTime() const {return m_endtime;}

    void set_how(rpcobtemp::How i) {m_how=i;}
    void set_which(std::vector<int> & i) { m_which.swap(i);}
    void set_starttime(float& i){m_starttime = i;}
    void set_endtime(float& i){m_endtime = i;}

  };


  template<>
  class ValueExtractor<RPCObTemp>: public  BaseValueExtractor<RPCObTemp> {
  public:

    static rpcobtemp::RPCObTempExtractor & extractor(rpcobtemp::How how) {
      static  rpcobtemp::RPCObTempExtractor fun[4] = { 
	rpcobtemp::RPCObTempExtractor(rpcobtemp::extractDetId),
	rpcobtemp::RPCObTempExtractor(rpcobtemp::extractDay),
	rpcobtemp::RPCObTempExtractor(rpcobtemp::extractTime),
	rpcobtemp::RPCObTempExtractor(rpcobtemp::extractTemp)
              };
      return fun[how];
    }

    typedef RPCObTemp Class;
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
  PayLoadInspector<RPCObTemp>::dump() const {}

  template<>
  std::string PayLoadInspector<RPCObTemp>::summary() const {
    std::stringstream ss;

    std::vector<RPCObTemp::T_Item> const & imon = object().ObTemp_rpc;

    for(unsigned int i = 0; i < imon.size(); ++i ){
      ss <<imon[i].dpid <<" "<<imon[i].value<<" "<<imon[i].time<<" "<<imon[i].day<<" ";
    }

    return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<RPCObTemp>::plot(std::string const & filename,
						   std::string const &, std::vector<int> const&, std::vector<float> const& ) const {
    std::string fname = filename + ".txt";
    std::ofstream f(fname.c_str());
    f << dump();
    return fname;
  }
  
}


namespace condPython {
  template<>
  void defineWhat<RPCObTemp>() {

    enum_<cond::rpcobtemp::How>("How")
      .value("detid",cond::rpcobtemp::detid)
      .value("day",cond::rpcobtemp::day) 
      .value("time",cond::rpcobtemp::time)
      .value("temp",cond::rpcobtemp::temp)
      ;

    typedef cond::ExtractWhat<RPCObTemp> What;
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


PYTHON_WRAPPER(RPCObTemp,RPCObTemp);



