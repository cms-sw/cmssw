#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

/////////////////
#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>
#include <fstream>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TH1D.h"
#include "TH2D.h"


namespace cond {

  namespace rpcobimon {
    enum How { detid, day, time, current};

    void extractDetId(RPCObVmon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObVmon::V_Item> const & imon = pl.ObVmon_rpc;
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
    
    void extractDay(RPCObVmon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObVmon::V_Item> const & imon = pl.ObVmon_rpc;
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

    void extractTime(RPCObVmon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime, const float& endtime) {
      std::vector<RPCObVmon::V_Item> const & imon = pl.ObVmon_rpc;
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

    void extractCurrent(RPCObVmon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime,const float& endtime) {
      std::vector<RPCObVmon::V_Item> const & imon = pl.ObVmon_rpc;
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

    typedef boost::function<void(RPCObVmon const & pl, std::vector<int> const & which,std::vector<float> & result,const float& starttime,const float& endtime)> RPCObVmonExtractor;
  }

  template<>
  struct ExtractWhat<RPCObVmon> {

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
  class ValueExtractor<RPCObVmon>: public  BaseValueExtractor<RPCObVmon> {
  public:

    static rpcobimon::RPCObVmonExtractor & extractor(rpcobimon::How how) {
      static  rpcobimon::RPCObVmonExtractor fun[4] = { 
	rpcobimon::RPCObVmonExtractor(rpcobimon::extractDetId),
	rpcobimon::RPCObVmonExtractor(rpcobimon::extractDay),
	rpcobimon::RPCObVmonExtractor(rpcobimon::extractTime),
	rpcobimon::RPCObVmonExtractor(rpcobimon::extractCurrent)
              };
      return fun[how];
    }

    typedef RPCObVmon Class;
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


//   template<>
//   std::string
//   PayLoadInspector<RPCObVmon>::dump() const {return std::string();}

  template<>
  std::string PayLoadInspector<RPCObVmon>::summary() const {
    std::stringstream ss;

    std::vector<RPCObVmon::V_Item> const & imon = object().ObVmon_rpc;
    
    ss <<"DetID\t"<<"Vval\t"<<"Time\t"<<"Day\n";
    for(unsigned int i = 0; i < imon.size(); ++i ){
      //      for(unsigned int p = 0; p < ipvss.size(); ++p){
      // 	if(imon[i].dpid()!=ipvss[p].dpid())continue;
      // 	RPCDetId rpcId(ipvss[p].region(),ipvss[p].ring(),ipvss[p].station(),ipvss[p].sector(),ipvss[p].layer(),ipvss[p].subsector(),1);
      // 	RPCGeomServ rGS(rpcId);
      // 	std::string chName(rGS.name().substr(0,rGS.name().find("_BACKWARD")));
      ss <<imon[i].dpid <<"\t"<<imon[i].value<<"\t"<<imon[i].time<<"\t"<<imon[i].day<<"\n";
      //      }
    }
    
    return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<RPCObVmon>::plot(std::string const & filename,
						std::string const &,
						std::vector<int> const&,
						std::vector<float> const& ) const {

    TCanvas canvas("iC","iC",800,800);    

    TH1D *iDistr=new TH1D("iDistr","IOV-averaged Vmon Distribution;Average Current(uA);Entries/1uA",100,0.,100.);

    std::vector<RPCObVmon::V_Item> const & imon = object().ObVmon_rpc;

    int tempId(0),count(0);
    double tempAve(0.);
    for(unsigned int i = 0;i < imon.size(); ++i){
      if(i==0){
	count++;
	tempAve+=imon[i].value;
	tempId=imon[i].dpid;
      }
      else {
	if(imon[i].dpid==tempId){
	  count++;
	  tempAve+=imon[i].value;	  
	}
	else{
	  iDistr->Fill(tempAve/(double)count);
	  count=1;
	  tempAve=imon[i].value;
	  tempId=imon[i].dpid;
	}
      }
    }


    iDistr->Draw();

    canvas.SaveAs(filename.c_str());
    return filename.c_str();

    iDistr->Draw();

    //     std::string fname = filename + ".txt";
    //     std::ofstream f(fname.c_str());
    //     f << summary();
    //     return fname;

   return std::string();

  }
  
}


namespace condPython {
  template<>
  void defineWhat<RPCObVmon>() {

    enum_<cond::rpcobimon::How>("How")
      .value("detid",cond::rpcobimon::detid)
      .value("day",cond::rpcobimon::day) 
      .value("time",cond::rpcobimon::time)
      .value("current",cond::rpcobimon::current)
      ;

    typedef cond::ExtractWhat<RPCObVmon> What;
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

PYTHON_WRAPPER(RPCObVmon,RPCObVmon);



