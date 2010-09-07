#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/DbSession.h"

#include "CondCore/ORA/interface/Database.h"
#include "CondCore/DBCommon/interface/PoolToken.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "TH1D.h"
#include "TH2D.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <iostream>
#include <fstream>

#include <utility>
using std::pair;
using std::make_pair;

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
  PayLoadInspector<RPCObTemp>::dump() const {return std::string();}

  template<>
  std::string PayLoadInspector<RPCObTemp>::summary() const {
    std::stringstream ss;

    //hardcoded values
    std::string authPath="/afs/cern.ch/cms/DB/conddb";
    std::string conString="oracle://cms_orcoff_prod/CMS_COND_31X_RPC";

    //frontend sends token instead of filename
    std::string token="[DB=00000000-0000-0000-0000-000000000000][CNT=RPCObPVSSmap][CLID=53B2D2D9-1F4E-9CA9-4D71-FFCCA123A454][TECH=00000B01][OID=0000000C-00000000]";

    //make connection object
    DbConnection dbConn;

    //set in configuration object authentication path
    dbConn.configuration().setAuthenticationPath(authPath);
    dbConn.configure();

    //create session object from connection
    DbSession dbSes=dbConn.createSession();

    //try to make connection
    dbSes.open(conString,true);
    
    //start a transaction (true=readOnly)
    dbSes.transaction().start(true);

    //get the actual object
    boost::shared_ptr<RPCObPVSSmap> pvssPtr;
    pvssPtr=dbSes.getTypedObject<RPCObPVSSmap>(token);

    //we have the object...
    std::vector<RPCObPVSSmap::Item> pvssCont=pvssPtr->ObIDMap_rpc;

    std::vector<RPCObTemp::T_Item> const & tmon = object().ObTemp_rpc;

    ss <<"DetID\t\t"<<"T(C)\t"<<"Time\t"<<"Day\n";
    for(unsigned int i = 0; i < tmon.size(); ++i ){
      for(unsigned int p = 0; p < pvssCont.size(); ++p){
       	if(tmon[i].dpid!=pvssCont[p].dpid || pvssCont[p].suptype!=4 || pvssCont[p].region!=0)continue;
	RPCDetId rpcId(pvssCont[p].region,pvssCont[p].ring,pvssCont[p].station,pvssCont[p].sector,pvssCont[p].layer,pvssCont[p].subsector,1);
	RPCGeomServ rGS(rpcId);
	std::string chName(rGS.name().substr(0,rGS.name().find("_Backward")));
	ss <<chName <<"\t"<<tmon[i].value<<"\t"<<tmon[i].time<<"\t"<<tmon[i].day<<"\n";
      }
    }

    dbSes.close();

    return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<RPCObTemp>::plot(std::string const & filename,
						   std::string const &, std::vector<int> const&, std::vector<float> const& ) const {

    TCanvas canvas("Temp","Temp",800,800);

    TH1D *tDistr=new TH1D("tDistr","IOV-averaged Temperature Distribution;Average Temp(C);Entries/0.5 C ",40,10.,30.);

    std::vector<RPCObTemp::T_Item> const & tmon = object().ObTemp_rpc;

    std::map<int,std::pair<int,double> > dpidMap;
    for(unsigned int i = 0;i < tmon.size(); ++i){
      if(dpidMap.find(tmon[i].dpid)==dpidMap.end())
	dpidMap[tmon[i].dpid]=make_pair(1,(double)tmon[i].value);
      else {
	dpidMap[tmon[i].dpid].first++;
	dpidMap[tmon[i].dpid].second+=tmon[i].value;
      }
    }

    for(std::map<int,std::pair<int,double> >::const_iterator mIt=dpidMap.begin();mIt!=dpidMap.end();mIt++)
      tDistr->Fill(mIt->second.second/(double)mIt->second.first);

    tDistr->Draw();

    canvas.SaveAs(filename.c_str());
    return filename.c_str();

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
