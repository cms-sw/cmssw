#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/DbSession.h"

#include "CondCore/ORA/interface/Database.h"
#include "CondCore/DBCommon/interface/PoolToken.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"
#include "CondCore/IOVService/interface/IOVProxy.h"

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
#include <boost/shared_ptr.hpp>
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

  //   template<>
  //   std::string
  //   PayLoadInspector<RPCObImon>::dump() const {return std::string();}

  template<>
  std::string PayLoadInspector<RPCObImon>::summary() const {

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

    std::stringstream ss;

    std::vector<RPCObImon::I_Item> const & imon = object().ObImon_rpc;
    
    ss <<"DetID\t\t"<<"I(uA)\t"<<"Time\t"<<"Day\n";
    for(unsigned int i = 0; i < imon.size(); ++i ){
      for(unsigned int p = 0; p < pvssCont.size(); ++p){
       	if(imon[i].dpid!=pvssCont[p].dpid ||
	   pvssCont[p].ring==0 || pvssCont[p].station==0 ||
	   pvssCont[p].sector==0 || pvssCont[p].layer==0 ||
	   pvssCont[p].subsector==0)continue;
       	RPCDetId rpcId(pvssCont[p].region,pvssCont[p].ring,pvssCont[p].station,pvssCont[p].sector,pvssCont[p].layer,pvssCont[p].subsector,1);
       	RPCGeomServ rGS(rpcId);
       	std::string chName(rGS.name().substr(0,rGS.name().find("_Backward")));
	ss <<chName <<"\t"<<imon[i].value<<"\t"<<imon[i].time<<"\t"<<imon[i].day<<"\n";
      }
    }
    
    //close db session
    dbSes.close();

    return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<RPCObImon>::plot(std::string const & filename,
						std::string const &,
						std::vector<int> const&,
						std::vector<float> const& ) const {

    TCanvas canvas("iC","iC",800,800);    

    TH1D *iDistr=new TH1D("iDistr","IOV-averaged Imon Distribution;Average Current(uA);Entries/1uA",100,0.,100.);

    std::vector<RPCObImon::I_Item> const & imon = object().ObImon_rpc;

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



