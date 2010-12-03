#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
//#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"

//#include "CondCore/ORA/interface/Database.h"
//#include "CondCore/DBCommon/interface/PoolToken.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"
//#include "CondCore/IOVService/interface/IOVProxy.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

//timestamp stuff
//#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
#include <sys/time.h>

/////////////////
#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "TVirtualPad.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TF1.h"
#include "TDatime.h"

#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>
#include <fstream>
#include <utility>
#include <iomanip>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/iterator/transform_iterator.hpp>

using std::pair;
using std::make_pair;

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
  std::string PayLoadInspector<RPCObImon>::summary() const {

    std::stringstream ss;
   
    //BEGIN OF NEW DB-SESSION PART
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
    //cond::DbTransaction dbTr=
    dbSes.transaction().start(true);
    
    //get the actual object
    boost::shared_ptr<RPCObPVSSmap> pvssPtr;
    pvssPtr=dbSes.getTypedObject<RPCObPVSSmap>(token);

    //we have the object...
    std::vector<RPCObPVSSmap::Item> pvssCont=pvssPtr->ObIDMap_rpc;
    //END OF NEW DB-SESSION PART

    std::vector<RPCObImon::I_Item> const & imon = object().ObImon_rpc;
    
    ss<<"DetID\t\t\tI(uA)\tTime\tDay\n";
    for(unsigned int i = 0; i < imon.size(); ++i ){
      for(unsigned int p = 0; p < pvssCont.size(); ++p){
	if(imon[i].dpid!=pvssCont[p].dpid || pvssCont[p].suptype!=0 || pvssCont[p].region!=0)continue;
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

  
  Double_t linearF(Double_t *x, Double_t *par){
    Double_t y=0.;
    y=par[0]*(*x);
    return y;
  }
  
  unsigned long long toUNIX(int date, int time)
  {
    int yea_ = (int)date/100; 
    int yea = 2000 + (date - yea_*100);
    int mon_ = (int)yea_/100;
    int mon = yea_ - mon_*100;
    int day = (int)yea_/100;
    int sec_ = (int)time/100;
    int sec = time - sec_*100;
    int min_ = (int)sec_/100;
    int min = sec_ - min_*100;
    int hou = (int)sec_/100;
    int nan = 0;
    coral::TimeStamp TS;  
    TS = coral::TimeStamp(yea, mon, day, hou, min, sec, nan);
    
    long long UT = (TS.year()-1970)*31536000+static_cast<int>(trunc((TS.year()-1972)/4))*86400+
      (((TS.month()-1)*31)*86400)+((TS.day()-1)*86400)+TS.hour()*3600+TS.minute()*60+TS.second();
    
    if (TS.month() == 3) UT = UT - 3*86400;
    if (TS.month() == 4) UT = UT - 3*86400;
    if (TS.month() == 5) UT = UT - 4*86400;
    if (TS.month() == 6) UT = UT - 4*86400;
    if (TS.month() == 7) UT = UT - 5*86400;
    if (TS.month() == 8) UT = UT - 5*86400;
    if (TS.month() == 9) UT = UT - 5*86400;
    if (TS.month() == 10) UT = UT - 6*86400;
    if (TS.month() == 11) UT = UT - 6*86400;
    if (TS.month() == 12) UT = UT - 7*86400;
    
    return UT;
  }
  
  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<RPCObImon>::plot(std::string const & filename,
						std::string const & str,
						std::vector<int> const & vInt,
						std::vector<float> const& vFlt) const {

    gStyle->SetPalette(1);

    TH1D *iDistr=new TH1D("iDistr","IOV-averaged Current Distribution;Average Current (#muA);Entries/0.2 #muA",500,0.,100.);
    TH1D *rmsDistr=new TH1D("rmsDistr","RMS over IOV-Current Distribution;Current RMS (#muA);Entries/0.2 #muA",5000,0.,1000.);

    //BEGIN OF NEW DB-SESSION PART
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
    //END OF NEW DB-SESSION PART

    std::vector<RPCObImon::I_Item> const & imon = object().ObImon_rpc;
    
    //    RPCRunIOV *iovHelp=new RPCRunIOV();
    
    std::map<int,std::pair<std::vector<double>,std::vector<double> > > dpidMap;
    for(unsigned int i = 0;i < imon.size(); ++i){
      for(unsigned int p = 0; p < pvssCont.size(); ++p){
	if(imon[i].dpid!=pvssCont[p].dpid || pvssCont[p].suptype!=0 || pvssCont[p].region!=0)continue;
	if(dpidMap.find(imon[i].dpid)==dpidMap.end()){
	  std::vector<double> dumVec1;dumVec1.push_back(imon[i].value);
	  std::vector<double> dumVec2;dumVec2.push_back((double)/*iovHelp->*/toUNIX(imon[i].day,imon[i].time));
	  dpidMap[imon[i].dpid]=make_pair(dumVec1,dumVec2);
	}
	else {
	  dpidMap[imon[i].dpid].first.push_back(imon[i].value);
	  dpidMap[imon[i].dpid].second.push_back((double)/*iovHelp->*/toUNIX(imon[i].day,imon[i].time));
	}
      }
    }
    
    //    delete iovHelp;

    double maxMean(-1.),maxRms(-1.);
    double minMean(9999.),minRms(9999.);
    std::map<int,std::pair<std::vector<double>,std::vector<double> > >::const_iterator minIt,maxIt;
    std::vector<float> means,rmss;
    for(std::map<int,std::pair<std::vector<double>,std::vector<double> > >::const_iterator mIt=dpidMap.begin();
	mIt!=dpidMap.end();mIt++){
      std::pair<double, std::vector<double> > meanAndVals =
	make_pair(accumulate(mIt->second.first.begin(),mIt->second.first.end(),0.)/(double)mIt->second.first.size(),mIt->second.first);
       
      iDistr->Fill(meanAndVals.first);
      if(meanAndVals.first>maxMean)maxMean=meanAndVals.first;
      if(meanAndVals.first<minMean)minMean=meanAndVals.first;
      double rms(0.);
      for(std::vector<double>::iterator rmsIt=meanAndVals.second.begin();
	  rmsIt!=meanAndVals.second.end();++rmsIt){
	rms+=pow((*rmsIt-meanAndVals.first)/(double)meanAndVals.second.size(),2);
      }
      rmsDistr->Fill(sqrt(rms));
      if(sqrt(rms)>maxRms){
	maxRms=sqrt(rms);
	maxIt=mIt;
      }
      if(sqrt(rms)<minRms){
	minRms=sqrt(rms);
	if(mIt->second.first.size()>10)
	  minIt=mIt;
      }
      means.push_back(meanAndVals.first);
      rmss.push_back(sqrt(rms));
    }

    if(maxMean<100.)
      iDistr->GetXaxis()->SetRangeUser(minMean-0.00001,maxMean+1.);
    if(maxRms<1000.)
      rmsDistr->GetXaxis()->SetRangeUser(minRms-0.00001,maxRms+1.);
    
    TCanvas c("Imon","Imon",1200,700);
    c.Divide(2,2);
    
    TVirtualPad *p1=c.cd(1);
    p1->SetLogy(1);
    iDistr->SetFillColor(4);
    iDistr->SetLineColor(4);
    iDistr->Draw();
    c.cd(1);
    
    TVirtualPad *p2=c.cd(2);
    p2->SetLogy(1);
    rmsDistr->SetFillColor(3);
    rmsDistr->SetLineColor(3);
    rmsDistr->Draw();

    c.cd(3);
    TGraph *iRmsDistr=new TGraph(means.size(),static_cast<const float *>(&rmss[0]),static_cast<const float *>(&means[0]));
    iRmsDistr->SetMarkerStyle(7);
    iRmsDistr->SetMarkerColor(2);
    TF1 *func=new TF1("linearF",linearF,minRms,maxRms,1);
    iRmsDistr->Fit("linearF","r");
    iRmsDistr->GetXaxis()->SetTitle("Current RMS (#muA)");
    iRmsDistr->GetYaxis()->SetTitle("Current Means (#muA)");
    iRmsDistr->Draw("AP");
    

    TVirtualPad *p4=c.cd(4);
    p4->SetLogy(1);
    TMultiGraph *mProf=new TMultiGraph();
    TLegend *leg=new TLegend(0.65,0.91,0.99,0.99);
    TGraph *minProf=new TGraph(minIt->second.first.size(),&minIt->second.second[0],&minIt->second.first[0]);
    TGraph *maxProf=new TGraph(maxIt->second.first.size(),&maxIt->second.second[0],&maxIt->second.first[0]);
    minProf->SetMarkerStyle(20);
    maxProf->SetMarkerStyle(20);
    minProf->SetMarkerColor(2);
    maxProf->SetMarkerColor(4);
    mProf->Add(minProf);
    leg->AddEntry(minProf,"I vs IOV for Min RMS case","lpf");
    mProf->Add(maxProf);
    leg->AddEntry(maxProf,"I vs IOV for Max RMS case","lpf");
    leg->Draw();
    mProf->Draw("AP");
    mProf->GetXaxis()->SetTitle("IOV");
    mProf->GetYaxis()->SetTitle("I (#muA)");

    c.SaveAs(filename.c_str());

    return filename.c_str();

  }
  
  template <>
  std::string PayLoadInspector<RPCObImon>::trend_plot(std::string const & filename,
						      std::string const & opt_string, 
						      std::vector<int> const& nts,
						      std::vector<float> const& floats,
						      std::vector<std::string> const& strings) const {

    std::stringstream ss("");
    
    if(strings.size()<2)
      return ("Error!Not enough data for initializing connection for making plots!(from template<> std::string PayLoadInspector<BeamSpotObjects>::trend_plot)");
   
    std::vector<std::string>::const_iterator strIt=strings.begin();

    std::string conString=(*strIt);strIt++;
    std::string authPath=(*strIt);strIt++;
 
    //BEGIN OF NEW DB-SESSION PART

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
    
    std::string pvssToken="[DB=00000000-0000-0000-0000-000000000000][CNT=RPCObPVSSmap][CLID=53B2D2D9-1F4E-9CA9-4D71-FFCCA123A454][TECH=00000B01][OID=0000000C-00000000]";
    //get the actual object
    boost::shared_ptr<RPCObImon> imonPtr;
    boost::shared_ptr<RPCObPVSSmap> pvssPtr;
    pvssPtr=dbSes.getTypedObject<RPCObPVSSmap>(pvssToken);

    //we get the objects...
    std::vector<RPCObPVSSmap::Item> pvssCont=pvssPtr->ObIDMap_rpc,pvssZero;

    for(unsigned int p=0;p<pvssCont.size();p++){
      if(pvssCont[p].suptype!=0)
	pvssZero.push_back(pvssCont[p]);
    }
    
    std::string token;
    std::vector<float> vecI;vecI.assign(floats.size(),0.);
    int elCount(0);

    for(;strIt!=strings.end();++strIt){
      
      token=(*strIt);
      imonPtr=dbSes.getTypedObject<RPCObImon>(token);      
      std::vector<RPCObImon::I_Item> const & imon = imonPtr->ObImon_rpc;

      float iCount(0.);
      for(unsigned int i=0;i<imon.size();i++){
	for(unsigned int p=0;p<pvssZero.size();p++){
	  if(imon[i].dpid!=pvssZero[p].dpid || pvssZero[p].region!=0)continue;
	  iCount++;
	  vecI[elCount]+=imon[i].value;
	}
      }
      if(iCount!=0)
	vecI[elCount]/=iCount;
      elCount++;
    }
    
    //END OF NEW DB-SESSION PART
    
    dbSes.close();
    
    std::vector<unsigned long long> lngs ;
    for(std::vector<float>::const_iterator fIt=floats.begin();fIt!=floats.end();fIt++){
      lngs.push_back((unsigned long long)*fIt);
      //      std::cout<<*fIt<<" "<<(long double)*fIt<<" "<<(unsigned long long)*fIt<<" "<<vecI[0]<<" "<<(unsigned long long)vecI[0]<<std::endl;
    }
    std::vector<unsigned long long> const longs=lngs;

    //    TGraph trend(vecI.size(),static_cast<const float *>(&longs[0]),static_cast<const float *>(&vecI[0]));
    std::cout<<(int)longs[longs.size()-1]-longs[0]<<std::endl;
    TH1F trend("trend","trend",(int)longs[longs.size()-1]-longs[0],longs[0],longs[longs.size()-1]);
    //TH1F trend("trend","trend",floats.size(),static_cast<const float* >(&floats[0]));
    trend.GetXaxis()->SetTimeDisplay(1);
    trend.GetXaxis()->SetTimeFormat("%d/%m/%y %H:%M");
    trend.SetLineColor(2);
    trend.SetLineWidth(2);
    trend.GetYaxis()->SetTitle("<I> (#muA)");

    std::cout<<"Bins "<<trend.GetNbinsX()<<std::endl;
    for(unsigned int fill=0;fill<=longs.size();fill++){
      trend.SetBinContent(longs[fill]-longs[0],vecI[fill]);
      std::cout<<fill<<" "<<floats[fill]<<" "<<longs[fill]-longs[0]<<" "<<vecI[fill]<<std::endl;
    }

    float min(*(floats.begin())),max(*(floats.end()-1));
    float scaleToggle((max-min)/max);
    TCanvas c("Itrend","Itrend",1200,700);

    if(scaleToggle>=0.1)
      c.SetLogx(1);

    trend.Draw(/*"LA*"*/);

    TLegend leg(0.65,0.91,0.99,0.99);
    leg.AddEntry(&trend,"Imon trend","lpf");

    leg.Draw("");

    ss.str("");
    ss<<filename<<".C";
    c.SaveAs((ss.str()).c_str());

    return ss.str();
    
  }
  
}


namespace condPython {
  template<>
  void defineWhat<RPCObImon>() {
    using namespace boost::python;
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
