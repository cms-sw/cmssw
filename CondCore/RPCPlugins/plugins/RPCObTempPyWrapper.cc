#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
//#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"

//#include "CondCore/ORA/interface/Database.h"
//#include "CondCore/DBCommon/interface/PoolToken.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
//#include "DataFormats/GeometryVector/interface/LocalPoint.h"
//#include "DataFormats/GeometrySurface/interface/Surface.h"

//#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

//timestamp stuff
//#include "DataFormats/Provenance/interface/Timestamp.h"
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
#include "TPaveStats.h"
#include "TPaveText.h"

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

  Double_t linearF(Double_t *x, Double_t *par){
    Double_t y=0.;
    y=par[0]*(*x);
    return y;
  }

  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<RPCObTemp>::plot(std::string const & filename,
						std::string const &, std::vector<int> const&, std::vector<float> const& ) const {

    std::map<std::string,std::pair<float,float> > geoMap;
    std::ifstream mapFile("/afs/cern.ch/user/s/stupputi/public/barDetPositions.txt",ifstream::in);
    std::string chamb;
    float xPos,yPos;
    while(mapFile >> chamb >> xPos >> yPos)
      geoMap[chamb]=make_pair(xPos,yPos);
    std::cout<<"size "<<geoMap.size()<<std::endl;
    gStyle->SetPalette(1);    

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

    std::vector<RPCObTemp::T_Item> const & temp = object().ObTemp_rpc;
    
    std::map<int,std::map<int,std::vector<double> > > valMap;
    std::map<int,std::string> detMap;
    for(unsigned int i = 0;i < temp.size(); ++i){
      for(unsigned int p = 0; p < pvssCont.size(); ++p){
	if(temp[i].dpid!=pvssCont[p].dpid || pvssCont[p].suptype!=4 || pvssCont[p].region!=0)continue;
	int whl(pvssCont[p].ring);
	if(valMap.find(whl)==valMap.end()){	  
	  std::map<int,std::vector<double> > dumMap;
	  std::vector<double> dumVec;dumVec.push_back(temp[i].value);
	  dumMap[temp[i].dpid]=dumVec;
	  valMap[whl]=dumMap;
	}
	else {
	  if(valMap[whl].find(temp[i].dpid)==valMap[whl].end()){
	    std::vector<double> dumVec;dumVec.push_back(temp[i].value);
	    (valMap[whl])[temp[i].dpid]=dumVec;
	  }
	  else 
	    (valMap[whl])[temp[i].dpid].push_back(temp[i].value);
	}
	if(detMap.find(temp[i].dpid)==detMap.end()){
	  RPCDetId rpcId(pvssCont[p].region,pvssCont[p].ring,pvssCont[p].station,pvssCont[p].sector,pvssCont[p].layer,pvssCont[p].subsector,1);
	  RPCGeomServ chGS(rpcId);
	  std::string chName(chGS.name());
	  std::string::size_type pos(chName.find("_S"));
	  if(pos!=std::string::npos)
	    chName=chName.substr(0,pos+5);
	  else
	    chName=chName.substr(0,chName.find("_CH")+5);	  	  
	  detMap[temp[i].dpid]=chName;
 	}	
      }
    }
    
    float histoScale(710),axisScale(750);
    int flip(-1);
    float incr(0.045*histoScale);
    float midSc(0.267949192431123),diagSc(sqrt((1.+(midSc*midSc))/2.));
    float dists[]={0.558*histoScale,0.607*histoScale,0.697*histoScale,/*0.6971*xSc,0.7464*xSc,*/0.823*histoScale,0.967*histoScale};

    TCanvas *c=new TCanvas("Temp","Temp",1200,700);
    c->Divide(3,2);

    for(int w=-2;w<3;w++){
      //  if(w!=1)continue;
      int canv(w+3);
      c->cd(canv);

      char wNumb[8];
      sprintf(wNumb,"Wheel %i",w);

      TH2F *histo= new TH2F("","",(2*axisScale)/incr,(-1)*axisScale,axisScale,(2*axisScale)/incr,(-1)*axisScale,axisScale);
      histo->GetXaxis()->SetTickLength(0);histo->GetXaxis()->SetLabelColor(0);histo->GetXaxis()->SetTitle("Temperature (#circC)");
      histo->GetYaxis()->SetTickLength(0);histo->GetYaxis()->SetLabelColor(0);
    
      double min(1000),max(-1);
      for(std::map<int,std::vector<double> >::const_iterator mIt=valMap[w].begin();
	  mIt!=valMap[w].end();mIt++){
       
	std::pair<double, std::vector<double> > meanAndVals =
	  make_pair(accumulate(mIt->second.begin(),mIt->second.end(),0.)/(double)mIt->second.size(),mIt->second);
	std::string chName(detMap[mIt->first]);
	histo->Fill(geoMap[chName].first,geoMap[chName].second,meanAndVals.first);
	if(meanAndVals.first<min)min=meanAndVals.first;
	if(meanAndVals.first>max)max=meanAndVals.first;
	std::cout<<chName<<" "<<geoMap[chName].first<<" "<<geoMap[chName].second<<" "<<meanAndVals.first<<std::endl;
      }

      histo->GetZaxis()->SetRangeUser(min,max);
      histo->Draw("colz");

      TPaveStats *ptstats = new TPaveStats(0.98,0.98,0.99,0.99,"br");
      histo->GetListOfFunctions()->Add(ptstats);
      histo->Draw("same");

      TPaveText *pt=new TPaveText(-235,763,248,860,"br");
      pt->SetFillColor(19);
      pt->AddText(wNumb);
      pt->Draw("same");

      TLine *line = new TLine();
      for(int i=0;i<5;i++){
	float scDist(midSc*dists[i]),onDiag(diagSc*dists[i]);
	float furDist(dists[i]+incr),scFurDist(midSc*furDist),furOnDiag(diagSc*furDist);
      
	line=new TLine(dists[i],flip*scDist,dists[i],scDist);line->Draw();//S01
	line=new TLine(dists[i],flip*scDist,furDist,flip*scDist);line->Draw();
	line=new TLine(dists[i],scDist,furDist,scDist);line->Draw();
	line=new TLine(furDist,flip*scDist,furDist,scDist);line->Draw();
	line=new TLine(flip*dists[i],flip*scDist,flip*dists[i],scDist);line->Draw();//S07		
	line=new TLine(flip*dists[i],flip*scDist,flip*furDist,flip*scDist);line->Draw();
	line=new TLine(flip*dists[i],scDist,flip*furDist,scDist);line->Draw();
	line=new TLine(flip*furDist,flip*scDist,flip*furDist,scDist);line->Draw();		
	line=new TLine(onDiag,onDiag,dists[i],scDist);line->Draw();//S02
	line=new TLine(onDiag,onDiag,furOnDiag,furOnDiag);line->Draw();
	line=new TLine(dists[i],scDist,furDist,scFurDist);line->Draw();
	line=new TLine(furOnDiag,furOnDiag,furDist,scFurDist);line->Draw();
	line=new TLine(flip*onDiag,flip*onDiag,flip*dists[i],flip*scDist);line->Draw();//S08		
	line=new TLine(flip*onDiag,flip*onDiag,flip*furOnDiag,flip*furOnDiag);line->Draw();
	line=new TLine(flip*dists[i],flip*scDist,flip*furDist,flip*scFurDist);line->Draw();
	line=new TLine(flip*furOnDiag,flip*furOnDiag,flip*furDist,flip*scFurDist);line->Draw();
	line=new TLine(onDiag,onDiag,scDist,dists[i]);line->Draw();//S03
	line=new TLine(onDiag,onDiag,furOnDiag,furOnDiag);line->Draw();
	line=new TLine(scDist,dists[i],scFurDist,furDist);line->Draw();
	line=new TLine(furOnDiag,furOnDiag,scFurDist,furDist);line->Draw();
	line=new TLine(flip*onDiag,flip*onDiag,flip*scDist,flip*dists[i]);line->Draw();//S09
	line=new TLine(flip*onDiag,flip*onDiag,flip*furOnDiag,flip*furOnDiag);line->Draw();
	line=new TLine(flip*scDist,flip*dists[i],flip*scFurDist,flip*furDist);line->Draw();
	line=new TLine(flip*furOnDiag,flip*furOnDiag,flip*scFurDist,flip*furDist);line->Draw();
	if(i==4){
	
	  line=new TLine(flip*scDist,dists[i],-0.005,dists[i]);line->Draw();//S04-
	  line=new TLine(flip*scDist,dists[i],flip*scDist,furDist);line->Draw();
	  line=new TLine(-0.005,dists[i],-0.005,furDist);line->Draw();
	  line=new TLine(flip*scDist,furDist,-0.005,furDist);line->Draw();			
	  line=new TLine(0.005,dists[i],scDist,dists[i]);line->Draw();//S04+
	  line=new TLine(scDist,dists[i],scDist,furDist);line->Draw();
	  line=new TLine(0.005,dists[i],0.005,furDist);line->Draw();
	  line=new TLine(scDist,furDist,0.005,furDist);line->Draw();			
	  line=new TLine(flip*scDist,flip*dists[i],-0.005,flip*dists[i]);line->Draw();//S10+
	  line=new TLine(flip*scDist,flip*dists[i],flip*scDist,flip*furDist);line->Draw();
	  line=new TLine(-0.005,flip*dists[i],-0.005,flip*furDist);line->Draw();
	  line=new TLine(flip*scDist,flip*furDist,-0.005,flip*furDist);line->Draw();
	  line=new TLine(0.005,flip*dists[i],scDist,flip*dists[i]);line->Draw();//S10-
	  line=new TLine(0.005,flip*dists[i],0.005,flip*furDist);line->Draw();
	  line=new TLine(scDist,flip*dists[i],scDist,flip*furDist);line->Draw();
	  line=new TLine(scDist,flip*furDist,0.005,flip*furDist);line->Draw();			
	}
	else{
	  line=new TLine(flip*scDist,dists[i],scDist,dists[i]);line->Draw();//S04
	  line=new TLine(flip*scDist,dists[i],flip*scDist,furDist);line->Draw();
	  line=new TLine(scDist,dists[i],scDist,furDist);line->Draw();
	  line=new TLine(flip*scDist,furDist,scDist,furDist);line->Draw();
	  line=new TLine(flip*scDist,flip*dists[i],scDist,flip*dists[i]);line->Draw();//S10
	  line=new TLine(flip*scDist,flip*dists[i],flip*scDist,flip*furDist);line->Draw();
	  line=new TLine(scDist,flip*dists[i],scDist,flip*furDist);line->Draw();
	  line=new TLine(flip*scDist,flip*furDist,scDist,flip*furDist);line->Draw();
	}
	line=new TLine(flip*onDiag,onDiag,flip*scDist,dists[i]);line->Draw();//S05
	line=new TLine(flip*onDiag,onDiag,flip*furOnDiag,furOnDiag);line->Draw();
	line=new TLine(flip*scDist,dists[i],flip*scFurDist,furDist);line->Draw();
	line=new TLine(flip*furOnDiag,furOnDiag,flip*scFurDist,furDist);line->Draw();
	line=new TLine(onDiag,flip*onDiag,scDist,flip*dists[i]);line->Draw();//S11		
	line=new TLine(onDiag,flip*onDiag,furOnDiag,flip*furOnDiag);line->Draw();
	line=new TLine(scDist,flip*dists[i],scFurDist,flip*furDist);line->Draw();
	line=new TLine(furOnDiag,flip*furOnDiag,scFurDist,flip*furDist);line->Draw();
	line=new TLine(flip*onDiag,onDiag,flip*dists[i],scDist);line->Draw();//S06
	line=new TLine(flip*onDiag,onDiag,flip*furOnDiag,furOnDiag);line->Draw();
	line=new TLine(flip*dists[i],scDist,flip*furDist,scFurDist);line->Draw();
	line=new TLine(flip*furOnDiag,furOnDiag,flip*furDist,scFurDist);line->Draw();
	line=new TLine(onDiag,flip*onDiag,dists[i],flip*scDist);line->Draw();//S12
	line=new TLine(onDiag,flip*onDiag,furOnDiag,flip*furOnDiag);line->Draw();
	line=new TLine(dists[i],flip*scDist,furDist,flip*scFurDist);line->Draw();
	line=new TLine(furOnDiag,flip*furOnDiag,furDist,flip*scFurDist);line->Draw();
      }
    }
        
    c->SaveAs(filename.c_str());
    return filename.c_str();

  }



  template <>
  std::string PayLoadInspector<RPCObTemp>::trend_plot(std::string const & filename,
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
    boost::shared_ptr<RPCObTemp> tempPtr;
    boost::shared_ptr<RPCObPVSSmap> pvssPtr;
    pvssPtr=dbSes.getTypedObject<RPCObPVSSmap>(pvssToken);

    //we get the objects...
    std::vector<RPCObPVSSmap::Item> pvssCont=pvssPtr->ObIDMap_rpc,pvssZero;

    for(unsigned int p=0;p<pvssCont.size();p++){
      if(pvssCont[p].suptype!=4)
	pvssZero.push_back(pvssCont[p]);
    }
    
    std::string token;
    std::vector<float> vecI;vecI.assign(floats.size(),0.);
    int elCount(0);

    for(;strIt!=strings.end();++strIt){
      
      token=(*strIt);
      tempPtr=dbSes.getTypedObject<RPCObTemp>(token);      
      std::vector<RPCObTemp::T_Item> const & temp = tempPtr->ObTemp_rpc;

      float iCount(0.);
      for(unsigned int i=0;i<temp.size();i++){
	for(unsigned int p=0;p<pvssZero.size();p++){
	  if(temp[i].dpid!=pvssZero[p].dpid || pvssZero[p].region!=0)continue;
	  iCount++;
	  vecI[elCount]+=temp[i].value;
	}
      }
      if(iCount!=0)
	vecI[elCount]/=iCount;
      elCount++;
    }
    
    //END OF NEW DB-SESSION PART
    
    dbSes.close();
    
    return ss.str();
    
  }
  
}


namespace condPython {
  template<>
  void defineWhat<RPCObTemp>() {
    using namespace boost::python;
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
