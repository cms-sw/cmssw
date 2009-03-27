#include <DQM/HcalMonitorClient/interface/HcalBaseClient.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>
#include <iostream>

HcalBaseClient::HcalBaseClient()
{
  dbe_ =NULL;
  clientName_ = "GenericHcalClient";
}

HcalBaseClient::~HcalBaseClient(){}

void HcalBaseClient::init(const ParameterSet& ps, DQMStore* dbe, 
			  string clientName)
{
  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  dbe_ = dbe;
  ievt_=0; jevt_=0;
  clientName_ = clientName;
  
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  debug_ = ps.getUntrackedParameter<int>("debug", 0);
  if(debug_>0) cout << clientName_ <<" debugging switch is on"<<endl;
  
  // timing switch
  showTiming_ = ps.getUntrackedParameter<bool>("showTiming",false); 

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "Hcal/");

  //Decide whether or not to fill unphysical iphi cells
  fillUnphysical_ = ps.getUntrackedParameter<bool>("fillUnphysicalIphi",true);
  
  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<4; i++)
    {
      subDetsOn_[i] = false;
    }

  for(unsigned int i=0; i<subdets.size(); i++)
    {
      if(subdets[i]=="HB") subDetsOn_[0] = true;
      else if(subdets[i]=="HE") subDetsOn_[1] = true;
      else if(subdets[i]=="HF") subDetsOn_[2] = true;
      else if(subdets[i]=="HO") subDetsOn_[3] = true;
    }
  
  // Define error palette, ranging from yellow for low to red for high. 
  for( int i=0; i<20; ++i )
    {
      //pcol_error_[19-i] = 901+i;
      TColor* color = gROOT->GetColor( 901+i );
      if( ! color ) color = new TColor( 901+i, 0, 0, 0, "" );
      color->SetRGB( 1.,
		     1.-.05*i,
		     0);
      pcol_error_[i]=901+i;
    } // for (int i=0;i<20;++i)

  return; 
} // void HcalBaseClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName)


void HcalBaseClient::errorOutput(){
  
  if(!dbe_) return;

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  
  for (map<string, string>::iterator testsMap=dqmQtests_.begin(); 
       testsMap!=dqmQtests_.end();testsMap++){
    string testName = testsMap->first;
    string meName = testsMap->second;
    MonitorElement* me = dbe_->get(meName);
    if(me){
      if (me->hasError()){
	vector<QReport*> report =  me->getQErrors();
	dqmReportMapErr_[meName] = report;
      }
      if (me->hasWarning()){
	vector<QReport*> report =  me->getQWarnings();
	dqmReportMapWarn_[meName] = report;
      }
      if(me->hasOtherReport()){
	vector<QReport*> report= me->getQOthers();
	dqmReportMapOther_[meName] = report;
      }
    }
  }

  if (debug_>0) cout << clientName_ << " Error Report: "<< dqmQtests_.size() << " tests, "<<dqmReportMapErr_.size() << " errors, " <<dqmReportMapWarn_.size() << " warnings, "<< dqmReportMapOther_.size() << " others" << endl;

  return;
}

void HcalBaseClient::getTestResults(int& totalTests, 
				    map<string, vector<QReport*> >& outE, 
				    map<string, vector<QReport*> >& outW, 
				    map<string, vector<QReport*> >& outO){
  this->errorOutput();
  //  outE.clear(); outW.clear(); outO.clear();

  for(map<string, vector<QReport*> >::iterator i=dqmReportMapErr_.begin(); i!=dqmReportMapErr_.end(); i++){
    outE[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapWarn_.begin(); i!=dqmReportMapWarn_.end(); i++){
    outW[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapOther_.begin(); i!=dqmReportMapOther_.end(); i++){
    outO[i->first] = i->second;
  }

  totalTests += dqmQtests_.size();

  return;
}


// ************************************************************************************************************ //

bool HcalBaseClient::validDetId(HcalSubdetector sd, int ies, int ip, int dp)
{
  // inputs are (subdetector, ieta, iphi, depth)
  // stolen from latest version of DataFormats/HcalDetId/src/HcalDetId.cc (not yet available in CMSSW_2_1_9)

  const int ie ( abs( ies ) ) ;

  return ( ( ip >=  1         ) &&
	   ( ip <= 72         ) &&
	   ( dp >=  1         ) &&
	   ( ie >=  1         ) &&
	   ( ( ( sd == HcalBarrel ) &&
	       ( ( ( ie <= 14         ) &&
		   ( dp ==  1         )    ) ||
		 ( ( ( ie == 15 ) || ( ie == 16 ) ) && 
		   ( dp <= 2          )                ) ) ) ||
	     (  ( sd == HcalEndcap ) &&
		( ( ( ie == 16 ) &&
		    ( dp ==  3 )          ) ||
		  ( ( ie == 17 ) &&
		    ( dp ==  1 )          ) ||
		  ( ( ie >= 18 ) &&
		    ( ie <= 20 ) &&
		    ( dp <=  2 )          ) ||
		  ( ( ie >= 21 ) &&
		    ( ie <= 26 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie >= 27 ) &&
		    ( ie <= 28 ) &&
		    ( dp <=  3 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie == 29 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         )          )      ) ||
	     (  ( sd == HcalOuter ) &&
		( ie <= 15 ) &&
		( dp ==  4 )           ) ||
	     (  ( sd == HcalForward ) &&
		( dp <=  2 )          &&
		( ( ( ie >= 29 ) &&
		    ( ie <= 39 ) &&
		    ( ip%2 == 1 )    ) ||
		  ( ( ie >= 40 ) &&
		    ( ie <= 41 ) &&
		    ( ip%4 == 3 )         )  ) ) ) ) ;



} // bool  HcalBaseClient::validDetId(HcalSubdetector sd, int ies, int ip, int dp)




void HcalBaseClient::getSJ6histos(char* dir, char* name, TH2F* h[6], char* units)
{
  if (debug_>2) cout <<"HcalBaseClient::getting SJ6histos (2D)"<<endl;
  TH2F* dummy = new TH2F();
  ostringstream hname;

  hname <<process_.c_str()<<dir<<"HB HF Depth 1 "<<name;
  if (units!="") hname<<" "<<units;
  if (debug_>3) cout <<"name = "<<hname.str()<<endl;
  h[0]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  hname.str("");

  hname <<process_.c_str()<<dir<<"HB HF Depth 2 "<<name;
  if (units!="") hname<<" "<<units;
  h[1]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  if (debug_>3) cout <<"name = "<<hname.str()<<endl;
  hname.str("");

  hname <<process_.c_str()<<dir<<"HE Depth 3 "<<name;
  if (units!="") hname<<" "<<units;
  h[2]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  if (debug_>3) cout <<"name = "<<hname.str()<<endl;
  hname.str("");

  hname <<process_.c_str()<<dir<<"HO ZDC "<<name;
  if (units!="") hname<<" "<<units;
  h[3]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  if (debug_>3) cout <<"name = "<<hname.str()<<endl;
  hname.str("");

  hname <<process_.c_str()<<dir<<"HE Depth 1 "<<name;
  if (units!="") hname<<" "<<units;
  h[4]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  if (debug_>3) cout <<"name = "<<hname.str()<<endl;
  hname.str("");

  hname <<process_.c_str()<<dir<<"HE Depth 2 "<<name;
  if (units!="") hname<<" "<<units;
  h[5]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  if (debug_>3) cout <<"name = "<<hname.str()<<endl;
  hname.str("");

  if (debug_>2) cout <<"Finished with getSJ6histos(2D)"<<endl;
  return;
} // void HcalBaseClient::getSJ6histos(2D)

void HcalBaseClient::getSJ6histos(char* dir, char* name, TH1F* h[6], char* units)
{
  TH1F* dummy = new TH1F();
  ostringstream hname;

  hname <<process_.c_str()<<dir<<"HB HF Depth 1 "<<name;
  if (units!="") hname << " "<<units;
  h[0]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  hname.str("");

  hname <<process_.c_str()<<dir<<"HB HF Depth 2 "<<name;
  if (units!="") hname << " "<<units;
  h[1]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  hname.str("");

  hname <<process_.c_str()<<dir<<"HE Depth 3 "<<name;
  if (units!="") hname << " "<<units;
  h[2]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  hname.str("");

  hname <<process_.c_str()<<dir<<"HO ZDC "<<name;
  if (units!="") hname << " "<<units;
  h[3]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  hname.str("");

  hname <<process_.c_str()<<dir<<"HE Depth 1 "<<name;
  if (units!="") hname << " "<<units;
  h[4]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  hname.str("");

  hname <<process_.c_str()<<dir<<"HE Depth 2 "<<name;
  if (units!="") hname << " "<<units;
  h[5]=getAnyHisto(dummy, hname.str(),process_,dbe_,debug_,cloneME_);
  hname.str("");
  return;
} // void HcalBaseClient::getSJ6histos(1D)




