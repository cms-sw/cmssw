#include <string>
#include <vector>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <cmath>
#include <TFile.h>
#include <TH1F.h>
#include <TF1.h>
#include <TStyle.h>
#include <TMath.h>

void GetMPV(char name[100],TH1F* histo, TDirectory* Dir, double& peak, double& error, double& sigma, double& err_sigma);
void GetMEAN(TH1F* histo, double& peak, double& error, double& sigma);
void CalculateResponse(bool UseRatioForResponse, double x, double ex, double y, double ey, double& r, double& e);
void CalculateCorrection(bool UseRatioForResponse, double x, double ex, double y, double ey, double& c, double& e);
void Invert(TF1* f, double Min, double Max, double y, double& x);
bool HistoExists(std::vector<std::string> LIST, std::string hname);
int  getBin(double x, std::vector<double> boundaries);

class CommandLine
{
public:
  //
  // construction/destruction
  //
  CommandLine();
  ~CommandLine();
  
  //
  // member functions
  //
  bool parse(int argc,char**argv);
  bool check();
  void print();
  
  template <class T> T getValue(const std::string& name);
  template <class T> T getValue(const std::string& name, T default_value);

  template <class T> std::vector<T> getVector(const std::string& name);
  template <class T> std::vector<T> getVector(const std::string& name,
					      const std::string& default_as_string);
  
private:
  bool parse_file(const std::string& file_name);
  
private:
  //
  // internal typedefs
  //
  typedef std::map<std::string,std::pair<std::string,bool> > OptionMap_t;
  typedef std::vector<std::string>                           StrVec_t;
  

  //
  // member data
  //
  std::string _exe;
  OptionMap_t _options;
  StrVec_t    _ordered_options;
  StrVec_t    _unknowns;

};

//
// implemenentation of inline functions
//

//______________________________________________________________________________
template <class T>
T CommandLine::getValue(const std::string& name)
{
  T result = T();
  OptionMap_t::iterator it=_options.find(name);
  if (it!=_options.end()) {
    it->second.second = true;
    _ordered_options.push_back(name);
    std::stringstream ss;
    ss<<it->second.first;
    ss>>result;
    return result;
  }
  _unknowns.push_back(name);
  return result;
}


//______________________________________________________________________________
template <class T>
T CommandLine::getValue(const std::string& name,T default_value)
{
  OptionMap_t::const_iterator it=_options.find(name);
  if (it!=_options.end()) return getValue<T>(name);
  std::string default_as_string;
  std::stringstream ss;
  ss<<default_value;
  ss>>default_as_string;
  _options[name] = std::make_pair(default_as_string,true);
  _ordered_options.push_back(name);
  return default_value;
}


//______________________________________________________________________________
template <>
bool CommandLine::getValue<bool>(const std::string& name)
{
  OptionMap_t::iterator it=_options.find(name);
  if (it!=_options.end()) {
    it->second.second = true;
    _ordered_options.push_back(name);
    std::string val_as_string = it->second.first;
    if (val_as_string=="true") return true;
    if (val_as_string=="false") return false;
    int val_as_int;
    std::stringstream ss;
    ss<<val_as_string;
    ss>>val_as_int;
    return val_as_int;
  }
  _unknowns.push_back(name);
  return false;
}


//______________________________________________________________________________
template <>
bool CommandLine::getValue(const std::string& name,bool default_value)
{
  OptionMap_t::const_iterator it=_options.find(name);
  if (it!=_options.end()) return getValue<bool>(name);
  _options[name] = (default_value) ?
    std::make_pair("true",true) : std::make_pair("false",true);
  _ordered_options.push_back(name);
  return default_value;
}


//______________________________________________________________________________
template <class T>
std::vector<T> CommandLine::getVector(const std::string& name)
{
  std::vector<T> result;
  OptionMap_t::iterator it=_options.find(name);
  if (it!=_options.end()) {
    it->second.second = true;
    _ordered_options.push_back(name);
    std::string tmp=it->second.first;
    std::string::size_type pos;
    if (!tmp.empty()) {
      do {
	pos = tmp.find(",");
	std::stringstream ss;
	ss<<tmp.substr(0,pos);
	tmp.erase(0,pos+1);
	T element;
	ss>>element;
	result.push_back(element);
      }
      while (pos!=std::string::npos);
    }
  }
  else {
    _unknowns.push_back(name);
  }
  return result;
}

//______________________________________________________________________________
template <class T>
std::vector<T> CommandLine::getVector(const std::string& name,
				       const std::string& default_as_string)
{
  OptionMap_t::iterator it=_options.find(name);
  if (it==_options.end()) _options[name] = std::make_pair(default_as_string,false);
  return getVector<T>(name);
}

////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
CommandLine::CommandLine()
{
  
}
//______________________________________________________________________________
CommandLine::~CommandLine()
{
  
}
////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
bool CommandLine::parse(int argc,char**argv)
{
  _exe = argv[0];
  _options.clear();
  _ordered_options.clear();
  _unknowns.clear();
  
  for (int i=1;i<argc;i++) {
    std::string opt=argv[i];
    if(0!=opt.find("-")) {
      if (i==1) {
	bool success = parse_file(opt);
	if (!success) return false;
	continue;
      }
      else {
	std::cout<<"CommandLine ERROR: options must start with '-'!"<<std::endl;
	return false;
      }
    }
    opt.erase(0,1);
    std::string next=argv[i+1];
    if (/*0==next.find("-")||*/i+1>=argc) {
      std::cout<<"ERROR: option '"<<opt<<"' requires value!"<<std::endl;
      return false;
    }
    _options[opt] = std::make_pair(next,false);
    i++;
    if (i<argc-1) {
      next=argv[i+1];
      while (next.find("-")!=0) {
	_options[opt].first += ","+next;
	i++;
	next = (i<argc-1) ? argv[i+1] : "-";
      }
    }
  }
  
  return true;
}
//______________________________________________________________________________
bool CommandLine::check()
{
  bool result = true;
  OptionMap_t::const_iterator it;
  for (it = _options.begin();it!=_options.end();++it) {
    if (!it->second.second) {
      std::cout<<"CommandLine WARNING: unused option '"<<it->first<<"'!"<<std::endl;
      result = false;
    }
  }
  
  if (_unknowns.size()>0) {
    result = false;
    std::cout<<"\nCommandLine WARNING: "<<_unknowns.size()
	<<" the followingparameters *must* be provided:"<<std::endl;
    for (StrVec_t::const_iterator it=_unknowns.begin();it!=_unknowns.end();++it)
      std::cout<<(*it)<<std::endl;
    std::cout<<std::endl;
  }
  return result;
}
//______________________________________________________________________________
void CommandLine::print()
{
  std::cout<<"------------------------------------------------------------"<<std::endl;
  std::cout<<_exe<<" options:"<<std::endl;
  std::cout<<"------------------------------------------------------------"<<std::endl;
  for (StrVec_t::const_iterator itvec=_ordered_options.begin();
       itvec!=_ordered_options.end();++itvec) {
    OptionMap_t::const_iterator it=_options.find(*itvec);
    assert(it!=_options.end());
    if (it->second.first.find(",")<std::string::npos) {
      std::string tmp=it->second.first;
      std::string::size_type length = tmp.length();
      std::string::size_type pos;
      do {
	pos = tmp.find(",");
	if (tmp.length()==length) {
	  std::cout<<std::setiosflags(std::ios::left)<<std::setw(22)
	      <<it->first
	      <<std::resetiosflags(std::ios::left)
	      <<std::setw(3)<<"="
	      <<std::setiosflags(std::ios::right)<<std::setw(35)
	      <<tmp.substr(0,pos)
	      <<std::resetiosflags(std::ios::right)
	      <<std::endl;
	}
	else {
	  std::cout<<std::setiosflags(std::ios::right)<<std::setw(60)
	      <<tmp.substr(0,pos)
	      <<std::resetiosflags(std::ios::right)
	      <<std::endl;
	}
	tmp.erase(0,pos+1);
      }
      while (pos!=std::string::npos);
    }
    else {
      std::cout<<std::setiosflags(std::ios::left)<<std::setw(22)
	  <<it->first
	  <<std::resetiosflags(std::ios::left)
	  <<std::setw(3)<<"="
	  <<std::setiosflags(std::ios::right)<<std::setw(35)
	  <<it->second.first
	  <<std::resetiosflags(std::ios::right)
	  <<std::endl;
    }
  }
  std::cout<<"------------------------------------------------------------"<<std::endl;
}
//______________________________________________________________________________
bool CommandLine::parse_file(const std::string& file_name)
{
  ifstream fin(file_name.c_str());
  if (!fin.is_open()) {
    std::cout<<"Can't open configuration file "<<file_name<<std::endl;
    return false;
  }

  std::stringstream ss;
  bool filter(false);
  while(!fin.eof()){
    char next;
    fin.get(next);
    if (!filter&&next=='$') filter=true;
    if(!filter) {
      if (next=='=') ss<<" "<<next<<" ";
      else ss<<next;
    }
    if (filter&&next=='\n') filter=false;
  }
  
  std::string token,last_token,key,value;
  ss>>token;
  while (!ss.eof()) {
    if (token=="=") {
      if (key!=""&&value!="") _options[key] = std::make_pair(value,false);
      key=last_token;
      last_token="";
      value="";
    }
    else if (last_token!="") {
      if (last_token.find("\"")==0) {
	if (last_token.rfind("\"")==last_token.length()-1) {
	  last_token=last_token.substr(1,last_token.length()-2);
	  value+=(value!="")?","+last_token:last_token;
	  last_token=token;
	}
	else last_token+=" "+token;
      }
      else {
	value+=(value!="")?","+last_token:last_token;
	last_token=(token=="=")?"":token;
      }
    }
    else last_token=(token=="=")?"":token;
    ss>>token;
  }
  if (last_token!="") {
    if (last_token.find("\"")==0&&last_token.rfind("\"")==last_token.length()-1)
      last_token=last_token.substr(1,last_token.length()-2);
    value+=(value!="")?","+last_token:last_token;
  }
  if (key!=""&&value!="") _options[key] = std::make_pair(value,false);

  return true;
}
//////////////////////////////////////////////////////////////////////
void GetMPV(char name[100],TH1F* histo, TDirectory* Dir, double& peak, double& error, double& sigma, double& err_sigma)
{
  double norm,mean,rms,integral,lowlimit,highlimit,LowResponse,HighResponse,a;
  int k;
  LowResponse = histo->GetXaxis()->GetXmin();
  HighResponse = histo->GetXaxis()->GetXmax();
  Dir->cd();
  TF1 *g;
  TStyle *myStyle = new TStyle("mystyle","mystyle");
  myStyle->Reset();
  myStyle->SetOptFit(1111);
  myStyle->SetOptStat(2200);
  myStyle->SetStatColor(0);
  myStyle->SetTitleFillColor(0);
  myStyle->cd(); 
  integral = histo->Integral();
  mean = histo->GetMean();
  rms = histo->GetRMS();
  a = 1.5;
  if (integral>0)
    { 
      lowlimit = TMath::Max(LowResponse,mean-a*rms);
      highlimit= TMath::Min(mean+a*rms,HighResponse); 
      norm = histo->GetMaximumStored();
      peak = mean;
      sigma = rms;
      for (k=0; k<3; k++)
       {
         g = new TF1("g","gaus",lowlimit, highlimit);
         g->SetParNames("N","#mu","#sigma");
         g->SetParameter(0,norm);
         g->SetParameter(1,peak);
         g->SetParameter(2,sigma);
         lowlimit = TMath::Max(LowResponse,peak-a*sigma);
         highlimit= TMath::Min(peak+a*sigma,HighResponse);  
         g->SetRange(lowlimit,highlimit);
         histo->Fit(g,"RQ");
         norm = g->GetParameter(0);
         peak = g->GetParameter(1);
         sigma = g->GetParameter(2);  
       }
      if (g->GetNDF()>5)
        {
          peak = g->GetParameter(1);
          sigma = g->GetParameter(2);
          error = g->GetParError(1);
          err_sigma = g->GetParError(2);
        }
      else
        {
          std::cout<<"FIT FAILURE: histogram "<<name<<"...Using MEAN and RMS."<<std::endl;
          peak = mean;
          sigma = rms;
          error = histo->GetMeanError();
          err_sigma = histo->GetRMSError();
        }
    }
  else
    {
      peak = 0;
      sigma = 0;
      error = 0;
      err_sigma = 0;
    }
  histo->Write();
}
//////////////////////////////////////////////////////////////////////
void GetMEAN(TH1F* histo, double& peak, double& error, double& sigma)
{
  double N = histo->Integral();
  if (N>2)
    {
      peak  = histo->GetMean();
      sigma = histo->GetRMS();
      error = histo->GetMeanError();
    }
  else
    {
      peak = 0;
      sigma = 0;
      error = 0; 
    }  
}
///////////////////////////////////////////////////////////////////////
void CalculateResponse(bool UseRatioForResponse, double x, double ex, double y, double ey, double& r, double& e)
{
  if (x>0 && fabs(y)>0)
    {
      if (UseRatioForResponse)
        {
          r = y;
          e = ey;
        }
      else
        {  
          r = (x+y)/x;
          e = fabs(r-1.)*sqrt(pow(ey/y,2)+pow(ex/x,2)); 
        }
    }
  else
    {
      r = 0;
      e = 0;
    }    
}
///////////////////////////////////////////////////////////////////////
void CalculateCorrection(bool UseRatioForResponse, double x, double ex, double y, double ey, double& c, double& e)
{
  if (x>0 && fabs(y)>0)
    {
      if (UseRatioForResponse)
        {
          c = 1./y;
          e = ey/(y*y);
        }
      else
        {  
          c = x/(x+y);
          e = (fabs(x*y)/pow(x+y,2))*sqrt(pow(ey/y,2)+pow(ex/x,2)); 
        }
    }
  else
    {
      c = 0;
      e = 0;
    }    
}
///////////////////////////////////////////////////////////////////////
bool HistoExists(std::vector<std::string> LIST, std::string hname)
{
  unsigned int i,N;
  bool found(false);
  N = LIST.size();
  if (N==0)
    std::cout<<"WARNING: empty file histogram list!!!!"<<std::endl;
  else
    for(i=0;i<N;i++)
     if (hname==LIST[i])
       found = true;
  if (!found)
    std::cout<<"Histogram: "<<hname<<" NOT FOUND!!! Check list of existing objects."<<std::endl;
  return found;
}
///////////////////////////////////////////////////////////////////////
int getBin(double x, std::vector<double> boundaries)
{
  int i;
  int n = boundaries.size()-1;
  if (n<=0) return -1;
  if (x<boundaries[0] || x>=boundaries[n])
    return -1;
  for(i=0;i<n;i++)
   {
     if (x>=boundaries[i] && x<boundaries[i+1])
       return i;
   }
  return 0; 
}
///////////////////////////////////////////////////////////////////////
