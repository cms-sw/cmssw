#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

//------------------------------------------------------------------------------

ScoutingAnalyzerBase::ScoutingAnalyzerBase( const edm::ParameterSet& conf ){
  m_MEsPath = conf.getUntrackedParameter<std::string>("rootPath","/DataScouting") ;
  m_modulePath = conf.getUntrackedParameter<std::string>("modulePath","DataScouting") ;
  m_verbosityLevel = conf.getUntrackedParameter<unsigned int>("verbosityLevel", 0) ;
  if (m_modulePath.size() != 0)
    m_MEsPath+="/"+m_modulePath; 
 }
 
//--------------------------------------------------------------------------

ScoutingAnalyzerBase::~ScoutingAnalyzerBase(){}

//------------------------------------------------------------------------------

void ScoutingAnalyzerBase::beginJob(){
  m_store = edm::Service<DQMStore>().operator->() ;
  if (!m_store)
   { edm::LogError("ScoutingAnalyzerBase::prepareStore")<<"No DQMStore found !" ; }
  m_store->setVerbose(m_verbosityLevel) ;
  m_store->setCurrentFolder(m_MEsPath) ;
  bookMEs() ;
  }

//------------------------------------------------------------------------------

inline std::string ScoutingAnalyzerBase::newName(const std::string & name) {  
  // let's keep it in case we need massage
  return name;  
}

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::bookH1
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = m_store->book1D(newName(name),title,nchX,lowX,highX) ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1F()->SetOption(option) ; }
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::bookH1withSumw2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
   
  std::cout << newName(name) << std::endl;
  MonitorElement * me = m_store->book1D(newName(name),title,nchX,lowX,highX) ;
  me->getTH1F()->Sumw2() ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1F()->SetOption(option) ; }
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::bookH2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = m_store->book2D(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH2F()->SetOption(option) ; }
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::bookH2withSumw2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = m_store->book2D(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
  me->getTH2F()->Sumw2() ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH2F()->SetOption(option) ; }
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::bookP1
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
             double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = m_store->bookProfile(newName(name),title,nchX,lowX,highX,lowY,highY," ") ;
  if (titleX!="") { me->getTProfile()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTProfile()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTProfile()->SetOption(option) ; }
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::bookH1andDivide
 ( const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 {
  std::string name2 = newName(name) ;
  TH1F * h_temp = (TH1F *)num->getTH1F()->Clone(name2.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (m_verbosityLevel>0) { h_temp->Print() ; }
  MonitorElement * me = m_store->book1D(name2,h_temp) ;
  delete h_temp ;
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::bookH2andDivide
 ( const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 {
  std::string name2 = newName(name) ;
  TH2F * h_temp = (TH2F *)num->getTH2F()->Clone(name2.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (m_verbosityLevel>0) { h_temp->Print() ; }
  MonitorElement * me = m_store->book2D(name2,h_temp) ;
  delete h_temp ;
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::profileX
 ( MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  std::string name2 = me2d->getName()+"_pfx" ;
  TProfile * p1_temp = me2d->getTH2F()->ProfileX() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = m_store->bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

//------------------------------------------------------------------------------

MonitorElement * ScoutingAnalyzerBase::profileY
 ( MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  std::string name2 = me2d->getName()+"_pfy" ;
  TProfile * p1_temp = me2d->getTH2F()->ProfileY() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = m_store->bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

//------------------------------------------------------------------------------
 
