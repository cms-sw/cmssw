#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

ScoutingAnalyzerBase::ScoutingAnalyzerBase( const edm::ParameterSet& conf ) {
  m_MEsPath = conf.getUntrackedParameter<std::string>("rootPath","DataScouting") ;
  m_modulePath = conf.getUntrackedParameter<std::string>("modulePath","DataScouting") ;
  m_verbosityLevel = conf.getUntrackedParameter<unsigned int>("verbosityLevel", 0) ;
  if (m_modulePath.size() != 0) {
    m_MEsPath+="/"+m_modulePath;
  }
}

ScoutingAnalyzerBase::~ScoutingAnalyzerBase() {}

inline std::string ScoutingAnalyzerBase::newName(const std::string & name) {
  // let's keep it in case we need massage
  return name;
}

void ScoutingAnalyzerBase::prepareBooking(DQMStore::IBooker & iBooker) {
  iBooker.setCurrentFolder(m_MEsPath);
}

MonitorElement * ScoutingAnalyzerBase::bookH1
( DQMStore::IBooker & iBooker,
  const std::string & name, const std::string & title,
  int nchX, double lowX, double highX,
  const std::string & titleX, const std::string & titleY,
  Option_t * option ) {
  MonitorElement * me = iBooker.book1DD(newName(name),title,nchX,lowX,highX) ;
  if (titleX!="") { me->getTH1()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1()->SetOption(option) ; }
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookH1withSumw2
( DQMStore::IBooker & iBooker,
  const std::string & name, const std::string & title,
  int nchX, double lowX, double highX,
  const std::string & titleX, const std::string & titleY,
  Option_t * option ) {
  std::cout << newName(name) << std::endl;
  MonitorElement * me = iBooker.book1DD(newName(name),title,nchX,lowX,highX) ;
  me->getTH1()->Sumw2() ;
  if (titleX!="") { me->getTH1()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1()->SetOption(option) ; }
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookH1BinArray
( DQMStore::IBooker & iBooker,
  const std::string & name, const std::string & title,
  int nchX, float *xbinsize,
  const std::string & titleX, const std::string & titleY,
  Option_t * option ) {
  MonitorElement * me = iBooker.book1D(newName(name),title,nchX,xbinsize) ;
  //book1DD not implemented in DQMServices/Core/src/DQMStore.cc
  if (titleX!="") { me->getTH1()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1()->SetOption(option) ; }
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookH1withSumw2BinArray
( DQMStore::IBooker & iBooker,
  const std::string & name, const std::string & title,
  int nchX, float *xbinsize,
  const std::string & titleX, const std::string & titleY,
  Option_t * option ) {
  std::cout << newName(name) << std::endl;
  MonitorElement * me = iBooker.book1D(newName(name),title,nchX,xbinsize) ;
  //book1DD not implemented in DQMServices/Core/src/DQMStore.cc
  me->getTH1()->Sumw2() ;
  if (titleX!="") { me->getTH1()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1()->SetOption(option) ; }
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookH2
( DQMStore::IBooker & iBooker,
  const std::string & name, const std::string & title,
  int nchX, double lowX, double highX,
  int nchY, double lowY, double highY,
  const std::string & titleX, const std::string & titleY,
  Option_t * option ) {
  MonitorElement * me = iBooker.book2DD(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
  if (titleX!="") { me->getTH1()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1()->SetOption(option) ; }
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookH2withSumw2
( DQMStore::IBooker & iBooker,
  const std::string & name, const std::string & title,
  int nchX, double lowX, double highX,
  int nchY, double lowY, double highY,
  const std::string & titleX, const std::string & titleY,
  Option_t * option ) {
  MonitorElement * me = iBooker.book2DD(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
  me->getTH1()->Sumw2() ;
  if (titleX!="") { me->getTH1()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1()->SetOption(option) ; }
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookP1
( DQMStore::IBooker & iBooker,
  const std::string & name, const std::string & title,
  int nchX, double lowX, double highX,
  double lowY, double highY,
  const std::string & titleX, const std::string & titleY,
  Option_t * option ) {
  MonitorElement * me = iBooker.bookProfile(newName(name),title,nchX,lowX,highX,lowY,highY," ") ;
  if (titleX!="") { me->getTProfile()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTProfile()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTProfile()->SetOption(option) ; }
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookH1andDivide
( DQMStore::IBooker & iBooker,
  const std::string & name, MonitorElement * num, MonitorElement * denom,
  const std::string & titleX, const std::string & titleY,
  const std::string & title ) {
  std::string name2 = newName(name) ;
  TH1D * h_temp = dynamic_cast<TH1D*>(num->getTH1()->Clone(name2.c_str()) );
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (m_verbosityLevel>0) { h_temp->Print() ; }
  MonitorElement * me = iBooker.book1DD(name2,h_temp) ;
  delete h_temp ;
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::bookH2andDivide
( DQMStore::IBooker & iBooker,
  const std::string & name, MonitorElement * num, MonitorElement * denom,
  const std::string & titleX, const std::string & titleY,
  const std::string & title ) {
  std::string name2 = newName(name) ;
  TH2D * h_temp = dynamic_cast<TH2D*>(num->getTH1()->Clone(name2.c_str()) );
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (m_verbosityLevel>0) { h_temp->Print() ; }
  MonitorElement * me = iBooker.book2DD(name2,h_temp) ;
  delete h_temp ;
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::profileX
( DQMStore::IBooker & iBooker,
  MonitorElement * me2d,
  const std::string & title, const std::string & titleX, const std::string & titleY,
  Double_t minimum, Double_t maximum ) {
  std::string name2 = me2d->getName()+"_pfx" ;
  TProfile * p1_temp = me2d->getTH2D()->ProfileX() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = iBooker.bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
}

MonitorElement * ScoutingAnalyzerBase::profileY
( DQMStore::IBooker & iBooker,
  MonitorElement * me2d,
  const std::string & title, const std::string & titleX, const std::string & titleY,
  Double_t minimum, Double_t maximum ) {
  std::string name2 = me2d->getName()+"_pfy" ;
  TProfile * p1_temp = me2d->getTH2D()->ProfileY() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = iBooker.bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
}
