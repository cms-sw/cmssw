
#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <iostream>

ElectronDqmAnalyzerBase::ElectronDqmAnalyzerBase( const edm::ParameterSet& conf )
 : finalDone_(false)
 {
  verbosity_ = conf.getUntrackedParameter<int>("Verbosity") ;
  finalStep_ = conf.getParameter<std::string>("FinalStep") ;
  inputFile_ = conf.getParameter<std::string>("InputFile") ;
  outputFile_ = conf.getParameter<std::string>("OutputFile") ;
  inputInternalPath_ = "Egamma/Electrons/" + conf.getParameter<std::string>("InputFolderName") ;
  outputInternalPath_ = "Egamma/Electrons/" + conf.getParameter<std::string>("OutputFolderName") ;
 }

ElectronDqmAnalyzerBase::~ElectronDqmAnalyzerBase()
 {}

void ElectronDqmAnalyzerBase::beginJob()
 {
  store_ = edm::Service<DQMStore>().operator->() ;
  if (!store_)
   { edm::LogError("ElectronDqmAnalyzerBase::prepareStore")<<"No DQMStore found !" ; }
  store_->setVerbose(verbosity_) ;
  if (inputFile_!="")
   { store_->open(inputFile_) ; }
  store_->setCurrentFolder(outputInternalPath_) ;
  book() ;
 }

void ElectronDqmAnalyzerBase::endRun( edm::Run const &, edm::EventSetup const & )
 {
  if (finalStep_=="AtRunEnd")
   {
    if (finalDone_)
     { edm::LogWarning("ElectronDqmAnalyzerBase::endRun")<<"finalize() already called" ; }
    store_->setCurrentFolder(outputInternalPath_) ;
    finalize() ;
    finalDone_ = true ;
   }
 }
void ElectronDqmAnalyzerBase::endLuminosityBlock( edm::LuminosityBlock const &, edm::EventSetup const & )
 {
  if (finalStep_=="AtLumiEnd")
   {
    if (finalDone_)
     { edm::LogWarning("ElectronDqmAnalyzerBase::endLuminosityBlock")<<"finalize() already called" ; }
    store_->setCurrentFolder(outputInternalPath_) ;
    finalize() ;
    finalDone_ = true ;
   }
 }

void ElectronDqmAnalyzerBase::endJob()
 {
  if (finalStep_=="AtJobEnd")
   {
    if (finalDone_)
     { edm::LogWarning("ElectronDqmAnalyzerBase::endJob")<<"finalize() already called" ; }
    store_->setCurrentFolder(outputInternalPath_) ;
    finalize() ;
    finalDone_ = true ;
   }
  if (outputFile_!="")
   { store_->save(outputFile_) ; }
 }

MonitorElement * ElectronDqmAnalyzerBase::get( const std::string & name )
 {
  MonitorElement * me = store_->get(inputInternalPath_+"/"+name) ;
  if (!me)
   { edm::LogWarning("ElectronDqmAnalyzerBase::get")<<"Unknown histogram "<<inputInternalPath_+"/"+name ; }
  return me ;
 }

void ElectronDqmAnalyzerBase::remove( const std::string & name )
 { store_->removeElement(name) ; }

MonitorElement * ElectronDqmAnalyzerBase::bookH1andDivide
 ( const std::string & name, const std::string & num, const std::string & denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 { return bookH1andDivide(name,get(num),get(denom),titleX,titleY,title) ;  }

MonitorElement * ElectronDqmAnalyzerBase::bookH2andDivide
 ( const std::string & name, const std::string & num, const std::string & denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 { return bookH2andDivide(name,get(num),get(denom),titleX,titleY,title) ; }

MonitorElement * ElectronDqmAnalyzerBase::cloneH1
 ( const std::string & clone, const std::string & original,
   const std::string & title )
 { return cloneH1(clone,get(original),title) ; }

MonitorElement * ElectronDqmAnalyzerBase::profileX
 ( const std::string & name, const std::string & me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 { return profileX(name,get(me2d),title,titleX,titleY,minimum,maximum) ; }

MonitorElement * ElectronDqmAnalyzerBase::profileY
 ( const std::string & name, const std::string & me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 { return profileY(name,get(me2d),title,titleX,titleY,minimum,maximum) ; }

MonitorElement * ElectronDqmAnalyzerBase::bookH1
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = store_->book1D(name,title,nchX,lowX,highX) ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1F()->SetOption(option) ; }
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::bookH1withSumw2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = store_->book1D(name,title,nchX,lowX,highX) ;
  me->getTH1F()->Sumw2() ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1F()->SetOption(option) ; }
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::bookH2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = store_->book2D(name,title,nchX,lowX,highX,nchY,lowY,highY) ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH2F()->SetOption(option) ; }
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::bookH2withSumw2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = store_->book2D(name,title,nchX,lowX,highX,nchY,lowY,highY) ;
  me->getTH2F()->Sumw2() ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH2F()->SetOption(option) ; }
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::bookP1
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
             double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = store_->bookProfile(name,title,nchX,lowX,highX,lowY,highY," ") ;
  if (titleX!="") { me->getTProfile()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTProfile()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTProfile()->SetOption(option) ; }
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::bookH1andDivide
 ( const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 {
  TH1F * h_temp = (TH1F *)num->getTH1F()->Clone(name.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (verbosity_>0) { h_temp->Print() ; }
  MonitorElement * me = store_->book1D(name,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::bookH2andDivide
 ( const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 {
  TH2F * h_temp = (TH2F *)num->getTH2F()->Clone(name.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (verbosity_>0) { h_temp->Print() ; }
  MonitorElement * me = store_->book2D(name,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::cloneH1
 ( const std::string & clone, MonitorElement * original,
   const std::string & title )
 {
  TH1F * h_temp = (TH1F *)original->getTH1F()->Clone(clone.c_str()) ;
  h_temp->Reset() ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  MonitorElement * me = store_->book1D(clone,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::profileX
 ( const std::string & name, MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  TProfile * p1_temp = me2d->getTH2F()->ProfileX() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = store_->bookProfile(name,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::profileY
 ( const std::string & name, MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  TProfile * p1_temp = me2d->getTH2F()->ProfileY() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = store_->bookProfile(name,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

