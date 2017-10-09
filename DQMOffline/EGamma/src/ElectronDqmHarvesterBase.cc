
#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h"
//#include "DQMServices/Core/interface/DQMStore.h"
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
#include <algorithm>
#include <sstream>

ElectronDqmHarvesterBase::ElectronDqmHarvesterBase( const edm::ParameterSet& conf )
 : bookPrefix_("ele"), bookIndex_(0), histoNamesReady(false), finalDone_(false)
 {
  verbosity_ = conf.getUntrackedParameter<int>("Verbosity") ;
  finalStep_ = conf.getParameter<std::string>("FinalStep") ;
  inputFile_ = conf.getParameter<std::string>("InputFile") ;
  outputFile_ = conf.getParameter<std::string>("OutputFile") ;
  inputInternalPath_ = conf.getParameter<std::string>("InputFolderName") ;
  outputInternalPath_ = conf.getParameter<std::string>("OutputFolderName") ;
 }

ElectronDqmHarvesterBase::~ElectronDqmHarvesterBase()
 { 
 }

void ElectronDqmHarvesterBase::setBookPrefix( const std::string & prefix )
 { bookPrefix_ = prefix ; }

void ElectronDqmHarvesterBase::setBookIndex( short index )
 { bookIndex_ = index ; }
 
void ElectronDqmHarvesterBase::setBookEfficiencyFlag( const bool & eff_flag )
 { bookEfficiencyFlag_ = eff_flag ;}

void ElectronDqmHarvesterBase::setBookStatOverflowFlag( const bool & statOverflow_flag )
 { bookStatOverflowFlag_ = statOverflow_flag ;}

std::string ElectronDqmHarvesterBase::newName( const std::string & name )
 {
  if (bookPrefix_.empty())
   { return name ; }
  std::ostringstream oss ;
  oss<<bookPrefix_ ;
  if (bookIndex_>=0)
   { oss<<bookIndex_++ ; }
  oss<<"_"<<name ;
  return oss.str() ;
 }

const std::string * ElectronDqmHarvesterBase::find( DQMStore::IGetter & iGetter, const std::string & name )
 {
  typedef std::vector<std::string> HistoNames ;
  typedef HistoNames::iterator HistoNamesItr ;
  if (!histoNamesReady)
   { histoNamesReady = true ; histoNames_ = iGetter.getMEs() ; }
  HistoNamesItr histoName ;
  std::vector<HistoNamesItr> res ;

  for ( histoName = histoNames_.begin() ; histoName != histoNames_.end() ; ++histoName )
   {
    std::size_t nsize = name.size(), lsize = histoName->size() ;
    if ( (lsize>=nsize) &&
         (histoName->find(name)==(lsize-nsize)) )
     { res.push_back(histoName) ; }
   }
  if (res.size()==0)
   {
    std::ostringstream oss ;
    oss<<"Histogram "<<name<<" not found in "<<outputInternalPath_ ;
    char sep = ':' ;
    for ( histoName = histoNames_.begin() ; histoName != histoNames_.end() ; ++histoName )
     { oss<<sep<<' '<<*histoName ; sep = ',' ; }
    oss<<'.' ;
    edm::LogWarning("ElectronDqmHarvesterBase::find")<<oss.str() ;
    return 0 ;
   }
  else if (res.size()>1)
   {
    std::ostringstream oss ;
    oss<<"Ambiguous histograms for "<<name<<" in "<<outputInternalPath_ ;
    char sep = ':' ;
    std::vector<HistoNamesItr>::iterator resItr ;
    for ( resItr = res.begin() ; resItr != res.end() ; ++resItr )
     { oss<<sep<<' '<<(**resItr) ; sep = ',' ; }
    oss<<'.' ;
    edm::LogWarning("ElectronDqmHarvesterBase::find")<<oss.str() ;
    return 0 ;
   }
  else
   {
    return &*res[0] ;
   }
 }

void ElectronDqmHarvesterBase::beginJob()
 {
 }

void ElectronDqmHarvesterBase::dqmEndLuminosityBlock( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, edm::LuminosityBlock const &, edm::EventSetup const& )
 {
  if (finalStep_=="AtLumiEnd")
   {
    if (finalDone_)
     { 
         edm::LogWarning("ElectronDqmHarvesterBase::endLuminosityBlock")<<"finalize() already called" ; 
     }
    iBooker.setCurrentFolder(outputInternalPath_) ;
    finalDone_ = true ;
   }

 }

void ElectronDqmHarvesterBase::dqmEndJob(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter)
 {

  if (finalStep_=="AtJobEnd")
   {
    if (finalDone_)
     { edm::LogWarning("ElectronDqmHarvesterBase::dqmEndJob")<<"finalize() already called" ; }
    iBooker.setCurrentFolder(outputInternalPath_) ;
    finalDone_ = true ;
   }
   iBooker.setCurrentFolder(outputInternalPath_) ;
   finalize( iBooker, iGetter ) ; 

 }

MonitorElement * ElectronDqmHarvesterBase::get( DQMStore::IGetter & iGetter, const std::string & name )
 {
  const std::string * fullName = find(iGetter, name) ;
  if (fullName)
   { return iGetter.get(inputInternalPath_+"/"+*fullName) ; }
  else
  { return 0 ; }
 }

void ElectronDqmHarvesterBase::remove( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const std::string & name )
 {
  const std::string * fullName = find(iGetter, name) ;
  if (fullName)
   {
    iBooker.setCurrentFolder(inputInternalPath_) ;
    iGetter.removeElement(*fullName) ;
   }
 }

MonitorElement * ElectronDqmHarvesterBase::bookH1andDivide
 ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter,
   const std::string & name, const std::string & num, const std::string & denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 { 
   return bookH1andDivide(iBooker, iGetter, name,get(iGetter, num),get(iGetter, denom),titleX,titleY,title) ;  
 }

MonitorElement * ElectronDqmHarvesterBase::bookH2andDivide
 ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter,
   const std::string & name, const std::string & num, const std::string & denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 { return bookH2andDivide(iBooker, iGetter, name,get(iGetter, num),get(iGetter, denom),titleX,titleY,title) ; }

MonitorElement * ElectronDqmHarvesterBase::cloneH1
 ( DQMStore::IBooker & iBooker,  DQMStore::IGetter & iGetter,
   const std::string & clone, const std::string & original,
   const std::string & title )
 { return cloneH1(iBooker, iGetter, clone,get(iGetter, original),title) ; }

MonitorElement * ElectronDqmHarvesterBase::profileX
 ( DQMStore::IBooker & iBooker,  DQMStore::IGetter & iGetter,
   const std::string & me2d, 
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 { return profileX(iBooker, iGetter, get(iGetter, me2d),title,titleX,titleY,minimum,maximum) ; }

MonitorElement * ElectronDqmHarvesterBase::profileY
 ( DQMStore::IBooker & iBooker,  DQMStore::IGetter & iGetter,
   const std::string & me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 { return profileY(iBooker, iGetter, get(iGetter, me2d),title,titleX,titleY,minimum,maximum) ; }

MonitorElement * ElectronDqmHarvesterBase::bookH1
 ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement * me = iBooker.book1D(newName(name),title,nchX,lowX,highX) ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1F()->SetOption(option) ; }
  if (bookStatOverflowFlag_) {me->getTH1F()->StatOverflows(kTRUE) ; }
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::bookH1withSumw2
 ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement * me = iBooker.book1D(newName(name),title,nchX,lowX,highX) ;
  me->getTH1F()->Sumw2() ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH1F()->SetOption(option) ; }
  if (bookStatOverflowFlag_) {me->getTH1F()->StatOverflows(kTRUE) ; }
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::bookH2
 ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement * me = iBooker.book2D(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH2F()->SetOption(option) ; }
  if (bookStatOverflowFlag_) {me->getTH1F()->StatOverflows(kTRUE) ; }
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::bookH2withSumw2
 ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement * me = iBooker.book2D(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
  me->getTH2F()->Sumw2() ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTH2F()->SetOption(option) ; }
  if (bookStatOverflowFlag_) {me->getTH1F()->StatOverflows(kTRUE) ; }
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::bookP1
 ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
             double lowY, double highY,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement * me = iBooker.bookProfile(newName(name),title,nchX,lowX,highX,lowY,highY," ") ;
  if (titleX!="") { me->getTProfile()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTProfile()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (TString(option)!="") { me->getTProfile()->SetOption(option) ; }
  if (bookStatOverflowFlag_) {me->getTH1F()->StatOverflows(kTRUE) ; }
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::bookH1andDivide
 ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, 
   const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 {
  if ((!num)||(!denom)) return 0 ;
  iBooker.setCurrentFolder(outputInternalPath_);
  std::string name2 = newName(name) ;
  TH1F * h_temp = (TH1F *)num->getTH1F()->Clone(name2.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (verbosity_>0) { h_temp->Print() ; }
  MonitorElement * me = iBooker.book1D(name2,h_temp) ;
  if (bookEfficiencyFlag_) { me->setEfficiencyFlag(); }
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::bookH2andDivide
 ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter,
   const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title )
 {
  if ((!num)||(!denom)) return 0 ;
  iBooker.setCurrentFolder(outputInternalPath_);
  std::string name2 = newName(name) ;
  TH2F * h_temp = (TH2F *)num->getTH2F()->Clone(name2.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (verbosity_>0) { h_temp->Print() ; }
  MonitorElement * me = iBooker.book2D(name2,h_temp) ;
  if (bookEfficiencyFlag_) { me->setEfficiencyFlag(); }
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::cloneH1
 ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter,
   const std::string & name, MonitorElement * original,
   const std::string & title )
 {
  if (!original) return 0 ;
  iBooker.setCurrentFolder(outputInternalPath_);
  std::string name2 = newName(name) ;
  TH1F * h_temp = (TH1F *)original->getTH1F()->Clone(name2.c_str()) ;
  h_temp->Reset() ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  MonitorElement * me = iBooker.book1D(name2,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::profileX
 ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  iBooker.setCurrentFolder(outputInternalPath_);
  std::string name2 = me2d->getName()+"_pfx" ;
  TProfile * p1_temp = me2d->getTH2F()->ProfileX() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = iBooker.bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

MonitorElement * ElectronDqmHarvesterBase::profileY
 ( DQMStore::IBooker & iBooker,  DQMStore::IGetter & iGetter,MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  iBooker.setCurrentFolder(outputInternalPath_);
  std::string name2 = me2d->getName()+"_pfy" ;
  TProfile * p1_temp = me2d->getTH2F()->ProfileY() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = iBooker.bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
 }


