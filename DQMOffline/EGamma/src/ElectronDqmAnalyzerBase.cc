
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
#include <algorithm>
#include <sstream>

ElectronDqmAnalyzerBase::ElectronDqmAnalyzerBase( const edm::ParameterSet& conf )
 : bookPrefix_("ele"), bookIndex_(0), histoNamesReady(false), finalDone_(false)
 {
  verbosity_ = conf.getUntrackedParameter<int>("Verbosity") ;
  finalStep_ = conf.getParameter<std::string>("FinalStep") ;
  inputFile_ = conf.getParameter<std::string>("InputFile") ;
  outputFile_ = conf.getParameter<std::string>("OutputFile") ;
  inputInternalPath_ = conf.getParameter<std::string>("InputFolderName") ;
  outputInternalPath_ = conf.getParameter<std::string>("OutputFolderName") ;
 }

ElectronDqmAnalyzerBase::~ElectronDqmAnalyzerBase()
 {}

void ElectronDqmAnalyzerBase::setBookPrefix( const std::string & prefix )
 { bookPrefix_ = prefix ; }

void ElectronDqmAnalyzerBase::setBookIndex( short index )
 { bookIndex_ = index ; }

std::string ElectronDqmAnalyzerBase::newName( const std::string & name )
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

const std::string * ElectronDqmAnalyzerBase::find( const std::string & name )
 {
  typedef std::vector<std::string> HistoNames ;
  typedef HistoNames::iterator HistoNamesItr ;
  if (!histoNamesReady)
   { histoNamesReady = true ; histoNames_ = store_->getMEs() ; }
  HistoNamesItr histoName ;
  std::vector<HistoNamesItr> res ;
  for ( histoName = histoNames_.begin() ; histoName != histoNames_.end() ; ++histoName )
   {
    std::size_t nsize = name.size(), lsize = histoName->size() ;
//    if ( (histoName->find(bookPrefix_)==0) &&
//         (lsize>=nsize) &&
//         (histoName->find(name)==(lsize-nsize)) )
//     { res.push_back(histoName) ; }
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
    edm::LogWarning("ElectronDqmAnalyzerBase::find")<<oss.str() ;
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
    edm::LogWarning("ElectronDqmAnalyzerBase::find")<<oss.str() ;
    return 0 ;
   }
  else
   {
    return &*res[0] ;
   }
 }

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
  const std::string * fullName = find(name) ;
  if (fullName)
   { return store_->get(inputInternalPath_+"/"+*fullName) ; }
  else
  { return 0 ; }
 }

void ElectronDqmAnalyzerBase::remove( const std::string & name )
 {
  const std::string * fullName = find(name) ;
  if (fullName)
   {
    store_->setCurrentFolder(inputInternalPath_) ;
    store_->removeElement(*fullName) ;
   }
 }

void ElectronDqmAnalyzerBase::remove_other_dirs()
 {
  std::string currentPath = store_->pwd() ;
  store_->cd() ;

  std::string currentFolder ;
  std::vector<std::string> subDirs ;
  std::vector<std::string>::iterator subDir ;
  std::string delimiter = "/" ;

  std::string::size_type lastPos = currentPath.find_first_not_of(delimiter,0) ;
  std::string::size_type pos = currentPath.find_first_of(delimiter,lastPos) ;
  while (std::string::npos != pos || std::string::npos != lastPos)
   {
    if (currentFolder.empty())
     { currentFolder = currentPath.substr(lastPos, pos - lastPos) ; }
    else
     {
      currentFolder += "/" ;
      currentFolder += currentPath.substr(lastPos, pos - lastPos) ;
     }
    lastPos = currentPath.find_first_not_of(delimiter, pos);
    pos = currentPath.find_first_of(delimiter, lastPos);

    subDirs = store_->getSubdirs() ;
    for ( subDir = subDirs.begin() ; subDir != subDirs.end() ; subDir++ )
     {
      if (currentFolder!=(*subDir))
       { store_->rmdir(*subDir) ; }
     }
    store_->cd(currentFolder) ;
   }
 }

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
 ( const std::string & me2d, const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 { return profileX(get(me2d),title,titleX,titleY,minimum,maximum) ; }

MonitorElement * ElectronDqmAnalyzerBase::profileY
 ( const std::string & me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 { return profileY(get(me2d),title,titleX,titleY,minimum,maximum) ; }

MonitorElement * ElectronDqmAnalyzerBase::bookH1
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY,
   Option_t * option )
 {
  MonitorElement * me = store_->book1D(newName(name),title,nchX,lowX,highX) ;
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
  MonitorElement * me = store_->book1D(newName(name),title,nchX,lowX,highX) ;
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
  MonitorElement * me = store_->book2D(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
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
  MonitorElement * me = store_->book2D(newName(name),title,nchX,lowX,highX,nchY,lowY,highY) ;
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
  MonitorElement * me = store_->bookProfile(newName(name),title,nchX,lowX,highX,lowY,highY," ") ;
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
  std::string name2 = newName(name) ;
  TH1F * h_temp = (TH1F *)num->getTH1F()->Clone(name2.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (verbosity_>0) { h_temp->Print() ; }
  MonitorElement * me = store_->book1D(name2,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::bookH2andDivide
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
  if (verbosity_>0) { h_temp->Print() ; }
  MonitorElement * me = store_->book2D(name2,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::cloneH1
 ( const std::string & name, MonitorElement * original,
   const std::string & title )
 {
  std::string name2 = newName(name) ;
  TH1F * h_temp = (TH1F *)original->getTH1F()->Clone(name2.c_str()) ;
  h_temp->Reset() ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  MonitorElement * me = store_->book1D(name2,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::profileX
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
  MonitorElement * me = store_->bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

MonitorElement * ElectronDqmAnalyzerBase::profileY
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
  MonitorElement * me = store_->bookProfile(name2,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

