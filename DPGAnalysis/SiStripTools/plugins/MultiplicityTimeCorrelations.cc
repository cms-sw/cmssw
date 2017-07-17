// -*- C++ -*-
//
// Package:    MultiplicityTimeCorrelations
// Class:      MultiplicityTimeCorrelations
//
/**\class MultiplicityTimeCorrelations MultiplicityTimeCorrelations.cc DPGAnalysis/SiStripTools/src/MultiplicityTimeCorrelations.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Oct 27 17:37:53 CET 2008
// $Id: MultiplicityTimeCorrelations.cc,v 1.1 2011/03/10 16:15:13 venturia Exp $
//
//


// system include files
#include <memory>

// user include files

#include <vector>
#include <map>
#include <limits.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH1F.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"
#include "DPGAnalysis/SiStripTools/interface/DigiBXCorrHistogramMaker.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistoryFilter.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

//
// class decleration
//

class MultiplicityTimeCorrelations : public edm::EDAnalyzer {
   public:
      explicit MultiplicityTimeCorrelations(const edm::ParameterSet&);
      ~MultiplicityTimeCorrelations();


private:
  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endJob() override ;

      // ----------member data ---------------------------

  DigiBXCorrHistogramMaker<EventWithHistory> _digibxcorrhmevent;
  EventWithHistoryFilter _evfilter;

  std::map<int,std::map<unsigned int,TH1F*> > _dbxhistos;
  /*
  std::map<int,TH1F*> _dbxtkhistos;
  std::map<int,TH1F*> _dbxtibhistos;
  std::map<int,TH1F*> _dbxtidhistos;
  std::map<int,TH1F*> _dbxtobhistos;
  std::map<int,TH1F*> _dbxtechistos;
  */

  edm::InputTag _hecollection;
  edm::EDGetTokenT<EventWithHistory> _hecollectionToken;
  edm::EDGetTokenT<APVCyclePhaseCollection> _apvphasecollToken;
  edm::EDGetTokenT<std::map<unsigned int, int> > _multiplicityMapToken;
  std::map<unsigned int, std::string> _subdets;
  std::map<unsigned int, int> _binmax;

  int _loworbit;
  int _highorbit;

  int _mindbx;
  int _mintrpltdbx;

  SiStripTKNumbers _trnumb;

  std::vector<int> _dbxbins;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MultiplicityTimeCorrelations::MultiplicityTimeCorrelations(const edm::ParameterSet& iConfig):
  _digibxcorrhmevent(iConfig, consumesCollector()),
  _evfilter(),
  _hecollection(iConfig.getParameter<edm::InputTag>("historyProduct")),
  _hecollectionToken(consumes<EventWithHistory>(_hecollection)),
  _apvphasecollToken(consumes<APVCyclePhaseCollection>(iConfig.getParameter<edm::InputTag>("apvPhaseCollection"))),
  _multiplicityMapToken(mayConsume<std::map<unsigned int, int> >(iConfig.getParameter<edm::InputTag>("multiplicityMap"))),
  _subdets(),
  _binmax(),
  _loworbit(iConfig.getUntrackedParameter<int>("lowedgeOrbit")),
  _highorbit(iConfig.getUntrackedParameter<int>("highedgeOrbit")),
  _mindbx(iConfig.getUntrackedParameter<int>("minDBX")),
  _mintrpltdbx(iConfig.getUntrackedParameter<int>("minTripletDBX")),
  _trnumb(),
  _dbxbins(iConfig.getUntrackedParameter<std::vector<int> >("dbxBins"))
{
   //now do what ever initialization is needed

  // configure the filter

  edm::ParameterSet filterConfig;
  filterConfig.addUntrackedParameter<edm::InputTag>("historyProduct",_hecollection);
  if(_mindbx>0) {
    std::vector<int> dbxrange;
    dbxrange.push_back(_mindbx+1);
    dbxrange.push_back(-1);
    filterConfig.addUntrackedParameter<std::vector<int> >("dbxRange",dbxrange);
  }
  if(_mintrpltdbx>0) {
    std::vector<int> dbxtrpltrange;
    dbxtrpltrange.push_back(_mintrpltdbx+1);
    dbxtrpltrange.push_back(-1);
    filterConfig.addUntrackedParameter<std::vector<int> >("dbxTripletRange",dbxtrpltrange);
  }

  _evfilter.set(filterConfig, consumesCollector());

  //

  edm::Service<TFileService> tfserv;

  // create map of labels

  std::vector<edm::ParameterSet> wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets"));

  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    _subdets[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    _binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
  }
  std::map<int,std::string> labels;

  for(std::map<unsigned int,std::string>::const_iterator subd=_subdets.begin();subd!=_subdets.end();++subd) {
    labels[int(subd->first)] = subd->second;
  }

  //

  _digibxcorrhmevent.book("EventProcs",labels);

  TFileDirectory subdbxbin = tfserv->mkdir("DBXDebugging");

  for(std::vector<int>::const_iterator bin=_dbxbins.begin();bin!=_dbxbins.end();bin++) {
    char hname[200]; char htitle[200];

    edm::LogInfo("DBXHistosBinMaxValue") << "Setting bin max values";

    for(std::map<unsigned int, std::string>::const_iterator subd=_subdets.begin();subd!=_subdets.end();++subd) {
      if(_binmax.find(subd->first)==_binmax.end()) {
	edm::LogVerbatim("DBXHistosNotConfiguredBinMax") << "Bin max for " << subd->second
						     << " not configured: " << _trnumb.nstrips(int(subd->first)) << " used";
	_binmax[subd->first] = _trnumb.nstrips(int(subd->first));
      }

      edm::LogVerbatim("DBXHistosBinMaxValue") << "Bin max for " << subd->second << " is " << _binmax[subd->first];



      sprintf(hname,"sumn%sdigi_%d",subd->second.c_str(),*bin);
      sprintf(htitle,"%s digi multiplicity at DBX = %d",subd->second.c_str(),*bin);
      LogDebug("DBXDebug") << "creating histogram " << hname << " " << htitle;
      _dbxhistos[*bin][subd->first]= subdbxbin.make<TH1F>(hname,htitle,1000,0.,_binmax[subd->first]/(20*1000)*1000);
      _dbxhistos[*bin][subd->first]->GetXaxis()->SetTitle("Number of Digis");
    }
    /*
    sprintf(hname,"sumntkdigi_%d",*bin);
    sprintf(htitle,"TK digi multiplicity at DBX = %d",*bin);
    LogDebug("DBXDebug") << "creating histogram " << hname << " " << htitle;
    _dbxtkhistos[*bin]= subdbxbin.make<TH1F>(hname,htitle,1000,0.,_trnumb.nstrips(0)/(20*1000)*1000);
    _dbxtkhistos[*bin]->GetXaxis()->SetTitle("Number of Digis");

    sprintf(hname,"sumntibdigi_%d",*bin);
    sprintf(htitle,"TIB digi multiplicity at DBX = %d",*bin);
    LogDebug("DBXDebug") << "creating histogram " << hname << " " << htitle;
    _dbxtibhistos[*bin]= subdbxbin.make<TH1F>(hname,htitle,1000,0.,_trnumb.nstrips(SiStripDetId::TIB)/(20*1000)*1000);
    _dbxtibhistos[*bin]->GetXaxis()->SetTitle("Number of Digis");

    sprintf(hname,"sumntiddigi_%d",*bin);
    sprintf(htitle,"TID digi multiplicity at DBX = %d",*bin);
    LogDebug("DBXDebug") << "creating histogram " << hname << " " << htitle;
    _dbxtidhistos[*bin]= subdbxbin.make<TH1F>(hname,htitle,1000,0.,_trnumb.nstrips(SiStripDetId::TID)/(20*1000)*1000);
    _dbxtidhistos[*bin]->GetXaxis()->SetTitle("Number of Digis");

    sprintf(hname,"sumntobdigi_%d",*bin);
    sprintf(htitle,"TOB digi multiplicity at DBX = %d",*bin);
    LogDebug("DBXDebug") << "creating histogram " << hname << " " << htitle;
    _dbxtobhistos[*bin]= subdbxbin.make<TH1F>(hname,htitle,1000,0.,_trnumb.nstrips(SiStripDetId::TOB)/(20*1000)*1000);
    _dbxtobhistos[*bin]->GetXaxis()->SetTitle("Number of Digis");

    sprintf(hname,"sumntecdigi_%d",*bin);
    sprintf(htitle,"TEC digi multiplicity at DBX = %d",*bin);
    LogDebug("DBXDebug") << "creating histogram " << hname << " " << htitle;
    _dbxtechistos[*bin]= subdbxbin.make<TH1F>(hname,htitle,1000,0.,_trnumb.nstrips(SiStripDetId::TEC)/(20*1000)*1000);
    _dbxtechistos[*bin]->GetXaxis()->SetTitle("Number of Digis");
    */
  }

}


MultiplicityTimeCorrelations::~MultiplicityTimeCorrelations()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MultiplicityTimeCorrelations::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // get Phase

  Handle<APVCyclePhaseCollection> apvphase;
  iEvent.getByToken(_apvphasecollToken,apvphase);

  // get HE

  Handle<EventWithHistory> he;
  iEvent.getByToken(_hecollectionToken,he);

  // check if the event is selected

  if((_loworbit < 0 || iEvent.orbitNumber() >= _loworbit) &&
     (_highorbit < 0 || iEvent.orbitNumber() <= _highorbit)) {

    if(_evfilter.selected(iEvent,iSetup)) {


      //Compute digi multiplicity
      /*
      int ntkdigi=0;
      int ntibdigi=0;
      int ntiddigi=0;
      int ntobdigi=0;
      int ntecdigi=0;
      */
      Handle<std::map<unsigned int, int> > mults;
      iEvent.getByToken(_multiplicityMapToken,mults);

      // create map of digi multiplicity

      std::map<int,int> digimap;
      for(std::map<unsigned int, int>::const_iterator mult=mults->begin();mult!=mults->end();++mult) {
	if(_subdets.find(mult->first)!=_subdets.end()) digimap[int(mult->first)] = mult->second;
      }

      _digibxcorrhmevent.fill(*he,digimap,apvphase);

      // fill debug histos

      if(he->depth()!=0) {

	long long dbx = he->deltaBX();

	if(_dbxhistos.find(dbx)!=_dbxhistos.end()) {
	  for(std::map<unsigned int,int>::const_iterator ndigi=mults->begin();ndigi!=mults->end();++ndigi) {
	  _dbxhistos[dbx][ndigi->first]->Fill(ndigi->second);
	  }
	}
	if(_dbxhistos.find(-1)!=_dbxhistos.end()) {
	  for(std::map<unsigned int,int>::const_iterator ndigi=mults->begin();ndigi!=mults->end();++ndigi) {
	  _dbxhistos[-1][ndigi->first]->Fill(ndigi->second);
	  }
	}
	/*
	if(_dbxtkhistos.find(dbx)!=_dbxtkhistos.end()) {
	  _dbxtkhistos[dbx]->Fill(ntkdigi);
	}
	if(_dbxtkhistos.find(-1)!=_dbxtkhistos.end()) {
	  _dbxtkhistos[-1]->Fill(ntkdigi);
	}

	if(_dbxtibhistos.find(dbx)!=_dbxtibhistos.end()) {
	  _dbxtibhistos[dbx]->Fill(ntibdigi);
	}
	if(_dbxtibhistos.find(-1)!=_dbxtibhistos.end()) {
	  _dbxtibhistos[-1]->Fill(ntibdigi);
	}

	if(_dbxtidhistos.find(dbx)!=_dbxtidhistos.end()) {
	  _dbxtidhistos[dbx]->Fill(ntiddigi);
	}
	if(_dbxtidhistos.find(-1)!=_dbxtidhistos.end()) {
	  _dbxtidhistos[-1]->Fill(ntiddigi);
	}
	if(_dbxtobhistos.find(dbx)!=_dbxtobhistos.end()) {
	  _dbxtobhistos[dbx]->Fill(ntobdigi);
	}
	if(_dbxtobhistos.find(-1)!=_dbxtobhistos.end()) {
	  _dbxtobhistos[-1]->Fill(ntobdigi);
	}
	if(_dbxtechistos.find(dbx)!=_dbxtechistos.end()) {
	  _dbxtechistos[dbx]->Fill(ntecdigi);
	}
	if(_dbxtechistos.find(-1)!=_dbxtechistos.end()) {
	  _dbxtechistos[-1]->Fill(ntecdigi);
	}
	*/
      }
    }
  }
}


// ------------ method called once each job just before starting event loop  ------------
void
MultiplicityTimeCorrelations::beginJob()
{

  LogDebug("IntegerDebug") << " int max and min " << INT_MIN << " " << INT_MAX;
  LogDebug("IntegerDebug") << " uint max and min " << UINT_MAX;
  LogDebug("IntegerDebug") << " long max and min " << LONG_MIN << " " << LONG_MAX;
  LogDebug("IntegerDebug") << " ulong max and min " << ULONG_MAX;
  LogDebug("IntegerDebug") << " long long max and min " << LLONG_MIN << " " << LLONG_MAX;
  LogDebug("IntegerDebug") << " u long long max and min " << ULLONG_MAX;


  edm::LogInfo("MultiplicityTimeCorrelations") << " Correlation studies performed only in the orbit # range " << _loworbit << " " << _highorbit ;

}

void
MultiplicityTimeCorrelations::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

  _digibxcorrhmevent.beginRun(iRun.run());

}
// ------------ method called once each job just after ending the event loop  ------------
void
MultiplicityTimeCorrelations::endJob() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(MultiplicityTimeCorrelations);
