#ifndef DPGAnalysis_SiStripTools_DigiBXCorrHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiBXCorrHistogramMaker_H

#include <map>
#include <string>
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

#include "TH2F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"

template <class T>
class DigiBXCorrHistogramMaker {
  
 public:
  DigiBXCorrHistogramMaker(const int ncorbins=1000);
  DigiBXCorrHistogramMaker(const edm::ParameterSet& iConfig);
  
  ~DigiBXCorrHistogramMaker() { };
  
  void book(const char* dirname, const std::map<int,std::string>& labels);
  void beginRun(const unsigned int nrun);
  void fill(const T& he, const std::map<int,int>& ndigi, const edm::Handle<APVCyclePhaseCollection>& phase);
  void fillcorr(const T& he1, const T& he2, const std::map<int,int>& ndigi);
  
 private:
  
  int _ncorbins;
  std::string _hitname;
  const bool _runHisto;

  std::map<int,std::string> _labels;
  std::map<unsigned int, int> _binmax;
  std::map<int,std::string> _phasepart;
  std::vector<int> _scalefact;
  const int _nbins;
  
  RunHistogramManager _rhm;

  std::map<int,TProfile*> _ndigivsdbx;
  std::map<int,TProfile*> _ndigivsdbxzoom2;
  std::map<int,TProfile*> _ndigivsdbxzoom;
  
  std::map<int,TProfile*> _ndigivsdbxincycle;
  std::map<int,TH2F*> _ndigivsdbxincycle2D;

  std::map<int,TProfile*> _nmeandigivscycle;

  std::map<int,TH2F*> _ndigivscycle;
  std::map<int,TH2F*> _ndigivscyclezoom;
  std::map<int,TH2F*> _ndigivscyclezoom2;
  
  std::map<int,TProfile*> _ndigivsbx;
  std::map<int,TH2F*> _ndigivsbx2D;
  std::map<int,TH2F*> _ndigivsbx2Dzoom;
  std::map<int,TH2F*> _ndigivsbx2Dzoom2;

  
  std::map<int,TProfile2D*> _ndigivscycledbx;
  
  std::map<int,TProfile2D*> _ndigivscycle2dbx;
  
  std::map<int,TProfile2D**> _ndigivscycletime;
  
  std::map<int,TH2F*> _ndigivsdbx2D;
  std::map<int,TH2F*> _ndigivsdbx2Dzoom2;
  std::map<int,TH2F*> _ndigivsdbx2Dzoom;
  
  std::map<int,TProfile2D*> _ndigivsdbx3zoom;
  
  std::map<int,TProfile*> _digicorr;
  
  
};

template <class T>
DigiBXCorrHistogramMaker<T>::DigiBXCorrHistogramMaker(const int ncorbins): 
  _ncorbins(ncorbins), _hitname("digi"), _runHisto(true), _labels(), _binmax(), _phasepart(), _scalefact(), _nbins(200),
  _rhm() { }

template <class T>
DigiBXCorrHistogramMaker<T>::DigiBXCorrHistogramMaker(const edm::ParameterSet& iConfig): 
  _ncorbins(iConfig.getUntrackedParameter<int>("corrNbins")),
  _hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  _runHisto(iConfig.getUntrackedParameter<bool>("runHisto",true)),
  _labels(),
  _scalefact(iConfig.getUntrackedParameter<std::vector<int> >("scaleFactors",std::vector<int>(1,5))),
  _nbins(iConfig.getUntrackedParameter<int>("numberOfBins",200)),
  _rhm()
{ 

  std::vector<edm::ParameterSet> 
    wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets",std::vector<edm::ParameterSet>()));
  
  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    _binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
    _phasepart[ps->getParameter<unsigned int>("detSelection")] = ps->getUntrackedParameter<std::string>("phasePartition","None");
  }
  
}

template <class T>
void DigiBXCorrHistogramMaker<T>::book(const char* dirname, const std::map<int,std::string>& labels) {
  
  _labels = labels;
  
  edm::Service<TFileService> tfserv;
  TFileDirectory subev = tfserv->mkdir(dirname);
  
  SiStripTKNumbers trnumb;
  
  edm::LogInfo("NumberOfBins") << "Number of Bins: " << _nbins;
  edm::LogInfo("ScaleFactors") << "x-axis range scale factors: ";
  for(std::vector<int>::const_iterator sf=_scalefact.begin();sf!=_scalefact.end();++sf) {
    edm::LogVerbatim("ScaleFactors") << *sf ;
  }
  edm::LogInfo("BinMaxValue") << "Setting bin max values";

  for(std::map<int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();lab++) {
    
    const int i = lab->first; const std::string slab = lab->second; const unsigned int ui = i;
    
    if(_binmax.find(ui)==_binmax.end()) {
      edm::LogVerbatim("NotConfiguredBinMax") << "Bin max for " << lab->second 
					      << " not configured: " << trnumb.nstrips(i) << " used";
      _binmax[ui] = trnumb.nstrips(i);
    }
 
    edm::LogVerbatim("BinMaxValue") << "Bin max for " << lab->second << " is " << _binmax[ui];

  }

  edm::LogInfo("PhasePartitions") << "Partitions for APV Cycle Phase";

  for(std::map<int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();lab++) {
    
    const int i = lab->first; const std::string slab = lab->second; const unsigned int ui = i;
    edm::LogVerbatim("PhasePartitions") << "Partition for " << lab->second << " is " << ((_phasepart.find(ui)!=_phasepart.end()) ? _phasepart[ui] : "not found") ;

  }

  for(std::map<int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();lab++) {
    
    const int i = lab->first; const std::string slab = lab->second; const unsigned int ui = i;
    
    char name[200];
    char title [500];
    
    // vs DBX
    
    if(_scalefact.size()>=1) {
      sprintf(title,"%s %s multiplicity vs BX separation",slab.c_str(),_hitname.c_str());
      sprintf(name,"n%sdigivsdbx2D",slab.c_str());
      _ndigivsdbx2D[i] = subev.make<TH2F>(name,title,1000,-0.5,500000-0.5,_nbins,0,_binmax[ui]/(_scalefact[0]*_nbins)*_nbins);
      sprintf(name,"n%sdigivsdbx2Dzoom2",slab.c_str());
      _ndigivsdbx2Dzoom2[i] = subev.make<TH2F>(name,title,1000,-0.5,50000-0.5,_nbins,0,_binmax[ui]/(_scalefact[0]*_nbins)*_nbins);
      sprintf(name,"n%sdigivsdbx2Dzoom",slab.c_str());
      _ndigivsdbx2Dzoom[i] = subev.make<TH2F>(name,title,1000,-0.5,999.5,_nbins,0,_binmax[ui]/(_scalefact[0]*_nbins)*_nbins);

      _ndigivsdbx2D[i]->GetXaxis()->SetTitle("#DeltaBX"); _ndigivsdbx2D[i]->GetYaxis()->SetTitle("Number of Hits"); 
      _ndigivsdbx2Dzoom2[i]->GetXaxis()->SetTitle("#DeltaBX"); _ndigivsdbx2Dzoom2[i]->GetYaxis()->SetTitle("Number of Hits"); 
      _ndigivsdbx2Dzoom[i]->GetXaxis()->SetTitle("#DeltaBX"); _ndigivsdbx2Dzoom[i]->GetYaxis()->SetTitle("Number of Hits"); 
    }
    
    sprintf(title,"%s %s multiplicity vs BX separation",slab.c_str(),_hitname.c_str());
    sprintf(name,"n%sdigivsdbx",slab.c_str());
    _ndigivsdbx[i] = subev.make<TProfile>(name,title,1000,-0.5,500000.-0.5);
    sprintf(name,"n%sdigivsdbxzoom2",slab.c_str());
    _ndigivsdbxzoom2[i] = subev.make<TProfile>(name,title,1000,-0.5,50000.-0.5);
    sprintf(name,"n%sdigivsdbxzoom",slab.c_str());
    _ndigivsdbxzoom[i] = subev.make<TProfile>(name,title,1000,-0.5,999.5);
    _ndigivsdbx[i]->GetXaxis()->SetTitle("#DeltaBX");   _ndigivsdbx[i]->GetYaxis()->SetTitle("Number of Hits"); 
    _ndigivsdbxzoom2[i]->GetXaxis()->SetTitle("#DeltaBX"); _ndigivsdbxzoom2[i]->GetYaxis()->SetTitle("Number of Hits");
    _ndigivsdbxzoom[i]->GetXaxis()->SetTitle("#DeltaBX"); _ndigivsdbxzoom[i]->GetYaxis()->SetTitle("Number of Hits"); 

    sprintf(name,"n%sdigivsdbx3zoom",slab.c_str());
    sprintf(title,"%s %s multiplicity vs Triplets BX separation",slab.c_str(),_hitname.c_str());
    _ndigivsdbx3zoom[i] = subev.make<TProfile2D>(name,title,100,-0.5,999.5,100,-0.5,999.5);
    _ndigivsdbx3zoom[i]->GetXaxis()->SetTitle("#DeltaBX(2-1)");  _ndigivsdbx3zoom[i]->GetYaxis()->SetTitle("#DeltaBX(3-2)");
    
    sprintf(name,"%sdigicorr",slab.c_str());
    sprintf(title,"%s %s DBX correlation",slab.c_str(),_hitname.c_str());
    _digicorr[i] = subev.make<TProfile>(name,title,_ncorbins,-0.5,_ncorbins-0.5);
    _digicorr[i]->GetXaxis()->SetTitle("#DeltaBX");   _digicorr[i]->GetYaxis()->SetTitle("Number of Hits"); 
    
    // vs DBX w.r.t. cycle

    if(_scalefact.size()>=1) {

      if(_phasepart.find(ui)!=_phasepart.end() && _phasepart[ui]!="None") {
	sprintf(name,"n%sdigivsdbxincycle",slab.c_str());
	sprintf(title,"%s %s multiplicity vs BX separation w.r.t. cycle",slab.c_str(),_hitname.c_str());
	_ndigivsdbxincycle[i] = subev.make<TProfile>(name,title,1000,-0.5,999.5);
	_ndigivsdbxincycle[i]->GetXaxis()->SetTitle("#DeltaBX w.r.t. cycle");   _ndigivsdbxincycle[i]->GetYaxis()->SetTitle("Number of Hits"); 
	
	sprintf(name,"n%sdigivsdbxincycle2D",slab.c_str());
	sprintf(title,"%s %s multiplicity vs BX separation w.r.t. cycle",slab.c_str(),_hitname.c_str());
	_ndigivsdbxincycle2D[i] = subev.make<TH2F>(name,title,1000,-0.5,999.5,_nbins,0.,_binmax[ui]/(_scalefact[0]*_nbins)*_nbins);
	_ndigivsdbxincycle2D[i]->GetXaxis()->SetTitle("#DeltaBX w.r.t. cycle");   _ndigivsdbxincycle2D[i]->GetYaxis()->SetTitle("Number of Hits"); 
      }

    }
    
    // vs absolute BX mod 70
    
    if(_phasepart.find(ui)!=_phasepart.end() && _phasepart[ui]!="None") {

      sprintf(title,"%s Mean %s multiplicity vs BX mod(70)",slab.c_str(),_hitname.c_str());
      sprintf(name,"n%smeandigivscycle",slab.c_str());
      _nmeandigivscycle[i] =subev.make<TProfile>(name,title,70,-0.5,69.5);
      _nmeandigivscycle[i]->GetXaxis()->SetTitle("absolute BX mod(70)"); 
      _nmeandigivscycle[i]->GetYaxis()->SetTitle("Mean number of Hits");

      sprintf(title,"%s %s multiplicity vs BX mod(70)",slab.c_str(),_hitname.c_str());

      if(_scalefact.size()>=1) {
	sprintf(name,"n%sdigivscycle",slab.c_str());
	_ndigivscycle[i] =subev.make<TH2F>(name,title,70,-0.5,69.5,_nbins,0,_binmax[ui]/(_scalefact[0]*_nbins)*_nbins);
	_ndigivscycle[i]->GetXaxis()->SetTitle("absolute BX mod(70)"); _ndigivscycle[i]->GetYaxis()->SetTitle("Number of Hits");
      }
      if(_scalefact.size()>=2) {
	sprintf(name,"n%sdigivscyclezoom",slab.c_str());
	_ndigivscyclezoom[i] =subev.make<TH2F>(name,title,70,-0.5,69.5,_nbins,0,_binmax[ui]/(_scalefact[1]*_nbins)*_nbins);
	_ndigivscyclezoom[i]->GetXaxis()->SetTitle("absolute BX mod(70)"); 
	_ndigivscyclezoom[i]->GetYaxis()->SetTitle("Number of Hits");
      }
      if(_scalefact.size()>=3) {
	sprintf(name,"n%sdigivscyclezoom2",slab.c_str());
	_ndigivscyclezoom2[i] =subev.make<TH2F>(name,title,70,-0.5,69.5,_nbins,0,_binmax[ui]/(_scalefact[2]*_nbins)*_nbins);
	_ndigivscyclezoom2[i]->GetXaxis()->SetTitle("absolute BX mod(70)"); 
	_ndigivscyclezoom2[i]->GetYaxis()->SetTitle("Number of Hits");
      }

      sprintf(name,"n%sdigivscycledbx",slab.c_str());
      sprintf(title,"%s %s multiplicity vs BX mod(70) and DBX",slab.c_str(),_hitname.c_str());
      _ndigivscycledbx[i] = subev.make<TProfile2D>(name,title,70,-0.5,69.5,1000,-0.5,999.5);
      _ndigivscycledbx[i]->GetXaxis()->SetTitle("Event 1 BX mod(70)"); _ndigivscycledbx[i]->GetYaxis()->SetTitle("#DeltaBX event 2-1");
      
      sprintf(name,"n%sdigivscycle2dbx",slab.c_str());
      sprintf(title,"%s %s multiplicity vs BX mod(70) and DBX",slab.c_str(),_hitname.c_str());
      _ndigivscycle2dbx[i] = subev.make<TProfile2D>(name,title,70,-0.5,69.5,1000,-0.5,999.5);
      _ndigivscycle2dbx[i]->GetXaxis()->SetTitle("Event 2 BX mod(70)"); _ndigivscycle2dbx[i]->GetYaxis()->SetTitle("#DeltaBX event 2-1");
      
    }

    // Multiplicity in cycle vs time is booked also if the phase is not corrected

    if(_runHisto) {
      sprintf(name,"n%sdigivscycletime",slab.c_str());
      sprintf(title,"%s %s multiplicity vs BX mod(70) and Orbit",slab.c_str(),_hitname.c_str());
      _ndigivscycletime[i] =  _rhm.makeTProfile2D(name,title,70,-0.5,69.5,90,0.,90.*11223);
      //      _ndigivscycletime[i]->GetXaxis()->SetTitle("Event 1 BX mod(70)"); _ndigivscycletime[i]->GetYaxis()->SetTitle("time [Orb#]"); 
      //      _ndigivscycletime[i]->SetBit(TH1::kCanRebin);
    }

    // vs BX number 

    sprintf(title,"%s %s mean multiplicity vs BX",slab.c_str(),_hitname.c_str());
    sprintf(name,"n%sdigivsbx",slab.c_str());
    _ndigivsbx[i] =subev.make<TProfile>(name,title,3564,-0.5,3563.5);
    _ndigivsbx[i]->GetXaxis()->SetTitle("BX#"); _ndigivsbx[i]->GetYaxis()->SetTitle("Mean Number of Hits");
    
    sprintf(title,"%s %s multiplicity vs BX",slab.c_str(),_hitname.c_str());
    
    if(_scalefact.size()>=1) {
      sprintf(name,"n%sdigivsbx2D",slab.c_str());
      _ndigivsbx2D[i] =subev.make<TH2F>(name,title,3564,-0.5,3563.5,_nbins,0,_binmax[ui]/(_scalefact[0]*_nbins)*_nbins);
      _ndigivsbx2D[i]->GetXaxis()->SetTitle("BX#"); _ndigivsbx2D[i]->GetYaxis()->SetTitle("Number of Hits");
    }
    if(_scalefact.size()>=2) {
      sprintf(name,"n%sdigivsbx2Dzoom",slab.c_str());
      _ndigivsbx2Dzoom[i] =subev.make<TH2F>(name,title,3564,-0.5,3563.5,_nbins,0,_binmax[ui]/(_scalefact[1]*_nbins)*_nbins);
      _ndigivsbx2Dzoom[i]->GetXaxis()->SetTitle("BX#"); _ndigivsbx2Dzoom[i]->GetYaxis()->SetTitle("Number of Hits");
    }
    if(_scalefact.size()>=3) {
      sprintf(name,"n%sdigivsbx2Dzoom2",slab.c_str());
      _ndigivsbx2Dzoom2[i] =subev.make<TH2F>(name,title,3564,-0.5,3563.5,_nbins,0,_binmax[ui]/(_scalefact[2]*_nbins)*_nbins);
      _ndigivsbx2Dzoom2[i]->GetXaxis()->SetTitle("BX#"); _ndigivsbx2Dzoom2[i]->GetYaxis()->SetTitle("Number of Hits");
    }

  }
  
}

template <class T>
void DigiBXCorrHistogramMaker<T>::beginRun(const unsigned int nrun) {

  _rhm.beginRun(nrun);

  for(std::map<int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();lab++) {
    
    const int i = lab->first; 
    if(_runHisto) {
      if(_ndigivscycletime[i]) {
	(*_ndigivscycletime[i])->GetXaxis()->SetTitle("Event 1 BX mod(70)"); (*_ndigivscycletime[i])->GetYaxis()->SetTitle("time [Orb#]"); 
	(*_ndigivscycletime[i])->SetBit(TH1::kCanRebin);
      }
    }
  }


}

template <class T>
void DigiBXCorrHistogramMaker<T>::fill(const T& he, const std::map<int,int>& ndigi, const edm::Handle<APVCyclePhaseCollection>& phase) {
  
  for(std::map<int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {
    
    if(_labels.find(digi->first) != _labels.end()) {
      const int i=digi->first; const unsigned int ui = i;
      
      int thephase = APVCyclePhaseCollection::invalid;
	if(_phasepart.find(ui)!=_phasepart.end() && _phasepart[ui]!="None") {
	  if(!phase.failedToGet() && phase.isValid()) {
	    thephase = phase->getPhase(_phasepart[ui]);
	  }
	}

      long long tbx = he.absoluteBX();
      if(thephase!=APVCyclePhaseCollection::nopartition &&
	 thephase!=APVCyclePhaseCollection::multiphase &&
	 thephase!=APVCyclePhaseCollection::invalid) tbx -= thephase;

      if(_nmeandigivscycle.find(i)!=_nmeandigivscycle.end()) _nmeandigivscycle[i]->Fill(tbx%70,digi->second);

      if(_ndigivscycle.find(i)!=_ndigivscycle.end()) _ndigivscycle[i]->Fill(tbx%70,digi->second);
      if(_ndigivscyclezoom.find(i)!=_ndigivscyclezoom.end()) _ndigivscyclezoom[i]->Fill(tbx%70,digi->second);
      if(_ndigivscyclezoom2.find(i)!=_ndigivscyclezoom2.end()) _ndigivscyclezoom2[i]->Fill(tbx%70,digi->second);
      
      if(_runHisto) {
	if(_ndigivscycletime.find(i)!=_ndigivscycletime.end()) {
	  if(_ndigivscycletime[i]!=0 && (*_ndigivscycletime[i])!=0 ) (*_ndigivscycletime[i])->Fill(tbx%70,(int)he._orbit,digi->second);
	}
      }

      _ndigivsbx[i]->Fill(he.bx(),digi->second);
      if(_ndigivsbx2D.find(i)!=_ndigivsbx2D.end()) _ndigivsbx2D[i]->Fill(he.bx(),digi->second);
      if(_ndigivsbx2Dzoom.find(i)!=_ndigivsbx2Dzoom.end()) _ndigivsbx2Dzoom[i]->Fill(he.bx(),digi->second);
      if(_ndigivsbx2Dzoom2.find(i)!=_ndigivsbx2Dzoom2.end()) _ndigivsbx2Dzoom2[i]->Fill(he.bx(),digi->second);
      
      
      if(he.depth()>0) {
	
	long long dbx = he.deltaBX();
	
	_ndigivsdbx[i]->Fill(dbx,digi->second);
	_ndigivsdbxzoom[i]->Fill(dbx,digi->second);
	_ndigivsdbxzoom2[i]->Fill(dbx,digi->second);
	
	if(_ndigivsdbx2D.find(i)!=_ndigivsdbx2D.end()) _ndigivsdbx2D[i]->Fill(dbx,digi->second);
	if(_ndigivsdbx2Dzoom.find(i)!=_ndigivsdbx2Dzoom.end()) _ndigivsdbx2Dzoom[i]->Fill(dbx,digi->second);
	if(_ndigivsdbx2Dzoom2.find(i)!=_ndigivsdbx2Dzoom2.end()) _ndigivsdbx2Dzoom2[i]->Fill(dbx,digi->second);
	
	if(thephase!=APVCyclePhaseCollection::nopartition &&
	   thephase!=APVCyclePhaseCollection::multiphase &&
	   thephase!=APVCyclePhaseCollection::invalid) {
	  long long dbxincycle = he.deltaBXinCycle(thephase);
	  if(_ndigivsdbxincycle2D.find(i)!=_ndigivsdbxincycle2D.end()) _ndigivsdbxincycle2D[i]->Fill(dbxincycle,digi->second);
	  if(_ndigivsdbxincycle.find(i)!=_ndigivsdbxincycle.end()) _ndigivsdbxincycle[i]->Fill(dbxincycle,digi->second);
	}

	long long prevtbx = he.absoluteBX(1);
	if(thephase!=APVCyclePhaseCollection::nopartition &&
	   thephase!=APVCyclePhaseCollection::multiphase &&
	   thephase!=APVCyclePhaseCollection::invalid) prevtbx -= thephase;

	
	if(_ndigivscycledbx.find(i)!=_ndigivscycledbx.end()) _ndigivscycledbx[i]->Fill(prevtbx%70,dbx,digi->second);
	if(_ndigivscycle2dbx.find(i)!=_ndigivscycle2dbx.end()) _ndigivscycle2dbx[i]->Fill(tbx%70,dbx,digi->second);
	
	if(he.depth()>1) {
	  
	  long long dbx2 = he.deltaBX(1,2);
	  _ndigivsdbx3zoom[i]->Fill(dbx2,dbx,digi->second);
	  
	}
      }
      
    }
    else {
      edm::LogWarning("MissingKey") << " Key " << digi->first << " is missing ";
    }
    
  }
}

template <class T>
void DigiBXCorrHistogramMaker<T>::fillcorr(const T& he1, const T& he2, const std::map<int,int>& ndigi) {
  
  for(std::map<int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {
    
    if(_labels.find(digi->first) != _labels.end()) {
      const int i=digi->first;
      
      long long dbx = he2.deltaBX(he1);
      _digicorr[i]->Fill(dbx,digi->second);
      
    }
    else {
      edm::LogWarning("MissingKey") << " Key " << digi->first << " is missing ";
    }
    
  }
}


#endif // DPGAnalysis_SiStripTools_DigiBXCorrHistogramMaker_H
