#ifndef DPGAnalysis_SiStripTools_DigiBXCorrHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiBXCorrHistogramMaker_H

#include <map>
#include <string>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

#include "TH2F.h"
#include "TH3F.h"
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
  DigiBXCorrHistogramMaker(edm::ConsumesCollector&& iC, const int ncorbins=1000);
  DigiBXCorrHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);

  ~DigiBXCorrHistogramMaker() { };

  void book(const char* dirname, const std::map<int,std::string>& labels);
  void beginRun(const unsigned int nrun);
  void fill(const T& he, const std::map<int,int>& ndigi, const edm::Handle<APVCyclePhaseCollection>& phase);
  void fillcorr(const T& he1, const T& he2, const std::map<int,int>& ndigi);

 private:

  int m_ncorbins;
  std::string m_hitname;
  const bool m_dbx3Histo;
  const bool m_dbx3Histo3D;
  const bool m_runHisto;

  std::map<int,std::string> m_labels;
  std::map<unsigned int, int> m_binmax;
  std::map<int,std::string> m_phasepart;
  std::vector<int> m_scalefact;
  const int m_nbins;

  RunHistogramManager m_rhm;

  std::map<int,TProfile*> m_ndigivsdbx;
  std::map<int,TProfile*> m_ndigivsdbxzoom2;
  std::map<int,TProfile*> m_ndigivsdbxzoom;

  std::map<int,TProfile*> m_ndigivsdbxincycle;
  std::map<int,TH2F*> m_ndigivsdbxincycle2D;

  std::map<int,TProfile*> m_nmeandigivscycle;

  std::map<int,TH2F*> m_ndigivscycle;
  std::map<int,TH2F*> m_ndigivscyclezoom;
  std::map<int,TH2F*> m_ndigivscyclezoom2;

  std::map<int,TProfile*> m_ndigivsbx;
  std::map<int,TH2F*> m_ndigivsbx2D;
  std::map<int,TH2F*> m_ndigivsbx2Dzoom;
  std::map<int,TH2F*> m_ndigivsbx2Dzoom2;


  std::map<int,TProfile2D*> m_ndigivscycledbx;

  std::map<int,TProfile2D*> m_ndigivscycle2dbx;

  std::map<int,TProfile2D**> m_ndigivscycletime;

  std::map<int,TH2F*> m_ndigivsdbx2D;
  std::map<int,TH2F*> m_ndigivsdbx2Dzoom2;
  std::map<int,TH2F*> m_ndigivsdbx2Dzoom;

  std::map<int,TProfile2D*> m_ndigivsdbx3zoom;
  std::map<int,TProfile2D*> m_ndigivsdbxincycle3;
  std::map<int,TH3F*> m_ndigivsdbxincycle33D;

  std::map<int,TProfile*> m_digicorr;


};

template <class T>
DigiBXCorrHistogramMaker<T>::DigiBXCorrHistogramMaker(edm::ConsumesCollector&& iC, const int ncorbins):
  m_ncorbins(ncorbins), m_hitname("digi"), m_dbx3Histo(false), m_dbx3Histo3D(false),
  m_runHisto(true), m_labels(), m_binmax(), m_phasepart(), m_scalefact(), m_nbins(200),
  m_rhm(iC) { }

template <class T>
DigiBXCorrHistogramMaker<T>::DigiBXCorrHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  m_ncorbins(iConfig.getUntrackedParameter<int>("corrNbins")),
  m_hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  m_dbx3Histo(iConfig.getUntrackedParameter<bool>("dbx3Histo",false)),
  m_dbx3Histo3D(iConfig.getUntrackedParameter<bool>("dbx3Histo3D",false)),
  m_runHisto(iConfig.getUntrackedParameter<bool>("runHisto",true)),
  m_labels(),
  m_scalefact(iConfig.getUntrackedParameter<std::vector<int> >("scaleFactors",std::vector<int>(1,5))),
  m_nbins(iConfig.getUntrackedParameter<int>("numberOfBins",200)),
  m_rhm(iC)
{

  std::vector<edm::ParameterSet>
    wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets",std::vector<edm::ParameterSet>()));

  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
    m_phasepart[ps->getParameter<unsigned int>("detSelection")] = ps->getUntrackedParameter<std::string>("phasePartition","None");
  }

}

template <class T>
void DigiBXCorrHistogramMaker<T>::book(const char* dirname, const std::map<int,std::string>& labels) {

  m_labels = labels;

  edm::Service<TFileService> tfserv;
  TFileDirectory subev = tfserv->mkdir(dirname);

  SiStripTKNumbers trnumb;

  edm::LogInfo("NumberOfBins") << "Number of Bins: " << m_nbins;
  edm::LogInfo("ScaleFactors") << "x-axis range scale factors: ";
  for(std::vector<int>::const_iterator sf=m_scalefact.begin();sf!=m_scalefact.end();++sf) {
    edm::LogVerbatim("ScaleFactors") << *sf ;
  }
  edm::LogInfo("BinMaxValue") << "Setting bin max values";

  for(std::map<int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {

    const int i = lab->first; const std::string slab = lab->second; const unsigned int ui = i;

    if(m_binmax.find(ui)==m_binmax.end()) {
      edm::LogVerbatim("NotConfiguredBinMax") << "Bin max for " << lab->second
					      << " not configured: " << trnumb.nstrips(i) << " used";
      m_binmax[ui] = trnumb.nstrips(i);
    }

    edm::LogVerbatim("BinMaxValue") << "Bin max for " << lab->second << " is " << m_binmax[ui];

  }

  edm::LogInfo("PhasePartitions") << "Partitions for APV Cycle Phase";

  for(std::map<int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {

    const int i = lab->first; const std::string slab = lab->second; const unsigned int ui = i;
    edm::LogVerbatim("PhasePartitions") << "Partition for " << lab->second << " is " << ((m_phasepart.find(ui)!=m_phasepart.end()) ? m_phasepart[ui] : "not found") ;

  }

  for(std::map<int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {

    const int i = lab->first; const std::string slab = lab->second; const unsigned int ui = i;

    char name[200];
    char title [500];

    // vs DBX

    if(m_scalefact.size()>=1) {
      sprintf(title,"%s %s multiplicity vs BX separation",slab.c_str(),m_hitname.c_str());
      sprintf(name,"n%sdigivsdbx2D",slab.c_str());
      m_ndigivsdbx2D[i] = subev.make<TH2F>(name,title,1000,-0.5,500000-0.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[0]*m_nbins))*m_nbins);
      sprintf(name,"n%sdigivsdbx2Dzoom2",slab.c_str());
      m_ndigivsdbx2Dzoom2[i] = subev.make<TH2F>(name,title,1000,-0.5,50000-0.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[0]*m_nbins))*m_nbins);
      sprintf(name,"n%sdigivsdbx2Dzoom",slab.c_str());
      m_ndigivsdbx2Dzoom[i] = subev.make<TH2F>(name,title,1000,-0.5,999.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[0]*m_nbins))*m_nbins);

      m_ndigivsdbx2D[i]->GetXaxis()->SetTitle("#DeltaBX"); m_ndigivsdbx2D[i]->GetYaxis()->SetTitle("Number of Hits");
      m_ndigivsdbx2Dzoom2[i]->GetXaxis()->SetTitle("#DeltaBX"); m_ndigivsdbx2Dzoom2[i]->GetYaxis()->SetTitle("Number of Hits");
      m_ndigivsdbx2Dzoom[i]->GetXaxis()->SetTitle("#DeltaBX"); m_ndigivsdbx2Dzoom[i]->GetYaxis()->SetTitle("Number of Hits");
    }

    sprintf(title,"%s %s multiplicity vs BX separation",slab.c_str(),m_hitname.c_str());
    sprintf(name,"n%sdigivsdbx",slab.c_str());
    m_ndigivsdbx[i] = subev.make<TProfile>(name,title,1000,-0.5,500000.-0.5);
    sprintf(name,"n%sdigivsdbxzoom2",slab.c_str());
    m_ndigivsdbxzoom2[i] = subev.make<TProfile>(name,title,1000,-0.5,50000.-0.5);
    sprintf(name,"n%sdigivsdbxzoom",slab.c_str());
    m_ndigivsdbxzoom[i] = subev.make<TProfile>(name,title,1000,-0.5,999.5);
    m_ndigivsdbx[i]->GetXaxis()->SetTitle("#DeltaBX");   m_ndigivsdbx[i]->GetYaxis()->SetTitle("Number of Hits");
    m_ndigivsdbxzoom2[i]->GetXaxis()->SetTitle("#DeltaBX"); m_ndigivsdbxzoom2[i]->GetYaxis()->SetTitle("Number of Hits");
    m_ndigivsdbxzoom[i]->GetXaxis()->SetTitle("#DeltaBX"); m_ndigivsdbxzoom[i]->GetYaxis()->SetTitle("Number of Hits");

    sprintf(name,"n%sdigivsdbx3zoom",slab.c_str());
    sprintf(title,"%s %s multiplicity vs Triplets BX separation",slab.c_str(),m_hitname.c_str());
    m_ndigivsdbx3zoom[i] = subev.make<TProfile2D>(name,title,100,-0.5,999.5,100,-0.5,999.5);
    m_ndigivsdbx3zoom[i]->GetXaxis()->SetTitle("#DeltaBX(n,n-1)");  m_ndigivsdbx3zoom[i]->GetYaxis()->SetTitle("#DeltaBX(n,n-2)");

    sprintf(name,"%sdigicorr",slab.c_str());
    sprintf(title,"%s %s DBX correlation",slab.c_str(),m_hitname.c_str());
    m_digicorr[i] = subev.make<TProfile>(name,title,m_ncorbins,-0.5,m_ncorbins-0.5);
    m_digicorr[i]->GetXaxis()->SetTitle("#DeltaBX");   m_digicorr[i]->GetYaxis()->SetTitle("Number of Hits");

    // vs DBX w.r.t. cycle

    if(m_scalefact.size()>=1) {

      if(m_phasepart.find(ui)!=m_phasepart.end() && m_phasepart[ui]!="None") {
	sprintf(name,"n%sdigivsdbxincycle",slab.c_str());
	sprintf(title,"%s %s multiplicity vs BX separation w.r.t. cycle",slab.c_str(),m_hitname.c_str());
	m_ndigivsdbxincycle[i] = subev.make<TProfile>(name,title,1000,-0.5,999.5);
	m_ndigivsdbxincycle[i]->GetXaxis()->SetTitle("#DeltaBX w.r.t. cycle");   m_ndigivsdbxincycle[i]->GetYaxis()->SetTitle("Number of Hits");

	sprintf(name,"n%sdigivsdbxincycle2D",slab.c_str());
	sprintf(title,"%s %s multiplicity vs BX separation w.r.t. cycle",slab.c_str(),m_hitname.c_str());
	m_ndigivsdbxincycle2D[i] = subev.make<TH2F>(name,title,1000,-0.5,999.5,m_nbins,0.,(1+m_binmax[ui]/(m_scalefact[0]*m_nbins))*m_nbins);
	m_ndigivsdbxincycle2D[i]->GetXaxis()->SetTitle("#DeltaBX w.r.t. cycle");   m_ndigivsdbxincycle2D[i]->GetYaxis()->SetTitle("Number of Hits");

	if(m_dbx3Histo) {
	  sprintf(name,"n%sdigivsdbxincycle3",slab.c_str());
	  sprintf(title,"%s %s multiplicity vs Triplets BX separation w.r.t. cycle",slab.c_str(),m_hitname.c_str());
	  m_ndigivsdbxincycle3[i] = subev.make<TProfile2D>(name,title,2000,-0.5,1999.5,30,-0.5,2099.5);
	  m_ndigivsdbxincycle3[i]->GetXaxis()->SetTitle("#DeltaBX(n,n-1)");
	  m_ndigivsdbxincycle3[i]->GetYaxis()->SetTitle("#DeltaBX(n,n-2)-#DeltaBX(n,n-1)");

	  if(m_dbx3Histo3D) {
	    sprintf(name,"n%sdigivsdbxincycle33D",slab.c_str());
	    sprintf(title,"%s %s multiplicity vs Triplets BX separation w.r.t. cycle",slab.c_str(),m_hitname.c_str());
	    m_ndigivsdbxincycle33D[i] = subev.make<TH3F>(name,title,2000,-0.5,1999.5,30,-0.5,2099.5,50,0.,(1+m_binmax[ui]/(m_scalefact[0]*50))*50);
	    m_ndigivsdbxincycle33D[i]->GetXaxis()->SetTitle("#DeltaBX(n,n-1)");
	    m_ndigivsdbxincycle33D[i]->GetYaxis()->SetTitle("#DeltaBX(n,n-2)-#DeltaBX(n,n-1)");
	  }
	}
      }

    }

    // vs absolute BX mod 70

    if(m_phasepart.find(ui)!=m_phasepart.end() && m_phasepart[ui]!="None") {

      sprintf(title,"%s Mean %s multiplicity vs BX mod(70)",slab.c_str(),m_hitname.c_str());
      sprintf(name,"n%smeandigivscycle",slab.c_str());
      m_nmeandigivscycle[i] =subev.make<TProfile>(name,title,70,-0.5,69.5);
      m_nmeandigivscycle[i]->GetXaxis()->SetTitle("absolute BX mod(70)");
      m_nmeandigivscycle[i]->GetYaxis()->SetTitle("Mean number of Hits");

      sprintf(title,"%s %s multiplicity vs BX mod(70)",slab.c_str(),m_hitname.c_str());

      if(m_scalefact.size()>=1) {
	sprintf(name,"n%sdigivscycle",slab.c_str());
	m_ndigivscycle[i] =subev.make<TH2F>(name,title,70,-0.5,69.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[0]*m_nbins))*m_nbins);
	m_ndigivscycle[i]->GetXaxis()->SetTitle("absolute BX mod(70)"); m_ndigivscycle[i]->GetYaxis()->SetTitle("Number of Hits");
      }
      if(m_scalefact.size()>=2) {
	sprintf(name,"n%sdigivscyclezoom",slab.c_str());
	m_ndigivscyclezoom[i] =subev.make<TH2F>(name,title,70,-0.5,69.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[1]*m_nbins))*m_nbins);
	m_ndigivscyclezoom[i]->GetXaxis()->SetTitle("absolute BX mod(70)");
	m_ndigivscyclezoom[i]->GetYaxis()->SetTitle("Number of Hits");
      }
      if(m_scalefact.size()>=3) {
	sprintf(name,"n%sdigivscyclezoom2",slab.c_str());
	m_ndigivscyclezoom2[i] =subev.make<TH2F>(name,title,70,-0.5,69.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[2]*m_nbins))*m_nbins);
	m_ndigivscyclezoom2[i]->GetXaxis()->SetTitle("absolute BX mod(70)");
	m_ndigivscyclezoom2[i]->GetYaxis()->SetTitle("Number of Hits");
      }

      sprintf(name,"n%sdigivscycledbx",slab.c_str());
      sprintf(title,"%s %s multiplicity vs BX mod(70) and DBX",slab.c_str(),m_hitname.c_str());
      m_ndigivscycledbx[i] = subev.make<TProfile2D>(name,title,70,-0.5,69.5,1000,-0.5,999.5);
      m_ndigivscycledbx[i]->GetXaxis()->SetTitle("Event 1 BX mod(70)"); m_ndigivscycledbx[i]->GetYaxis()->SetTitle("#DeltaBX event 2-1");

      sprintf(name,"n%sdigivscycle2dbx",slab.c_str());
      sprintf(title,"%s %s multiplicity vs BX mod(70) and DBX",slab.c_str(),m_hitname.c_str());
      m_ndigivscycle2dbx[i] = subev.make<TProfile2D>(name,title,70,-0.5,69.5,1000,-0.5,999.5);
      m_ndigivscycle2dbx[i]->GetXaxis()->SetTitle("Event 2 BX mod(70)"); m_ndigivscycle2dbx[i]->GetYaxis()->SetTitle("#DeltaBX event 2-1");

    }

    // Multiplicity in cycle vs time is booked also if the phase is not corrected

    if(m_runHisto) {
      sprintf(name,"n%sdigivscycletime",slab.c_str());
      sprintf(title,"%s %s multiplicity vs BX mod(70) and Orbit",slab.c_str(),m_hitname.c_str());
      m_ndigivscycletime[i] =  m_rhm.makeTProfile2D(name,title,70,-0.5,69.5,90,0.,90*262144);
      //      m_ndigivscycletime[i]->GetXaxis()->SetTitle("Event 1 BX mod(70)"); m_ndigivscycletime[i]->GetYaxis()->SetTitle("time [Orb#]");
      //      m_ndigivscycletime[i]->SetCanExtend(TH1::kYaxis);
    }

    // vs BX number

    sprintf(title,"%s %s mean multiplicity vs BX",slab.c_str(),m_hitname.c_str());
    sprintf(name,"n%sdigivsbx",slab.c_str());
    m_ndigivsbx[i] =subev.make<TProfile>(name,title,3564,-0.5,3563.5);
    m_ndigivsbx[i]->GetXaxis()->SetTitle("BX#"); m_ndigivsbx[i]->GetYaxis()->SetTitle("Mean Number of Hits");

    sprintf(title,"%s %s multiplicity vs BX",slab.c_str(),m_hitname.c_str());

    if(m_scalefact.size()>=1) {
      sprintf(name,"n%sdigivsbx2D",slab.c_str());
      m_ndigivsbx2D[i] =subev.make<TH2F>(name,title,3564,-0.5,3563.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[0]*m_nbins))*m_nbins);
      m_ndigivsbx2D[i]->GetXaxis()->SetTitle("BX#"); m_ndigivsbx2D[i]->GetYaxis()->SetTitle("Number of Hits");
    }
    if(m_scalefact.size()>=2) {
      sprintf(name,"n%sdigivsbx2Dzoom",slab.c_str());
      m_ndigivsbx2Dzoom[i] =subev.make<TH2F>(name,title,3564,-0.5,3563.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[1]*m_nbins))*m_nbins);
      m_ndigivsbx2Dzoom[i]->GetXaxis()->SetTitle("BX#"); m_ndigivsbx2Dzoom[i]->GetYaxis()->SetTitle("Number of Hits");
    }
    if(m_scalefact.size()>=3) {
      sprintf(name,"n%sdigivsbx2Dzoom2",slab.c_str());
      m_ndigivsbx2Dzoom2[i] =subev.make<TH2F>(name,title,3564,-0.5,3563.5,m_nbins,0,(1+m_binmax[ui]/(m_scalefact[2]*m_nbins))*m_nbins);
      m_ndigivsbx2Dzoom2[i]->GetXaxis()->SetTitle("BX#"); m_ndigivsbx2Dzoom2[i]->GetYaxis()->SetTitle("Number of Hits");
    }

  }

}

template <class T>
void DigiBXCorrHistogramMaker<T>::beginRun(const unsigned int nrun) {

  m_rhm.beginRun(nrun);

  for(std::map<int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {

    const int i = lab->first;
    if(m_runHisto) {
      if(m_ndigivscycletime[i]) {
	(*m_ndigivscycletime[i])->GetXaxis()->SetTitle("Event 1 BX mod(70)"); (*m_ndigivscycletime[i])->GetYaxis()->SetTitle("time [Orb#]");
	(*m_ndigivscycletime[i])->SetCanExtend(TH1::kAllAxes);
      }
    }
  }


}

template <class T>
void DigiBXCorrHistogramMaker<T>::fill(const T& he, const std::map<int,int>& ndigi, const edm::Handle<APVCyclePhaseCollection>& phase) {

  for(std::map<int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {

    if(m_labels.find(digi->first) != m_labels.end()) {
      const int i=digi->first; const unsigned int ui = i;

      int thephase = APVCyclePhaseCollection::invalid;
	if(m_phasepart.find(ui)!=m_phasepart.end() && m_phasepart[ui]!="None") {
	  if(!phase.failedToGet() && phase.isValid()) {
	    thephase = phase->getPhase(m_phasepart[ui]);
	  }
	}

	long long tbx = he.absoluteBX();
	if(thephase!=APVCyclePhaseCollection::nopartition &&
	   thephase!=APVCyclePhaseCollection::multiphase &&
	   thephase!=APVCyclePhaseCollection::invalid) {

	  tbx -= thephase;

	  if(m_nmeandigivscycle.find(i)!=m_nmeandigivscycle.end()) m_nmeandigivscycle[i]->Fill(tbx%70,digi->second);

	  if(m_ndigivscycle.find(i)!=m_ndigivscycle.end()) m_ndigivscycle[i]->Fill(tbx%70,digi->second);
	  if(m_ndigivscyclezoom.find(i)!=m_ndigivscyclezoom.end()) m_ndigivscyclezoom[i]->Fill(tbx%70,digi->second);
	  if(m_ndigivscyclezoom2.find(i)!=m_ndigivscyclezoom2.end()) m_ndigivscyclezoom2[i]->Fill(tbx%70,digi->second);

	}

	if(m_runHisto) {
	  if(m_ndigivscycletime.find(i)!=m_ndigivscycletime.end()) {
	    if(m_ndigivscycletime[i]!=0 && (*m_ndigivscycletime[i])!=0 ) (*m_ndigivscycletime[i])->Fill(tbx%70,(int)he._orbit,digi->second);
	  }
	}

	m_ndigivsbx[i]->Fill(he.bx()%3564,digi->second);
	if(m_ndigivsbx2D.find(i)!=m_ndigivsbx2D.end()) m_ndigivsbx2D[i]->Fill(he.bx()%3564,digi->second);
	if(m_ndigivsbx2Dzoom.find(i)!=m_ndigivsbx2Dzoom.end()) m_ndigivsbx2Dzoom[i]->Fill(he.bx()%3564,digi->second);
	if(m_ndigivsbx2Dzoom2.find(i)!=m_ndigivsbx2Dzoom2.end()) m_ndigivsbx2Dzoom2[i]->Fill(he.bx()%3564,digi->second);


	if(he.depth()>0) {

	  long long dbx = he.deltaBX();

	  m_ndigivsdbx[i]->Fill(dbx,digi->second);
	  m_ndigivsdbxzoom[i]->Fill(dbx,digi->second);
	  m_ndigivsdbxzoom2[i]->Fill(dbx,digi->second);

	  if(m_ndigivsdbx2D.find(i)!=m_ndigivsdbx2D.end()) m_ndigivsdbx2D[i]->Fill(dbx,digi->second);
	  if(m_ndigivsdbx2Dzoom.find(i)!=m_ndigivsdbx2Dzoom.end()) m_ndigivsdbx2Dzoom[i]->Fill(dbx,digi->second);
	  if(m_ndigivsdbx2Dzoom2.find(i)!=m_ndigivsdbx2Dzoom2.end()) m_ndigivsdbx2Dzoom2[i]->Fill(dbx,digi->second);

	  long long prevtbx = he.absoluteBX(1);
	  if(thephase!=APVCyclePhaseCollection::nopartition &&
	     thephase!=APVCyclePhaseCollection::multiphase &&
	     thephase!=APVCyclePhaseCollection::invalid) {

	    long long dbxincycle = he.deltaBXinCycle(thephase);
	    if(m_ndigivsdbxincycle2D.find(i)!=m_ndigivsdbxincycle2D.end()) m_ndigivsdbxincycle2D[i]->Fill(dbxincycle,digi->second);
	    if(m_ndigivsdbxincycle.find(i)!=m_ndigivsdbxincycle.end()) m_ndigivsdbxincycle[i]->Fill(dbxincycle,digi->second);

	    prevtbx -= thephase;
	    if(m_ndigivscycledbx.find(i)!=m_ndigivscycledbx.end()) m_ndigivscycledbx[i]->Fill(prevtbx%70,dbx,digi->second);
	    if(m_ndigivscycle2dbx.find(i)!=m_ndigivscycle2dbx.end()) m_ndigivscycle2dbx[i]->Fill(tbx%70,dbx,digi->second);

	  }

	  if(he.depth()>1) {

	    long long dbx2 = he.deltaBX(2);
	    m_ndigivsdbx3zoom[i]->Fill(dbx2,dbx,digi->second);

	    if(thephase!=APVCyclePhaseCollection::nopartition &&
	       thephase!=APVCyclePhaseCollection::multiphase &&
	       thephase!=APVCyclePhaseCollection::invalid) {
	      long long dbxincycle = he.deltaBXinCycle(thephase);
	      long long dbxincycle2 = he.deltaBXinCycle(2,thephase);
	      if(m_dbx3Histo) {
		if(m_ndigivsdbxincycle3.find(i)!=m_ndigivsdbxincycle3.end()) m_ndigivsdbxincycle3[i]->Fill(dbxincycle,dbxincycle2-dbxincycle,digi->second);
		if(m_dbx3Histo3D) {
		  if(m_ndigivsdbxincycle33D.find(i)!=m_ndigivsdbxincycle33D.end()) m_ndigivsdbxincycle33D[i]->Fill(dbxincycle,dbxincycle2-dbxincycle,digi->second);
		}
	      }
	    }
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

    if(m_labels.find(digi->first) != m_labels.end()) {
      const int i=digi->first;

      long long dbx = he2.deltaBX(he1);
      m_digicorr[i]->Fill(dbx,digi->second);

    }
    else {
      edm::LogWarning("MissingKey") << " Key " << digi->first << " is missing ";
    }

  }
}


#endif // DPGAnalysis_SiStripTools_DigiBXCorrHistogramMaker_H
