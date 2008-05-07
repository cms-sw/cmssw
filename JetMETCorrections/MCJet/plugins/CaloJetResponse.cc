#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "CaloJetResponse.h"
#include "JetUtilMC.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

namespace cms
{
typedef CaloJetCollection::const_iterator CalJetIter;
typedef GenJetCollection::const_iterator GenJetIter;

CaloJetResponse::CaloJetResponse(edm::ParameterSet const& cfg) 
{
  MatchRadius_   = 0.2;
  RecJetPtMin_   = 5.;
  NJetMax_       = 2;
  genjets_       = cfg.getParameter<std::string> ("genjets");
  recjets_       = cfg.getParameter<std::string> ("recjets");
  NJetMax_       = cfg.getParameter<int> ("NJetMax");
  MatchRadius_   = cfg.getParameter<double> ("MatchRadius");
  RecJetPtMin_   = cfg.getParameter<double> ("RecJetPtMin");
  GenJetPtBins_  = cfg.getParameter< std::vector<double> >("GenJetPtBins");
  EtaBoundaries_ = cfg.getParameter< std::vector<double> >("EtaBoundaries");
  histogramFile_ = cfg.getParameter<std::string> ("HistogramFile"),
  hist_file_     = new TFile(histogramFile_.c_str(),"RECREATE");
  NGenPtBins_    = GenJetPtBins_.size()-1;
  NEtaBins_      = EtaBoundaries_.size()-1;

  bookHistograms();
}
void CaloJetResponse::endJob() 
{
  done();
}

void CaloJetResponse::analyze(edm::Event const& event, edm::EventSetup const& iSetup) 
{
  edm::Handle<GenJetCollection> genjets;
  edm::Handle<CaloJetCollection> recjets;
  edm::Handle<GenMETCollection> genmet;
  edm::Handle<CaloMETCollection> recmet;
  event.getByLabel (genjets_,genjets);
  event.getByLabel (recjets_,recjets);
  analyze(*genjets,*recjets);
}

CaloJetResponse::CaloJetResponse() 
{
  hist_file_=0; // set to null
}

////////////////////////////////////////////////////////////////////////////////////////
void CaloJetResponse::fillHist1D(const TString& histName,const Double_t& value) 
{
  std::map<TString, TH1*>::iterator hid=m_HistNames1D.find(histName);
  if (hid==m_HistNames1D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(value);
}
/////////////////////////////////////////////////////////////////////////////////////////
void CaloJetResponse::fillHist2D(const TString& histName, const Double_t& x,const Double_t& y) 
{
  std::map<TString, TH2*>::iterator hid=m_HistNames2D.find(histName);
  if (hid==m_HistNames2D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else 
    hid->second->Fill(x,y);
}
/////////////////////////////////////////////////////////////////////////////////////////
void CaloJetResponse::analyze(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets)
{
  CalculateJetResponse(genjets,recjets);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void CaloJetResponse::done() 
{
  if (hist_file_!=0) 
    {
      hist_file_->cd();
      for (std::map<TString, TH1*>::iterator hid = m_HistNames1D.begin(); hid != m_HistNames1D.end(); hid++)
        hid->second->Write(); 
      delete hist_file_;
      hist_file_=0;      
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
void CaloJetResponse::bookHistograms() 
{
  TString hname;
  int ip,NRespBins(600);
  double RespLow=-1000,RespHigh=200;
  double PtLow = 0.;
  double PtMax = 6000.;
  double DPt = 5;
  int N = (int)((PtMax-PtLow)/DPt);
  for(ip=0;ip<NGenPtBins_;ip++)
    { 
      std::ostringstream oip;
      oip << GenJetPtBins_[ip];
      ////////////// Barrel //////////////////////////////
      hname="JetGenPt_GenPt"+oip.str()+"_Barrel";
      m_HistNames1D[hname] = new TH1F(hname,hname,N,PtLow,PtMax);
      hname="JetResponseGenPt"+oip.str()+"_Barrel";
      m_HistNames1D[hname] = new TH1F(hname,hname,NRespBins,RespLow,RespHigh);
      ///////////// Eta ///////////////////////////////
      for(int it=0; it<NEtaBins_; it++)
        {
          std::ostringstream oit; 
          oit << it;
          hname="JetGenPt_GenPt"+oip.str()+"_EtaBin"+oit.str();
          m_HistNames1D[hname] = new TH1F(hname,hname,N,PtLow,PtMax);
          hname="JetResponseGenPt"+oip.str()+"_EtaBin"+oit.str();
          m_HistNames1D[hname] = new TH1F(hname,hname,NRespBins,RespLow,RespHigh);
        }
    }// end of GenPt histograms 
}
//////////////////////////////////////////////////////////////////
int  CaloJetResponse::GetBin(double x, std::vector<double> a)
{
  int i,N;
  N = a.size()-1; 
  for(i=0;i<N;i++)
    {
      if (a[i] < x && x < a[i+1])
        return i;
    }
  return 0;
}
///////////////////////////////////////////////////////////////////////////
void CaloJetResponse::CalculateJetResponse(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets)
{
  if (genjets.size()==0) 
    return;
  if (recjets.size()==0) 
    return;
  const double GenJetEtaMax=5.5; 
  double GenJetPtMin=GenJetPtBins_[0];
  double GenJetPt,CaloJetPt,Response,GenJetEta,CaloJetEta;
  TString hname;
  int njet(0);
  int ipt_gen,ieta;
  float rr;
  
  for(GenJetIter i=genjets.begin();i!=genjets.end(); i++) 
    {
      njet++;
      if (njet>NJetMax_) return; 
      GenJetPt = i->pt();
      GenJetEta = i->eta();
      if (GenJetPt>GenJetPtMin) 
        {
          if (fabs(GenJetEta)<GenJetEtaMax)
            {
              float rmin(99);
	      CalJetIter caljet;
	      for(CalJetIter j=recjets.begin();j!=recjets.end();j++)
                {
	          rr=radius(i,j);
	          if (rr<rmin)
                    {
                      rmin=rr;
                      caljet=j;
                    }
	        }
              CaloJetPt = caljet->pt();
              CaloJetEta = caljet->eta();
              ipt_gen  = GetBin(GenJetPt,GenJetPtBins_);
	      ieta     = GetBin(CaloJetEta,EtaBoundaries_);
              if (CaloJetPt<RecJetPtMin_) continue;
              Response = CaloJetPt - GenJetPt; 
              std::ostringstream oipt_gen;
              std::ostringstream oieta; 
	      oipt_gen  << GenJetPtBins_[ipt_gen];
	      oieta << ieta;
	      if (rmin<MatchRadius_)
                {
	          hname="JetResponseGenPt"+oipt_gen.str()+"_EtaBin"+oieta.str();
	          fillHist1D(hname,Response);
                  hname="JetGenPt_GenPt"+oipt_gen.str()+"_EtaBin"+oieta.str();
                  fillHist1D(hname,GenJetPt);
                  if (fabs(CaloJetEta)<1.3)
                    {
                      hname="JetGenPt_GenPt"+oipt_gen.str()+"_Barrel";
                      fillHist1D(hname,GenJetPt);
                      hname="JetResponseGenPt"+oipt_gen.str()+"_Barrel";
                      fillHist1D(hname,Response);
                    }  
	        }
            }
        }
    }
}
}
