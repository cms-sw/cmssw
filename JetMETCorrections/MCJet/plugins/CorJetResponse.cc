#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "CorJetResponse.h"
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

CorJetResponse::CorJetResponse(edm::ParameterSet const& cfg) 
{
  MatchRadius_      = 0.2;
  RecJetPtMin_      = 5.;
  NJetMax_          = 2;
  genjets_          = cfg.getParameter<std::string> ("genjets");
  recjets_          = cfg.getParameter<std::string> ("recjets");
  NJetMax_          = cfg.getParameter<int> ("NJetMax");
  MatchRadius_      = cfg.getParameter<double> ("MatchRadius");
  RecJetPtMin_      = cfg.getParameter<double> ("RecJetPtMin");
  JetPtBins_        = cfg.getParameter< std::vector<double> >("JetPtBins");
  EtaBoundaries_    = cfg.getParameter< std::vector<double> >("EtaBoundaries");
  histogramFile_    = cfg.getParameter<std::string>("HistogramFile");
  hist_file_        = new TFile(histogramFile_.c_str(),"RECREATE");
  NPtBins_          = JetPtBins_.size()-1;
  NEtaBins_         = EtaBoundaries_.size()-1;

  BookHistograms();
}
//////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::endJob() 
{
  done();
}
//////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::analyze(edm::Event const& event, edm::EventSetup const& iSetup) 
{
  edm::Handle<GenJetCollection> genjets;
  edm::Handle<CaloJetCollection> recjets;
  edm::Handle<GenMETCollection> genmet;
  edm::Handle<CaloMETCollection> recmet;
  event.getByLabel (genjets_,genjets);
  event.getByLabel (recjets_,recjets);
  analyze(*genjets,*recjets);
}
//////////////////////////////////////////////////////////////////////////////////////////
CorJetResponse::CorJetResponse() 
{
  hist_file_=0;
}
////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::FillHist1D(const TString& histName,const Double_t& value) 
{
  std::map<TString, TH1*>::iterator hid=m_HistNames1D.find(histName);
  if (hid==m_HistNames1D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(value);
}
/////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::FillHist2D(const TString& histName, const Double_t& x,const Double_t& y) 
{
  std::map<TString, TH2*>::iterator hid=m_HistNames2D.find(histName);
  if (hid==m_HistNames2D.end()){
  }
  else
    hid->second->Fill(x,y);
}
/////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::BookHistograms() 
{
  BookJetResponse();
  BookJetPt();
}
//////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::analyze(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets)
{
  CalculateJetResponse(genjets,recjets);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::done() 
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
//////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::BookJetResponse() 
{
  TString hname;
  int ip,NRespBins(200);
  double RespLow=0,RespHigh=2;
  for(ip=0;ip<NPtBins_;ip++)
    { 
      std::ostringstream oip;
      oip << JetPtBins_[ip];
      ///////////// EtaBins ///////////////////////////////
      for(int ieta=0; ieta<NEtaBins_; ieta++)
        {
          std::ostringstream oieta; 
          oieta << ieta;
          hname="JetResponsePt_GenPt"+oip.str()+"_EtaBin"+oieta.str();
          m_HistNames1D[hname] = new TH1F(hname,hname,NRespBins,RespLow,RespHigh);
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::BookJetPt() 
{
  TString hname;
  int ip;
  double PtLow = 0.;
  double PtMax = 6000.;
  double DPt = 5;
  int N = (int)((PtMax-PtLow)/DPt);
  for(ip=0;ip<NPtBins_;ip++)
    { 
      std::ostringstream oip;
      oip << JetPtBins_[ip];
      ///////////// EtaBins ///////////////////////////////
      for(int ieta=0; ieta<NEtaBins_; ieta++)
        {
          std::ostringstream oieta; 
          oieta << ieta; 
          hname="JetGenPt_GenPt"+oip.str()+"_EtaBin"+oieta.str();
          m_HistNames1D[hname] = new TH1F(hname,hname,N,PtLow,PtMax);
          hname="JetCaloPt_CaloPt"+oip.str()+"_EtaBin"+oieta.str();
          m_HistNames1D[hname] = new TH1F(hname,hname,N,PtLow,PtMax);
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
int CorJetResponse::GetBin(double x, std::vector<double> boundaries)
{
  int i;
  int n = boundaries.size()-1;
  if (x<boundaries[0])
    return 0;
  if (x>=boundaries[n])
    return n-1;
  for(i=0;i<n;i++)
   {
     if (x>=boundaries[i] && x<boundaries[i+1])
       return i;
   }
  return 0; 
}
//////////////////////////////////////////////////////////////////////////////////////////
void CorJetResponse::CalculateJetResponse(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets)
{
  if (genjets.size()==0) 
    return;
  if (recjets.size()==0) 
    return;
  const double GenJetEtaMax=5.5; 
  double GenJetPtMin = JetPtBins_[0];
  TString hname;
  int njet(0);
  for(GenJetIter i=genjets.begin();i!=genjets.end();i++) 
   {
     njet++;
     if (njet>NJetMax_) 
       return; 
     Double_t GenJetPt  = i->pt();
     Double_t GenJetEta = i->eta();
     if (GenJetPt>GenJetPtMin) 
       {
         if (fabs(GenJetEta)<GenJetEtaMax)
           {
	     float rmin(99);
	     CalJetIter caljet;
	     for (CalJetIter j=recjets.begin();j!=recjets.end();j++)
              {
	        float rr=radius(i,j);
	        if (rr<rmin)
                  {
                    rmin=rr;
                    caljet=j;
                  }
	      }
             double CaloJetEta = caljet->eta();
	     double CaloJetPt  = caljet->pt();
             double ResponsePt;
             if (CaloJetPt<RecJetPtMin_) 
               continue;
             ResponsePt   = CaloJetPt/GenJetPt;
             int ipt_gen  = GetBin(GenJetPt,JetPtBins_);
             int ipt_calo = GetBin(CaloJetPt,JetPtBins_);
	     int ieta     = GetBin(CaloJetEta,EtaBoundaries_);
	     std::ostringstream oipt_gen; 
             oipt_gen << JetPtBins_[ipt_gen];
             std::ostringstream oipt_calo; 
             oipt_calo << JetPtBins_[ipt_calo];
	     std::ostringstream oieta; 
             oieta << ieta;
	     if (rmin<MatchRadius_)
               {
                 hname="JetResponsePt_GenPt"+oipt_gen.str()+"_EtaBin"+oieta.str();
	         FillHist1D(hname,ResponsePt);
                 hname="JetGenPt_GenPt"+oipt_gen.str()+"_EtaBin"+oieta.str();
                 FillHist1D(hname,GenJetPt);
                 hname="JetCaloPt_CaloPt"+oipt_gen.str()+"_EtaBin"+oieta.str();
                 FillHist1D(hname,CaloJetPt);      
               }	
           }//if (fabs(GenJetEta)<GenJetEtaMax)
       }//if (GenJetPt>GenJetPtMin)
   }//for(GenJetIter i=genjets.begin();i!=genjets.end(); i++)
}
}
