#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "CaloJetsCorExample.h"
#include "JetUtilMC.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

using namespace edm;
using namespace reco;
using namespace std;
namespace cms
{
CaloJetsCorExample::CaloJetsCorExample(edm::ParameterSet const& cfg)
{
   GenJetAlgorithm_      = cfg.getParameter<string>("GenJetAlgorithm");
   CaloJetAlgorithm_     = cfg.getParameter<string>("CaloJetAlgorithm"); 
   CorJetAlgorithm_      = cfg.getParameter<string>("CorJetAlgorithm");
   JetCorrectionService_ = cfg.getParameter<string>("JetCorrectionService");
   HistogramFile_        = cfg.getParameter<string>("HistogramFile");

   TString hname,htitle;
   m_file = new TFile(HistogramFile_.c_str(),"RECREATE");
   hname = "h_ptGen";
   htitle = "p_{T} of leading GenJets";
   m_HistNames[hname] = new TH1F(hname,htitle,500,0,1000); 
   hname = "h_ptCalo";
   htitle = "p_{T} of leading CaloJets";
   m_HistNames[hname] = new TH1F(hname,htitle,500,0,1000);
   hname = "h_ptCor";
   htitle = "p_{T} of leading CorJets";
   m_HistNames[hname] = new TH1F(hname,htitle,500,0,1000);
   hname = "h_ptCorOnTheFly";
   htitle = "p_{T} of leading CaloJets Corrected on the Fly";
   m_HistNames[hname] = new TH1F(hname,htitle,500,0,1000);
}

void CaloJetsCorExample::fillHist(const TString& histName, const Double_t& value)
{
  std::map<TString, TH1*>::iterator hid = m_HistNames.find(histName);
  if (hid==m_HistNames.end())
    std::cout<<"%fillHist -- Could not find histogram with name: "<<histName<<std::endl;
  else
    hid->second->Fill(value);  
}

void CaloJetsCorExample::analyze(edm::Event const& event, edm::EventSetup const& iSetup) 
{
  edm::Handle<GenJetCollection> genJets;
  edm::Handle<CaloJetCollection> CaloJets;
  edm::Handle<CaloJetCollection> CorJets;
  event.getByLabel(GenJetAlgorithm_, genJets);
  event.getByLabel(CaloJetAlgorithm_, CaloJets);
  event.getByLabel(CorJetAlgorithm_, CorJets);
  CaloJetCollection::const_iterator i_Calojet;
  CaloJetCollection::const_iterator i_Corjet;
  GenJetCollection::const_iterator i_genjet; 
  ///////////////////////  CaloJet Collection //////////////////////////
  int jetInd = 0;
  for(i_Calojet = CaloJets->begin(); i_Calojet != CaloJets->end(); i_Calojet++) 
    { 
      fillHist("h_ptCalo",i_Calojet->pt());
      jetInd++;
      if (jetInd==2) break;
      
    }
  ///////////////////////  CorJet Collection //////////////////////////
  jetInd = 0;
  for(i_Corjet = CorJets->begin(); i_Corjet != CorJets->end(); i_Corjet++) 
    { 
      fillHist("h_ptCor",i_Corjet->pt());
      jetInd++;
      if (jetInd==2) break;
    }
  /////////////////////// Gen Jet Collection //////////////////////////
  jetInd = 0;
  for(i_genjet = genJets->begin(); i_genjet != genJets->end(); i_genjet++) 
    {  
      fillHist("h_ptGen",i_genjet->pt());   
      jetInd++;
      if (jetInd==2) break;
    } 
  /////////////////////// Correction on the fly //////////////////////////
  const JetCorrector* corrector = JetCorrector::getJetCorrector (JetCorrectionService_,iSetup);
  double highestPt=0.0;
  double nextPt=0.0;
  for(i_Calojet = CaloJets->begin(); i_Calojet != CaloJets->end(); i_Calojet++) 
   {
     double scale = corrector->correction(i_Calojet->p4());
     double corPt = scale*i_Calojet->pt();
     if (corPt>highestPt)
       {
         nextPt = highestPt;
         highestPt = corPt;
       }
     else if (corPt>nextPt)
       nextPt = corPt;
   } 
  fillHist("h_ptCorOnTheFly",highestPt);
  fillHist("h_ptCorOnTheFly",nextPt);
}
void CaloJetsCorExample::endJob() 
{
  if (m_file !=0) 
    {
      m_file->cd();
      for (std::map<TString, TH1*>::iterator hid = m_HistNames.begin(); hid != m_HistNames.end(); hid++)
        hid->second->Write();
      delete m_file;
      m_file=0;      
    }
}
CaloJetsCorExample::CaloJetsCorExample() 
{
  m_file=0;
}
}
