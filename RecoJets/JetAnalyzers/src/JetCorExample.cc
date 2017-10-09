// Implementation of template class: JetCorExample
// Description:  Example of simple EDAnalyzer correcting jets "on the fly".
// Author: K. Kousouris
// Date:  25 - August - 2008
#include "RecoJets/JetAnalyzers/interface/JetCorExample.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include <TFile.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
JetCorExample<Jet>::JetCorExample(edm::ParameterSet const& cfg)
{
  JetAlgorithm         = cfg.getParameter<std::string> ("JetAlgorithm"); 
  HistoFileName        = cfg.getParameter<std::string> ("HistoFileName");
  JetCorrectionService = cfg.getParameter<std::string> ("JetCorrectionService");
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetCorExample<Jet>::beginJob() 
{
  TString hname;
  m_file = new TFile(HistoFileName.c_str(),"RECREATE"); 
  /////////// Booking histograms //////////////////////////
  hname = "JetPt";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,0,1000);
  hname = "CorJetPt";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,0,1000);
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetCorExample<Jet>::analyze(edm::Event const& evt, edm::EventSetup const& iSetup) 
{
  /////////// Get the jet collection //////////////////////
  Handle<JetCollection> jets;
  evt.getByLabel(JetAlgorithm,jets);
  typename JetCollection::const_iterator i_jet;
  TString hname; 
  const JetCorrector* corrector = JetCorrector::getJetCorrector (JetCorrectionService,iSetup);
  double scale;
  /////////// Loop over all jets and apply correction /////
  for(i_jet = jets->begin(); i_jet != jets->end(); i_jet++) 
    {
      scale = corrector->correction(i_jet->p4()); 
      hname = "JetPt";
      FillHist1D(hname,i_jet->pt());   
      hname = "CorJetPt";
      FillHist1D(hname,scale*i_jet->pt()); 
    }
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetCorExample<Jet>::endJob() 
{
  /////////// Write Histograms in output ROOT file ////////
  if (m_file !=0) 
    {
      m_file->cd();
      for (std::map<TString, TH1*>::iterator hid = m_HistNames1D.begin(); hid != m_HistNames1D.end(); hid++)
        hid->second->Write();
      delete m_file;
      m_file = 0;      
    }
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetCorExample<Jet>::FillHist1D(const TString& histName,const Double_t& value) 
{
  std::map<TString, TH1*>::iterator hid=m_HistNames1D.find(histName);
  if (hid==m_HistNames1D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(value);
}
/////////// Register Modules ////////
#include "FWCore/Framework/interface/MakerMacros.h"
/////////// Calo Jet Instance ////////
typedef JetCorExample<CaloJet> CaloJetCorExample;
DEFINE_FWK_MODULE(CaloJetCorExample);
/////////// PF Jet Instance ////////
typedef JetCorExample<PFJet> PFJetCorExample;
DEFINE_FWK_MODULE(PFJetCorExample);
