// Implementation of template class: JetPlotsExample
// Description:  Example of simple EDAnalyzer for jets.
// Author: K. Kousouris
// Date:  25 - August - 2008
#include "RecoJets/JetAnalyzers/interface/JetPlotsExample.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TFile.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
JetPlotsExample<Jet>::JetPlotsExample(edm::ParameterSet const& cfg)
{
  JetAlgorithm  = cfg.getParameter<std::string> ("JetAlgorithm"); 
  HistoFileName = cfg.getParameter<std::string> ("HistoFileName");
  NJets         = cfg.getParameter<int> ("NJets");
  useJecLevels  = cfg.exists("jecLevels");
  if ( useJecLevels )
    jecLevels     = cfg.getParameter<std::string> ("jecLevels");
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetPlotsExample<Jet>::beginJob() 
{
  TString hname;
  m_file = new TFile(HistoFileName.c_str(),"RECREATE"); 
  /////////// Booking histograms //////////////////////////
  hname = "JetPt";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,0,1000);
  hname = "JetEta";
  m_HistNames1D[hname] = new TH1F(hname,hname,120,-6,6);
  hname = "JetPhi";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,-M_PI,M_PI);
  hname = "NumberOfJets";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,0,100);
  hname = "EtaEtaMoment";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,-0.1,0.1);
  hname = "EtaPhiMoment";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,-0.1,0.1);
  hname = "PhiPhiMoment";
  m_HistNames1D[hname] = new TH1F(hname,hname,100,-0.1,0.1);
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetPlotsExample<Jet>::analyze(edm::Event const& evt, edm::EventSetup const& iSetup) 
{
  /////////// Get the jet collection //////////////////////
  //Handle<JetCollection> jets;
   edm::Handle<edm::View<reco::Jet> > jets;
  evt.getByLabel(JetAlgorithm,jets);
  // typename JetCollection::const_iterator i_jet;
  int index = 0;
  TString hname; 
  /////////// Count the jets in the event /////////////////
  hname = "NumberOfJets";
  FillHist1D(hname,jets->size()); 
  /////////// Fill Histograms for the leading NJet jets ///
  edm::View<reco::Jet>::const_iterator i_jet, endpjets = jets->end(); 
   for (i_jet = jets->begin();  i_jet != endpjets && index < NJets;  ++i_jet) {

      double jecFactor = 1.0;
      if ( useJecLevels ) {
	edm::Ptr<reco::Jet> ptrToJet = jets->ptrAt( i_jet - jets->begin() );
	if ( ptrToJet.isNonnull() && ptrToJet.isAvailable() ) {
	  pat::Jet const * patJet = dynamic_cast<pat::Jet const *>( ptrToJet.get() );
          if ( patJet != 0 ) {
	    jecFactor = patJet->jecFactor( jecLevels );
	  }
        }
      }
      hname = "JetPt";
      FillHist1D(hname,(*i_jet).pt() * jecFactor );   
      hname = "JetEta";
      FillHist1D(hname,(*i_jet).eta());
      hname = "JetPhi";
      FillHist1D(hname,(*i_jet).phi());
      hname = "EtaEtaMoment";
      FillHist1D(hname,(*i_jet).etaetaMoment());
      hname = "EtaPhiMoment";
      FillHist1D(hname,(*i_jet).etaphiMoment());
      hname = "PhiPhiMoment";
      FillHist1D(hname,(*i_jet).phiphiMoment());
      index++;
    }
}
////////////////////////////////////////////////////////////////////////////////////////
template<class Jet>
void JetPlotsExample<Jet>::endJob() 
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
void JetPlotsExample<Jet>::FillHist1D(const TString& histName,const Double_t& value) 
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
typedef JetPlotsExample<CaloJet> CaloJetPlotsExample;
DEFINE_FWK_MODULE(CaloJetPlotsExample);
/////////// Cen Jet Instance ////////
typedef JetPlotsExample<GenJet> GenJetPlotsExample;
DEFINE_FWK_MODULE(GenJetPlotsExample);
/////////// PF Jet Instance ////////
typedef JetPlotsExample<PFJet> PFJetPlotsExample;
DEFINE_FWK_MODULE(PFJetPlotsExample);
/////////// JPT Jet Instance ////////
typedef JetPlotsExample<JPTJet> JPTJetPlotsExample;
DEFINE_FWK_MODULE(JPTJetPlotsExample);
