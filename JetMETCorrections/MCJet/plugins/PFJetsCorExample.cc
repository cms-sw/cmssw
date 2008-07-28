#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "PFJetsCorExample.h"
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
PFJetsCorExample::PFJetsCorExample(edm::ParameterSet const& cfg)
{
   GenJetAlgorithm_      = cfg.getParameter<string>("GenJetAlgorithm");
   PFJetAlgorithm_       = cfg.getParameter<string>("PFJetAlgorithm"); 
   JetCorrectionService_ = cfg.getParameter<string>("JetCorrectionService");
   HistogramFile_        = cfg.getParameter<string>("HistogramFile");

   TString hname,htitle;
   m_file = new TFile(HistogramFile_.c_str(),"RECREATE");
   hname = "h_ptGen";
   htitle = "p_{T} of leading GenJets";
   m_HistNames[hname] = new TH1F(hname,htitle,500,0,1000); 
   hname = "h_ptPF";
   htitle = "p_{T} of leading PFJets";
   m_HistNames[hname] = new TH1F(hname,htitle,500,0,1000);
   hname = "h_ptCorPF";
   htitle = "p_{T} of leading PFJets Corrected on the Fly";
   m_HistNames[hname] = new TH1F(hname,htitle,500,0,1000);
}

void PFJetsCorExample::fillHist(const TString& histName, const Double_t& value)
{
  std::map<TString, TH1*>::iterator hid = m_HistNames.find(histName);
  if (hid==m_HistNames.end())
    std::cout<<"%fillHist -- Could not find histogram with name: "<<histName<<std::endl;
  else
    hid->second->Fill(value);  
}

void PFJetsCorExample::analyze(edm::Event const& event, edm::EventSetup const& iSetup) 
{
  edm::Handle<GenJetCollection> genJets;
  edm::Handle<PFJetCollection> pfJets;
  event.getByLabel(GenJetAlgorithm_, genJets);
  event.getByLabel(PFJetAlgorithm_, pfJets);
  PFJetCollection::const_iterator i_pfjet;
  GenJetCollection::const_iterator i_genjet; 
  ///////////////////////  PFJet Collection //////////////////////////
  int jetInd = 0;
  for(i_pfjet = pfJets->begin(); i_pfjet != pfJets->end(); i_pfjet++) 
    { 
      fillHist("h_ptPF",i_pfjet->pt());
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
  for(i_pfjet = pfJets->begin(); i_pfjet != pfJets->end(); i_pfjet++) 
   {
     double scale = corrector->correction(i_pfjet->p4());
     double corPt = scale*i_pfjet->pt();
     if (corPt>highestPt)
       {
         nextPt = highestPt;
         highestPt = corPt;
       }
     else if (corPt>nextPt)
       nextPt = corPt;
   } 
  fillHist("h_ptCorPF",highestPt);
  fillHist("h_ptCorPF",nextPt);
}
void PFJetsCorExample::endJob() 
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
PFJetsCorExample::PFJetsCorExample() 
{
  m_file=0;
}
}
