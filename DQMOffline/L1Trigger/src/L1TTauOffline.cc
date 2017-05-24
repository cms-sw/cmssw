/** 
 *  @file     L1TTauOffline.cc
 *  @authors  Olivier Davignon (University of Bristol), Cécile Caillol (University of Wisconsin - Madison)
 *  @date     24/05/2017  
 *  @version  1.0 
 *  
 */

#include "DQMOffline/L1Trigger/interface/L1TTauOffline.h"
#include "DQMOffline/L1Trigger/interface/L1TFillWithinLimits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "TLorentzVector.h"

#include "DataFormats/L1Trigger/interface/Muon.h"							
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/TauReco/interface/CaloTau.h"

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "TMath.h"
#include "TLorentzVector.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <algorithm>

using namespace reco;
using namespace trigger;
using namespace edm;
using namespace std;

TauL1TPair::TauL1TPair(const TauL1TPair& tauL1tPair) {

  m_tau    = tauL1tPair.m_tau;			
  m_regTau  = tauL1tPair.m_regTau;		

  m_eta     = tauL1tPair.m_eta;
  m_phi_bar = tauL1tPair.m_phi_bar;
  m_phi_end = tauL1tPair.m_phi_end;

}

double TauL1TPair::dR() {
  
  float dEta = m_regTau ? (m_regTau->eta() - eta()) : 999.;					
  float dPhi = m_regTau ? TMath::ACos(TMath::Cos(m_regTau->phi() - phi())) : 999.; 		
  float dr = sqrt(dEta*dEta + dPhi*dPhi);
  return dr;
}


//
// -------------------------------------- Constructor --------------------------------------------
//
L1TTauOffline::L1TTauOffline(const edm::ParameterSet& ps) :
        theTauCollection_(consumes < reco::PFTauCollection > (ps.getUntrackedParameter < edm::InputTag > ("tauInputTag"))),
	AntiMuInputTag_(consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("antiMuInputTag"))),
	AntiEleInputTag_(consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("antiEleInputTag"))),
	DecayModeFindingInputTag_(consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("decayModeFindingInputTag"))),
	comb3TInputTag_(consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("comb3TInputTag"))),
	MuonInputTag_(consumes<reco::MuonCollection>(ps.getUntrackedParameter<edm::InputTag>("muonInputTag"))),
	MetInputTag_(consumes<reco::PFMETCollection>(ps.getUntrackedParameter<edm::InputTag>("metInputTag"))),
	VtxInputTag_(consumes<reco::VertexCollection>(ps.getUntrackedParameter<edm::InputTag>("vtxInputTag"))),
	BsInputTag_(consumes<reco::BeamSpot>(ps.getUntrackedParameter<edm::InputTag>("bsInputTag"))),
        triggerEvent_(consumes < trigger::TriggerEvent > (ps.getUntrackedParameter < edm::InputTag > ("trigInputTag"))),
	trigProcess_(ps.getUntrackedParameter<string>("trigProcess")),
        triggerResults_(consumes < edm::TriggerResults > (ps.getUntrackedParameter < edm::InputTag > ("trigProcess_token"))),
        triggerPath_(ps.getUntrackedParameter < vector<std::string> > ("triggerNames")),
        histFolder_(ps.getParameter < std::string > ("histFolder")),
        efficiencyFolder_(histFolder_ + "/efficiency_raw"),
        stage2CaloLayer2TauToken_(consumes < l1t::TauBxCollection > (ps.getUntrackedParameter < edm::InputTag > ("l1tInputTag"))),
        tauEfficiencyThresholds_(ps.getParameter < std::vector<double> > ("tauEfficiencyThresholds")),
        tauEfficiencyBins_(ps.getParameter < std::vector<double> > ("tauEfficiencyBins"))
{
  edm::LogInfo("L1TTauOffline") << "Constructor " << "L1TTauOffline::L1TTauOffline " << std::endl;
}

//
// -- Destructor
//
L1TTauOffline::~L1TTauOffline()
{
  edm::LogInfo("L1TTauOffline") << "Destructor L1TTauOffline::~L1TTauOffline " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void L1TTauOffline::dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup)
// void L1TTauOffline::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  bool changed = true;
  m_hltConfig.init(run,iSetup,trigProcess_,changed);

  edm::LogInfo("L1TTauOffline") << "L1TTauOffline::beginRun" << std::endl;
}

//
// -------------------------------------- bookHistos --------------------------------------------
//
void L1TTauOffline::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("L1TTauOffline") << "L1TTauOffline::bookHistograms" << std::endl;

  //book at beginRun
  bookTauHistos(ibooker);
 
  vector<string>::const_iterator trigNamesIt  = triggerPath_.begin();
  vector<string>::const_iterator trigNamesEnd = triggerPath_.end();

  for (; trigNamesIt!=trigNamesEnd; ++trigNamesIt) { 
    
    TString tNameTmp = TString(*trigNamesIt); 
    TRegexp tNamePattern = TRegexp(tNameTmp,true);
    int tIndex = -1;
    
    for (unsigned ipath = 0; ipath < m_hltConfig.size(); ++ipath) {
      
      TString tmpName = TString(m_hltConfig.triggerName(ipath));
      if (tmpName.Contains(tNamePattern)) {
	tIndex = int(ipath);
	m_trigIndices.push_back(tIndex);
      }
    }
        
  }
   
}
//
// -------------------------------------- beginLuminosityBlock --------------------------------------------
//
void L1TTauOffline::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context)
{
  edm::LogInfo("L1TTauOffline") << "L1TTauOffline::beginLuminosityBlock" << std::endl;
}

//
// -------------------------------------- Analyze --------------------------------------------
//
void L1TTauOffline::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{

  m_MaxTauEta   = 2.1;
  m_MaxL1tTauDR = 0.5;
  m_MaxHltTauDR = 0.5;

  edm::Handle<reco::PFTauCollection> taus;
  e.getByToken(theTauCollection_, taus);

  edm::Handle<reco::MuonCollection> muons;
  e.getByToken(MuonInputTag_, muons);

  edm::Handle<reco::BeamSpot> beamSpot;
  e.getByToken(BsInputTag_, beamSpot);

  edm::Handle<reco::VertexCollection> vertex;
  e.getByToken(VtxInputTag_, vertex);
  edm::Handle<l1t::TauBxCollection> l1tCands;				

  e.getByToken(stage2CaloLayer2TauToken_,l1tCands);
  
  edm::Handle<edm::TriggerResults> trigResults;
  e.getByToken(triggerResults_,trigResults);
  
  edm::Handle<trigger::TriggerEvent> trigEvent;
  e.getByToken(triggerEvent_,trigEvent);

  edm::Handle<reco::PFMETCollection> mets;
  e.getByToken(MetInputTag_, mets);

  eSetup.get<IdealMagneticFieldRecord>().get(m_BField);									
  const reco::Vertex primaryVertex = getPrimaryVertex(vertex,beamSpot);

  getTightMuons(muons,mets,primaryVertex,trigEvent);
  getProbeTaus(e,taus,muons,primaryVertex);
  getTauL1tPairs(l1tCands);

  reco::PFTauCollection::const_iterator tauIt  = taus->begin();
  reco::PFTauCollection::const_iterator tauEnd = taus->end();

  for(; tauIt!=tauEnd; ++tauIt) {

    // float eta = tauIt->eta();
    // float phi = tauIt->phi();
    // float pt  = tauIt->pt();


    // m_EfficiencyHistos[0]["RecoEtaNEW_Histo"]->Fill(eta);	
    // m_EfficiencyHistos[0]["RecoPhiNEW_Histo"]->Fill(phi);	
    // m_EfficiencyHistos[0]["RecoPtNEW_Histo"]->Fill(pt);	
    

  }
  vector<l1t::Tau> l1tContainer;
  
  for (auto tau = l1tCands->begin(0); tau != l1tCands->end(0); ++tau) {
     l1tContainer.push_back(*tau);
  }

  vector<l1t::Tau>::const_iterator l1tIt = l1tContainer.begin();;				
  vector<l1t::Tau>::const_iterator l1tEnd = l1tContainer.end();				
    
  for(; l1tIt!=l1tEnd; ++l1tIt) {

    // float eta = l1tIt->eta();
    // float phi = l1tIt->phi();
    // float pt  = l1tIt->pt();

    // m_EfficiencyHistos[0]["L1TEtaNEW_Histo"]->Fill(eta);	
    // m_EfficiencyHistos[0]["L1TPhiNEW_Histo"]->Fill(phi);	
    // m_EfficiencyHistos[0]["L1TPtNEW_Histo"]->Fill(pt);
    

  }

  vector<TauL1TPair>::const_iterator tauL1tPairsIt  = m_TauL1tPairs.begin();
  vector<TauL1TPair>::const_iterator tauL1tPairsEnd = m_TauL1tPairs.end(); 
     
  for(; tauL1tPairsIt!=tauL1tPairsEnd; ++tauL1tPairsIt) {

    float eta = tauL1tPairsIt->eta();
    float phi = tauL1tPairsIt->phi();
    float pt  = tauL1tPairsIt->pt();

    // unmatched gmt cands have l1tPt = -1.	
    float l1tPt  = tauL1tPairsIt->l1tPt();
    // cout<<pt<<" "<<l1tPt<<endl;

    vector<int>::const_iterator l1tPtCutsIt  = m_L1tPtCuts.begin();
    vector<int>::const_iterator l1tPtCutsEnd = m_L1tPtCuts.end();

    int counter = 0;

    for (auto threshold : tauEfficiencyThresholds_) 
      {
	std::string str_threshold = std::to_string(int(threshold));
      
	int l1tPtCut = 0;
	std::istringstream ss(str_threshold);
	ss >> l1tPtCut;
	bool l1tAboveCut = (l1tPt >= l1tPtCut);

	stringstream ptCutToTag; ptCutToTag << l1tPtCut;
	string ptTag = ptCutToTag.str();

	if (fabs(eta) < m_MaxTauEta) {

	  if(counter==0)
	    {
	      if(fabs(eta)<1.5) h_L1TauETvsTauET_EB_->Fill(pt,l1tPt);
	      else h_L1TauETvsTauET_EE_->Fill(pt,l1tPt);
	      h_L1TauETvsTauET_EB_EE_->Fill(pt,l1tPt);
	      
	      if(fabs(eta)<1.5) h_L1TauPhivsTauPhi_EB_->Fill(phi,tauL1tPairsIt->l1tPhi());
	      else h_L1TauPhivsTauPhi_EE_->Fill(phi,tauL1tPairsIt->l1tPhi());
	      h_L1TauPhivsTauPhi_EB_EE_->Fill(phi,tauL1tPairsIt->l1tPhi());
	      
	      h_L1TauEtavsTauEta_->Fill(phi,tauL1tPairsIt->l1tEta());

	      if(fabs(eta)<1.5) h_resolutionTauET_EB_->Fill((l1tPt-pt)/pt);
	      else h_L1TauETvsTauET_EE_->Fill((l1tPt-pt)/pt);
	      h_resolutionTauET_EB_EE_->Fill((l1tPt-pt)/pt);

	      if(fabs(eta)<1.5) h_resolutionTauPhi_EB_->Fill(tauL1tPairsIt->l1tPhi()-phi);
	      else h_L1TauPhivsTauPhi_EE_->Fill(tauL1tPairsIt->l1tPhi()-phi);
	      h_resolutionTauPhi_EB_EE_->Fill(tauL1tPairsIt->l1tPhi()-phi);
	      
	      h_resolutionTauEta_->Fill(tauL1tPairsIt->l1tEta()-eta);

	      counter++;
	    }

	  if(fabs(eta)<1.5) h_efficiencyNonIsoTauET_EB_total_[threshold]->Fill(pt);
	  else h_efficiencyNonIsoTauET_EE_total_[threshold]->Fill(pt);
	  h_efficiencyNonIsoTauET_EB_EE_total_[threshold]->Fill(pt);

	  if(fabs(eta)<1.5) h_efficiencyIsoTauET_EB_total_[threshold]->Fill(pt);
	  else h_efficiencyIsoTauET_EE_total_[threshold]->Fill(pt);
	  h_efficiencyIsoTauET_EB_EE_total_[threshold]->Fill(pt);	  

	  if(l1tAboveCut)
	    {

	      if(fabs(eta)<1.5) h_efficiencyNonIsoTauET_EB_pass_[threshold]->Fill(pt);
	      else h_efficiencyNonIsoTauET_EE_pass_[threshold]->Fill(pt);
	      h_efficiencyNonIsoTauET_EB_EE_pass_[threshold]->Fill(pt);

	      if(tauL1tPairsIt->l1tIso()>0.5)
		{
		  if(fabs(eta)<1.5) h_efficiencyIsoTauET_EB_pass_[threshold]->Fill(pt);
		  else h_efficiencyIsoTauET_EE_pass_[threshold]->Fill(pt);
		  h_efficiencyIsoTauET_EB_EE_pass_[threshold]->Fill(pt);
		}

	    }	
	}
      }
  }//loop over tau-L1 pairs
 
}

//
// -------------------------------------- endLuminosityBlock --------------------------------------------
//
void L1TTauOffline::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TTauOffline") << "L1TTauOffline::endLuminosityBlock" << std::endl;
}

//
// -------------------------------------- endRun --------------------------------------------
//
void L1TTauOffline::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TTauOffline") << "L1TTauOffline::endRun" << std::endl;
}

//
// -------------------------------------- book histograms --------------------------------------------
//
void L1TTauOffline::bookTauHistos(DQMStore::IBooker & ibooker)
{
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_.c_str());
  h_nVertex_ = ibooker.book1D("nVertex", "Number of event vertices in collection", 40, -0.5, 39.5);
  h_tagAndProbeMass_ = ibooker.book1D("tagAndProbeMass", "Invariant mass of tag & probe pair", 100, 40, 140);

  h_L1TauETvsTauET_EB_ = ibooker.book2D("L1TauETvsTauET_EB",
      "L1 Tau E_{T} vs PFTau E_{T} (EB); PFTau E_{T} (GeV); L1 Tau E_{T} (GeV)", 300, 0, 300, 300,
      0, 300);
  h_L1TauETvsTauET_EE_ = ibooker.book2D("L1TauETvsTauET_EE",
      "L1 Tau E_{T} vs PFTau E_{T} (EE); PFTau E_{T} (GeV); L1 Tau E_{T} (GeV)", 300, 0, 300, 300,
      0, 300);
  h_L1TauETvsTauET_EB_EE_ = ibooker.book2D("L1TauETvsTauET_EB_EE",
      "L1 Tau E_{T} vs PFTau E_{T} (EB+EE); PFTau E_{T} (GeV); L1 Tau E_{T} (GeV)", 300, 0, 300,
      300, 0, 300);

  h_L1TauPhivsTauPhi_EB_ = ibooker.book2D("L1TauPhivsTauPhi_EB",
      "#phi_{tau}^{L1} vs #phi_{tau}^{offline} (EB); #phi_{tau}^{offline}; #phi_{tau}^{L1}", 100,
      -4, 4, 100, -4, 4);
  h_L1TauPhivsTauPhi_EE_ = ibooker.book2D("L1TauPhivsTauPhi_EE",
      "#phi_{tau}^{L1} vs #phi_{tau}^{offline} (EE); #phi_{tau}^{offline}; #phi_{tau}^{L1}", 100,
      -4, 4, 100, -4, 4);
  h_L1TauPhivsTauPhi_EB_EE_ = ibooker.book2D("L1TauPhivsTauPhi_EB_EE",
      "#phi_{tau}^{L1} vs #phi_{tau}^{offline} (EB+EE); #phi_{tau}^{offline}; #phi_{tau}^{L1}", 100,
      -4, 4, 100, -4, 4);

  h_L1TauEtavsTauEta_ = ibooker.book2D("L1TauEtavsTauEta",
      "L1 Tau #eta vs PFTau #eta; PFTau #eta; L1 Tau #eta", 100, -3, 3, 100, -3, 3);

  // tau resolutions
  h_resolutionTauET_EB_ = ibooker.book1D("resolutionTauET_EB",
      "tau ET resolution (EB); (L1 Tau E_{T} - PFTau E_{T})/PFTau E_{T}; events", 50, -1, 1.5);
  h_resolutionTauET_EE_ = ibooker.book1D("resolutionTauET_EE",
      "tau ET resolution (EE); (L1 Tau E_{T} - PFTau E_{T})/PFTau E_{T}; events", 50, -1, 1.5);
  h_resolutionTauET_EB_EE_ = ibooker.book1D("resolutionTauET_EB_EE",
      "tau ET resolution (EB+EE); (L1 Tau E_{T} - PFTau E_{T})/PFTau E_{T}; events", 50, -1, 1.5);

  h_resolutionTauPhi_EB_ =
      ibooker.book1D("resolutionTauPhi_EB",
          "#phi_{tau} resolution (EB); #phi_{tau}^{L1} - #phi_{tau}^{offline}; events",
          120, -0.3, 0.3);
  h_resolutionTauPhi_EE_ =
      ibooker.book1D("resolutionTauPhi_EE",
          "tau #phi resolution (EE); #phi_{tau}^{L1} - #phi_{tau}^{offline}; events",
          120, -0.3, 0.3);
  h_resolutionTauPhi_EB_EE_ =
      ibooker.book1D("resolutionTauPhi_EB_EE",
          "tau #phi resolution (EB+EE); #phi_{tau}^{L1} - #phi_{tau}^{offline}; events",
          120, -0.3, 0.3);

  h_resolutionTauEta_ = ibooker.book1D("resolutionTauEta",
      "tau #eta resolution  (EB); L1 Tau #eta - PFTau #eta; events", 120, -0.3, 0.3);

  // tau turn-ons
  ibooker.setCurrentFolder(efficiencyFolder_.c_str());
  std::vector<float> tauBins(tauEfficiencyBins_.begin(), tauEfficiencyBins_.end());
  int nBins = tauBins.size() - 1;
  float* tauBinArray = &(tauBins[0]);

  for (auto threshold : tauEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyIsoTauET_EB_pass_[threshold] = ibooker.book1D(
        "efficiencyIsoTauET_EB_threshold_" + str_threshold + "_Num",
        "iso tau efficiency (EB); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyIsoTauET_EE_pass_[threshold] = ibooker.book1D(
        "efficiencyIsoTauET_EE_threshold_" + str_threshold + "_Num",
        "iso tau efficiency (EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyIsoTauET_EB_EE_pass_[threshold] = ibooker.book1D(
        "efficiencyIsoTauET_EB_EE_threshold_" + str_threshold + "_Num",
        "iso tau efficiency (EB+EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);

    h_efficiencyIsoTauET_EB_total_[threshold] = ibooker.book1D(
        "efficiencyIsoTauET_EB_threshold_" + str_threshold + "_Den",
        "iso tau efficiency (EB); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyIsoTauET_EE_total_[threshold] = ibooker.book1D(
        "efficiencyIsoTauET_EE_threshold_" + str_threshold + "_Den",
        "iso tau efficiency (EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyIsoTauET_EB_EE_total_[threshold] = ibooker.book1D(
        "efficiencyIsoTauET_EB_EE_threshold_" + str_threshold + "_Den",
        "iso tau efficiency (EB+EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);


    //non iso
    h_efficiencyNonIsoTauET_EB_pass_[threshold] = ibooker.book1D(
        "efficiencyNonIsoTauET_EB_threshold_" + str_threshold + "_Num",
        "inclusive tau efficiency (EB); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyNonIsoTauET_EE_pass_[threshold] = ibooker.book1D(
        "efficiencyNonIsoTauET_EE_threshold_" + str_threshold + "_Num",
        "inclusive tau efficiency (EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyNonIsoTauET_EB_EE_pass_[threshold] = ibooker.book1D(
        "efficiencyNonIsoTauET_EB_EE_threshold_" + str_threshold + "_Num",
        "inclusive tau efficiency (EB+EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);

    h_efficiencyNonIsoTauET_EB_total_[threshold] = ibooker.book1D(
        "efficiencyNonIsoTauET_EB_threshold_" + str_threshold + "_Den",
        "inclusive tau efficiency (EB); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyNonIsoTauET_EE_total_[threshold] = ibooker.book1D(
        "efficiencyNonIsoTauET_EE_threshold_" + str_threshold + "_Den",
        "inclusive tau efficiency (EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);
    h_efficiencyNonIsoTauET_EB_EE_total_[threshold] = ibooker.book1D(
        "efficiencyNonIsoTauET_EB_EE_threshold_" + str_threshold + "_Den",
        "inclusive tau efficiency (EB+EE); PFTau E_{T} (GeV); events", nBins, tauBinArray);

  }

  ibooker.cd();

  return;

}

const reco::Vertex L1TTauOffline::getPrimaryVertex( edm::Handle<reco::VertexCollection> & vertex, edm::Handle<reco::BeamSpot> & beamSpot ) {
  
  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  
  bool hasPrimaryVertex = false;

  if (vertex.isValid())
    {

      vector<reco::Vertex>::const_iterator vertexIt  = vertex->begin();
      vector<reco::Vertex>::const_iterator vertexEnd = vertex->end();

      for (;vertexIt!=vertexEnd;++vertexIt) 
	{
	  if (vertexIt->isValid() && 
	      !vertexIt->isFake()) 
	    {
	      posVtx = vertexIt->position();
	      errVtx = vertexIt->error();
	      hasPrimaryVertex = true;	      
	      break;
	    }
	}
    }

  if ( !hasPrimaryVertex ) {
    posVtx = beamSpot->position();
    errVtx(0,0) = beamSpot->BeamWidthX();
    errVtx(1,1) = beamSpot->BeamWidthY();
    errVtx(2,2) = beamSpot->sigmaZ();
  }

  const reco::Vertex primaryVertex(posVtx,errVtx);
  
  return primaryVertex;
}

bool L1TTauOffline::matchHlt(edm::Handle<trigger::TriggerEvent>  & triggerEvent, const reco::Muon * muon) {


  double matchDeltaR = 9999;

  trigger::TriggerObjectCollection trigObjs = triggerEvent->getObjects();

  vector<int>::const_iterator trigIndexIt  = m_trigIndices.begin();
  vector<int>::const_iterator trigIndexEnd = m_trigIndices.end();
  
  for(; trigIndexIt!=trigIndexEnd; ++trigIndexIt) {

    const vector<string> moduleLabels(m_hltConfig.moduleLabels(*trigIndexIt));
    const unsigned moduleIndex = m_hltConfig.size((*trigIndexIt))-2;

    const unsigned hltFilterIndex = triggerEvent->filterIndex(InputTag(moduleLabels[moduleIndex],"",trigProcess_));
    
    if (hltFilterIndex < triggerEvent->sizeFilters()) {
      const Keys triggerKeys(triggerEvent->filterKeys(hltFilterIndex));
      const Vids triggerVids(triggerEvent->filterIds(hltFilterIndex));
      
      const unsigned nTriggers = triggerVids.size();
      for (size_t iTrig = 0; iTrig < nTriggers; ++iTrig) {
        const TriggerObject trigObject = trigObjs[triggerKeys[iTrig]];
	
        double dRtmp = deltaR((*muon),trigObject);
        if (dRtmp < matchDeltaR) matchDeltaR = dRtmp;
	
      }
    }
  }
  
  return (matchDeltaR < m_MaxHltTauDR);

}

void L1TTauOffline::getTauL1tPairs(edm::Handle<l1t::TauBxCollection> & l1tCands) {					

  m_TauL1tPairs.clear();
  
  vector<const reco::PFTau*>::const_iterator probeTauIt  = m_ProbeTaus.begin();	
  vector<const reco::PFTau*>::const_iterator probeTauEnd = m_ProbeTaus.end();		
  vector<l1t::Tau> l1tContainer;
  
  for (auto tau = l1tCands->begin(0); tau != l1tCands->end(0); ++tau) {
    l1tContainer.push_back(*tau);
  }

  vector<l1t::Tau>::const_iterator l1tIt;					
  vector<l1t::Tau>::const_iterator l1tEnd = l1tContainer.end();				
  
  for (; probeTauIt!=probeTauEnd; ++probeTauIt) {

    // float eta = (*probeTauIt)->eta();		
    // float phi = (*probeTauIt)->phi();		
    // float pt  = (*probeTauIt)->pt();
    

    // m_EfficiencyHistos[0]["ProbeTauEta_Histo"]->Fill(eta);	
    // m_EfficiencyHistos[0]["ProbeTauPhi_Histo"]->Fill(phi);	
    // m_EfficiencyHistos[0]["ProbeTauPt_Histo"]->Fill(pt);
    
    
    TauL1TPair pairBestCand((*probeTauIt),0);    
    l1tIt = l1tContainer.begin();
    
    for(; l1tIt!=l1tEnd; ++l1tIt) {
      
      TauL1TPair pairTmpCand((*probeTauIt),&(*l1tIt));

      if (pairTmpCand.dR() < m_MaxL1tTauDR && pairTmpCand.l1tPt() > pairBestCand.l1tPt())
	pairBestCand = pairTmpCand;

    }

    m_TauL1tPairs.push_back(pairBestCand);
  
    //m_ControlHistos["TauL1tDeltaR"]->Fill(pairBestCand.dR());

  }

}

void L1TTauOffline::getTightMuons(edm::Handle<reco::MuonCollection> & muons, edm::Handle<reco::PFMETCollection> &mets,  const reco::Vertex & vertex, edm::Handle<trigger::TriggerEvent> & trigEvent) {

  m_TightMuons.clear();
  reco::MuonCollection::const_iterator muonIt  = muons->begin();
  reco::MuonCollection::const_iterator muonEnd = muons->end();

  reco::MuonCollection::const_iterator muonIt2  = muons->begin();
  reco::MuonCollection::const_iterator muonEnd2 = muons->end();

  const reco::PFMET *pfmet=NULL;
  pfmet=&(mets->front());

  int nb_mu=0;

  for(; muonIt2!=muonEnd2; ++muonIt2) {
    // cout<<"eta = "<<muonIt2->eta()<<endl;
    // cout<<"muonIt2->pt() = "<<muonIt2->pt()<<endl;
    // cout<<"muon::isLooseMuon((*muonIt2)) = "<< muon::isLooseMuon((*muonIt2))<<endl;
    // cout<<"iso = "<<(muonIt2->pfIsolationR04().sumChargedHadronPt+max(muonIt2->pfIsolationR04().sumNeutralHadronEt+muonIt2->pfIsolationR04().sumPhotonEt-0.5*muonIt2->pfIsolationR04().sumPUPt,0.0))/muonIt2->pt()<<endl;
    if (fabs(muonIt2->eta())< 2.4 && muonIt2->pt()>10 && muon::isLooseMuon((*muonIt2)) && (muonIt2->pfIsolationR04().sumChargedHadronPt+max(muonIt2->pfIsolationR04().sumNeutralHadronEt+muonIt2->pfIsolationR04().sumPhotonEt-0.5*muonIt2->pfIsolationR04().sumPUPt,0.0))/muonIt2->pt()<0.3) {
      ++nb_mu;
    }
  }
  bool foundTightMu=false;
  for(; muonIt!=muonEnd; ++muonIt) {
    // cout<<"eta = "<<muonIt->eta()<<endl;
    // cout<<"muonIt->pt() = "<<muonIt->pt()<<endl;
    // cout<<"muon::isLooseMuon((*muonIt)) = "<< muon::isLooseMuon((*muonIt))<<endl;
    // cout<<"iso = "<<(muonIt->pfIsolationR04().sumChargedHadronPt+max(muonIt->pfIsolationR04().sumNeutralHadronEt+muonIt->pfIsolationR04().sumPhotonEt-0.5*muonIt->pfIsolationR04().sumPUPt,0.0))/muonIt->pt()<<endl;
    //if(0) continue;//HERE!!!!
    if (!matchHlt(trigEvent,&(*muonIt))) continue;
    float muiso=(muonIt->pfIsolationR04().sumChargedHadronPt+max(muonIt->pfIsolationR04().sumNeutralHadronEt+muonIt->pfIsolationR04().sumPhotonEt-0.5*muonIt->pfIsolationR04().sumPUPt,0.0))/muonIt->pt();

    if (muiso<0.1 && nb_mu<2 && !foundTightMu && fabs(muonIt->eta())< 2.1 && muonIt->pt()>24 && muon::isLooseMuon((*muonIt))) {
      float mt=sqrt(pow(muonIt->pt() + pfmet->pt(), 2) - pow(muonIt->px() + pfmet->px(),2) - pow(muonIt->py() + pfmet->py(), 2));
      if (mt<30){
         m_TightMuons.push_back(&(*muonIt));
         foundTightMu=true;
         // m_EfficiencyHistos[0]["TagMuonEta_Histo"]->Fill(muonIt->eta());
         // m_EfficiencyHistos[0]["TagMuonPhi_Histo"]->Fill(muonIt->phi());
         // m_EfficiencyHistos[0]["TagMuonPt_Histo"]->Fill(muonIt->pt());
      } 
    }
  }
  // m_ControlHistos["NTightVsAll"]->Fill(muons->size(),m_TightMuons.size());
  vector<const reco::Muon*>::const_iterator tightMuIt  = m_TightMuons.begin();
  vector<const reco::Muon*>::const_iterator tightMuEnd  = m_TightMuons.end();
  
  for(; tightMuIt!=tightMuEnd; ++tightMuIt) {
    // float eta = (*tightMuIt)->eta();
    // float phi = (*tightMuIt)->phi();
    // float pt  = (*tightMuIt)->pt();

    // m_EfficiencyHistos[0]["RecoEtaNEWtight_Histo"]->Fill(eta);	
    // m_EfficiencyHistos[0]["RecoPhiNEWtight_Histo"]->Fill(phi);	
    // m_EfficiencyHistos[0]["RecoPtNEWtight_Histo"]->Fill(pt);
    
  }
}

void L1TTauOffline::getProbeTaus(const edm::Event & iEvent,edm::Handle<reco::PFTauCollection> & taus, edm::Handle<reco::MuonCollection> & muons, const reco::Vertex & vertex) {

  m_ProbeTaus.clear();
  reco::PFTauCollection::const_iterator tauIt  = taus->begin();
  reco::PFTauCollection::const_iterator tauEnd = taus->end();

  edm::Handle<reco::PFTauDiscriminator> antimu;
  iEvent.getByToken(AntiMuInputTag_, antimu);
  edm::Handle<reco::PFTauDiscriminator> dmf;
  iEvent.getByToken(DecayModeFindingInputTag_, dmf);
  edm::Handle<reco::PFTauDiscriminator> antiele;
  iEvent.getByToken(AntiEleInputTag_, antiele);
  edm::Handle<reco::PFTauDiscriminator> comb3T;
  iEvent.getByToken(comb3TInputTag_, comb3T);

  if (m_TightMuons.size()>0){
     TLorentzVector mymu;
     mymu.SetPtEtaPhiE(m_TightMuons[0]->pt(),m_TightMuons[0]->eta(),m_TightMuons[0]->phi(),m_TightMuons[0]->energy());
     for(unsigned iTau=0; tauIt!=tauEnd; ++tauIt,++iTau) {
        reco::PFTauRef tauCandidate(taus, iTau);
	TLorentzVector mytau;
	mytau.SetPtEtaPhiE(tauIt->pt(),tauIt->eta(),tauIt->phi(),tauIt->energy());

	// cout<<"fabs(tauIt->charge()) = "<<fabs(tauIt->charge())<<endl;
	// cout<<"fabs(tauIt->eta()) = "<<fabs(tauIt->eta())<<endl;
	// cout<<"tauIt->pt() = "<<tauIt->pt()<<endl;
	// cout<<"(*antimu)[tauCandidate] = "<<(*antimu)[tauCandidate]<<endl;
	// cout<<"(*antiele)[tauCandidate] = "<<(*antiele)[tauCandidate]<<endl;
	// cout<<"(*dmf)[tauCandidate] = "<<(*dmf)[tauCandidate]<<endl;
	// cout<<"(*comb3T)[tauCandidate] = "<<(*comb3T)[tauCandidate]<<endl;
	// cout<<"mymu.DeltaR(mytau) = "<<mymu.DeltaR(mytau)<<endl;
	// cout<<"(mymu+mytau).M() = "<<(mymu+mytau).M()<<endl;
	// cout<<"m_TightMuons[0]->charge()*tauIt->charge() = "<<m_TightMuons[0]->charge()*tauIt->charge()<<endl;

        if (fabs(tauIt->charge())==1 && fabs(tauIt->eta())< 2.1 && tauIt->pt()>20 && (*antimu)[tauCandidate] > 0.5 && (*antiele)[tauCandidate] > 0.5 && (*dmf)[tauCandidate] > 0.5 && (*comb3T)[tauCandidate] > 0.5) {
	    if (mymu.DeltaR(mytau)>0.5 && (mymu+mytau).M()>40 && (mymu+mytau).M()<80 && m_TightMuons[0]->charge()*tauIt->charge()<0){
               m_ProbeTaus.push_back(&(*tauIt));
	       // cout<<"********** tau found ************"<<endl;
	    }
        }
	// cout<<"--"<<endl;
     }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE (L1TTauOffline);
