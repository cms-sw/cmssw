#include "DQM/DataScouting/plugins/DiJetVarAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include <cmath>

//------------------------------------------------------------------------------
// A simple constructor which takes as inoput only the name of the PF jet collection
DiJetVarAnalyzer::DiJetVarAnalyzer( const edm::ParameterSet & conf ):
  ScoutingAnalyzerBase(conf),
  jetCollectionTag_        (conf.getUntrackedParameter<edm::InputTag>("jetCollectionTag")),
  //dijetVarCollectionTag_   (conf.getUntrackedParameter<edm::InputTag>("dijetVarCollectionTag")),
  widejetsCollectionTag_   (conf.getUntrackedParameter<edm::InputTag>("widejetsCollectionTag")),
  numwidejets_             (conf.getParameter<unsigned int>("numwidejets")),
  etawidejets_             (conf.getParameter<double>("etawidejets")),
  ptwidejets_              (conf.getParameter<double>("ptwidejets")),
  detawidejets_            (conf.getParameter<double>("detawidejets")),
  dphiwidejets_            (conf.getParameter<double>("dphiwidejets"))
{
}

//------------------------------------------------------------------------------
// Nothing to destroy: the DQM service thinks about everything
DiJetVarAnalyzer::~DiJetVarAnalyzer(){}

//------------------------------------------------------------------------------
// Usual analyze method
void DiJetVarAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ){
  
  using namespace std;
  using namespace edm;
  using namespace reco;
  
  // ## Get jet collection
  edm::Handle<reco::CaloJetCollection> calojets_handle;
  iEvent.getByLabel(jetCollectionTag_,calojets_handle);

  //   // Loop over all the jets
  //   for(reco::CaloJetCollection::const_iterator it = calojets_handle->begin(); it != calojets_handle->end(); ++it)
  //     {
  //       cout << "== jet: " << it->pt() << " " << it->eta() << " " << it->phi() << endl;
  //     }
  
  // ## Get widejets 
  edm::Handle< vector<math::PtEtaPhiMLorentzVector> > widejets_handle;
  iEvent.getByLabel (widejetsCollectionTag_,widejets_handle);
  
  TLorentzVector wj1;
  TLorentzVector wj2;
  TLorentzVector wdijet;

  double MJJWide = -1;
  double DeltaEtaJJWide = -1;
  double DeltaPhiJJWide = -1;

  if( widejets_handle->size() >= 2 )
    {
      wj1.SetPtEtaPhiM(widejets_handle->at(0).pt(),
		       widejets_handle->at(0).eta(),
		       widejets_handle->at(0).phi(),
		       widejets_handle->at(0).mass()
		       );
      wj2.SetPtEtaPhiM(widejets_handle->at(1).pt(),
		       widejets_handle->at(1).eta(),
		       widejets_handle->at(1).phi(),
		       widejets_handle->at(1).mass()
		       );

      wdijet = wj1 + wj2;

      MJJWide = wdijet.M();
      DeltaEtaJJWide = fabs(wj1.Eta()-wj2.Eta());
      DeltaPhiJJWide = fabs(wj1.DeltaPhi(wj2));

      //       cout << "== j1 wide: " << wj1.Pt() << " " << wj1.Eta() << " " << wj1.Phi() << " " << wj1.M() << endl;
      //       cout << "== j2 wide: " << wj2.Pt() << " " << wj2.Eta() << " " << wj2.Phi() << " " << wj2.M() << endl;
      //       cout << "== MJJWide: " << MJJWide << endl;
      //       cout << "== DeltaEtaJJWide: " << DeltaEtaJJWide << endl;
      //       cout << "== DeltaPhiJJWide: " << DeltaPhiJJWide << endl;
    }
  
  // ## Event Selection
  bool pass_nocut=false;
  bool pass_twowidejets=false;
  bool pass_etaptcuts=false;
  bool pass_deta=false;
  bool pass_JetIDtwojets=false;
  bool pass_dphi=false;

  bool pass_fullsel=false;

  // No cut
  pass_nocut=true;

  // Two wide jets
  if( widejets_handle->size() >= numwidejets_ )
    {
      // Two wide jets
      pass_twowidejets=true;

      // Eta/pt cuts
      if( fabs(wj1.Eta())<etawidejets_ && wj1.Pt()>ptwidejets_ 
	  &&
	  fabs(wj2.Eta())<etawidejets_ && wj2.Pt()>ptwidejets_
	  )
	{
	  pass_etaptcuts=true;
	}
      
      // Deta cut
      if( DeltaEtaJJWide < detawidejets_ )
	pass_deta=true;

      // Dphi cut
      if( DeltaPhiJJWide > dphiwidejets_ )
	pass_dphi=true;
    }

  // Jet id two leading jets
  pass_JetIDtwojets=true;

  // Full selection
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_deta && pass_JetIDtwojets && pass_dphi )
    pass_fullsel=true;

  // ## Fill Histograms 

  // Cut-flow plot
  if( pass_nocut )
    m_cutFlow->Fill(0);
  if( pass_nocut && pass_twowidejets )
    m_cutFlow->Fill(1);
  if( pass_nocut && pass_twowidejets && pass_etaptcuts )
    m_cutFlow->Fill(2);
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_deta )
    m_cutFlow->Fill(3);
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_deta && pass_JetIDtwojets )
    m_cutFlow->Fill(4);
  if( pass_fullsel )
    m_cutFlow->Fill(5);

  // After full selection
  if( pass_fullsel ) 
    {
      // 1D histograms
      m_MjjWide->Fill(MJJWide);
      m_DetajjWide->Fill(DeltaEtaJJWide);
      m_DphijjWide->Fill(DeltaPhiJJWide);
      
      // 2D histograms
      m_DetajjVsMjjWide->Fill(MJJWide,DeltaEtaJJWide);      
    }      

}

void DiJetVarAnalyzer::endRun( edm::Run const &, edm::EventSetup const & ){
}

//------------------------------------------------------------------------------
// Function to book the Monitoring Elements.
void DiJetVarAnalyzer::bookMEs(){
  

  // 1D histograms
  m_cutFlow = bookH1withSumw2( "h1_cutFlow",
			       "Cut Flow",
			       6,0.,6.,
			       "Cut",
			       "Number of events"
			       );

  m_MjjWide = bookH1withSumw2( "h1_MjjWide",
			       "M_{jj} WideJets",
			       500,0.,5000.,
			       "M_{jj} WideJets [GeV]",
			       "Number of events"
			       );

  m_DetajjWide = bookH1withSumw2( "h1_DetajjWide",
				  "#Delta#eta_{jj} WideJets",
				  100,0.,5.,
				  "#Delta#eta_{jj} WideJets",
				  "Number of events"
				  );

  m_DphijjWide = bookH1withSumw2( "h1_DphijjWide",
				  "#Delta#phi_{jj} WideJets",
				  100,0.,3.15,
				  "#Delta#phi_{jj} WideJets [rad.]",
				  "Number of events"
				  );

  // 2D histograms
  m_DetajjVsMjjWide = bookH2withSumw2("h2_DetajjVsMjjWide",
				      "#Delta#eta_{jj} vs M_{jj} WideJets",
				      500,0.,5000.,
				      100,0.,5.,
				      "M_{jj} WideJets [GeV]",
				      "#Delta#eta_{jj} WideJets");

}

