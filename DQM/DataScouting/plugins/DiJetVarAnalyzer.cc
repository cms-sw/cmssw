#include "DQM/DataScouting/plugins/DiJetVarAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"

#include <cmath>

// A simple constructor which takes as inoput only the name of the PF jet collection
DiJetVarAnalyzer::DiJetVarAnalyzer( const edm::ParameterSet & conf ):
  ScoutingAnalyzerBase(conf),
  jetCollectionTag_        (conf.getUntrackedParameter<edm::InputTag>("jetCollectionTag")),
  widejetsCollectionTag_   (conf.getUntrackedParameter<edm::InputTag>("widejetsCollectionTag")),
  metCollectionTag_        (conf.getUntrackedParameter<edm::InputTag>("metCollectionTag")),
  metCleanCollectionTag_   (conf.getUntrackedParameter<edm::InputTag>("metCleanCollectionTag")),
  numwidejets_             (conf.getParameter<unsigned int>("numwidejets")),
  etawidejets_             (conf.getParameter<double>("etawidejets")),
  ptwidejets_              (conf.getParameter<double>("ptwidejets")),
  detawidejets_            (conf.getParameter<double>("detawidejets")),
  dphiwidejets_            (conf.getParameter<double>("dphiwidejets")),
  maxEMfraction_           (conf.getParameter<double>("maxEMfraction")),
  maxHADfraction_          (conf.getParameter<double>("maxHADfraction")),
  HLTpathMain_             (triggerExpression::parse( conf.getParameter<std::string>("HLTpathMain") )),
  HLTpathMonitor_          (triggerExpression::parse( conf.getParameter<std::string>("HLTpathMonitor") )),
  triggerConfiguration_    (conf.getParameterSet("triggerConfiguration"),consumesCollector()) {
  //set Token(-s)
  jetCollectionTagToken_ = consumes<reco::CaloJetCollection>(conf.getUntrackedParameter<edm::InputTag>("jetCollectionTag"));
  widejetsCollectionTagToken_ = consumes<std::vector<math::PtEtaPhiMLorentzVector> >(conf.getUntrackedParameter<edm::InputTag>("widejetsCollectionTag"));
  metCollectionTagToken_ = consumes<reco::CaloMETCollection>(conf.getUntrackedParameter<edm::InputTag>("metCollectionTag"));
  metCleanCollectionTagToken_ = consumes<reco::CaloMETCollection>(conf.getUntrackedParameter<edm::InputTag>("metCleanCollectionTag"));
}

DiJetVarAnalyzer::~DiJetVarAnalyzer() {} 

// Function to book the Monitoring Elements.
void DiJetVarAnalyzer::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const &) {
  ScoutingAnalyzerBase::prepareBooking(iBooker);
  // ==> TO BE UPDATED FOR sqrt(s)=8 TeV
  const int N_mass_bins=83;
  float massBins[N_mass_bins+1] = {1, 3, 6, 10, 16, 23, 31, 40, 50, 61, 74, 88, 103, 119, 137, 156, 176, 197, 220, 244, 270, 296, 325, 354, 386, 419, 453, 489, 526, 565, 606, 649, 693, 740, 788, 838, 890, 944, 1000, 1058, 1118, 1181, 1246, 1313, 1383, 1455, 1530, 1607, 1687, 1770, 1856, 1945, 2037, 2132, 2231, 2332, 2438, 2546, 2659, 2775, 2895, 3019, 3147, 3279, 3416, 3558, 3704, 3854, 4010, 4171, 4337, 4509, 4686, 4869, 5058, 5253, 5455, 5663, 5877, 6099, 6328, 6564, 6808, 7000};

  // 1D histograms
  m_cutFlow = bookH1withSumw2(
      iBooker,
      "h1_cutFlow",
      "Cut Flow",
      7,0.,7.,
      "Cut",
      "Number of events"
      );
  m_cutFlow->getTH1()->GetXaxis()->SetBinLabel(1,"No cut");
  m_cutFlow->getTH1()->GetXaxis()->SetBinLabel(2,"N(WideJets)>=2");
  m_cutFlow->getTH1()->GetXaxis()->SetBinLabel(3,"|#eta|<2.5 , pT>30");
  m_cutFlow->getTH1()->GetXaxis()->SetBinLabel(4,"|#Delta#eta|<1.3");
  m_cutFlow->getTH1()->GetXaxis()->SetBinLabel(5,"JetID");
  m_cutFlow->getTH1()->GetXaxis()->SetBinLabel(6,"|#Delta#phi|>#pi/3");
  m_cutFlow->getTH1()->GetXaxis()->SetBinLabel(7,"|met-metClean|>0.1");

  m_MjjWide_finalSel = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_finalSel",
      "M_{jj} WideJets (final selection)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_finalSel_varbin = bookH1withSumw2BinArray(
      iBooker,
      "h1_MjjWide_finalSel_varbin",
      "M_{jj} WideJets (final selection)",
      N_mass_bins, massBins,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_finalSel_WithoutNoiseFilter = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_finalSel_WithoutNoiseFilter",
      "M_{jj} WideJets (final selection, without noise filters)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_finalSel_WithoutNoiseFilter_varbin = bookH1withSumw2BinArray(
      iBooker,
      "h1_MjjWide_finalSel_WithoutNoiseFilter_varbin",
      "M_{jj} WideJets (final selection, without noise filters)",
      N_mass_bins, massBins,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_deta_0p0_0p5 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_deta_0p0_0p5",
      "M_{jj} WideJets (0.0<=#Delta#eta<0.5)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_deta_0p5_1p0 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_deta_0p5_1p0",
      "M_{jj} WideJets (0.5<=#Delta#eta<1.0)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_deta_1p0_1p5 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_deta_1p0_1p5",
      "M_{jj} WideJets (1.0<=#Delta#eta<1.5)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_deta_1p5_2p0 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_deta_1p5_2p0",
      "M_{jj} WideJets (1.5<=#Delta#eta<2.0)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_deta_2p0_2p5 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_deta_2p0_2p5",
      "M_{jj} WideJets (2.0<=#Delta#eta<2.5)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_deta_2p5_3p0 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_deta_2p5_3p0",
      "M_{jj} WideJets (2.5<=#Delta#eta<3.0)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_deta_3p0_inf = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_deta_3p0_inf",
      "M_{jj} WideJets (#Delta#eta>=3.0)",
      8000,0.,8000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_den_NOdeta = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_den_NOdeta",
      "HLT Efficiency Studies (no deta cut)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_num_NOdeta = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_num_NOdeta",
      "HLT Efficiency Studies (no deta cut)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_den_detaL4 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_den_detaL4",
      "HLT Efficiency Studies (deta cut < 4.0)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_num_detaL4 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_num_detaL4",
      "HLT Efficiency Studies (deta cut < 4.0)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_den_detaL3 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_den_detaL3",
      "HLT Efficiency Studies (deta cut < 3.0)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_num_detaL3 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_num_detaL3",
      "HLT Efficiency Studies (deta cut < 3.0)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_den_detaL2 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_den_detaL2",
      "HLT Efficiency Studies (deta cut < 2.0)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_num_detaL2 = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_num_detaL2",
      "HLT Efficiency Studies (deta cut < 2.0)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_den = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_den",
      "HLT Efficiency Studies (default deta cut)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_MjjWide_num = bookH1withSumw2(
      iBooker,
      "h1_MjjWide_num",
      "HLT Efficiency Studies (default deta cut)",
      400,0.,2000.,
      "M_{jj} WideJets [GeV]",
      "Number of events"
      );

  m_DetajjWide_finalSel = bookH1withSumw2(
      iBooker,
      "h1_DetajjWide_finalSel",
      "#Delta#eta_{jj} WideJets (final selection)",
      100,0.,5.,
      "#Delta#eta_{jj} WideJets",
      "Number of events"
      );

  m_DetajjWide = bookH1withSumw2(
      iBooker,
      "h1_DetajjWide",
      "#Delta#eta_{jj} WideJets (final selection except #Delta#eta cut)",
      100,0.,5.,
      "#Delta#eta_{jj} WideJets",
      "Number of events"
      );

  m_DphijjWide_finalSel = bookH1withSumw2(
      iBooker,
      "h1_DphijjWide_finalSel",
      "#Delta#phi_{jj} WideJets (final selection)",
      100,0.,TMath::Pi()+0.0001,
      "#Delta#phi_{jj} WideJets [rad.]",
      "Number of events"
      );


  m_selJets_pt = bookH1withSumw2(
      iBooker,
      "h1_selJets_pt",
      "Selected CaloJets",
      500,0.,5000.,
      "Jet Pt [GeV]",
      "Number of events"
      );

  m_selJets_eta = bookH1withSumw2(
      iBooker,
      "h1_selJets_eta",
      "Selected CaloJets",
      100,-5.,5.,
      "#eta",
      "Number of events"
      );

  m_selJets_phi = bookH1withSumw2(
      iBooker,
      "h1_selJets_phi",
      "Selected CaloJets",
      100,-TMath::Pi(),TMath::Pi(),
      "#phi (rad.)",
      "Number of events"
      );

  m_selJets_hadEnergyFraction = bookH1withSumw2(
      iBooker,
      "h1_selJets_hadEnergyFraction",
      "Selected CaloJets",
      110,0.,1.1,
      "HAD Energy Fraction",
      "Number of events"
      );

  m_selJets_emEnergyFraction = bookH1withSumw2(
      iBooker,
      "h1_selJets_emEnergyFraction",
      "Selected CaloJets",
      110,0.,1.1,
      "EM Energy Fraction",
      "Number of events"
      );

  m_selJets_towersArea = bookH1withSumw2(
      iBooker,
      "h1_selJets_towersArea",
      "Selected CaloJets",
      200,0.,2.,
      "towers area",
      "Number of events"
      );

  m_metDiff = bookH1withSumw2(
      iBooker,
      "h1_metDiff",
      "Met - MetCleaned",
      500,-1000.,1000.,
      "met - metcleaned [GeV]",
      "Number of events"
      );

  m_metCases = bookH1withSumw2(
      iBooker,
      "h1_metCases",
      "Met cases",
      3,0.,3.,
      "case",
      "Number of events"
      );
  m_metCases->getTH1()->GetXaxis()->SetBinLabel(1,"met , metclean");
  m_metCases->getTH1()->GetXaxis()->SetBinLabel(2,"met , !metclean");
  m_metCases->getTH1()->GetXaxis()->SetBinLabel(3,"!met , metclean");

  m_metCaseNoMetClean = bookH1withSumw2(
      iBooker,
      "h1_metCaseNoMetClean",
      "Met - MetCleaned",
      1000,0.,2000.,
      "MET [GeV]",
      "Number of events"
      );  

  m_HT_inclusive = bookH1withSumw2(
      iBooker,
      "h1_HT_inclusive",
      "HT (inclusive)",
      150,0.,15000.,
      "HT [GeV]",
      "Number of events"
      );  

  m_HT_finalSel = bookH1withSumw2(
      iBooker,
      "h1_HT_finalSel",
      "HT (final selection)",
      150,0.,15000.,
      "HT [GeV]",
      "Number of events"
      );  

  // 2D histograms
  m_DetajjVsMjjWide = bookH2withSumw2(
      iBooker,
      "h2_DetajjVsMjjWide",
      "#Delta#eta_{jj} vs M_{jj} WideJets",
      8000,0.,8000.,
      100,0.,5.,
      "M_{jj} WideJets [GeV]",
      "#Delta#eta_{jj} WideJets");

  m_DetajjVsMjjWide_rebin = bookH2withSumw2(
      iBooker,
      "h2_DetajjVsMjjWide_rebin",
      "#Delta#eta_{jj} vs M_{jj} WideJets",
      400,0.,8000.,
      50,0.,5.,
      "M_{jj} WideJets [GeV]",
      "#Delta#eta_{jj} WideJets");

  m_metVSmetclean = bookH2withSumw2(
      iBooker,
      "h2_metVSmetclean",
      "MET clean vs MET",
      100,0.,2000.,
      100,0.,2000.,
      "MET [GeV]",
      "MET clean [GeV]");
}

// Usual analyze method
void DiJetVarAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ) {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // ## Get jet collection
  edm::Handle<reco::CaloJetCollection> calojets_handle;
  iEvent.getByToken(jetCollectionTagToken_,calojets_handle);

  // Loop over all the selected jets ( defined at DQM/DataScouting/python/dijetScouting_cff.py )  
  double thisHT = 0.;
  for(reco::CaloJetCollection::const_iterator it = calojets_handle->begin(); it != calojets_handle->end(); ++it)
  {
    //cout << "== jet: " << it->pt() << " " << it->eta() << " " << it->phi() << endl;
    m_selJets_pt->Fill( it->pt() );
    m_selJets_eta->Fill( it->eta() );
    m_selJets_phi->Fill( it->phi() );
    m_selJets_hadEnergyFraction->Fill( it->energyFractionHadronic() );
    m_selJets_emEnergyFraction->Fill( it->emEnergyFraction() );
    m_selJets_towersArea->Fill( it->towersArea() );
    thisHT += it->pt();
  }

  // HT
  m_HT_inclusive->Fill(thisHT);      

  // ## Get widejets 
  edm::Handle< vector<math::PtEtaPhiMLorentzVector> > widejets_handle;
  iEvent.getByToken(widejetsCollectionTagToken_, widejets_handle);

  TLorentzVector wj1;
  TLorentzVector wj2;
  TLorentzVector wdijet;

  double MJJWide = -1;
  double DeltaEtaJJWide = -1;
  double DeltaPhiJJWide = -1;

  if( widejets_handle->size() >= 2 ) {
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
  }

  // ## Get met collection
  // met
  edm::Handle<reco::CaloMETCollection> calomet_handle;
  iEvent.getByToken(metCollectionTagToken_, calomet_handle);
  // met cleaned
  edm::Handle<reco::CaloMETCollection> calometClean_handle;
  iEvent.getByToken(metCleanCollectionTagToken_, calometClean_handle);

  if( calomet_handle.isValid() && calometClean_handle.isValid() ) {
    //       std::cout << "---" << std::endl;
    //       std::cout << "== calomet: " << (calomet_handle->front()).pt() << " " << (calomet_handle->front()).phi() << std::endl;
    //       std::cout << "== calometClean: " << (calometClean_handle->front()).pt() << " " << (calometClean_handle->front()).phi() << std::endl;
    //       std::cout << "== calomet - calometClean: " << (calomet_handle->front()).pt() - (calometClean_handle->front()).pt() << std::endl;
    //       std::cout << "---" << std::endl;
    m_metCases->Fill(0);
    m_metDiff->Fill( (calomet_handle->front()).pt() - (calometClean_handle->front()).pt() );
    m_metVSmetclean->Fill( (calomet_handle->front()).pt() , (calometClean_handle->front()).pt() );      
  }
  else if( calomet_handle.isValid() && !calometClean_handle.isValid() )
  {
    m_metCases->Fill(1);
    m_metCaseNoMetClean->Fill((calomet_handle->front()).pt());
  }
  else if( !calomet_handle.isValid() && calometClean_handle.isValid() )
  {
    m_metCases->Fill(2);
  }

  // ## Event Selection
  bool pass_nocut=false;
  bool pass_twowidejets=false;
  bool pass_etaptcuts=false;
  bool pass_deta=false;
  bool pass_JetIDtwojets=true;
  bool pass_dphi=false;
  bool pass_metFilter=true;
  //--
  bool pass_deta_L4=false;
  bool pass_deta_L3=false;
  bool pass_deta_L2=false;

  bool pass_fullsel_NOdeta=false;
  bool pass_fullsel_detaL4=false;
  bool pass_fullsel_detaL3=false;
  bool pass_fullsel_detaL2=false;
  bool pass_fullsel=false;

  // No cut
  pass_nocut=true;

  // Two wide jets
  if( widejets_handle->size() >= numwidejets_ ) {
    // Two wide jets
    pass_twowidejets=true;

    // Eta/pt cuts
    if( fabs(wj1.Eta())<etawidejets_ && wj1.Pt()>ptwidejets_ 
        &&
        fabs(wj2.Eta())<etawidejets_ && wj2.Pt()>ptwidejets_
      ) {
      pass_etaptcuts=true;
    }

    // Deta cut
    if( DeltaEtaJJWide < detawidejets_ )
      pass_deta=true;

    // Dphi cut
    if( DeltaPhiJJWide > dphiwidejets_ )
      pass_dphi=true;

    // Other Deta cuts
    if( DeltaEtaJJWide < 4.0 )
      pass_deta_L4=true;

    if( DeltaEtaJJWide < 3.0 )
      pass_deta_L3=true;

    if( DeltaEtaJJWide < 2.0 )
      pass_deta_L2=true;
  }
  // Jet id two leading jets
  if( calojets_handle->size() >= numwidejets_ )
  {
    //   first jet
    reco::CaloJetCollection::const_iterator thisJet = calojets_handle->begin();
    //cout << "== thisJet1: " << thisJet->pt() << " " << thisJet->eta() << " " << thisJet->phi() << endl;
    if( thisJet->energyFractionHadronic()>maxHADfraction_ || thisJet->emEnergyFraction()>maxEMfraction_ )
      pass_JetIDtwojets=false;

    //   second jet
    ++thisJet;
    //cout << "== thisJet2: " << thisJet->pt() << " " << thisJet->eta() << " " << thisJet->phi() << endl;
    if( thisJet->energyFractionHadronic()>maxHADfraction_ || thisJet->emEnergyFraction()>maxEMfraction_ )
      pass_JetIDtwojets=false;
  }      
  // Met filter
  if( calomet_handle.isValid() && calometClean_handle.isValid() )
  {
    if( fabs ( (calomet_handle->front()).pt() - (calometClean_handle->front()).pt() ) > 0.1 )
      pass_metFilter=false;
  }
  // Full selection (no deta cut)
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_JetIDtwojets && pass_dphi && pass_metFilter )
    pass_fullsel_NOdeta=true;
  // Full selection (various deta cuts)
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_JetIDtwojets && pass_dphi && pass_metFilter && pass_deta_L4 )
    pass_fullsel_detaL4=true;
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_JetIDtwojets && pass_dphi && pass_metFilter && pass_deta_L3 )
    pass_fullsel_detaL3=true;
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_JetIDtwojets && pass_dphi && pass_metFilter && pass_deta_L2 )
    pass_fullsel_detaL2=true;
  // Full selection (default deta cut)
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_deta && pass_JetIDtwojets && pass_dphi && pass_metFilter )
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
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_deta && pass_JetIDtwojets && pass_dphi )
    m_cutFlow->Fill(5);
  if( pass_fullsel )
    m_cutFlow->Fill(6);

  // After full selection
  if( pass_fullsel ) 
  {
    // 1D histograms
    m_MjjWide_finalSel->Fill(MJJWide);
    m_MjjWide_finalSel_varbin->Fill(MJJWide);
    m_DetajjWide_finalSel->Fill(DeltaEtaJJWide);
    m_DphijjWide_finalSel->Fill(DeltaPhiJJWide);      
    m_HT_finalSel->Fill(thisHT);      
  }      
  // After full selection (without "noise" filters)
  if( pass_nocut && pass_twowidejets && pass_etaptcuts && pass_deta )
  {
    m_MjjWide_finalSel_WithoutNoiseFilter->Fill(MJJWide);
    m_MjjWide_finalSel_WithoutNoiseFilter_varbin->Fill(MJJWide);
  }
  // After full selection (except DeltaEta cut)
  if( pass_fullsel_NOdeta )
  {
    // 1D histograms
    m_DetajjWide->Fill(DeltaEtaJJWide);
    if( DeltaEtaJJWide >= 0.0 && DeltaEtaJJWide < 0.5 )
      m_MjjWide_deta_0p0_0p5->Fill(MJJWide);
    if( DeltaEtaJJWide >= 0.5 && DeltaEtaJJWide < 1.0 )
      m_MjjWide_deta_0p5_1p0->Fill(MJJWide);
    if( DeltaEtaJJWide >= 1.0 && DeltaEtaJJWide < 1.5 )
      m_MjjWide_deta_1p0_1p5->Fill(MJJWide);
    if( DeltaEtaJJWide >= 1.5 && DeltaEtaJJWide < 2.0 )
      m_MjjWide_deta_1p5_2p0->Fill(MJJWide);
    if( DeltaEtaJJWide >= 2.0 && DeltaEtaJJWide < 2.5 )
      m_MjjWide_deta_2p0_2p5->Fill(MJJWide);
    if( DeltaEtaJJWide >= 2.5 && DeltaEtaJJWide < 3.0 )
      m_MjjWide_deta_2p5_3p0->Fill(MJJWide);
    if( DeltaEtaJJWide >= 3.0 )
      m_MjjWide_deta_3p0_inf->Fill(MJJWide);

    // 2D histograms
    m_DetajjVsMjjWide->Fill(MJJWide,DeltaEtaJJWide);            
    m_DetajjVsMjjWide_rebin->Fill(MJJWide,DeltaEtaJJWide);
  }

  // ## Get Trigger Info

  // HLT paths for DataScouting
  //  DST_HT250_v1
  //  DST_L1HTT_Or_L1MultiJet_v1
  //  DST_Mu5_HT250_v1
  //  DST_Ele8_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT250_v1
  int HLTpathMain_fired    = -1;
  int HLTpathMonitor_fired = -1;
  if (HLTpathMain_ and HLTpathMonitor_ and triggerConfiguration_.setEvent(iEvent, c)) {
    // invalid HLT configuration, skip the processing

    // if the L1 or HLT configurations have changed, (re)initialize the filters (including during the first event)
    if (triggerConfiguration_.configurationUpdated()) {
      HLTpathMain_->init(triggerConfiguration_);
      HLTpathMonitor_->init(triggerConfiguration_);

      // log the expanded configuration
      // std::cout << "HLT selector configurations updated" << std::endl;
      // std::cout << "HLTpathMain:    " << *HLTpathMain_    << std::endl;
      // std::cout << "HLTpathMonitor: " << *HLTpathMonitor_ << std::endl;
    }

    HLTpathMain_fired    = (*HLTpathMain_)(triggerConfiguration_);
    HLTpathMonitor_fired = (*HLTpathMonitor_)(triggerConfiguration_);

    // The OR of the two should always be "1"
    // std::cout << *HLTpathMain_ << ": " << HLTpathMain_fired << " -- " << *HLTpathMonitor_ << ": " << HLTpathMonitor_fired << std::endl;
  }

  // ## Trigger Efficiency Curves

  //denominator - full sel NO deta cut
  if( pass_fullsel_NOdeta && HLTpathMonitor_fired == 1 )
  {
    m_MjjWide_den_NOdeta->Fill(MJJWide);

    //numerator  
    if( HLTpathMain_fired == 1)
    {
      m_MjjWide_num_NOdeta->Fill(MJJWide);
    }
  }

  //denominator - full sel deta < 4.0
  if( pass_fullsel_detaL4 && HLTpathMonitor_fired == 1 )
  {
    m_MjjWide_den_detaL4->Fill(MJJWide);

    //numerator  
    if( HLTpathMain_fired == 1)
    {
      m_MjjWide_num_detaL4->Fill(MJJWide);
    }
  }

  //denominator - full sel deta < 3.0
  if( pass_fullsel_detaL3 && HLTpathMonitor_fired == 1 )
  {
    m_MjjWide_den_detaL3->Fill(MJJWide);

    //numerator  
    if( HLTpathMain_fired == 1)
    {
      m_MjjWide_num_detaL3->Fill(MJJWide);
    }
  }

  //denominator - full sel deta < 2.0
  if( pass_fullsel_detaL2 && HLTpathMonitor_fired == 1 )
  {
    m_MjjWide_den_detaL2->Fill(MJJWide);

    //numerator  
    if( HLTpathMain_fired == 1)
    {
      m_MjjWide_num_detaL2->Fill(MJJWide);
    }
  }

  //denominator - full sel default deta cut (typically 1.3)
  if( pass_fullsel && HLTpathMonitor_fired == 1 )
  {
    m_MjjWide_den->Fill(MJJWide);

    //numerator  
    if( HLTpathMain_fired == 1)
    {
      m_MjjWide_num->Fill(MJJWide);
    }
  }
}
