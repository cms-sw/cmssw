#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_alphaT.h"
//#include "HLTriggerOffline/SUSYBSM/interface/AlphaT.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> LorentzV;

SUSY_HLT_alphaT::SUSY_HLT_alphaT(const edm::ParameterSet &ps) {
  edm::LogInfo("SUSY_HLT_alphaT") << "Constructor SUSY_HLT_alphaT::SUSY_HLT_alphaT " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
  // theCaloJetCollection_ =
  // consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  triggerPreFilter_ = ps.getParameter<edm::InputTag>("TriggerPreFilter");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
  // caloAlphaTThrTurnon_ =
  // ps.getUntrackedParameter<double>("caloAlphaTThrTurnon"); caloHtThrTurnon_ =
  // ps.getUntrackedParameter<double>("caloHtThrTurnon");
  pfAlphaTThrTurnon_ = ps.getUntrackedParameter<double>("pfAlphaTThrTurnon");
  pfHtThrTurnon_ = ps.getUntrackedParameter<double>("pfHtThrTurnon");
}

SUSY_HLT_alphaT::~SUSY_HLT_alphaT() {
  edm::LogInfo("SUSY_HLT_alphaT") << "Destructor SUSY_HLT_alphaT::~SUSY_HLT_alphaT " << std::endl;
}

void SUSY_HLT_alphaT::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e) {
  bool changed;

  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_alphaT") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  bool pathFound = false;
  const std::vector<std::string> allTrigNames = fHltConfig.triggerNames();
  for (size_t j = 0; j < allTrigNames.size(); ++j) {
    if (allTrigNames[j].find(triggerPath_) != std::string::npos) {
      pathFound = true;
    }
  }

  if (!pathFound) {
    LogDebug("SUSY_HLT_alphaT") << "Path not found"
                                << "\n";
    return;
  }
  // std::vector<std::string> filtertags = fHltConfig.moduleLabels( triggerPath_
  // ); triggerFilter_ =
  // edm::InputTag(filtertags[filtertags.size()-1],"",fHltConfig.processName());
  // triggerFilter_ = edm::InputTag("hltPFMET120Mu5L3PreFiltered", "",
  // fHltConfig.processName());

  edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::beginRun" << std::endl;
}

void SUSY_HLT_alphaT::bookHistograms(DQMStore::IBooker &ibooker_, edm::Run const &, edm::EventSetup const &) {
  edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::bookHistograms" << std::endl;
  // book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_alphaT::analyze(edm::Event const &e, edm::EventSetup const &eSetup) {
  edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::analyze" << std::endl;

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_, hltresults);
  if (!hltresults.isValid()) {
    edm::LogWarning("SUSY_HLT_alphaT") << "invalid collection: TriggerResults"
                                       << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if (!triggerSummary.isValid()) {
    edm::LogWarning("SUSY_HLT_alphaT") << "invalid collection: TriggerSummary"
                                       << "\n";
    return;
  }

  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  e.getByToken(thePfJetCollection_, pfJetCollection);
  if (!pfJetCollection.isValid()) {
    edm::LogWarning("SUSY_HLT_alphaT") << "invalid collection: PFJets"
                                       << "\n";
    return;
  }
  // edm::Handle<reco::CaloJetCollection> caloJetCollection;
  // e.getByToken (theCaloJetCollection_,caloJetCollection);
  // if ( !caloJetCollection.isValid() ){
  //     edm::LogWarning ("SUSY_HLT_alphaT") << "invalid collection: CaloJets"
  //     << "\n"; return;
  // }

  // get online objects
  // For now just get the jets and recalculate ht and alphaT
  size_t filterIndex = triggerSummary->filterIndex(triggerFilter_);
  // size_t preFilterIndex = triggerSummary->filterIndex( triggerPreFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();

  // Get the PF objects from the filter
  double hltPfHt = 0.;
  std::vector<LorentzV> hltPfJets;
  if (!(filterIndex >= triggerSummary->sizeFilters())) {
    const trigger::Keys &keys = triggerSummary->filterKeys(filterIndex);

    for (size_t j = 0; j < keys.size(); ++j) {
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];

      //  if(foundObject.id() == 85){ //It's a jet
      if (foundObject.pt() > ptThrJet_ && fabs(foundObject.eta()) < etaThrJet_) {
        hltPfHt += foundObject.pt();
        LorentzV JetLVec(foundObject.pt(), foundObject.eta(), foundObject.phi(), foundObject.mass());
        hltPfJets.push_back(JetLVec);
      }
      //   }
    }
  }

  // //Get the Calo objects from the prefilter
  // double hltCaloHt=0.;
  // std::vector<LorentzV> hltCaloJets;
  // if( !(preFilterIndex >= triggerSummary->sizeFilters()) ){
  //     const trigger::Keys& keys = triggerSummary->filterKeys( preFilterIndex
  //     );

  //     for( size_t j = 0; j < keys.size(); ++j ){
  //         trigger::TriggerObject foundObject = triggerObjects[keys[j]];

  //         //  if(foundObject.id() == 85){ //It's a jet
  //         if(foundObject.pt()>ptThrJet_ && fabs(foundObject.eta()) <
  //         etaThrJet_){
  //             hltCaloHt += foundObject.pt();
  //             LorentzV
  //             JetLVec(foundObject.pt(),foundObject.eta(),foundObject.phi(),foundObject.mass());
  //             hltCaloJets.push_back(JetLVec);
  //         }
  //         //   }
  //     }
  // }

  // Fill the alphaT and HT histograms
  if (!hltPfJets.empty()) {
    double hltPfAlphaT = AlphaT(hltPfJets, true).value();
    h_triggerPfAlphaT->Fill(hltPfAlphaT);
    h_triggerPfHt->Fill(hltPfHt);
    h_triggerPfAlphaT_triggerPfHt->Fill(hltPfHt, hltPfAlphaT);
  }

  // if(hltCaloJets.size()>0){
  //     double hltCaloAlphaT = AlphaT(hltCaloJets,true).value();
  //     h_triggerCaloAlphaT->Fill(hltCaloAlphaT);
  //     h_triggerCaloHt->Fill(hltCaloHt);
  //     h_triggerCaloAlphaT_triggerCaloHt->Fill(hltCaloHt, hltCaloAlphaT);
  // }

  bool hasFired = false;
  bool hasFiredAuxiliaryForHadronicLeg = false;
  const edm::TriggerNames &trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  for (unsigned int hltIndex = 0; hltIndex < numTriggers; ++hltIndex) {
    if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) &&
        hltresults->accept(hltIndex))
      hasFired = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForHadronic_) != std::string::npos &&
        hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex))
      hasFiredAuxiliaryForHadronicLeg = true;
  }

  if (hasFiredAuxiliaryForHadronicLeg) {
    float pfHT = 0.0;
    std::vector<LorentzV> pfJets;
    for (reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end();
         ++i_pfjet) {
      if (i_pfjet->pt() < ptThrJet_)
        continue;
      if (fabs(i_pfjet->eta()) > etaThrJet_)
        continue;
      pfHT += i_pfjet->pt();
      LorentzV JetLVec(i_pfjet->pt(), i_pfjet->eta(), i_pfjet->phi(), i_pfjet->mass());
      pfJets.push_back(JetLVec);
    }

    // //Make the gen Calo HT and AlphaT
    // float caloHT = 0.0;
    // std::vector<LorentzV> caloJets;
    // for (reco::CaloJetCollection::const_iterator i_calojet =
    // caloJetCollection->begin(); i_calojet != caloJetCollection->end();
    // ++i_calojet){
    //   if (i_calojet->pt() < ptThrJet_) continue;
    //   if (fabs(i_calojet->eta()) > etaThrJet_) continue;
    //   caloHT += i_calojet->pt();
    //   LorentzV
    //   JetLVec(i_calojet->pt(),i_calojet->eta(),i_calojet->phi(),i_calojet->mass());
    //   caloJets.push_back(JetLVec);
    // }

    // AlphaT aT = AlphaT(jets);
    // double caloAlphaT = AlphaT(caloJets).value();
    double pfAlphaT = AlphaT(pfJets).value();

    // Fill the turnons
    if (hasFired) {
      // if(caloHT>caloHtThrTurnon_) h_caloAlphaTTurnOn_num-> Fill(caloAlphaT);
      // if(caloAlphaT>caloAlphaTThrTurnon_) h_caloHtTurnOn_num-> Fill(caloHT);

      if (pfHT > pfHtThrTurnon_)
        h_pfAlphaTTurnOn_num->Fill(pfAlphaT);
      if (pfAlphaT > pfAlphaTThrTurnon_)
        h_pfHtTurnOn_num->Fill(pfHT);
    }
    // if(caloHT>caloHtThrTurnon_) h_caloAlphaTTurnOn_den-> Fill(caloAlphaT);
    // if(caloAlphaT>caloAlphaTThrTurnon_) h_caloHtTurnOn_den-> Fill(caloHT);

    if (pfHT > pfHtThrTurnon_)
      h_pfAlphaTTurnOn_den->Fill(pfAlphaT);
    if (pfAlphaT > pfAlphaTThrTurnon_)
      h_pfHtTurnOn_den->Fill(pfHT);
  }
}

void SUSY_HLT_alphaT::bookHistos(DQMStore::IBooker &ibooker_) {
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  // offline quantities

  // online quantities
  // h_triggerCaloHt = ibooker_.book1D("triggerCaloHt", "Trigger Calo Ht; HT
  // (GeV)", 60, 0.0, 1500.0); h_triggerCaloAlphaT =
  // ibooker_.book1D("triggerCaloAlphaT", "Trigger Calo AlphaT; AlphaT", 80,
  // 0., 1.0); h_triggerCaloAlphaT_triggerCaloHt =
  // ibooker_.book2D("triggerCaloAlphaT_triggerCaloHt","Trigger Calo HT vs
  // Trigger Calo AlphaT; HT (GeV); AlphaT", 60,0.0,1500.,80,0.,1.0);
  h_triggerPfHt = ibooker_.book1D("triggerPfHt", "Trigger PF Ht; HT (GeV)", 60, 0.0, 1500.0);
  h_triggerPfAlphaT = ibooker_.book1D("triggerPfAlphaT", "Trigger PF AlphaT; AlphaT", 80, 0., 1.0);
  h_triggerPfAlphaT_triggerPfHt = ibooker_.book2D("triggerPfAlphaT_triggerPfHt",
                                                  "Trigger PF HT vs Trigger PF AlphaT; HT (GeV); AlphaT",
                                                  60,
                                                  0.0,
                                                  1500.,
                                                  80,
                                                  0.,
                                                  1.0);

  // num and den hists to be divided in harvesting step to make turn on curves
  // h_caloAlphaTTurnOn_num = ibooker_.book1D("caloAlphaTTurnOn_num", "Calo
  // AlphaT Turn On Numerator; AlphaT", 40, 0.0, 1.0 ); h_caloAlphaTTurnOn_den =
  // ibooker_.book1D("caloAlphaTTurnOn_den", "Calo AlphaT Turn OnDenominator;
  // AlphaT", 40, 0.0, 1.0 ); h_caloHtTurnOn_num =
  // ibooker_.book1D("caloHtTurnOn_num", "Calo HT Turn On Numerator; HT (GeV)",
  // 30, 0.0, 1500.0 ); h_caloHtTurnOn_den = ibooker_.book1D("caloHtTurnOn_den",
  // "Calo HT Turn On Denominator; HT (GeV)", 30, 0.0, 1500.0 );

  h_pfAlphaTTurnOn_num = ibooker_.book1D("pfAlphaTTurnOn_num", "PF AlphaT Turn On Numerator; AlphaT", 40, 0.0, 1.0);
  h_pfAlphaTTurnOn_den = ibooker_.book1D("pfAlphaTTurnOn_den", "PF AlphaT Turn OnDenominator; AlphaT", 40, 0.0, 1.0);
  h_pfHtTurnOn_num = ibooker_.book1D("pfHtTurnOn_num", "PF HT Turn On Numerator; HT (GeV)", 30, 0.0, 1500.0);
  h_pfHtTurnOn_den = ibooker_.book1D("pfHtTurnOn_den", "PF HT Turn On Denominator; HT (GeV)", 30, 0.0, 1500.0);

  ibooker_.cd();
}

// define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_alphaT);
