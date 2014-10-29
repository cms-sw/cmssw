#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_alphaT.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > LorentzV;

SUSY_HLT_alphaT::SUSY_HLT_alphaT(const edm::ParameterSet& ps)
{
  edm::LogInfo("SUSY_HLT_alphaT") << "Constructor SUSY_HLT_alphaT::SUSY_HLT_alphaT " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  //thePfMETCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  //theCaloMETCollection_ = consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("caloMETCollection"));
  //thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  //triggerPathAuxiliaryForMuon_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForMuon");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
  alphaTThrTurnon_ = ps.getUntrackedParameter<double>("alphaTThrTurnon");
  htThrTurnon_ = ps.getUntrackedParameter<double>("htThrTurnon");
}

SUSY_HLT_alphaT::~SUSY_HLT_alphaT()
{
   edm::LogInfo("SUSY_HLT_alphaT") << "Destructor SUSY_HLT_alphaT::~SUSY_HLT_alphaT " << std::endl;
}

void SUSY_HLT_alphaT::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e)
{
 
  bool changed;
  
  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_alphaT") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  bool pathFound = false;
  const std::vector<std::string> allTrigNames = fHltConfig.triggerNames();
  for(size_t j = 0; j <allTrigNames.size(); ++j) {
    if(allTrigNames[j].find(triggerPath_) != std::string::npos) {
      pathFound = true;
    }
  }

  if(!pathFound) {
    edm::LogError ("SUSY_HLT_alphaT") << "Path not found" << "\n";
    return;
  }
  //std::vector<std::string> filtertags = fHltConfig.moduleLabels( triggerPath_ );
  //triggerFilter_ = edm::InputTag(filtertags[filtertags.size()-1],"",fHltConfig.processName());  
  //triggerFilter_ = edm::InputTag("hltPFMET120Mu5L3PreFiltered", "", fHltConfig.processName());

  edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::beginRun" << std::endl;
}

 void SUSY_HLT_alphaT::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::bookHistograms" << std::endl;
  //book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_alphaT::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
  edm::EventSetup const& context)
{
   edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::beginLuminosityBlock" << std::endl;
}



void SUSY_HLT_alphaT::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::analyze" << std::endl;


  //-------------------------------
  //--- MET
  //-------------------------------
 // edm::Handle<reco::PFMETCollection> pfMETCollection;
 // e.getByToken(thePfMETCollection_, pfMETCollection);
 // if ( !pfMETCollection.isValid() ){
 //   edm::LogError ("SUSY_HLT_alphaT") << "invalid collection: PFMET" << "\n";
 //  return;
 // }
 // edm::Handle<reco::CaloMETCollection> caloMETCollection;
 // e.getByToken(theCaloMETCollection_, caloMETCollection);
 // if ( !caloMETCollection.isValid() ){
 //   edm::LogError ("SUSY_HLT_alphaT") << "invalid collection: CaloMET" << "\n";
 //   return;
 // }
    

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  if(!hltresults.isValid()){
    edm::LogError ("SUSY_HLT_alphaT") << "invalid collection: TriggerResults" << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()) {
    edm::LogError ("SUSY_HLT_alphaT") << "invalid collection: TriggerSummary" << "\n";
    return;
  }

  //-------------------------------
  //--- Jets
  //-------------------------------
  //edm::Handle<reco::PFJetCollection> pfJetCollection;
  //e.getByToken (thePfJetCollection_,pfJetCollection);
  //if ( !pfJetCollection.isValid() ){
  //  edm::LogError ("SUSY_HLT_alphaT") << "invalid collection: PFJets" << "\n";
  //  return;
  //}
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken (theCaloJetCollection_,caloJetCollection);
  if ( !caloJetCollection.isValid() ){
      edm::LogError ("SUSY_HLT_alphaT") << "invalid collection: CaloJets" << "\n";
      return;
  }

  //get online objects
  //For now just get the jets and recalculate ht and alphaT
  size_t filterIndex = triggerSummary->filterIndex( triggerFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();

  double hltHt=0.;
  std::vector<LorentzV> hltJets;
  if( !(filterIndex >= triggerSummary->sizeFilters()) ){
      const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex );

      for( size_t j = 0; j < keys.size(); ++j ){
          trigger::TriggerObject foundObject = triggerObjects[keys[j]];

          //  if(foundObject.id() == 85){ //It's a jet 
          if(foundObject.pt()>ptThrJet_ && fabs(foundObject.eta()) < etaThrJet_){
              hltHt += foundObject.pt();
              LorentzV JetLVec(foundObject.pt(),foundObject.eta(),foundObject.phi(),foundObject.mass());
              hltJets.push_back(JetLVec);
          }
          //   }
      }
  }

  //Fill the alphaT and HT histograms
  if(hltJets.size()>0){
      double hltAlphaT = AlphaT(hltJets).value();
      h_triggerAlphaT->Fill(hltAlphaT);
      h_triggerHt->Fill(hltHt);
      h_triggerAlphaT_triggerHt->Fill(hltHt, hltAlphaT);
  }

  bool hasFired = false;
  bool hasFiredAuxiliaryForHadronicLeg = false;
  const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ){
      if (trigNames.triggerName(hltIndex)==triggerPath_ && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFired = true;
      if (trigNames.triggerName(hltIndex)==triggerPathAuxiliaryForHadronic_ && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFiredAuxiliaryForHadronicLeg = true;

  }

  if(hasFiredAuxiliaryForHadronicLeg) {

      float caloHT = 0.0;
      //float pfHT = 0.0;
      //for (reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end(); ++i_pfjet){
      //  if (i_pfjet->pt() < ptThrJet_) continue;
      //  if (fabs(i_pfjet->eta()) > etaThrJet_) continue;
      //  pfHT += i_pfjet->pt();
      //}

      //Make the gen Calo HT and AlphaT
      std::vector<LorentzV> jets;
      for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); ++i_calojet){
          if (i_calojet->pt() < ptThrJet_) continue;
          if (fabs(i_calojet->eta()) > etaThrJet_) continue;
          caloHT += i_calojet->pt();
          LorentzV JetLVec(i_calojet->pt(),i_calojet->eta(),i_calojet->phi(),i_calojet->mass());
          jets.push_back(JetLVec);
      }

      //AlphaT aT = AlphaT(jets);
      double caloAlphaT = AlphaT(jets).value();

      //Fill the turnons
      if(hasFired) {
          if(caloHT>htThrTurnon_) h_alphaTTurnOn_num-> Fill(caloAlphaT);
          if(caloAlphaT>alphaTThrTurnon_) h_htTurnOn_num-> Fill(caloHT);
      } 
      if(caloHT>htThrTurnon_) h_alphaTTurnOn_den-> Fill(caloAlphaT);
      if(caloAlphaT>alphaTThrTurnon_) h_htTurnOn_den-> Fill(caloHT);
  }
}


void SUSY_HLT_alphaT::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
    edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::endLuminosityBlock" << std::endl;
}


void SUSY_HLT_alphaT::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
    edm::LogInfo("SUSY_HLT_alphaT") << "SUSY_HLT_alphaT::endRun" << std::endl;
}

void SUSY_HLT_alphaT::bookHistos(DQMStore::IBooker & ibooker_)
{
    ibooker_.cd();
    ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

    //offline quantities

    //online quantities 
    h_triggerHt = ibooker_.book1D("triggerHt", "Trigger Ht; HT (GeV)", 60, 0.0, 1500.0);
    h_triggerAlphaT = ibooker_.book1D("triggerAlphaT", "Trigger AlphaT; AlphaT", 80, 0., 1.0);
    h_triggerAlphaT_triggerHt = ibooker_.book2D("triggerAlphaT_triggerHt","Trigger HT vs Trigger AlphaT; HT (GeV); AlphaT", 60,0.0,1500.,80,0.,1.0);
    
    //h_triggerMht = ibooker_.book1D("triggerMht", "Trigger Mht", 20, -3.5, 3.5);

    //num and den hists to be divided in harvesting step to make turn on curves
    h_alphaTTurnOn_num = ibooker_.book1D("alphaTTurnOn_num", "AlphaT Turn On Numerator; AlphaT", 40, 0.0, 1.0 );
    h_alphaTTurnOn_den = ibooker_.book1D("alphaTTurnOn_den", "AlphaT Turn OnDenominator; AlphaT", 40, 0.0, 1.0 );
    h_htTurnOn_num = ibooker_.book1D("htTurnOn_num", "HT Turn On Numerator; HT (GeV)", 30, 0.0, 1500.0 );
    h_htTurnOn_den = ibooker_.book1D("htTurnOn_den", "HT Turn On Denominator; HT (GeV)", 30, 0.0, 1500.0 );

    ibooker_.cd();
}

//define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_alphaT);
