#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_InclusiveCaloHT.h"


SUSY_HLT_InclusiveCaloHT::SUSY_HLT_InclusiveCaloHT(const edm::ParameterSet& ps)
{
  edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "Constructor SUSY_HLT_InclusiveCaloHT::SUSY_HLT_InclusiveCaloHT " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  theCaloMETCollection_ = consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("caloMETCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
}

SUSY_HLT_InclusiveCaloHT::~SUSY_HLT_InclusiveCaloHT()
{
   edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "Destructor SUSY_HLT_InclusiveCaloHT::~SUSY_HLT_InclusiveCaloHT " << std::endl;
}

void SUSY_HLT_InclusiveCaloHT::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "SUSY_HLT_InclusiveCaloHT::beginRun" << std::endl;
}

 void SUSY_HLT_InclusiveCaloHT::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "SUSY_HLT_InclusiveCaloHT::bookHistograms" << std::endl;
  //book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_InclusiveCaloHT::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
  edm::EventSetup const& context)
{
   edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "SUSY_HLT_InclusiveCaloHT::beginLuminosityBlock" << std::endl;
}

void SUSY_HLT_InclusiveCaloHT::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "SUSY_HLT_InclusiveCaloHT::analyze" << std::endl;


  //-------------------------------
  //--- MET
  //-------------------------------
  edm::Handle<reco::CaloMETCollection> CaloMETCollection;
  e.getByToken(theCaloMETCollection_, CaloMETCollection);
  if ( !CaloMETCollection.isValid() ){
    edm::LogError ("SUSY_HLT_InclusiveCaloHT") << "invalid collection: CaloMET" << "\n";
   return;
  }
  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken (theCaloJetCollection_,caloJetCollection);
  if ( !caloJetCollection.isValid() ){
  edm::LogError ("SUSY_HLT_InclusiveCaloHT") << "invalid collection: CaloJets" << "\n";
  return;
  }

  //check what is in the menu
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  if(!hltresults.isValid()){
    edm::LogError ("SUSY_HLT_InclusiveCaloHT") << "invalid collection: TriggerResults" << "\n";
    return;
  }
  
  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()) {
    edm::LogError ("SUSY_HLT_InclusiveCaloHT") << "invalid collection: TriggerSummary" << "\n";
    return;
  }

  //get online objects
  /*
  size_t filterIndex = triggerSummary->filterIndex( triggerFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
  if( !(filterIndex >= triggerSummary->sizeFilters()) ){
      const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex );
      for( size_t j = 0; j < keys.size(); ++j ){
        trigger::TriggerObject foundObject = triggerObjects[keys[j]];
        //if(foundObject.id() == 85 && foundObject.pt() > 40.0 && fabs(foundObject.eta()) < 3.0){
        //  h_triggerJetPt->Fill(foundObject.pt());
        //  h_triggerJetEta->Fill(foundObject.eta());
        //  h_triggerJetPhi->Fill(foundObject.phi());
        //}
        //if(foundObject.id() == 87){
        //  h_triggerMetPt->Fill(foundObject.pt());
          h_triggerMetPhi->Fill(foundObject.phi());
        }
        if(foundObject.id() == 89){
          h_triggerHT->Fill(foundObject.pt());
        }
      }
    }
  */

  bool hasFired = false, hasFiredAuxiliaryForHadronicLeg=false;
  const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ){
    if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFired = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForHadronic_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFiredAuxiliaryForHadronicLeg = true;
  }

  if(hasFiredAuxiliaryForHadronicLeg || !e.isRealData()) {
    float caloHT = 0.0;
    float caloMET = CaloMETCollection->begin()->et();
    for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); ++i_calojet){
        if (i_calojet->pt() < ptThrJet_) continue;
        caloHT += i_calojet->pt();
    }
    std::cout << "Value: " << caloHT << std::endl;
    if(hasFired){
      for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); ++i_calojet){
        if (i_calojet->pt() < ptThrJet_) continue;
        if (fabs(i_calojet->eta()) > etaThrJet_) continue;
        h_caloJetPt ->Fill(i_calojet->pt());
        h_caloJetEta ->Fill(i_calojet->eta());
        h_caloJetPhi ->Fill(i_calojet->phi());
      }
      h_caloMet -> Fill(CaloMETCollection->begin()->et());
      h_caloHT -> Fill(caloHT);

      if(caloHT > 250) h_caloMetTurnOn_num -> Fill(caloMET);
      if(caloMET > 70) h_caloHTTurnOn_num -> Fill(caloHT);
    }
    //fill denominator histograms for all events, used for turn on curves
    if(caloHT > 250) h_caloMetTurnOn_den -> Fill(caloMET);
    if(caloMET > 70) h_caloHTTurnOn_den -> Fill(caloHT);
  }
}


void SUSY_HLT_InclusiveCaloHT::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "SUSY_HLT_InclusiveCaloHT::endLuminosityBlock" << std::endl;
}


void SUSY_HLT_InclusiveCaloHT::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_InclusiveCaloHT") << "SUSY_HLT_InclusiveCaloHT::endRun" << std::endl;
}

void SUSY_HLT_InclusiveCaloHT::bookHistos(DQMStore::IBooker & ibooker_)
{
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  //offline quantities
  h_caloMet = ibooker_.book1D("caloMet", "Calo Missing E_{T}; GeV", 20, 0.0, 300.0 );
  h_caloHT = ibooker_.book1D("caloHT", "Calo H_{T}; GeV", 30, 0.0, 1000.0);
  h_caloJetPt = ibooker_.book1D("caloJetPt", "CaloJet P_{T}; GeV", 20, 0.0, 300.0 );
  h_caloJetEta = ibooker_.book1D("caloJetEta", "CaloJet Eta", 20, -3.0, 3.0 );
  h_caloJetPhi = ibooker_.book1D("caloJetPhi", "CaloJet Phi", 20, -3.5, 3.5 );


  //num and den hists to be divided in harvesting step to make turn on curves
  h_caloMetTurnOn_num = ibooker_.book1D("caloMetTurnOn_num", "Calo MET Turn On Numerator", 20, 0.0, 300.0 );
  h_caloMetTurnOn_den = ibooker_.book1D("caloMetTurnOn_den", "Calo MET Turn OnDenominator", 20, 0.0, 300.0 );
  h_caloHTTurnOn_num = ibooker_.book1D("caloHTTurnOn_num", "Calo HT Turn On Numerator", 30, 0.0, 1000.0 );
  h_caloHTTurnOn_den = ibooker_.book1D("caloHTTurnOn_den", "Calo HT Turn On Denominator", 30, 0.0, 1000.0 );

  ibooker_.cd();
}

 //define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_InclusiveCaloHT);
