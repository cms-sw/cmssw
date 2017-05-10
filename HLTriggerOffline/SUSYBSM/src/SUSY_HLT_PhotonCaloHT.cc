#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_PhotonCaloHT.h"


SUSY_HLT_PhotonCaloHT::SUSY_HLT_PhotonCaloHT(const edm::ParameterSet& ps)
{
  edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "Constructor SUSY_HLT_PhotonCaloHT::SUSY_HLT_PhotonCaloHT " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  thePhotonCollection_ = consumes<reco::PhotonCollection>(ps.getParameter<edm::InputTag>("photonCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilterPhoton_ = ps.getParameter<edm::InputTag>("TriggerFilterPhoton");
  triggerFilterHt_ = ps.getParameter<edm::InputTag>("TriggerFilterHt");
  ptThrOffline_ = ps.getUntrackedParameter<double>("ptThrOffline");
  htThrOffline_ = ps.getUntrackedParameter<double>("htThrOffline");
}

SUSY_HLT_PhotonCaloHT::~SUSY_HLT_PhotonCaloHT()
{
   edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "Destructor SUSY_HLT_PhotonCaloHT::~SUSY_HLT_PhotonCaloHT " << std::endl;
}

void SUSY_HLT_PhotonCaloHT::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "SUSY_HLT_PhotonCaloHT::beginRun" << std::endl;
}

 void SUSY_HLT_PhotonCaloHT::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "SUSY_HLT_PhotonCaloHT::bookHistograms" << std::endl;
  //book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_PhotonCaloHT::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
  edm::EventSetup const& context)
{
   edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "SUSY_HLT_PhotonCaloHT::beginLuminosityBlock" << std::endl;
}

void SUSY_HLT_PhotonCaloHT::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "SUSY_HLT_PhotonCaloHT::analyze" << std::endl;

  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::CaloJetCollection> CaloJetCollection;
  e.getByToken (theCaloJetCollection_,CaloJetCollection);
  if ( !CaloJetCollection.isValid() ){
    edm::LogError ("SUSY_HLT_PhotonCaloHT") << "invalid collection: CaloJets" << "\n";
    return;
  }
 
 
  //-------------------------------
  //--- Photon
  //-------------------------------
  edm::Handle<reco::PhotonCollection> photonCollection;
  e.getByToken (thePhotonCollection_, photonCollection);
  if ( !photonCollection.isValid() ){
    edm::LogError ("SUSY_HLT_PhotonCaloHT") << "invalid egamma collection" << "\n";
    return;
  }


  //check what is in the menu
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  if(!hltresults.isValid()){
    edm::LogError ("SUSY_HLT_PhotonCaloHT") << "invalid collection: TriggerResults" << "\n";
    return;
  }

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()) {
    edm::LogError ("SUSY_HLT_PhotonCaloHT") << "invalid collection: TriggerSummary" << "\n";
    return;
  }

  // get online objects
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();

  // get the photon object
  size_t filterIndexPhoton = triggerSummary->filterIndex( triggerFilterPhoton_ );
  if( filterIndexPhoton < triggerSummary->sizeFilters() ) {
      const trigger::Keys& keys = triggerSummary->filterKeys( filterIndexPhoton );
      if( keys.size() ) {
          // take the leading photon
          float pt = triggerObjects[ keys[0] ].pt();
          h_photonPt->Fill( pt );
      }
  }

  // get ht
  size_t filterIndexHt = triggerSummary->filterIndex( triggerFilterHt_ );
  if( filterIndexHt < triggerSummary->sizeFilters() ) {
      const trigger::Keys& keys = triggerSummary->filterKeys( filterIndexHt );
      if( keys.size() ) {
          float ht = triggerObjects[ keys[0] ].pt();
          h_ht->Fill( ht );
      }
  }

  bool hasFired = false, hasFiredAuxiliaryForHadronicLeg=false;
  const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ){
    if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFired = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForHadronic_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFiredAuxiliaryForHadronicLeg = true;
  }


  if((hasFiredAuxiliaryForHadronicLeg  || !e.isRealData())) {
    float recoHt = 0;
    for (reco::CaloJetCollection::const_iterator i_calojet = CaloJetCollection->begin(); i_calojet != CaloJetCollection->end(); ++i_calojet){
      if (i_calojet->pt() < 30) continue;
      if (fabs(i_calojet->eta()) > 3.0) continue;
      recoHt += i_calojet->pt();
    }

    float recoPhotonPt = photonCollection->size() ? photonCollection->begin()->et() : 0;
    if(hasFired){
      if( photonCollection->size() && recoHt >= htThrOffline_ ) h_photonTurnOn_num -> Fill( recoPhotonPt );
      if( CaloJetCollection->size() && recoPhotonPt >= ptThrOffline_ ) h_htTurnOn_num -> Fill( recoHt );
    }
    if( photonCollection->size() && recoHt >= htThrOffline_ ) h_photonTurnOn_den -> Fill( recoPhotonPt );
    if( CaloJetCollection->size() && recoPhotonPt >= ptThrOffline_ ) h_htTurnOn_den -> Fill( recoHt );
  }

}

void SUSY_HLT_PhotonCaloHT::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "SUSY_HLT_PhotonCaloHT::endLuminosityBlock" << std::endl;
}


void SUSY_HLT_PhotonCaloHT::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_PhotonCaloHT") << "SUSY_HLT_PhotonCaloHT::endRun" << std::endl;
}

void SUSY_HLT_PhotonCaloHT::bookHistos(DQMStore::IBooker & ibooker_)
{
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  //offline quantities
  h_photonPt = ibooker_.book1D("photonPt", "Photon transverse momentum; p_{T} (GeV)", 50, 0, 800 );
  h_ht = ibooker_.book1D("ht", "Hadronic activity;H_{T} (GeV)", 20, 0, 2000 );
  h_htTurnOn_num = ibooker_.book1D("HtTurnOn_num", "Calo HT Turn On Numerator",   20, 0, 800 );
  h_htTurnOn_den = ibooker_.book1D("HtTurnOn_den", "Calo HT Turn On Denominator", 20, 0, 800 );
  h_photonTurnOn_num = ibooker_.book1D("photonTurnOn_num", "Photon Turn On Numerator",   20, 0, 800 );
  h_photonTurnOn_den = ibooker_.book1D("photonTurnOn_den", "Photon Turn On Denominator", 20, 0, 800 );

  ibooker_.cd();
}

 //define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_PhotonCaloHT);
