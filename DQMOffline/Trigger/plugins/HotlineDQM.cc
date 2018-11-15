#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DQMOffline/Trigger/interface/HotlineDQM.h"

HotlineDQM::HotlineDQM(const edm::ParameterSet& ps)
{

  edm::LogInfo("HotlineDQM") << "Constructor HotlineDQM::HotlineDQM " << std::endl;
  // Get parameters from configuration file
  theMuonCollection_ = consumes<reco::MuonCollection>(ps.getParameter<edm::InputTag>("muonCollection")); 
  thePfMETCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMetCollection"));
  theMETCollection_ = consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("caloMetCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  thePhotonCollection_ = consumes<reco::PhotonCollection>(ps.getParameter<edm::InputTag>("photonCollection"));
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("triggerResults"));
  triggerPath_ = ps.getParameter<std::string>("triggerPath");
  triggerFilter_ = ps.getParameter<edm::InputTag>("triggerFilter");
  useMuons = ps.getParameter<bool>("useMuons");
  useMet = ps.getParameter<bool>("useMet");
  usePFMet = ps.getParameter<bool>("usePFMet");
  useHT = ps.getParameter<bool>("useHT");
  usePhotons = ps.getParameter<bool>("usePhotons");
}

HotlineDQM::~HotlineDQM()
{
   edm::LogInfo("HotlineDQM") << "Destructor HotlineDQM::~HotlineDQM " << std::endl;
}

void HotlineDQM::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
    edm::LogInfo("HotlineDQM") << "HotlineDQM::bookHistograms" << std::endl;

    ibooker_.cd();
    ibooker_.setCurrentFolder("HLT/Hotline/" + triggerPath_);

    //online quantities 
    h_MuPt = ibooker_.book1D("MuPt", "Muon Pt; GeV", 20, 0.0, 2000.0);
    h_PhotonPt = ibooker_.book1D("PhotonPt", "Photon Pt; GeV", 20, 0.0, 4000.0);
    h_HT = ibooker_.book1D("HT", "HT; GeV", 20, 0.0, 6000.0);
    h_MetPt = ibooker_.book1D("MetPt", "Calo MET; GeV", 20, 0.0, 2000);
    h_PFMetPt = ibooker_.book1D("PFMetPt", "PF MET; GeV", 20, 0.0, 2000);

    if(useMuons) h_OnlineMuPt = ibooker_.book1D("OnlineMuPt", "Online Muon Pt; GeV", 20, 0.0, 2000.0);
    if(usePhotons) h_OnlinePhotonPt = ibooker_.book1D("OnlinePhotonPt", "Online Photon Pt; GeV", 20, 0.0, 4000.0);
    if(useHT) h_OnlineHT = ibooker_.book1D("OnlineHT", "Online HT; GeV", 20, 0.0, 6000.0);
    if(useMet) h_OnlineMetPt = ibooker_.book1D("OnlineMetPt", "Online Calo MET; GeV", 20, 0.0, 2000);
    if(usePFMet) h_OnlinePFMetPt = ibooker_.book1D("OnlinePFMetPt", "Online PF MET; GeV", 20, 0.0, 2000);

    ibooker_.cd();
}

void HotlineDQM::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
    edm::LogInfo("HotlineDQM") << "HotlineDQM::analyze" << std::endl;

    //-------------------------------
    //--- MET
    //-------------------------------
    edm::Handle<reco::PFMETCollection> pfMETCollection;
    e.getByToken(thePfMETCollection_, pfMETCollection);
    if ( !pfMETCollection.isValid() ){
        edm::LogError ("HotlineDQM") << "invalid collection: PFMET" << "\n";
        return;
    }
    edm::Handle<reco::CaloMETCollection> caloMETCollection;
    e.getByToken(theMETCollection_, caloMETCollection);
    if ( !caloMETCollection.isValid() ){
        edm::LogError ("HotlineDQM") << "invalid collection: CaloMET" << "\n";
        return;
    }

    //-------------------------------
    //--- Jets
    //-------------------------------
    edm::Handle<reco::CaloJetCollection> caloJetCollection;
    e.getByToken (theCaloJetCollection_,caloJetCollection);
    if ( !caloJetCollection.isValid() ){
        edm::LogError ("HotlineDQM") << "invalid collection: CaloJets" << "\n";
        return;
    }

    //-------------------------------
    //--- Muon
    //-------------------------------
    edm::Handle<reco::MuonCollection> MuonCollection;
    e.getByToken (theMuonCollection_, MuonCollection);
    if ( !MuonCollection.isValid() ){
        edm::LogError ("HotlineDQM") << "invalid collection: Muons " << "\n";
        return;
    }

    //-------------------------------
    //--- Photon 
    //-------------------------------
    edm::Handle<reco::PhotonCollection> PhotonCollection;
    e.getByToken (thePhotonCollection_, PhotonCollection);
    if ( !PhotonCollection.isValid() ){
        edm::LogError ("HotlineDQM") << "invalid collection: Photons " << "\n";
        return;
    }

    //-------------------------------
    //--- Trigger
    //-------------------------------
    edm::Handle<edm::TriggerResults> hltresults;
    e.getByToken(triggerResults_,hltresults);
    if(!hltresults.isValid()){
        edm::LogError ("HotlineDQM") << "invalid collection: TriggerResults" << "\n";
        return;
    }
    edm::Handle<trigger::TriggerEvent> triggerSummary;
    e.getByToken(theTrigSummary_, triggerSummary);
    if(!triggerSummary.isValid()) {
        edm::LogError ("HotlineDQM") << "invalid collection: TriggerSummary" << "\n";
        return;
    }

    bool hasFired = false;
    const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
    unsigned int numTriggers = trigNames.size();
    for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ){
        if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)){
            hasFired = true;
        }
    }

    //get online objects
    float ptMuon=-1, ptPhoton=-1, met=-1, pfMet=-1, ht = 0;
    size_t filterIndex = triggerSummary->filterIndex( triggerFilter_ );
    trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
    if( !(filterIndex >= triggerSummary->sizeFilters()) ){
        const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex );
        for(unsigned short key : keys){
            trigger::TriggerObject foundObject = triggerObjects[key];
            if(useMuons && fabs(foundObject.id()) == 13){ //muon
                if(foundObject.pt() > ptMuon) ptMuon = foundObject.pt();
            }
            else if(usePhotons && fabs(foundObject.id()) == 0){ //photon
                if(foundObject.pt() > ptPhoton) ptPhoton = foundObject.pt();
            }
            else if(useMet && fabs(foundObject.id()) == 0){ //MET
                met = foundObject.pt(); 
            }
            else if(usePFMet && fabs(foundObject.id()) == 0){ //PFMET
                pfMet = foundObject.pt();
            }
            else if(useHT && fabs(foundObject.id()) == 89){ //HT
                ht = foundObject.pt();
            }
        }
    }

    if(hasFired) {
        //fill appropriate online histogram
        if(useMuons) h_OnlineMuPt->Fill(ptMuon);
        if(usePhotons) h_OnlinePhotonPt->Fill(ptPhoton);
        if(useMet) h_OnlineMetPt->Fill(met);
        if(usePFMet) h_OnlinePFMetPt->Fill(pfMet);
        if(useHT) h_OnlineHT->Fill(ht);

        //fill muon pt histogram
        if(!MuonCollection->empty()){
            float maxMuPt = -1.0;
            for(auto &mu : *MuonCollection){
                if(mu.pt() > maxMuPt) maxMuPt = mu.pt();
            }
            h_MuPt->Fill(maxMuPt);
        }

        //fill photon pt histogram
        if(!PhotonCollection->empty()){
            float maxPhoPt = -1.0;
            for(auto &pho : *PhotonCollection){
                if(pho.pt() > maxPhoPt) maxPhoPt = pho.pt();
            }
            h_PhotonPt->Fill(maxPhoPt);
        }

        //fill HT histogram
        float caloHT = 0.0;
        for (auto const & i_calojet : *caloJetCollection){
            if (i_calojet.pt() < 40) continue;
            if (fabs(i_calojet.eta()) > 3.0) continue;
            caloHT += i_calojet.pt();
        }
        h_HT->Fill(caloHT);

        //fill CaloMET histogram
        h_MetPt->Fill(caloMETCollection->front().et()); 

        //fill PFMET histogram
        h_PFMetPt->Fill(pfMETCollection->front().et());
    }
}

void HotlineDQM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("photonCollection", edm::InputTag("photons"));
  desc.add<edm::InputTag>("muonCollection", edm::InputTag("muons"));
  desc.add<edm::InputTag>("caloJetCollection", edm::InputTag("ak4CaloJets"));
  desc.add<edm::InputTag>("pfMetCollection", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("caloMetCollection", edm::InputTag("caloMet"));
  desc.add<edm::InputTag>("triggerResults",edm::InputTag("TriggerResults","","HLT"));
  desc.add<edm::InputTag>("trigSummary",edm::InputTag("hltTriggerSummaryAOD"));
  desc.add<std::string>("triggerPath","HLT_HT2000_v")->setComment("trigger path name");
  desc.add<edm::InputTag>("triggerFilter",edm::InputTag("hltHt2000","","HLT"))->setComment("name of the last filter in the path");
  desc.add<bool>("useMuons", false);
  desc.add<bool>("usePhotons", false);
  desc.add<bool>("useMet", false);
  desc.add<bool>("usePFMet", false);
  desc.add<bool>("useHT", false);
  descriptions.add("HotlineDQM",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HotlineDQM);
