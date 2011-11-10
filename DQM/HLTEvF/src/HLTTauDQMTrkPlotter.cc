#include "DQM/HLTEvF/interface/HLTTauDQMTrkPlotter.h"

HLTTauDQMTrkPlotter::HLTTauDQMTrkPlotter(const edm::ParameterSet& iConfig, int etbins, int etabins, int phibins, double maxpt, bool ref, double dr, std::string dqmBaseFolder ) {
    //Initialize Plotter
    name_ = "HLTTauDQMTrkPlotter";
    
    //Process PSet
    try {
        jetTagSrc_       = iConfig.getUntrackedParameter<edm::InputTag>("ConeIsolation");
        isolJets_        = iConfig.getUntrackedParameter<edm::InputTag>("IsolatedJets");
        triggerTag_      = iConfig.getUntrackedParameter<std::string>("DQMFolder");
        triggerTagAlias_ = iConfig.getUntrackedParameter<std::string>("Alias","");
        type_            = iConfig.getUntrackedParameter<std::string>("Type");
        mcMatch_         = dr;
        EtMax_           = maxpt;
        NPtBins_         = etbins;
        NEtaBins_        = etabins;
        NPhiBins_        = phibins;
        dqmBaseFolder_   = dqmBaseFolder;
        doRef_           = ref;
        validity_        = true;
    } catch ( cms::Exception &e ) {
        edm::LogWarning("HLTTauDQMTrkPlotter::HLTTauDQMTrkPlotter") << e.what() << std::endl;
        validity_ = false;
        return;
    }
    
    if (store_) {
        //Create the histograms
        store_->setCurrentFolder(triggerTag());
        store_->removeContents();
        
        jetEt = store_->book1D((type_+"TauEt").c_str(), "#tau E_{T}",NPtBins_,0,EtMax_);
        jetEta = store_->book1D((type_+"TauEta").c_str(), "#tau #eta", NEtaBins_, -2.5, 2.5);
        jetPhi = store_->book1D((type_+"TauPhi").c_str(), "#tau #phi", NPhiBins_, -3.2, 3.2);
        isoJetEt = store_->book1D((type_+"IsolJetEt").c_str(), "Selected Jet E_{T}", NPtBins_, 0,EtMax_);
        isoJetEta = store_->book1D((type_+"IsolJetEta").c_str(), "Selected Jet #eta", NEtaBins_, -2.5, 2.5);
        isoJetPhi = store_->book1D((type_+"IsolJetPhi").c_str(), "Selected jet #phi", NPhiBins_, -3.2, 3.2);
        
        nPxlTrksInL25Jet = store_->book1D((type_+"nTracks").c_str(), "# RECO Tracks", 30, 0, 30);
        nQPxlTrksInL25Jet = store_->book1D((type_+"nQTracks").c_str(),"# Quality RECO Tracks", 15, 0, 15);
        signalLeadTrkPt = store_->book1D((type_+"LeadTrackPt").c_str(), "Lead Track p_{T}", 75, 0, 150);
        hasLeadTrack = store_->book1D((type_+"HasLeadTrack").c_str(), "Lead Track ?", 2, 0, 2);
        
        EtEffNum = store_->book1D((type_+"TauEtEffNum").c_str(),"Efficiency vs E_{T} (Numerator)",NPtBins_,0,EtMax_);
        EtEffNum->getTH1F()->Sumw2();
        
        EtEffDenom = store_->book1D((type_+"TauEtEffDenom").c_str(),"Efficiency vs E_{T} (Denominator)",NPtBins_,0,EtMax_);
        EtEffDenom->getTH1F()->Sumw2();
        
        EtaEffNum = store_->book1D((type_+"TauEtaEffNum").c_str(),"Efficiency vs #eta (Numerator)",NEtaBins_,-2.5,2.5);
        EtaEffNum->getTH1F()->Sumw2();
        
        EtaEffDenom = store_->book1D((type_+"TauEtaEffDenom").c_str(),"Efficiency vs #eta (Denominator)",NEtaBins_,-2.5,2.5);
        EtaEffDenom->getTH1F()->Sumw2();
        
        PhiEffNum = store_->book1D((type_+"TauPhiEffNum").c_str(),"Efficiency vs #phi (Numerator)",NPhiBins_,-3.2,3.2);
        PhiEffNum->getTH1F()->Sumw2();
        
        PhiEffDenom = store_->book1D((type_+"TauPhiEffDenom").c_str(),"Efficiency vs #phi (Denominator)",NPhiBins_,-3.2,3.2);
        PhiEffDenom->getTH1F()->Sumw2();
    }
}


HLTTauDQMTrkPlotter::~HLTTauDQMTrkPlotter() {
}

void HLTTauDQMTrkPlotter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::map<int,LVColl>& mcInfo ) {
    using namespace edm;
    using namespace reco;
    
    //Tau reference
    std::map<int,LVColl>::const_iterator iref;
    iref = mcInfo.find(15);
    
    Handle<IsolatedTauTagInfoCollection> tauTagInfos;
    Handle<CaloJetCollection> isolJets;			   
    
    bool gotL3 = iEvent.getByLabel(jetTagSrc_, tauTagInfos) && tauTagInfos.isValid();
    
    if ( gotL3 ) {
        for ( unsigned int i = 0; i < tauTagInfos->size(); ++i ) {
            IsolatedTauTagInfo tauTagInfo = (*tauTagInfos)[i];
            if ( &(*tauTagInfo.jet()) ) {
                LV theJet = tauTagInfo.jet()->p4();  		         

                std::pair<bool,LV> m(false,LV());
                if ( iref != mcInfo.end() ) m = match(theJet,iref->second,mcMatch_);
                                
                if ( (doRef_ && m.first) || (!doRef_) ) {
                    jetEt->Fill(theJet.Et()); 		  							         
                    jetEta->Fill(theJet.Eta());		  						         
                    jetPhi->Fill(theJet.Phi());		  						         
                    nPxlTrksInL25Jet->Fill(tauTagInfo.allTracks().size());								    
                    nQPxlTrksInL25Jet->Fill(tauTagInfo.selectedTracks().size());							    
                    
                    const TrackRef leadTrk = tauTagInfo.leadingSignalTrack();
                    if ( !leadTrk ) { 
                        hasLeadTrack->Fill(0.5);
                    } else {
                        hasLeadTrack->Fill(1.5);
                        signalLeadTrkPt->Fill(leadTrk->pt());				 
                    }
                    
                    LV refV;
                    if ( doRef_ ) {
                        refV = m.second;   
                    } else {
                        refV = theJet;   
                    } 
                    
                    EtEffDenom->Fill(refV.pt());
                    EtaEffDenom->Fill(refV.eta());
                    PhiEffDenom->Fill(refV.phi());
                    
                    bool gotIsoL3 = iEvent.getByLabel(isolJets_, isolJets) && isolJets.isValid();
                    
                    if ( gotIsoL3 ) {
                        if ( matchJet(*(tauTagInfo.jet()),*isolJets) ) {
                            isoJetEta->Fill(theJet.Eta());
                            isoJetEt->Fill(theJet.Et());
                            isoJetPhi->Fill(theJet.Phi());
                            
                            EtEffNum->Fill(refV.pt());
                            EtaEffNum->Fill(refV.eta());
                            PhiEffNum->Fill(refV.phi());
                        }
                    }
                }
            }
        }
    }
}

bool HLTTauDQMTrkPlotter::matchJet(const reco::Jet& jet,const reco::CaloJetCollection& McInfo) {
    //Loop On the Collection and see if your tau jet is matched to one there
    //Also find the nearest Matched MC Particle to your Jet (to be complete)
    
    bool matched = false;
    
    for ( reco::CaloJetCollection::const_iterator it = McInfo.begin(); it != McInfo.end(); ++it ) {
        double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),it->p4().Vect());
        if ( delta < mcMatch_ ) {
            matched = true;
        }
    }
    return matched;
}
