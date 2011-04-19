// -*- C++ -*-
//
// Package:    AnalyseDisplacedJetTrigger
// Class:      AnalyseDisplacedJetTrigger
// 
/**\class AnalyseDisplacedJetTrigger AnalyseDisplacedJetTrigger.cc
 Description: EDAnalyzer to analyze Displaced Jet Exotica Trigger.
*/
//
// Original Author:  Ian Tomalin
//

#include "HLTriggerOffline/SUSYBSM/interface/AnalyseDisplacedJetTrigger.h"
#include "HLTriggerOffline/SUSYBSM/interface/GoodJetDisplacedJetTrigger.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <exception>

AnalyseDisplacedJetTrigger::AnalyseDisplacedJetTrigger(const edm::ParameterSet& iConfig)
 : 
   nPromptTkMax_(iConfig.getParameter<double> ("nPromptTkMax")),
   ptMin_(iConfig.getParameter<double> ("ptMin")),
   etaMax_(iConfig.getParameter<double> ("etaMax"))

   //  datasetNameString_(iConfig.getParameter< vector<string> > ("datasetName")),
   //  dataPeriod_(iConfig.getParameter<int> ("dataPeriod")),
{  
  // Define offline cuts used to select good jets.
  GoodJetDisplacedJetTrigger::setCuts(nPromptTkMax_, ptMin_, etaMax_);

  // Book some histos.
  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogWarning("AnalyseDisplacedJetTrigger") << "ERROR: unable to get DQMStore service?";
  }
    
  trigEffi_ = dbe_->bookProfile("trigEffi","Trigger efficiency",50,0.5,50.5,-9.9,9.9);
}

void AnalyseDisplacedJetTrigger::beginRun(const edm::Run& run, const edm::EventSetup& c) {
}

void AnalyseDisplacedJetTrigger::endJob() {

  // Set bin names on trigger efficiency histogram
  map<string, int>::const_iterator iter;
  for (iter = trigNameList_.begin(); iter != trigNameList_.end(); iter++) {
    trigEffi_->setBinLabel(iter->second, iter->first);            
  }
}

void AnalyseDisplacedJetTrigger::bookHistos(string trigName) {
  //=========================================================
  // Book histograms to study performance of given trigger.
  //=========================================================
  dbe_->setCurrentFolder(trigName);

  //
  // Event related variables.
  //

  histos_[trigName].nPV_ = dbe_->book1D("nPV","Number of primary vertices",30,-0.5,29.5);
  histos_[trigName].PVz_ = dbe_->book1D("PVz","primary vertex z",60,-30.0,30.0);

  // Ditto if displaced jet trigger fired.
  histos_[trigName].nPVPassed_ = dbe_->book1D("nPVPassed","Number of primary vertices when displaced jet trigger fired",30,-0.5,29.5);
  histos_[trigName].PVzPassed_ = dbe_->book1D("PVzPassed","primary vertex z when displaced jet trigger fired",60,-30.0,30.0);

  //
  // Jet related variables
  //

  // If MC truth is available, this is the true production radius of the jet.
  histos_[trigName].trueJetProdRadius_ = dbe_->book1D("trueJetProdRadius","true production radius of jet (cm)",202,-1.01,99.99);
  // If MC truth is available, this is the number of true displaced jets in the event.
  histos_[trigName].trueNumDispJets_ = dbe_->book1D("trueNumDispJets","Number of true displaced jets per event",20,-0.5,19.5);
  // Histograms of offline recoJets that are useful for displaced exotica jet search.
  histos_[trigName].recoJetNpromptTk_ = dbe_->book1D("recoJetNpromptTk","recoJet prompt tracks",50,-0.5,49.5);
  histos_[trigName].recoJetPt_ = dbe_->book1D("recoJetPt","recoJet Pt",200,0.0,1000.);
  histos_[trigName].recoJetEta_ = dbe_->book1D("recoJetEta","recoJet Eta",30,0.0,3.0);
  histos_[trigName].recoJetEMfraction_ = dbe_->book1D("recoJetEMfraction","recoJet EM fraction",102,-0.01,1.01);
  histos_[trigName].recoJetHPDfraction_ = dbe_->book1D("recoJetHPDfraction","recoJet HPD fraction",102,-0.01,1.01);
  histos_[trigName].recoJetN90_ = dbe_->book1D("recoJetN90","recoJet N90",100,-0.5,99.5);

  // Ditto, but only if recoJet is matched to a trigJet found by displaced jet trigger.
  histos_[trigName].trueJetProdRadiusMatched_ = dbe_->book1D("trueJetProdRadiusMatched","true production radius of jet (cm) if found by trigger",202,-1.01,99.99);
  histos_[trigName].trueNumDispJetsMatched_ = dbe_->book1D("trueNumDispJetsMatched","Number of true displaced jets found by trigger per event",20,-0.5,19.5);
  histos_[trigName].recoJetNpromptTkMatched_ = dbe_->book1D("recoJetNpromptTkMatched","recoJet prompt tracks if found by trigger",50,-0.5,49.5);
  histos_[trigName].recoJetPtMatched_ = dbe_->book1D("recoJetPtMatched","recoJet Pt if found by trigger",200,0.0,1000.);
  histos_[trigName].recoJetEtaMatched_ = dbe_->book1D("recoJetEtaMatched","recoJet Eta if found by trigger",30,0.0,3.0);
  histos_[trigName].recoJetEMfractionMatched_ = dbe_->book1D("recoJetEMfractionMatched","recoJet EM fraction if found by trigger",102,-0.01,1.01);
  histos_[trigName].recoJetHPDfractionMatched_ = dbe_->book1D("recoJetHPDfractionMatched","recoJet HPD fraction if found by trigger",102,-0.01,1.01);
  histos_[trigName].recoJetN90Matched_ = dbe_->book1D("recoJetN90Matched","recoJet N90 if found by trigger",100,-0.5,99.5);

  // Sundry jet related variables.
  histos_[trigName].trigJetVsRecoJetPt_ = dbe_->book2D("trigJetVsRecoJetPt","trigJet vs. recoJet Pt",50,0.0,1000.,50,0.0,1000.);
}

//====================//
// ANALYSE EACH EVENT //
//====================//

void AnalyseDisplacedJetTrigger::analyze(const edm::Event& iEvent, 
					 const edm::EventSetup& iSetup) {
  
  iEvent.getByLabel("selectedPatJets", patJets_);
  iEvent.getByLabel("patTriggerEvent", patTriggerEvent_);
  iEvent.getByLabel ("offlinePrimaryVerticesWithBS", primaryVertex_);
  unsigned int nPV = primaryVertex_->size();
  float PVz = 29.9;
  if (primaryVertex_->size() > 0) PVz = primaryVertex_->begin()->z();

  // Print debug PV info if required.
  this->debugPrintPV(iEvent);

  // For each displaced jet trigger (specified by name) return a boolean indicating
  // if it passed the trigger and the trigger objects it used.
  map<string, pair<bool, TriggerObjectRefVector> > trigJetsInAllTrigs = this->getTriggerInfo();

  // Loop over displaced jet triggers

  map<string, pair<bool, TriggerObjectRefVector> >::const_iterator iter;
  for (iter = trigJetsInAllTrigs.begin(); iter != trigJetsInAllTrigs.end(); iter++) {
    string trigName = iter->first;
    const bool passedTrig(iter->second.first);
    const TriggerObjectRefVector& trigJets(iter->second.second);

    // Plot event related variables and ditto if the displaced jet trigger fired.
    histos_[trigName].nPV_->Fill(nPV);
    histos_[trigName].PVz_->Fill(PVz);
    if (passedTrig) {
      histos_[trigName].nPVPassed_->Fill(nPV);
      histos_[trigName].PVzPassed_->Fill(PVz);
    }

    // Analyse offline reco jets and see if they match a trigger jet.

    unsigned int nTrueDisplacedJets = 0;
    unsigned int nTrueDisplacedJetsMatched = 0;
    const float trueDecayLengthCut = 1.0;

    for(unsigned int j=0; j<patJets_->size(); j++) {
      const pat::JetRef recoJet(patJets_,j);

      // Require that at least one other jet in the event is likely to pass the kinematic
      // requirements of the displaced jet trigger. This is because unless at least 2 jets
      // do so, no trigger objects will be stored for this trigger.
      if (this->anotherGoodJet(recoJet)) {

	// Find closest trigJet to recoJet, if any.
	TriggerObjectRef trigJet = this->matchJets(recoJet, trigJets);
	bool match = trigJet.isNonnull();

	// Note if this jet was good for displaced jet exotica search.
	GoodJetDisplacedJetTrigger good(recoJet);
	// Also if it remains good on relaxing each cut in turn.
	GoodJetDisplacedJetTrigger goodNoNpromptTkCut(good); goodNoNpromptTkCut.passNpromptTk = true;
	GoodJetDisplacedJetTrigger goodNoPtCut(good)       ; goodNoPtCut.passPt = true;
	GoodJetDisplacedJetTrigger goodNoEtaCut(good)      ; goodNoEtaCut.passEta = true;
	GoodJetDisplacedJetTrigger goodNoJetIDCut(good)    ; goodNoJetIDCut.passJetID = true;

	// Check if this jet was produced by a displaced parton, and if so, note its production radius.

	LogDebug("AnalyseDisplacedJetTrigger") <<"Jet="<<j<<" nPromptTk="<<recoJet->bDiscriminator("displacedJetTags");
	float trueDecayLength = -1.;
        if (! iEvent.isRealData()) {
	  const reco::GenParticle* gen = recoJet->genParton();
	  if (gen != 0 && gen->numberOfDaughters() > 0) {
	    reco::Candidate::Point trueDecayVert  = gen->daughter(0)->vertex(); // decay vertex of parton
            reco::Candidate::Point trueOrigVert = gen->vertex(); // true primary vertex 
	    if (gen->numberOfMothers() > 0) trueOrigVert = gen->mother(0)->vertex();
	    // Transverse production radius of parton.
            trueDecayLength = (trueDecayVert - trueOrigVert).rho();

	    LogInfo("AnalyseDisplacedJetTrigger") <<"Jet matched to GenParton: id="<<gen->pdgId()<<" Gen/Reco Pt = "<<gen->pt()<< "/" <<recoJet->pt()<<
	      " P.V. radius="<<trueOrigVert.rho()<<" decay radius="<<trueDecayVert.rho();
	  } else {
	    LogInfo("AnalyseDisplacedJetTrigger") <<"Jet not matched to GenParton: Reco Pt = " <<recoJet->pt();
	  }
	}

	// Plot recoJet properties for jets useful to exotica search. Relax cut on quantity being plotted.
	// Then repeat if matched to trigger jet.
	if (goodNoNpromptTkCut.ok()) {
	  float nPromptTk = recoJet->bDiscriminator("displacedJetTags");
	  histos_[trigName].trueJetProdRadius_->Fill(trueDecayLength);
	  histos_[trigName].recoJetNpromptTk_->Fill(nPromptTk);
	  if (trueDecayLength >  trueDecayLengthCut) nTrueDisplacedJets++;

	  /*
	  // For debug purposes, examine jets with no offline prompt tracks, which surprisingly have no corresponding trigger object.
	  // Check if the trigger path created the corresponding jets and jetTags.
	  if (good.ok() && !match && nPromptTk == 0 && trigName == "HLT_HT250_DoubleDisplacedJet60_v2") {
            Handle<reco::CaloJetCollection> hltJets1;
            iEvent.getByLabel("hltJetIDPassedCorrJets", hltJets1);
            Handle<reco::CaloJetCollection> hltJets2;
            iEvent.getByLabel("hltAntiKT5L2L3CorrCaloJetsPt60Eta2V2", hltJets2);
	    edm::Handle<reco::JetTagCollection> jetTags;
	    iEvent.getByLabel("hltDisplacedHT250L3JetTagsV2", jetTags);

	    double bestDelRCJ = 0.3;
	    edm::Ref<reco::CaloJetCollection> caloJetMatch;
	    if (hltJets1.isValid()) {
              for(unsigned int jt=0; jt<hltJets1->size(); jt++) {
		edm::Ref<reco::CaloJetCollection> caloJet(hltJets1,jt);
		double delR = deltaR(*recoJet, *caloJet);
		if (bestDelRCJ > delR) {
		  bestDelRCJ = delR;
		  caloJetMatch = caloJet;
		}
	      }
              LogInfo("AnalyseDisplacedJetTrigger")<<"FAILED TO TRIGGER: CaloJetMatch matchedJetPt/Eta="<<caloJetMatch->pt()<<"/"<<caloJetMatch->eta()<<" n90="<<caloJetMatch->n90()<<" em="<<caloJetMatch->emEnergyFraction();
            }

	    double bestDelRJT = 0.3;
	    edm::Ref<reco::JetTagCollection> jetTagMatch;
	    if (jetTags.isValid()) {
              for(unsigned int jt=0; jt<jetTags->size(); jt++) {
		edm::Ref<reco::JetTagCollection> jetTag(jetTags,jt);
		double delR = deltaR(*recoJet, *(jetTag->first));
		if (bestDelRJT > delR) {
		  bestDelRJT = delR;
		  jetTagMatch = jetTag;
		}
	      }
              if (jetTagMatch.isNonnull()) {
  	        LogInfo("AnalyseDisplacedJetTrigger")<<" JET FAILED TO TRIGGER: pt="<<recoJet->pt()<<" eta="<<recoJet->eta()<<" found"<<hltJets1.isValid()<<"/"<<hltJets2.isValid()<<"/"<<jetTags.isValid()<<" matchedJetPt/Eta="<<jetTagMatch->first->pt()<<"/"<<jetTagMatch->first->eta()<<" matchedNtrk="<<jetTagMatch->second;
              } else {
  	        LogInfo("AnalyseDisplacedJetTrigger")<<" JET FAILED TO TRIGGER: pt="<<recoJet->pt()<<" eta="<<recoJet->eta()<<" found"<<hltJets1.isValid()<<"/"<<hltJets2.isValid()<<"/"<<jetTags.isValid()<<" NOT MATCHED";
              }
	    } else {
  	        LogInfo("AnalyseDisplacedJetTrigger")<<" JET FAILED TO TRIGGER: pt="<<recoJet->pt()<<" eta="<<recoJet->eta()<<" found"<<hltJets1.isValid()<<"/"<<hltJets2.isValid()<<"/"<<jetTags.isValid()<<" NOT RUN";
            }
          }
	  */

	  if (match) {
	    histos_[trigName].trueJetProdRadiusMatched_->Fill(trueDecayLength);
	    histos_[trigName].recoJetNpromptTkMatched_->Fill(nPromptTk);
	    if (trueDecayLength >  trueDecayLengthCut) nTrueDisplacedJetsMatched++;
	    /*
            // For debug purposes, print out the offline tracks in all triggered jets.
	    const reco::TrackIPTagInfo* tagInfo = recoJet->tagInfoTrackIP("displacedJet");
	    LogInfo("AnalyseDisplacedJetTrigger") << "NPROMPT offline tracks = "<<nPromptTk; 
	    if (tagInfo != 0) {
	      const edm::RefVector<reco::TrackCollection>& tracks = tagInfo->selectedTracks();
	      const std::vector<reco::TrackIPTagInfo::TrackIPData>& ip = tagInfo->impactParameterData();
	      for (unsigned int itrk = 0; itrk < tracks.size(); itrk++) {
		LogVerbatim("AnalyseDisplacedJetTrigger") <<"Track "<<itrk<<" pt="<<tracks[itrk]->pt()<<" d0="<<ip[itrk].ip3d.value()<<" pixelHits="<<tracks[itrk]->hitPattern().pixelLayersWithMeasurement();
	      }
	    }
            */
	  }
	}
	if (goodNoPtCut.ok()) {
	  histos_[trigName].recoJetPt_->Fill(recoJet->pt());
	  if (match) histos_[trigName].recoJetPtMatched_->Fill(recoJet->pt());
	}
	if (goodNoEtaCut.ok()) {
	  histos_[trigName].recoJetEta_->Fill(fabs(recoJet->eta()));
	  if (match) histos_[trigName].recoJetEtaMatched_->Fill(fabs(recoJet->eta()));
	}
	if (goodNoJetIDCut.ok()) {
	  histos_[trigName].recoJetEMfraction_->Fill(recoJet->emEnergyFraction());
	  histos_[trigName].recoJetHPDfraction_->Fill(recoJet->jetID().fHPD);
	  histos_[trigName].recoJetN90_->Fill(recoJet->jetID().n90Hits);
	  if (match) {
	    histos_[trigName].recoJetEMfractionMatched_->Fill(recoJet->emEnergyFraction());
	    histos_[trigName].recoJetHPDfractionMatched_->Fill(recoJet->jetID().fHPD);
	    histos_[trigName].recoJetN90Matched_->Fill(recoJet->jetID().n90Hits);
	  }
	}

	// Sundry histos
	if (match) histos_[trigName].trigJetVsRecoJetPt_->Fill(recoJet->pt(), trigJet->pt());
      }

      // True displaced jets per event count
      histos_[trigName].trueNumDispJets_->Fill(nTrueDisplacedJets);
      histos_[trigName].trueNumDispJetsMatched_->Fill(nTrueDisplacedJetsMatched);
    }
  }
}

bool AnalyseDisplacedJetTrigger::anotherGoodJet(pat::JetRef thisRecoJet) {

  // Check if there is another jet in the event aside from this one which is likely
  // to pass the kinematic requirements of the displaced jet trigger.

  bool foundAnother = false;
  for(unsigned int j=0; j<patJets_->size(); j++) {
    const pat::JetRef recoJet(patJets_,j);
    if (recoJet != thisRecoJet) {

      // Note if this jet is likely to pass kinematic requirements of displaced jet trigger.
      GoodJetDisplacedJetTrigger goodNoNpromptTkCut(recoJet); goodNoNpromptTkCut.passNpromptTk = true;

      if (goodNoNpromptTkCut.ok()) foundAnother = true;
    }
  }
  return foundAnother;
}

TriggerObjectRef AnalyseDisplacedJetTrigger::matchJets(pat::JetRef recoJet, const TriggerObjectRefVector& trigJets) {
  // Find closest trigger jet to reco jet, if any.
  double bestDelR = 0.3;
  TriggerObjectRef matchedJet;
  for (unsigned n = 0; n < trigJets.size(); n++) {
    double delR = deltaR(*recoJet, *(trigJets[n]));
    if (bestDelR > delR) {
      bestDelR = delR;
      matchedJet = trigJets[n];
    }
  }
  return matchedJet;
}

 map<string, pair<bool, TriggerObjectRefVector> >  AnalyseDisplacedJetTrigger::getTriggerInfo() {
  // Analyse triggers. For each displaced jet trigger (specified by name) return 
  // a boolean indicating if it passed the trigger and the trigger objects it used.
   map<string, pair<bool, TriggerObjectRefVector> >  trigJetsInAllTrigs;

  if (patTriggerEvent_->wasRun()) {    

    // Loop over all triggers and note which ones fired.
    const pat::TriggerPathCollection& trigPaths = *(patTriggerEvent_->paths());
    unsigned int nTrig = trigPaths.size();
    for (unsigned int i=0; i < nTrig; i++) {
      if (trigPaths[i].wasRun()) {
        string trigName = trigPaths[i].name();
        if (this->isTrigDisplaced(trigName) || this->isTrigJet(trigName) ||
            this->isTrigHT(trigName))
	  {
	    // Check if we have encountered this trigger before.
	    map<string, int>::const_iterator iter = trigNameList_.find(trigName); 
	    bool newTrig = (iter == trigNameList_.end());
            if (newTrig) {
	      // If new, note its name and book histograms for it.
              unsigned int nFound = trigNameList_.size() + 1;
	      trigNameList_[trigName] = nFound; 
   	      if (this->isTrigDisplaced(trigName)) this->bookHistos(trigName);
            }

	    // Plot trigger efficiency of all triggers.
	    bool pass = trigPaths[i].wasAccept();
	    trigEffi_->Fill(trigNameList_[trigName], pass);

	    // This boolean could be set to true only if a second trigger, used to collect an unbiased
	    // data sample for monitoring the performance of the dispaced jet trigger, has fired.
	    // e.g. The normal HT or jet trigger. This is not yet implemented.
            const bool studyThisTrig = true;

	    if (studyThisTrig && this->isTrigDisplaced(trigName)) {

	      trigJetsInAllTrigs[trigName].first = pass;
	      trigJetsInAllTrigs[trigName].second = TriggerObjectRefVector();

	      const pat::TriggerObjectRefVector trigObjs( patTriggerEvent_->pathObjects(trigName) );
              vector<float> killL25objects;
  	      LogInfo("AnalyseDisplacedJetTrigger") <<"Trigger="<<trigName<<" pass="<<pass<<" no. trig. objs. = "<<trigObjs.size();
	      for (unsigned nObj = 0; nObj < trigObjs.size(); nObj++) {
		// Jet object will only exist if trigger passed.
		/*
		  vector<trigger::TriggerObjectType> killL25objects = trigObjs[nObj]->filterIds();
		  for (unsigned int k = 0; k < killL25objects.size(); k++) {
                  LogInfo("AnalyseDisplacedJetTrigger")<<"Trigger object type = "<<k<<" "<<killL25objects[k];
		  }
		*/
		if (trigObjs[nObj]->hasFilterId( trigger::TriggerBJet )) {
                  bool l3 = true;
		  /*
                  // useful if L25 objects also stored in event. 
                  bool l3 = false;
                  for (unsigned int is = 0; is < killL25objects.size(); is++) {
		    if (fabs(killL25objects[is] - trigObjs[nObj]->pt()) < 0.1) l3 = true;
                  }
                  if (!l3) killL25objects.push_back(trigObjs[nObj]->pt());
		  */
		  LogVerbatim("AnalyseDisplacedJetTrigger")<<"     trig obj="<<nObj<<" Pt="<<trigObjs[nObj]->pt()<<" type="<<trigObjs[nObj]->collection()<<" l3="<<l3<<endl;
                  if (l3) trigJetsInAllTrigs[trigName].second.push_back(trigObjs[nObj]);
		}
	      }
	    }
	  }
      }
    }
  }
  return trigJetsInAllTrigs;
}

void AnalyseDisplacedJetTrigger::debugPrintPV(const edm::Event& iEvent) {
  // Print debug info comparing HLT and RECO primary vertices if required (and if stored in input data file).

  LogInfo("AnalyseDisplacedJetTrigger")<<"--- primary vertex position ---";
  for (unsigned int ipv = 0; ipv < primaryVertex_->size(); ipv++) {
    const reco::VertexRef vtx(primaryVertex_, ipv);
    LogVerbatim("AnalyseDisplacedJetTrigger")<<"Offline PV "<<ipv<<" "<<primaryVertex_<<" x="<<vtx->x()<<" y="<<vtx->y()<<" z="<<vtx->z();
  }
  Handle<reco::VertexCollection> hltPixelVertex;
  iEvent.getByLabel ("hltPixelVertices", hltPixelVertex);
  if (hltPixelVertex.isValid()) {
    for (unsigned int ipv = 0; ipv < hltPixelVertex->size(); ipv++) {
      const reco::VertexRef vtx(hltPixelVertex, ipv);
      LogVerbatim("AnalyseDisplacedJetTrigger")<<"HLT Pixel PV "<<ipv<<" "<<hltPixelVertex<<" x="<<vtx->x()<<" y="<<vtx->y()<<" z="<<vtx->z();
    }
  }
  Handle<reco::VertexCollection> pixelVertex;
  iEvent.getByLabel ("pixelVertices", pixelVertex);
  if (pixelVertex.isValid()) {
    for (unsigned int ipv = 0; ipv < pixelVertex->size(); ipv++) {
      const reco::VertexRef vtx(pixelVertex, ipv);
      LogVerbatim("AnalyseDisplacedJetTrigger")<<"Offline Pixel PV "<<ipv<<" "<<pixelVertex<<" x="<<vtx->x()<<" y="<<vtx->y()<<" z="<<vtx->z();
    }
  }
}

DEFINE_FWK_MODULE( AnalyseDisplacedJetTrigger );
