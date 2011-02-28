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

#include "FWCore/Utilities/interface/Exception.h"
#include <exception>

AnalyseDisplacedJetTrigger::AnalyseDisplacedJetTrigger(const edm::ParameterSet& iConfig)
// : 

  //  datasetNameString_(iConfig.getParameter< vector<string> > ("datasetName")),
    //  dataPeriod_(iConfig.getParameter<int> ("dataPeriod")),
{  
  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("AnalyseDisplacedJetTrigger") << "ERROR: unable to get DQMStore service?";
  }
    
  trigEffi_ = dbe_->bookProfile("trigEffi","Trigger efficiency",30,0.5,30.5,-9.9,9.9);
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

  // If MC truth is available, this is the true production radius of the jet.
  histos_[trigName].trueJetProdRadius_ = dbe_->book1D("trueJetProdRadius","true production radius of jet (cm)",202,-1.25,99.75);
  // Histograms of offline recoJets that are useful for displaced exotica jet search.
  histos_[trigName].recoJetNpromptTk_ = dbe_->book1D("recoJetNpromptTk","recoJet prompt tracks",50,-0.5,49.5);
  histos_[trigName].recoJetPt_ = dbe_->book1D("recoJetPt","recoJet Pt",100,0.0,1000.);
  histos_[trigName].recoJetEta_ = dbe_->book1D("recoJetEta","recoJet Eta",30,0.0,3.0);
  histos_[trigName].recoJetEMfraction_ = dbe_->book1D("recoJetEMfraction","recoJet EM fraction",102,-0.01,1.01);
  histos_[trigName].recoJetHPDfraction_ = dbe_->book1D("recoJetHPDfraction","recoJet HPD fraction",102,-0.01,1.01);
  histos_[trigName].recoJetN90_ = dbe_->book1D("recoJetN90","recoJet N90",100,-0.5,99.5);

  // Ditto, but only if recoJet is matched to a trigJet found by displaced jet trigger.
  histos_[trigName].trueJetProdRadiusMatched_ = dbe_->book1D("trueJetProdRadiusMatched","true production radius of jet (cm) if found by trigger",202,-1.25,99.75);
  histos_[trigName].recoJetNpromptTkMatched_ = dbe_->book1D("recoJetNpromptTkMatched","recoJet prompt tracks if found by trigger",50,-0.5,49.5);
  histos_[trigName].recoJetPtMatched_ = dbe_->book1D("recoJetPtMatched","recoJet Pt if found by trigger",100,0.0,1000.);
  histos_[trigName].recoJetEtaMatched_ = dbe_->book1D("recoJetEtaMatched","recoJet Eta if found by trigger",30,0.0,3.0);
  histos_[trigName].recoJetEMfractionMatched_ = dbe_->book1D("recoJetEMfractionMatched","recoJet EM fraction if found by trigger",102,-0.01,1.01);
  histos_[trigName].recoJetHPDfractionMatched_ = dbe_->book1D("recoJetHPDfractionMatched","recoJet HPD fraction if found by trigger",102,-0.01,1.01);
  histos_[trigName].recoJetN90Matched_ = dbe_->book1D("recoJetN90Matched","recoJet N90 if found by trigger",100,-0.5,99.5);

  // Sundry
  histos_[trigName].trigJetVsRecoJetPt_ = dbe_->book2D("trigJetVsRecoJetPt","trigJet vs. recoJet Pt",50,0.0,1000.,50,0.0,1000.);
}

//====================//
// ANALYSE EACH EVENT //
//====================//

void AnalyseDisplacedJetTrigger::analyze(const edm::Event& iEvent, 
					 const edm::EventSetup& iSetup) {
  
  iEvent.getByLabel("selectedPatJets", patJets_);
  iEvent.getByLabel("patTriggerEvent", patTriggerEvent_);

  // Get trigger objects for each displaced jet trigger.
  map<string, TriggerObjectRefVector> trigJetsInAllTrigs = this->getTriggerInfo();

  // Loop over displaced jet triggers

  map<string, TriggerObjectRefVector>::const_iterator iter;
  for (iter = trigJetsInAllTrigs.begin(); iter != trigJetsInAllTrigs.end(); iter++) {
    string trigName = iter->first;
    const TriggerObjectRefVector& trigJets(iter->second);

    // Analyse offline reco jets.
    for(unsigned int j=0; j<patJets_->size(); j++) {
      const pat::JetRef recoJet(patJets_,j);

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

      LogDebug("AnalyseDisplacedJetTrigger") <<"BTAG "<<j<<" "<<recoJet->bDiscriminator("displacedJetTags")<<endl;
      float trueRadius = -1.;
      const reco::GenParticle* gen = recoJet->genParton();
      if (gen != 0 && gen->numberOfDaughters() > 0) {
        trueRadius = gen->daughter(0)->vertex().rho(); // decay radius of parton
        LogDebug("AnalyseDisplacedJetTrigger") <<"Matched to GenParton with Pt = "<<gen->pt()<< "/" <<recoJet->pt()<<" R="<<gen->vertex().rho()<<" id="<<gen->pdgId()<<" ndaugh = "<<gen->numberOfDaughters()<<" RD="<<gen->daughter(0)->vertex().rho()<<" idd="<<gen->daughter(0)->pdgId()<<" moth="<<gen->mother()->pdgId();
      } else {
        LogDebug("AnalyseDisplacedJetTrigger") <<"Unmatched to GenParton with Pt = 0/" <<recoJet->pt();
      }

      // Plot recoJet properties for jets useful to exotica search. Relax cut on quantity being plotted.
      // Then repeat if matched to trigger jet.
      if (goodNoNpromptTkCut.ok()) {
 	 float nPromptTk = recoJet->bDiscriminator("displacedJetTags");
         histos_[trigName].trueJetProdRadius_->Fill(trueRadius);
         histos_[trigName].recoJetNpromptTk_->Fill(nPromptTk);
         if (match) {
           histos_[trigName].trueJetProdRadiusMatched_->Fill(trueRadius);
           histos_[trigName].recoJetNpromptTkMatched_->Fill(nPromptTk);
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
  }
}

TriggerObjectRef AnalyseDisplacedJetTrigger::matchJets(pat::JetRef recoJet, const TriggerObjectRefVector& trigJets) {
  // Find closest trigger jet to reco jet, if any.
  double bestDelR = 0.5;
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

map<string, TriggerObjectRefVector> AnalyseDisplacedJetTrigger::getTriggerInfo() {
  // Analyse triggers and return trigger objects for displaced jet triggers.

  map<string, TriggerObjectRefVector> trigJetsInAllTrigs;

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

	      trigJetsInAllTrigs[trigName] = TriggerObjectRefVector();

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
                  bool l3 = false;
                  for (unsigned int is = 0; is < killL25objects.size(); is++) {
		    if (fabs(killL25objects[is] - trigObjs[nObj]->pt()) < 0.1) l3 = true;
                  }
                  if (!l3) killL25objects.push_back(trigObjs[nObj]->pt());
		  LogVerbatim("AnalyseDisplacedJetTrigger")<<"     trig obj="<<nObj<<" Pt="<<trigObjs[nObj]->pt()<<" type="<<trigObjs[nObj]->collection()<<" l3="<<l3<<endl;
                  if (l3) trigJetsInAllTrigs[trigName].push_back(trigObjs[nObj]);
		}
	      }
	    }
	  }
      }
    }
  }
  return trigJetsInAllTrigs;
}

DEFINE_FWK_MODULE( AnalyseDisplacedJetTrigger );
