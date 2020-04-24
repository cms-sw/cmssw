// system include files
#include <memory>
#include <map>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TTree.h>
#include <Math/VectorUtil.h>


class matchGenHFHadrons : public edm::EDAnalyzer {
    public: 
        explicit matchGenHFHadrons(const edm::ParameterSet & );
        ~matchGenHFHadrons() {};
        
    private:
        virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
        virtual void beginJob();
        void bookHadronBranches(TTree* tree, const int flavour);
        void clearEventVariables();
        void clearHadronVariables();
        void fillHadronTree( TTree* tree, const int flavour, const reco::GenJetCollection& genJets,
                             const std::vector<int>& genHadIndex, const std::vector<int>& genHadJetIndex,
                             const std::vector<int>& genHadFlavour, const std::vector<int>& genHadFromTopWeakDecay,
                             const std::vector<reco::GenParticle>& genHadPlusMothers, const std::vector<std::vector<int> >& genHadPlusMothersIndices,
                             const std::vector<int>& genHadLeptonHadronIndex, const std::vector<int>& genHadLeptonViaTau,
                             const std::vector<int>& genHadBHadronId = std::vector<int>(0) );
        // A recursive function that scans through the particle chain to find a particle with specific pdgId or abs(pdgId)
        void findAncestorParticles(const int startId, std::set<int>& resultIds, 
                                   const std::vector<reco::GenParticle>& genParticles, const std::vector<std::vector<int> >& motherIndices, 
                                   const int endPdgId, const bool endPdgIdIsAbs = false, const int firstAnyLast = 0);
        
        
        // Jets configuration
        const double genJetPtMin_;
        const double genJetAbsEtaMax_;
        
        // Input tags
        const edm::EDGetTokenT<reco::GenJetCollection> genJetsToken_;
        
        const edm::EDGetTokenT<std::vector<int> > genBHadJetIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadFlavourToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadFromTopWeakDecayToken_;
        const edm::EDGetTokenT<std::vector<reco::GenParticle> > genBHadPlusMothersToken_;
        const edm::EDGetTokenT<std::vector<std::vector<int> > > genBHadPlusMothersIndicesToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadLeptonHadronIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadLeptonViaTauToken_;
        
        const edm::EDGetTokenT<std::vector<int> > genCHadJetIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadFlavourToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadFromTopWeakDecayToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadBHadronIdToken_;
        const edm::EDGetTokenT<std::vector<reco::GenParticle> > genCHadPlusMothersToken_;
        const edm::EDGetTokenT<std::vector<std::vector<int> > > genCHadPlusMothersIndicesToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadLeptonHadronIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadLeptonViaTauToken_;
        
        // Output to be written to the trees
        TTree* tree_events_;
        TTree* tree_bHadrons_;
        TTree* tree_cHadrons_;
        
        int nJets_;
        
        int nBJets_;
        int nBHadrons_;
        int nBHadronsTop_;
        int nBHadronsAdditional_;
        int nBHadronsPseudoadditional_;
        
        int nCJets_;
        int nCHadrons_;
        int nCHadronsAdditional_;
        int nCHadronsPseudoadditional_;
        
        int nBHadronsHiggs_;
        int additionalJetEventId_;
        
        // Additional performance information
        int hadron_flavour_;
        int hadron_nQuarksAll_;
        int hadron_nQuarksSameFlavour_;
        int hadron_fromTopWeakDecay_;
        int hadron_bHadronId_;
        
        double hadron_quark1_dR_;
        double hadron_quark2_dR_;
        double hadron_quark1_ptRatio_;
        double hadron_quark2_ptRatio_;
        
        int hadron_nLeptonsAll_;
        int hadron_nLeptonsViaTau_;
        
        double hadron_jetPtRatio_;
        double hadron_jetDR_;
};

bool sort_pairs_second(const std::pair<int, double>& a, const std::pair<int, double>& b) 
{
    return a.second < b.second;
}

matchGenHFHadrons::matchGenHFHadrons ( const edm::ParameterSet& config):
    genJetPtMin_(config.getParameter<double>("genJetPtMin")),
    genJetAbsEtaMax_(config.getParameter<double>("genJetAbsEtaMax")),
    genJetsToken_(consumes<reco::GenJetCollection>(config.getParameter<edm::InputTag>("genJets"))),
    genBHadJetIndexToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genBHadJetIndex"))),
    genBHadFlavourToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genBHadFlavour"))),
    genBHadFromTopWeakDecayToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genBHadFromTopWeakDecay"))),
    genBHadPlusMothersToken_(consumes<std::vector<reco::GenParticle> >(config.getParameter<edm::InputTag>("genBHadPlusMothers"))),
    genBHadPlusMothersIndicesToken_(consumes<std::vector<std::vector<int> > >(config.getParameter<edm::InputTag>("genBHadPlusMothersIndices"))),
    genBHadIndexToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genBHadIndex"))),
    genBHadLeptonHadronIndexToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genBHadLeptonHadronIndex"))),
    genBHadLeptonViaTauToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genBHadLeptonViaTau"))),
    genCHadJetIndexToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genCHadJetIndex"))),
    genCHadFlavourToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genCHadFlavour"))),
    genCHadFromTopWeakDecayToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genCHadFromTopWeakDecay"))),
    genCHadBHadronIdToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genCHadBHadronId"))),
    genCHadPlusMothersToken_(consumes<std::vector<reco::GenParticle> >(config.getParameter<edm::InputTag>("genCHadPlusMothers"))),
    genCHadPlusMothersIndicesToken_(consumes<std::vector<std::vector<int> > >(config.getParameter<edm::InputTag>("genCHadPlusMothersIndices"))),
    genCHadIndexToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genCHadIndex"))),
    genCHadLeptonHadronIndexToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genCHadLeptonHadronIndex"))),
    genCHadLeptonViaTauToken_(consumes<std::vector<int> >(config.getParameter<edm::InputTag>("genCHadLeptonViaTau")))
{

}


void matchGenHFHadrons::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    this->clearEventVariables();
    
    // Reading gen jets from the event
    edm::Handle<reco::GenJetCollection> genJets;
    event.getByToken(genJetsToken_, genJets);
    
    // Reading B hadrons related information
    edm::Handle<std::vector<int> > genBHadFlavour;
    event.getByToken(genBHadFlavourToken_, genBHadFlavour);
    
    edm::Handle<std::vector<int> > genBHadJetIndex;
    event.getByToken(genBHadJetIndexToken_, genBHadJetIndex);
    
    edm::Handle<std::vector<int> > genBHadFromTopWeakDecay;
    event.getByToken(genBHadFromTopWeakDecayToken_, genBHadFromTopWeakDecay);
    
    edm::Handle<std::vector<reco::GenParticle> > genBHadPlusMothers;
    event.getByToken(genBHadPlusMothersToken_, genBHadPlusMothers);
    
    edm::Handle<std::vector<std::vector<int> > > genBHadPlusMothersIndices;
    event.getByToken(genBHadPlusMothersIndicesToken_, genBHadPlusMothersIndices);
    
    edm::Handle<std::vector<int> > genBHadIndex;
    event.getByToken(genBHadIndexToken_, genBHadIndex);
    
    edm::Handle<std::vector<int> > genBHadLeptonHadronIndex;
    event.getByToken(genBHadLeptonHadronIndexToken_, genBHadLeptonHadronIndex);
    
    edm::Handle<std::vector<int> > genBHadLeptonViaTau;
    event.getByToken(genBHadLeptonViaTauToken_, genBHadLeptonViaTau);
    
    // Reading C hadrons related information
    edm::Handle<std::vector<int> > genCHadFlavour;
    event.getByToken(genCHadFlavourToken_, genCHadFlavour);
    
    edm::Handle<std::vector<int> > genCHadJetIndex;
    event.getByToken(genCHadJetIndexToken_, genCHadJetIndex);
    
    edm::Handle<std::vector<int> > genCHadFromTopWeakDecay;
    event.getByToken(genCHadFromTopWeakDecayToken_, genCHadFromTopWeakDecay);
    
    edm::Handle<std::vector<int> > genCHadBHadronId;
    event.getByToken(genCHadBHadronIdToken_, genCHadBHadronId);

    edm::Handle<std::vector<reco::GenParticle> > genCHadPlusMothers;
    event.getByToken(genCHadPlusMothersToken_, genCHadPlusMothers);
    
    edm::Handle<std::vector<std::vector<int> > > genCHadPlusMothersIndices;
    event.getByToken(genCHadPlusMothersIndicesToken_, genCHadPlusMothersIndices);
    
    edm::Handle<std::vector<int> > genCHadIndex;
    event.getByToken(genCHadIndexToken_, genCHadIndex);
    
    edm::Handle<std::vector<int> > genCHadLeptonHadronIndex;
    event.getByToken(genCHadLeptonHadronIndexToken_, genCHadLeptonHadronIndex);
    
    edm::Handle<std::vector<int> > genCHadLeptonViaTau;
    event.getByToken(genCHadLeptonViaTauToken_, genCHadLeptonViaTau);
    

    // Counting number of jets of different flavours in the event
    for(size_t jetId = 0; jetId < genJets->size(); ++jetId) {
        if(genJets->at(jetId).pt() < genJetPtMin_) continue;
        if(std::fabs(genJets->at(jetId).eta()) < genJetAbsEtaMax_) continue;
        nJets_++;
        if(std::find(genBHadJetIndex->begin(), genBHadJetIndex->end(), jetId) != genBHadJetIndex->end()) nBJets_++;
        else if(std::find(genCHadJetIndex->begin(), genCHadJetIndex->end(), jetId) != genCHadJetIndex->end()) nCJets_++;
    }
    // Counting number of different b hadrons
    for(size_t hadronId = 0; hadronId < genBHadFlavour->size(); ++hadronId) {
        const int flavour = genBHadFlavour->at(hadronId);
        const int flavourAbs = std::abs(flavour);
	// 0 - not from top decay; 1 - from top decay; 2 - not identified;
        const bool afterTopDecay = genBHadFromTopWeakDecay->at(hadronId) > 0 ? true : false;
        nBHadrons_++;
        if(flavourAbs==25) nBHadronsHiggs_++;
        if(flavourAbs==6) nBHadronsTop_++;
        if(flavourAbs!=6) {
            if(afterTopDecay) nBHadronsPseudoadditional_++;
            else nBHadronsAdditional_++;
        }
    }
    // Counting number of different c hadrons
    for(size_t hadronId = 0; hadronId < genCHadJetIndex->size(); ++hadronId) {
	// 0 - not from top decay; 1 - from top decay; 2 - not identified;
        const bool afterTopDecay = genCHadFromTopWeakDecay->at(hadronId) > 0 ? true : false;
        nCHadrons_++;
        // Skipping c hadrons that are coming from b hadrons
        if(genCHadBHadronId->at(hadronId) >= 0) continue;
        
        if(afterTopDecay) nCHadronsPseudoadditional_++;
        else nCHadronsAdditional_++;
    }
    
    // Map <jet index, number of specific hadrons in the jet>
    // B jets with b hadrons directly from top quark decay (including those not in acceptance)
    std::map<int, int> bJetFromTopIds_all;
    // B jets with b hadrons directly from top quark decay
    std::map<int, int> bJetFromTopIds;
    // B jets with b hadrons after top quark decay
    std::map<int, int> bJetAfterTopIds;
    // B jets with b hadrons before top quark decay chain
    std::map<int, int> bJetBeforeTopIds;
    // C jets with c hadrons before top quark decay chain
    std::map<int, int> cJetBeforeTopIds;
    // C jets with c hadrons after top quark decay
    std::map<int, int> cJetAfterTopIds;
    
    // Counting number of specific hadrons in each b jet
    for(size_t hadronId = 0; hadronId < genBHadIndex->size(); ++hadronId) {
        // Flavour of the hadron's origin
        const int flavour = genBHadFlavour->at(hadronId);
        // Skipping b hadrons from W-boson decays (in semileptonic or full-hadronic channels)
        if(std::abs(flavour) == 24) continue;
        // Whether hadron radiated before top quark decay
        const bool afterTopDecay = genBHadFromTopWeakDecay->at(hadronId) > 0 ? true : false;
        // Index of a jet associated to the hadron
        const int jetIndex = genBHadJetIndex->at(hadronId);
        // Skipping hadrons which have no associated jet
        if(jetIndex < 0) continue;
        // Jet from direct top quark decay [pdgId(top)=6]
        if(std::abs(flavour) == 6) {
            if(bJetFromTopIds_all.count(jetIndex) < 1) bJetFromTopIds_all[jetIndex] = 1;
            else bJetFromTopIds_all[jetIndex]++;
        }
        // Skipping if jet is not in acceptance
        if(genJets->at(jetIndex).pt() < genJetPtMin_) continue;
        if(std::fabs(genJets->at(jetIndex).eta()) > genJetAbsEtaMax_) continue;
        // Identifying jets with b hadrons not from top quark decay
        // Jet from direct top quark decay [pdgId(top)=6]
        if(std::abs(flavour) == 6) {
            if(bJetFromTopIds.count(jetIndex) < 1) bJetFromTopIds[jetIndex] = 1;
            else bJetFromTopIds[jetIndex]++;
        }
        // Skipping if jet is from top quark decay
        if(std::abs(flavour) == 6) continue;
        // Jet before top quark decay
        if(!afterTopDecay) {
            if(bJetBeforeTopIds.count(jetIndex) < 1) bJetBeforeTopIds[jetIndex] = 1;
            else bJetBeforeTopIds[jetIndex]++;
        }
        // Jet after top quark decay but not directly from top
        else if(afterTopDecay) {
            if(bJetAfterTopIds.count(jetIndex) < 1) bJetAfterTopIds[jetIndex] = 1;
            else bJetAfterTopIds[jetIndex]++;
        }
    }
    
    // Counting number of specific hadrons in each c jet
    for(size_t hadronId = 0; hadronId < genCHadJetIndex->size(); ++hadronId) {
        // Skipping c hadrons that are coming from b hadrons
        if(genCHadBHadronId->at(hadronId) >= 0) continue;
        // Flavour of the hadron's origin
        const int flavour = genCHadFlavour->at(hadronId);
        // Skipping c hadrons from W-boson decays (in semileptonic or full-hadronic channels)
        if(std::abs(flavour) == 24) continue;
        // Index of a jet associated to the hadron
        const int jetIndex = genCHadJetIndex->at(hadronId);
        // Whether hadron radiated after top quark decay
        const bool afterTopDecay = genCHadFromTopWeakDecay->at(hadronId) > 0 ? true : false;
        // Skipping hadrons which have no associated jet
        if(jetIndex < 0) continue;
        // Skipping if jet is not in acceptance
        if(genJets->at(jetIndex).pt() < genJetPtMin_) continue;
        if(std::fabs(genJets->at(jetIndex).eta()) > genJetAbsEtaMax_) continue;
        // Jet before top quark decay
        if(!afterTopDecay) {
            if(cJetBeforeTopIds.count(jetIndex) < 1) cJetBeforeTopIds[jetIndex] = 1;
            else cJetBeforeTopIds[jetIndex]++;
        }
        // Jet after top quark decay but not directly from top
        else if(afterTopDecay) {
            if(cJetAfterTopIds.count(jetIndex) < 1) cJetAfterTopIds[jetIndex] = 1;
            else cJetAfterTopIds[jetIndex]++;
        }
    }
    
    // Finding additional b jets (before top decay)
    std::vector<int> additionalBJetIds;
    for(std::map<int, int>::iterator it = bJetBeforeTopIds.begin(); it != bJetBeforeTopIds.end(); ++it) {
        const int jetId = it->first;
        // Skipping the jet if it contains a b hadron directly from top quark decay
        if(bJetFromTopIds.count(jetId) > 0) continue;
        additionalBJetIds.push_back(jetId);
    }
    // Finding pseudo-additional b jets (after top decay)
    std::vector<int> pseudoadditionalBJetIds;
    for(std::map<int, int>::iterator it = bJetAfterTopIds.begin(); it != bJetAfterTopIds.end(); ++it) {
        const int jetId = it->first;
        // Skipping the jet if it contains a b hadron directly from top quark decay
        if(bJetFromTopIds.count(jetId) > 0) continue;
        pseudoadditionalBJetIds.push_back(jetId);
    }
    // Finding additional c jets
    std::vector<int> additionalCJetIds;
    for(std::map<int, int>::iterator it = cJetBeforeTopIds.begin(); it != cJetBeforeTopIds.end(); ++it) {
        const int jetId = it->first;
        // Skipping the jet if it contains a b hadron directly from top quark decay
        if(bJetFromTopIds.count(jetId) > 0) continue;
        additionalCJetIds.push_back(jetId);
    }
    // Finding pseudo-additional c jets (after top decay)
    std::vector<int> pseudoadditionalCJetIds;
    for(std::map<int, int>::iterator it = cJetAfterTopIds.begin(); it != cJetAfterTopIds.end(); ++it) {
        const int jetId = it->first;
        // Skipping the jet if it contains a b hadron directly from top quark decay
        if(bJetFromTopIds.count(jetId) > 0) continue;
        pseudoadditionalCJetIds.push_back(jetId);
    }
    
    // Categorizing event based on number of additional b/c jets and number of corresponding hadrons in each of them 
    // Exclusively for dileptonic final states of the ttbar system (may be unapplicable for other decay channels)
    additionalJetEventId_ = bJetFromTopIds.size()*100;
    // tt + 1 additional b jet
    if (additionalBJetIds.size() == 1) {
        int nHadronsInJet = bJetBeforeTopIds[additionalBJetIds.at(0)];
        // tt + 1 additional b jet from 1 additional b hadron
        if(nHadronsInJet == 1) additionalJetEventId_ += 51;
        // tt + 1 additional b jet from >=2 additional b hadrons
        else additionalJetEventId_ += 52;
    }
    // tt + 2 additional b jets
    else if (additionalBJetIds.size() > 1) {
        int nHadronsInJet1 = bJetBeforeTopIds[additionalBJetIds.at(0)];
        int nHadronsInJet2 = bJetBeforeTopIds[additionalBJetIds.at(1)];
        // tt + 2 additional b jets each from 1 additional b hadron
        if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId_ += 53;
        // tt + 2 additional b jets one of which from >=2 overlapping additional b hadrons
        else if(std::min(nHadronsInJet1, nHadronsInJet2) == 1 && std::max(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId_ += 54;
        // tt + 2 additional b jets each from >=2 additional b hadrons
        else if(std::min(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId_ += 55;
    }
    // tt + no additional b jets
    else if(additionalBJetIds.size() == 0) {
        // tt + >=1 pseudo-additional b jet with b hadrons after top quark decay
        if(pseudoadditionalBJetIds.size() > 0) additionalJetEventId_ += 56;
        // tt + 1 additional c jet
        else if(additionalCJetIds.size() == 1) {
            int nHadronsInJet = cJetBeforeTopIds[additionalCJetIds.at(0)];
            // tt + 1 additional c jet from 1 additional c hadron
            if(nHadronsInJet == 1) additionalJetEventId_ += 41;
            // tt + 1 additional c jet from >=2 overlapping additional c hadrons
            else additionalJetEventId_ += 42;
        }
        // tt + >=2 additional c jets
        else if(additionalCJetIds.size() > 1) {
            int nHadronsInJet1 = cJetBeforeTopIds[additionalCJetIds.at(0)];
            int nHadronsInJet2 = cJetBeforeTopIds[additionalCJetIds.at(1)];
            // tt + 2 additional c jets each from 1 additional c hadron
            if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId_ += 43;
            // tt + 2 additional c jets one of which from >=2 overlapping additional c hadrons
            else if(std::min(nHadronsInJet1, nHadronsInJet2) == 1 && std::max(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId_ += 44;
            // tt + 2 additional c jets each from >=2 additional c hadrons
            else if(std::min(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId_ += 45;
        }
        // tt + no additional c jets
        else if(additionalCJetIds.size() == 0) {
            // tt + >=1 pseudo-additional c jet with c hadrons after top quark decay
            if(pseudoadditionalCJetIds.size() > 0) additionalJetEventId_ += 46;
            // tt + light jets
            else additionalJetEventId_ += 0;
        }
    }
    
    
    // Performance information filled in a tree with entry for each b hadron #######################################
    fillHadronTree(tree_bHadrons_, 5, *genJets, *genBHadIndex, *genBHadJetIndex, *genBHadFlavour, *genBHadFromTopWeakDecay, *genBHadPlusMothers, *genBHadPlusMothersIndices, *genBHadLeptonHadronIndex, *genBHadLeptonViaTau);
    
    // Performance information filled in a tree with entry for each c hadron #######################################
    fillHadronTree(tree_cHadrons_, 4, *genJets, *genCHadIndex, *genCHadJetIndex, *genCHadFlavour, *genCHadFromTopWeakDecay, *genCHadPlusMothers, *genCHadPlusMothersIndices, *genCHadLeptonHadronIndex, *genCHadLeptonViaTau, *genCHadBHadronId);
    
    tree_events_->Fill();
}


void matchGenHFHadrons::findAncestorParticles( const int startId, std::set<int>& resultIds, 
                                               const std::vector<reco::GenParticle>& genParticles, const std::vector<std::vector<int> >& motherIndices, 
                                               const int endPdgId, const bool endPdgIdIsAbs, const int firstAnyLast)
{
    if(startId < 0) return;
    int startPdgId = genParticles.at(startId).pdgId();
    if(endPdgIdIsAbs) startPdgId = std::abs(startPdgId);
    
    // No mothers stored for the requested particle
    if((int)motherIndices.size() < startId+1) return;
    
    // Getting ids of mothers of the starting particle
    const std::vector<int>& motherIds = motherIndices.at(startId);
    
    // In case the starting particle has pdgId as requested and it should not have same mothers
    if(startPdgId == endPdgId && firstAnyLast == 1) {
        int nSamePdgIdMothers = 0;
        // Counting how many mothers with the same pdgId the particle has
        for(size_t motherId = 0; motherId < motherIds.size(); ++motherId) {
            const int motherParticleId = motherIds.at(motherId);
            if(motherParticleId < 0) continue;
            int motherParticlePdgId = genParticles.at(motherParticleId).pdgId();
            if(endPdgIdIsAbs) motherParticlePdgId = std::abs(motherParticlePdgId);
            if(motherParticlePdgId == startPdgId) nSamePdgIdMothers++;
        }
        if(nSamePdgIdMothers < 1) {
            resultIds.insert(startId);
            return;
        }
    }
    
    // Looping over all mothers of the particle
    for(size_t motherId = 0; motherId < motherIds.size(); ++motherId) {
        const int motherParticleId = motherIds.at(motherId);
        if(motherParticleId < 0) continue;
        int motherParticlePdgId = genParticles.at(motherParticleId).pdgId();
        if(endPdgIdIsAbs) motherParticlePdgId = std::abs(motherParticlePdgId);
        // Checking whether pdgId of this particle matches the requested one
        bool isCandidate = false;
        bool inProperPlace = false;
        if(motherParticlePdgId == endPdgId) isCandidate = true;
        // Checking whether the candidate particle is first/last according to the request
        if(isCandidate) {
            if(firstAnyLast == -1 && startPdgId != motherParticlePdgId) inProperPlace = true;
            else if(firstAnyLast == 0) inProperPlace = true;
        }
        // Adding the mother if it has proper pdgId and place in chain
        if(isCandidate && inProperPlace) {
            resultIds.insert(motherParticleId);
            if(firstAnyLast < 0) continue;
        }
        
        // Checking mothers of this mother
        findAncestorParticles(motherParticleId, resultIds, genParticles, motherIndices, endPdgId, endPdgIdIsAbs, firstAnyLast);
    }
    
    
}


void matchGenHFHadrons::beginJob()
{
    edm::Service<TFileService> fileService;
    if(!fileService) throw edm::Exception(edm::errors::Configuration, "TFileService is not registered in cfg file");
    tree_events_ = fileService->make<TTree>("tree_events", "tree_events");
    tree_bHadrons_ = fileService->make<TTree>("tree_bHadrons", "tree_bHadrons");
    tree_cHadrons_ = fileService->make<TTree>("tree_cHadrons", "tree_cHadrons");
    
    // Creating output branches
    tree_events_->Branch("nJets", &nJets_);
    
    tree_events_->Branch("nBJets", &nBJets_);
    tree_events_->Branch("nBHadrons", &nBHadrons_);
    tree_events_->Branch("nBHadronsTop", &nBHadronsTop_);
    tree_events_->Branch("nBHadronsAdditional", &nBHadronsAdditional_);
    tree_events_->Branch("nBHadronsPseudoadditional", &nBHadronsPseudoadditional_);
    
    tree_events_->Branch("nCJets", &nCJets_);
    tree_events_->Branch("nCHadrons", &nCHadrons_);
    tree_events_->Branch("nCHadronsAdditional", &nCHadronsAdditional_);
    tree_events_->Branch("nCHadronsPseudoadditional", &nCHadronsPseudoadditional_);
    
    tree_events_->Branch("nBHadronsHiggs", &nBHadronsHiggs_);
    tree_events_->Branch("additionalJetEventId", &additionalJetEventId_);
    
    bookHadronBranches(tree_bHadrons_, 5);
    bookHadronBranches(tree_cHadrons_, 4);
}


void matchGenHFHadrons::bookHadronBranches(TTree* tree, const int flavour)
{
    tree->Branch("flavour", &hadron_flavour_);
    tree->Branch("nQuarksAll", &hadron_nQuarksAll_);
    tree->Branch("nQuarksSameFlavour", &hadron_nQuarksSameFlavour_);
    tree->Branch("fromTopWeakDecay", &hadron_fromTopWeakDecay_);
    
    tree->Branch("quark1_dR", &hadron_quark1_dR_);
    tree->Branch("quark2_dR", &hadron_quark2_dR_);
    tree->Branch("quark1_ptRatio", &hadron_quark1_ptRatio_);
    tree->Branch("quark2_ptRatio", &hadron_quark2_ptRatio_);
    
    tree->Branch("nLeptonsAll", &hadron_nLeptonsAll_);
    tree->Branch("nLeptonsViaTau", &hadron_nLeptonsViaTau_);
    
    tree->Branch("jetPtRatio", &hadron_jetPtRatio_);
    tree->Branch("jetDR", &hadron_jetDR_);
    
    if(flavour == 4) tree->Branch("bHadronId", &hadron_bHadronId_);
}


void matchGenHFHadrons::clearEventVariables()
{
    nJets_ = 0;
    
    nBJets_ = 0;
    nBHadrons_ = 0;
    nBHadronsTop_ = 0;
    nBHadronsAdditional_ = 0;
    nBHadronsPseudoadditional_ = 0;
    
    nCJets_ = 0;
    nCHadrons_ = 0;
    nCHadronsAdditional_ = 0;
    nCHadronsPseudoadditional_ = 0;
    
    nBHadronsHiggs_ = 0;
    additionalJetEventId_ += -1;
}


void matchGenHFHadrons::clearHadronVariables()
{
    hadron_flavour_ = 0;
    hadron_nQuarksAll_ = 0;
    hadron_nQuarksSameFlavour_ = 0;
    hadron_fromTopWeakDecay_ = -1;
    hadron_bHadronId_ = -1;
    
    hadron_quark1_dR_ = -0.1;
    hadron_quark2_dR_ = -0.1;
    hadron_quark1_ptRatio_ = -0.1;
    hadron_quark2_ptRatio_ = -0.1;
    
    
    hadron_nLeptonsAll_ = 0;
    hadron_nLeptonsViaTau_ = 0;
    
    hadron_jetPtRatio_ = -0.1;
    hadron_jetDR_ = -0.1;
}


void matchGenHFHadrons::fillHadronTree( TTree* tree, const int flavour, const reco::GenJetCollection& genJets,
                                        const std::vector<int>& genHadIndex, const std::vector<int>& genHadJetIndex,
                                        const std::vector<int>& genHadFlavour, const std::vector<int>& genHadFromTopWeakDecay,
                                        const std::vector<reco::GenParticle>& genHadPlusMothers, 
                                        const std::vector<std::vector<int> >& genHadPlusMothersIndices,
                                        const std::vector<int>& genHadLeptonHadronIndex, const std::vector<int>& genHadLeptonViaTau,
                                        const std::vector<int>& genHadBHadronId )
{
    for(size_t hadronId = 0; hadronId < genHadIndex.size(); ++hadronId) {
        clearHadronVariables();

        const int hadronParticleId = genHadIndex.at(hadronId);
        if(hadronParticleId < 0) continue;
        
        // Calculating hadron/jet pt ratio
        const int hadronJetId = genHadJetIndex.at(hadronId);
        if(hadronJetId >= 0) {
            hadron_jetPtRatio_ = genHadPlusMothers.at(genHadIndex.at(hadronId)).pt() / genJets.at(hadronJetId).pt();
            hadron_jetDR_ = ROOT::Math::VectorUtil::DeltaR(genHadPlusMothers.at(genHadIndex.at(hadronId)).polarP4(), genJets.at(hadronJetId).polarP4());
        }
        
        if(genHadBHadronId.size() > 0) hadron_bHadronId_ = genHadBHadronId.at(hadronId);
        hadron_fromTopWeakDecay_ = genHadFromTopWeakDecay.at(hadronId);
        
        const int hadronParticlePdgId = genHadPlusMothers.at(hadronParticleId).pdgId();
        hadron_flavour_ = genHadFlavour.at(hadronId);
        // Finding all b quark candidates
        std::set<int> hadronLastQuarks;
        findAncestorParticles(hadronParticleId, hadronLastQuarks, genHadPlusMothers, genHadPlusMothersIndices, flavour, true, 1);
        hadron_nQuarksAll_ = hadronLastQuarks.size();
        // Identifying flavour of the proper candidate quark
        int quarkCandidateFlavourSign = hadronParticlePdgId / std::abs(hadronParticlePdgId);
        // Inverting flavour if this is for a b meson
        if(std::abs(hadronParticlePdgId)/100 == 5 && std::abs(hadronParticlePdgId)%1000/10/11 != 5) quarkCandidateFlavourSign *= -1;
        // Selecting candidates that have proper flavour sign
        std::vector<std::pair<int, double> > hadronLastQuarks_sameFlavour;
        for(std::set<int>::iterator it=hadronLastQuarks.begin(); it!=hadronLastQuarks.end(); ++it) {
            const int quarkParticleId = *it;
            if(quarkParticleId < 0) continue;
            const int quarkParticlePdgId = genHadPlusMothers.at(quarkParticleId).pdgId();
            // Skipping quarks that have opposite flavour sign
            if(quarkParticlePdgId * quarkCandidateFlavourSign < 0) continue;
            double hadron_quark_dR = ROOT::Math::VectorUtil::DeltaR(genHadPlusMothers.at(hadronParticleId).polarP4(), genHadPlusMothers.at(quarkParticleId).polarP4());
            hadronLastQuarks_sameFlavour.push_back( std::pair<int, double>(quarkParticleId, hadron_quark_dR) );
        }
        hadron_nQuarksSameFlavour_ = hadronLastQuarks_sameFlavour.size();
        
        // Sorting quarks with proper flavour by their dR
        std::sort(hadronLastQuarks_sameFlavour.begin(), hadronLastQuarks_sameFlavour.end(), sort_pairs_second);
        const int hadronQuark1ParticleId = hadronLastQuarks_sameFlavour.size() > 0 ? hadronLastQuarks_sameFlavour.at(0).first : -1;
        const int hadronQuark2ParticleId = hadronLastQuarks_sameFlavour.size() > 1 ? hadronLastQuarks_sameFlavour.at(1).first : -1;
        
        if(hadronQuark1ParticleId >= 0) {
            hadron_quark1_dR_ = ROOT::Math::VectorUtil::DeltaR(genHadPlusMothers.at(hadronParticleId).polarP4(), genHadPlusMothers.at(hadronQuark1ParticleId).polarP4());
            hadron_quark1_ptRatio_ = genHadPlusMothers.at(hadronParticleId).pt() / genHadPlusMothers.at(hadronQuark1ParticleId).pt();
        }
        
        if(hadronQuark2ParticleId >= 0) {
            hadron_quark2_dR_ = ROOT::Math::VectorUtil::DeltaR(genHadPlusMothers.at(hadronParticleId).polarP4(), genHadPlusMothers.at(hadronQuark2ParticleId).polarP4());
            hadron_quark2_ptRatio_ = genHadPlusMothers.at(hadronParticleId).pt() / genHadPlusMothers.at(hadronQuark2ParticleId).pt();
        }
        
        // Counting number of leptons coming from the b hadron decay
        for(size_t leptonId = 0; leptonId < genHadLeptonHadronIndex.size(); ++leptonId) {
            const size_t leptonHadronId = genHadLeptonHadronIndex.at(leptonId);
            // Skipping the lepton id doesn't come from the current b hadron
            if(leptonHadronId != hadronId) continue;
            hadron_nLeptonsAll_++;
            if(genHadLeptonViaTau.at(leptonId) == 1) hadron_nLeptonsViaTau_++;
        }
        
    tree->Fill();
    }
}



DEFINE_FWK_MODULE( matchGenHFHadrons );
