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


class genHFHadronMatcher : public edm::EDAnalyzer {
    public: 
        explicit genHFHadronMatcher(const edm::ParameterSet & );
        ~genHFHadronMatcher() {};
        
    private:
        virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
        virtual void beginJob();
        void clearEventVariables();
        void clearBHadronVariables();
        // A recursive function that scans through the particle chain to find a particle with specific pdgId or abs(pdgId)
        void findAncestorParticles(const int startId, std::set<int>& resultIds, 
                                   const std::vector<reco::GenParticle>& genParticles, const std::vector<std::vector<int> >& motherIndices, 
                                   const int endPdgId, const bool endPdgIdIsAbs = false, const int firstAnyLast = 0);
        
        
        // Jets configuration
        const double genJetPtMin_;
        const double genJetAbsEtaMax_;
        
        // Input tags
        const edm::InputTag genJetsTag_;
        
        const edm::InputTag genBHadJetIndexTag_;
        const edm::InputTag genBHadFlavourTag_;
        const edm::InputTag genBHadFromTopWeakDecayTag_;
        const edm::InputTag genBHadPlusMothersTag_;
        const edm::InputTag genBHadPlusMothersIndicesTag_;
        const edm::InputTag genBHadIndexTag_;
        const edm::InputTag genBHadLeptonHadronIndexTag_;
        const edm::InputTag genBHadLeptonViaTauTag_;
        
        const edm::InputTag genCHadJetIndexTag_;
        const edm::InputTag genCHadFlavourTag_;
        const edm::InputTag genCHadFromTopWeakDecayTag_;
        const edm::InputTag genCHadBHadronIdTag_;
        
        // Output to be written to the trees
        TTree* tree_events_;
        TTree* tree_bHadrons_;
        
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
        int bHadron_flavour_;
        int bHadron_nQuarksAll_;
        int bHadron_nQuarksSameFlavour_;
        
        double bHadron_quark1_dR_;
        double bHadron_quark2_dR_;
        double bHadron_quark1_ptRatio_;
        double bHadron_quark2_ptRatio_;
        
        int bHadron_nLeptonsAll_;
        int bHadron_nLeptonsViaTau_;
        
        double bHadron_jetPtRatio_;
};

bool sort_pairs_second(const std::pair<int, double>& a, const std::pair<int, double>& b) 
{
    return a.second < b.second;
}

genHFHadronMatcher::genHFHadronMatcher ( const edm::ParameterSet& config):
    genJetPtMin_(config.getParameter<double>("genJetPtMin")),
    genJetAbsEtaMax_(config.getParameter<double>("genJetAbsEtaMax")),
    genJetsTag_(config.getParameter<edm::InputTag>("genJets")),
    genBHadJetIndexTag_(config.getParameter<edm::InputTag>("genBHadJetIndex")),
    genBHadFlavourTag_(config.getParameter<edm::InputTag>("genBHadFlavour")),
    genBHadFromTopWeakDecayTag_(config.getParameter<edm::InputTag>("genBHadFromTopWeakDecay")),
    genBHadPlusMothersTag_(config.getParameter<edm::InputTag>("genBHadPlusMothers")),
    genBHadPlusMothersIndicesTag_(config.getParameter<edm::InputTag>("genBHadPlusMothersIndices")),
    genBHadIndexTag_(config.getParameter<edm::InputTag>("genBHadIndex")),
    genBHadLeptonHadronIndexTag_(config.getParameter<edm::InputTag>("genBHadLeptonHadronIndex")),
    genBHadLeptonViaTauTag_(config.getParameter<edm::InputTag>("genBHadLeptonViaTau")),
    genCHadJetIndexTag_(config.getParameter<edm::InputTag>("genCHadJetIndex")),
    genCHadFromTopWeakDecayTag_(config.getParameter<edm::InputTag>("genCHadFromTopWeakDecay")),
    genCHadBHadronIdTag_(config.getParameter<edm::InputTag>("genCHadBHadronId"))
{

}


void genHFHadronMatcher::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    this->clearEventVariables();
    
    // Reading gen jets from the event
    edm::Handle<reco::GenJetCollection> genJets;
    event.getByLabel(genJetsTag_, genJets);
    
    // Reading B hadrons related information
    edm::Handle<std::vector<int> > genBHadFlavour;
    event.getByLabel(genBHadFlavourTag_, genBHadFlavour);
    
    edm::Handle<std::vector<int> > genBHadJetIndex;
    event.getByLabel(genBHadJetIndexTag_, genBHadJetIndex);
    
    edm::Handle<std::vector<int> > genBHadFromTopWeakDecay;
    event.getByLabel(genBHadFromTopWeakDecayTag_, genBHadFromTopWeakDecay);
    
    edm::Handle<std::vector<reco::GenParticle> > genBHadPlusMothers;
    event.getByLabel(genBHadPlusMothersTag_, genBHadPlusMothers);
    
    edm::Handle<std::vector<std::vector<int> > > genBHadPlusMothersIndices;
    event.getByLabel(genBHadPlusMothersIndicesTag_, genBHadPlusMothersIndices);
    
    edm::Handle<std::vector<int> > genBHadIndex;
    event.getByLabel(genBHadIndexTag_, genBHadIndex);
    
    edm::Handle<std::vector<int> > genBHadLeptonHadronIndex;
    event.getByLabel(genBHadLeptonHadronIndexTag_, genBHadLeptonHadronIndex);
    
    edm::Handle<std::vector<int> > genBHadLeptonViaTau;
    event.getByLabel(genBHadLeptonViaTauTag_, genBHadLeptonViaTau);
    
    // Reading C hadrons related information
    edm::Handle<std::vector<int> > genCHadFlavour;
    event.getByLabel(genCHadFlavourTag_, genCHadFlavour);
    
    edm::Handle<std::vector<int> > genCHadJetIndex;
    event.getByLabel(genCHadJetIndexTag_, genCHadJetIndex);
    
    edm::Handle<std::vector<int> > genCHadFromTopWeakDecay;
    event.getByLabel(genCHadFromTopWeakDecayTag_, genCHadFromTopWeakDecay);
    
    edm::Handle<std::vector<int> > genCHadBHadronId;
    event.getByLabel(genCHadBHadronIdTag_, genCHadBHadronId);
    

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
        const bool afterTop = genBHadFromTopWeakDecay->at(hadronId) == 1 ? true : false;
        nBHadrons_++;
        if(flavourAbs==25) nBHadronsHiggs_++;
        if(flavourAbs==6) nBHadronsTop_++;
        if(flavourAbs!=6) {
            if(afterTop) nBHadronsPseudoadditional_++;
            else nBHadronsAdditional_++;
        }
    }
    // Counting number of different c hadrons
    for(size_t hadronId = 0; hadronId < genCHadJetIndex->size(); ++hadronId) {
        const bool afterTop = genCHadFromTopWeakDecay->at(hadronId) == 1 ? true : false;
        nCHadrons_++;
        // Skipping c hadrons that are coming from b hadrons
        if(genCHadBHadronId->at(hadronId) >= 0) continue;
        
        if(afterTop) nCHadronsPseudoadditional_++;
        else nCHadronsAdditional_++;
    }
    
    // Map <jet index, number of specific hadrons in the jet>
    // B jets with b hadrons directly from top quark decay
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
        // Whether hadron radiated before top quark decay
        const bool fromTopDecay = genBHadFromTopWeakDecay->at(hadronId);
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
        if(!fromTopDecay) {
            if(bJetBeforeTopIds.count(jetIndex) < 1) bJetBeforeTopIds[jetIndex] = 1;
            else bJetBeforeTopIds[jetIndex]++;
        }
        // Jet after top quark decay but not directly from top
        else if(fromTopDecay) {
            if(bJetAfterTopIds.count(jetIndex) < 1) bJetAfterTopIds[jetIndex] = 1;
            else bJetAfterTopIds[jetIndex]++;
        }
    }
    
    // Counting number of specific hadrons in each c jet
    for(size_t hadronId = 0; hadronId < genCHadJetIndex->size(); ++hadronId) {
        // Skipping c hadrons that are coming from b hadrons
        if(genCHadBHadronId->at(hadronId) >= 0) continue;
        // Index of a jet associated to the hadron
        const int jetIndex = genCHadJetIndex->at(hadronId);
        // Whether hadron radiated before top quark decay
        const bool fromTopDecay = genCHadFromTopWeakDecay->at(hadronId);
        // Skipping hadrons which have no associated jet
        if(jetIndex < 0) continue;
        // Skipping if jet is not in acceptance
        if(genJets->at(jetIndex).pt() < genJetPtMin_) continue;
        if(std::fabs(genJets->at(jetIndex).eta()) > genJetAbsEtaMax_) continue;
        // Jet before top quark decay
        if(!fromTopDecay) {
            if(cJetBeforeTopIds.count(jetIndex) < 1) cJetBeforeTopIds[jetIndex] = 1;
            else cJetBeforeTopIds[jetIndex]++;
        }
        // Jet after top quark decay but not directly from top
        else if(fromTopDecay) {
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
        if(bJetAfterTopIds.count(jetId) > 0) continue;
        pseudoadditionalBJetIds.push_back(jetId);
    }
    // Finding additional c jets
    std::vector<int> additionalCJetIds;
    for(std::map<int, int>::iterator it = cJetBeforeTopIds.begin(); it != cJetBeforeTopIds.end(); ++it) {
        const int jetId = it->first;
        additionalCJetIds.push_back(jetId);
    }
    // Finding pseudo-additional c jets (after top decay)
    std::vector<int> pseudoadditionalCJetIds;
    for(std::map<int, int>::iterator it = cJetAfterTopIds.begin(); it != cJetAfterTopIds.end(); ++it) {
        const int jetId = it->first;
        // Skipping the jet if it contains a b hadron directly from top quark decay
        if(cJetAfterTopIds.count(jetId) > 0) continue;
        pseudoadditionalCJetIds.push_back(jetId);
    }
    
    // Categorizing event based on number of additional b/c jets 
    // and number of corresponding hadrons in each of them
    additionalJetEventId_ = bJetFromTopIds.size()*100;
    // tt + 1 additional b jet
    if (additionalBJetIds.size() == 1) {
        int nHadronsInJet = bJetBeforeTopIds[additionalBJetIds.at(0)];
        // tt + 1 additional b jet from 1 additional b hadron
        if(nHadronsInJet == 1) additionalJetEventId_ = 51;
        // tt + 1 additional b jet from >=2 additional b hadrons
        else additionalJetEventId_ = 52;
    }
    // tt + 2 additional b jets
    else if (additionalBJetIds.size() > 1) {
        int nHadronsInJet1 = bJetBeforeTopIds[additionalBJetIds.at(0)];
        int nHadronsInJet2 = bJetBeforeTopIds[additionalBJetIds.at(1)];
        // tt + 2 additional b jets each from 1 additional b hadron
        if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId_ = 53;
        // tt + 2 additional b jets one of which from >=2 overlapping additional b hadrons
        else if(std::min(nHadronsInJet1, nHadronsInJet2) == 1 && std::max(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId_ = 54;
        // tt + 2 additional b jets each from >=2 additional b hadrons
        else if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId_ = 55;
    }
    // tt + no additional b jets
    else if(additionalBJetIds.size() == 0) {
        // tt + >=1 pseudo-additional b jet with b hadrons after top quark decay
        if(pseudoadditionalBJetIds.size() > 0) additionalJetEventId_ = 56;
        // tt + 1 additional c jet
        else if(additionalCJetIds.size() == 1) {
            int nHadronsInJet = cJetBeforeTopIds[additionalCJetIds.at(0)];
            // tt + 1 additional c jet from 1 additional c hadron
            if(nHadronsInJet == 1) additionalJetEventId_ = 41;
            // tt + 1 additional c jet from >=2 overlapping additional c hadrons
            else additionalJetEventId_ = 42;
        }
        // tt + >=2 additional c jets
        else if(additionalCJetIds.size() > 1) {
            int nHadronsInJet1 = cJetBeforeTopIds[additionalCJetIds.at(0)];
            int nHadronsInJet2 = cJetBeforeTopIds[additionalCJetIds.at(1)];
            // tt + 2 additional c jets each from 1 additional c hadron
            if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId_ = 43;
            // tt + 2 additional c jets one of which from >=2 overlapping additional c hadrons
            else if(std::min(nHadronsInJet1, nHadronsInJet2) == 1 && std::max(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId_ = 44;
            // tt + 2 additional c jets each from >=2 additional c hadrons
            else if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId_ = 45;
        }
        // tt + no additional c jets
        else if(additionalCJetIds.size() == 0) {
            // tt + >=1 pseudo-additional c jet with c hadrons after top quark decay
            if(pseudoadditionalCJetIds.size() > 0) additionalJetEventId_ = 46;
            // tt + light jets
            else additionalJetEventId_ = 0;
        }
    }
    
    
    // Performance information filled in a tree with entry for each b hadron #######################################
    // Looping over all b hadrons
    for(size_t hadronId = 0; hadronId < genBHadIndex->size(); ++hadronId) {
        clearBHadronVariables();

        const int hadronParticleId = genBHadIndex->at(hadronId);
        if(hadronParticleId < 0) continue;
        
        // Calculating hadron/jet pt ratio
        const int hadronJetId = genBHadJetIndex->at(hadronId);
        if(hadronJetId >= 0) {
            bHadron_jetPtRatio_ = genBHadPlusMothers->at(genBHadIndex->at(hadronId)).pt() / genJets->at(hadronJetId).pt();
        }
        
        const int hadronParticlePdgId = genBHadPlusMothers->at(hadronParticleId).pdgId();
        bHadron_flavour_ = genBHadFlavour->at(hadronId);
        // Finding all b quark candidates
        std::set<int> hadronLastQuarks;
        findAncestorParticles(hadronParticleId, hadronLastQuarks, *genBHadPlusMothers, *genBHadPlusMothersIndices, 5, true, 1);
        bHadron_nQuarksAll_ = hadronLastQuarks.size();
        // Identifying flavour of the proper candidate quark
        int quarkCandidateFlavourSign = hadronParticlePdgId > 0 ? 1 : -1;
        // Inverting flavour if this is a b meson
        if(hadronParticlePdgId / 1000 == 0) quarkCandidateFlavourSign *= -1;
        // Selecting candidates that have proper flavour sign
        std::vector<std::pair<int, double> > hadronLastQuarks_sameFlavour;
        for(std::set<int>::iterator it=hadronLastQuarks.begin(); it!=hadronLastQuarks.end(); ++it) {
            const int quarkParticleId = *it;
            if(quarkParticleId < 0) continue;
            const int quarkParticlePdgId = genBHadPlusMothers->at(quarkParticleId).pdgId();
            // Skipping quarks that have opposite flavour sign
            if(quarkParticlePdgId * quarkCandidateFlavourSign < 0) continue;
            double hadron_quark_dR = ROOT::Math::VectorUtil::DeltaR(genBHadPlusMothers->at(hadronParticleId).polarP4(), genBHadPlusMothers->at(quarkParticleId).polarP4());
            hadronLastQuarks_sameFlavour.push_back( std::pair<int, double>(quarkParticleId, hadron_quark_dR) );
        }
        bHadron_nQuarksSameFlavour_ = hadronLastQuarks_sameFlavour.size();
        
        // Sorting quarks with proper flavour by their dR
        std::sort(hadronLastQuarks_sameFlavour.begin(), hadronLastQuarks_sameFlavour.end(), sort_pairs_second);
        const int hadronQuark1ParticleId = hadronLastQuarks_sameFlavour.size() > 0 ? hadronLastQuarks_sameFlavour.at(0).first : -1;
        const int hadronQuark2ParticleId = hadronLastQuarks_sameFlavour.size() > 1 ? hadronLastQuarks_sameFlavour.at(1).first : -1;
        
        if(hadronQuark1ParticleId >= 0) {
            bHadron_quark1_dR_ = ROOT::Math::VectorUtil::DeltaR(genBHadPlusMothers->at(hadronParticleId).polarP4(), genBHadPlusMothers->at(hadronQuark1ParticleId).polarP4());
            bHadron_quark1_ptRatio_ = genBHadPlusMothers->at(hadronParticleId).pt() / genBHadPlusMothers->at(hadronQuark1ParticleId).pt();
        }
        
        if(hadronQuark2ParticleId >= 0) {
            bHadron_quark2_dR_ = ROOT::Math::VectorUtil::DeltaR(genBHadPlusMothers->at(hadronParticleId).polarP4(), genBHadPlusMothers->at(hadronQuark2ParticleId).polarP4());
            bHadron_quark2_ptRatio_ = genBHadPlusMothers->at(hadronParticleId).pt() / genBHadPlusMothers->at(hadronQuark2ParticleId).pt();
        }
        
        // Counting number of leptons coming from the b hadron decay
        for(size_t leptonId = 0; leptonId < genBHadLeptonHadronIndex->size(); ++leptonId) {
            const size_t leptonHadronId = genBHadLeptonHadronIndex->at(leptonId);
            // Skipping the lepton id doesn't come from the current b hadron
            if(leptonHadronId != hadronId) continue;
            bHadron_nLeptonsAll_++;
            if(genBHadLeptonViaTau->at(leptonId) == 1) bHadron_nLeptonsViaTau_++;
        }
        
        
    tree_bHadrons_->Fill();
    }
    
    tree_events_->Fill();
}


void genHFHadronMatcher::findAncestorParticles(const int startId, std::set<int>& resultIds, 
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


void genHFHadronMatcher::beginJob()
{
    edm::Service<TFileService> fileService;
    if(!fileService) throw edm::Exception(edm::errors::Configuration, "TFileService is not registered in cfg file");
    tree_events_ = fileService->make<TTree>("tree_events", "tree_events");
    tree_bHadrons_ = fileService->make<TTree>("tree_bHadrons", "tree_bHadrons");
    
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
    
    tree_bHadrons_->Branch("bHadron_flavour", &bHadron_flavour_);
    tree_bHadrons_->Branch("bHadron_nQuarksAll", &bHadron_nQuarksAll_);
    tree_bHadrons_->Branch("bHadron_nQuarksSameFlavour", &bHadron_nQuarksSameFlavour_);
    
    tree_bHadrons_->Branch("bHadron_quark1_dR", &bHadron_quark1_dR_);
    tree_bHadrons_->Branch("bHadron_quark2_dR", &bHadron_quark2_dR_);
    tree_bHadrons_->Branch("bHadron_quark1_ptRatio", &bHadron_quark1_ptRatio_);
    tree_bHadrons_->Branch("bHadron_quark2_ptRatio", &bHadron_quark2_ptRatio_);
    
    tree_bHadrons_->Branch("bHadron_nLeptonsAll", &bHadron_nLeptonsAll_);
    tree_bHadrons_->Branch("bHadron_nLeptonsViaTau", &bHadron_nLeptonsViaTau_);
    
    tree_bHadrons_->Branch("bHadron_jetPtRatio", &bHadron_jetPtRatio_);
}


void genHFHadronMatcher::clearEventVariables()
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
    additionalJetEventId_ = -1;
}


void genHFHadronMatcher::clearBHadronVariables()
{
    bHadron_flavour_ = 0;
    bHadron_nQuarksAll_ = 0;
    bHadron_nQuarksSameFlavour_ = 0;
    
    bHadron_quark1_dR_ = -0.1;
    bHadron_quark2_dR_ = -0.1;
    bHadron_quark1_ptRatio_ = -0.1;
    bHadron_quark2_ptRatio_ = -0.1;
    
    
    bHadron_nLeptonsAll_ = 0;
    bHadron_nLeptonsViaTau_ = 0;
    
    bHadron_jetPtRatio_ = -0.1;
}




DEFINE_FWK_MODULE( genHFHadronMatcher );