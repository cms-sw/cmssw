////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

class SimpleMassShift: public edm::EDFilter {
public:
      SimpleMassShift(const edm::ParameterSet& pset);
      virtual ~SimpleMassShift();
      virtual bool filter(edm::Event &, const edm::EventSetup&);
      virtual void beginJob() ;
      virtual void endJob() ;
private:
      std::string selectorPath_;
      edm::InputTag genParticlesTag_;
      std::vector<edm::InputTag> weightTags_;
      unsigned int originalEvents_;
      unsigned int selectedEvents_;
      double mass_;
      std::vector<double> weightedEvents_;
      std::vector<double> weightedMass_;
};

////////// Source code ////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

/////////////////////////////////////////////////////////////////////////////////////
SimpleMassShift::SimpleMassShift(const edm::ParameterSet& pset) :
  selectorPath_(pset.getUntrackedParameter<std::string> ("SelectorPath","")),
  genParticlesTag_(pset.getUntrackedParameter<edm::InputTag> ("GenParticlesTag", edm::InputTag("genParticles"))),
  weightTags_(pset.getUntrackedParameter<std::vector<edm::InputTag> > ("WeightTags")) { 
}

/////////////////////////////////////////////////////////////////////////////////////
SimpleMassShift::~SimpleMassShift(){}

/////////////////////////////////////////////////////////////////////////////////////
void SimpleMassShift::beginJob(){
      originalEvents_ = 0;
      selectedEvents_ = 0;
      mass_ = 0.;
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "Uncertainties will be determined for the following tags: ";
      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\t" << weightTags_[i].encode();
            weightedEvents_.push_back(0.);
            weightedMass_.push_back(0.);
      }
}

/////////////////////////////////////////////////////////////////////////////////////
void SimpleMassShift::endJob(){
      if (originalEvents_==0) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "NO EVENTS => NO RESULTS";
            return;
      }
      if (selectedEvents_==0) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "NO SELECTED EVENTS => NO RESULTS";
            return;
      }

      edm::LogVerbatim("SimpleSystematicsAnalysis") << "\n>>>> Begin of Weight systematics summary >>>>";
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "Total number of analyzed data: " << originalEvents_ << " [events]";
      double originalAcceptance = double(selectedEvents_)/originalEvents_;
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "Total number of selected data: " << selectedEvents_ << " [events], corresponding to acceptance: " << originalAcceptance*100 << " [%]";

      double mass_ref = mass_/selectedEvents_;
      
      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "Results for Weight Tag: " << weightTags_[i].encode() << " ---->";

            double mass_central = 0.;
            if (weightedEvents_[i]>0) mass_central = weightedMass_[i]/weightedEvents_[i]; 
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\tMass shift with respect to reference: " << mass_central-mass_ref << " [GeV]";

      }
      edm::LogVerbatim("SimpleSystematicsAnalysis") << ">>>> End of Weight systematics summary >>>>";

}

/////////////////////////////////////////////////////////////////////////////////////
bool SimpleMassShift::filter(edm::Event & ev, const edm::EventSetup&){
      originalEvents_++;

      bool selectedEvent = false;
      edm::Handle<edm::TriggerResults> triggerResults;
      if (!ev.getByLabel(edm::InputTag("TriggerResults"), triggerResults)) {
            edm::LogError("SimpleSystematicsAnalysis") << ">>> TRIGGER collection does not exist !!!";
            return false;
      }

      const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);
      unsigned int pathIndex = trigNames.triggerIndex(selectorPath_);
      bool pathFound = (pathIndex>=0 && pathIndex<trigNames.size());
      if (pathFound) {
            if (triggerResults->accept(pathIndex)) selectedEvent = true;
      }
      //edm::LogVerbatim("SimpleSystematicsAnalysis") << ">>>> Path Name: " << selectorPath_ << ", selected? " << selectedEvent;

      if (selectedEvent) {
            selectedEvents_++;
      } else {
            return true;
      }

      edm::Handle<reco::GenParticleCollection> genParticles;
      ev.getByLabel(genParticlesTag_, genParticles);
      unsigned int gensize = genParticles->size();

      unsigned int nleptonsFound = 0;
      double mass = 0.;
      double en = 0.;
      double px = 0.;
      double py = 0.;
      double pz = 0.;
      for(unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& part = (*genParticles)[i];
            int status = part.status();
            if (status!=3) break;
            int id = part.pdgId();
            if (abs(id)!=13 && abs(id)!=11) continue;
            unsigned int nmothers = part.numberOfMothers();
            if (nmothers!=1) continue;
            size_t key = part.motherRef(0).key();
            int bosonId = (*genParticles)[key].pdgId();
            if (bosonId!=23 && abs(bosonId)!=24) continue;
            unsigned int ndaughters = part.numberOfDaughters();
            for(unsigned int j = 0; j<ndaughters; ++j) {
                  size_t key = part.daughterRef(j).key();
                  const reco::GenParticle* lepton = &((*genParticles)[key]);
                  if (abs(lepton->pdgId())!=13 && abs(lepton->pdgId())!=11) continue;
                  nleptonsFound += 1;
                  //printf("nleptons %d id %d\n", nleptonsFound, lepton->pdgId());
                  en += lepton->energy();
                  px += lepton->px();
                  py += lepton->py();
                  pz += lepton->pz();
            }
      }

      if (nleptonsFound==2) {
            mass = sqrt(en*en-px*px-py*py-pz*pz);
            mass_ += mass;
            //printf("Mass = %f [GeV]\n", mass);
      } else {
            edm::LogError("PDFAnalysis") << ">>> Boson not found!!";
            return false;
      }

      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::Handle<double> weightHandle;
            ev.getByLabel(weightTags_[i], weightHandle);
            if (selectedEvent) {
                  weightedEvents_[i] += (*weightHandle);
                  weightedMass_[i] += (*weightHandle)*mass;
            }
      }

      return true;
}

DEFINE_FWK_MODULE(SimpleMassShift);
