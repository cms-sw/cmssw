////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

class PdfMassShift: public edm::EDFilter {
public:
      PdfMassShift(const edm::ParameterSet& pset);
      virtual ~PdfMassShift();
      virtual bool filter(edm::Event &, const edm::EventSetup&);
      virtual void beginJob() ;
      virtual void endJob() ;
private:
      std::string selectorPath_;
      edm::InputTag genParticlesTag_;
      std::vector<edm::InputTag> pdfWeightTags_;
      unsigned int originalEvents_;
      unsigned int selectedEvents_;
      double massSelectedEvents_;
      std::vector<int> pdfStart_;
      std::vector<double> weightedSelectedEvents_;
      std::vector<double> weightedMassSelectedEvents_;
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
PdfMassShift::PdfMassShift(const edm::ParameterSet& pset) :
  selectorPath_(pset.getUntrackedParameter<std::string> ("SelectorPath","")),
  genParticlesTag_(pset.getUntrackedParameter<edm::InputTag> ("GenParticlesTag", edm::InputTag("genParticles"))),
  pdfWeightTags_(pset.getUntrackedParameter<std::vector<edm::InputTag> > ("PdfWeightTags")) { 
}

/////////////////////////////////////////////////////////////////////////////////////
PdfMassShift::~PdfMassShift(){}

/////////////////////////////////////////////////////////////////////////////////////
void PdfMassShift::beginJob(){
      originalEvents_ = 0;
      selectedEvents_ = 0;
      massSelectedEvents_ = 0;
      edm::LogVerbatim("PDFAnalysis") << "PDF uncertainties will be determined for the following sets: ";
      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            edm::LogVerbatim("PDFAnalysis") << "\t" << pdfWeightTags_[i].instance();
            pdfStart_.push_back(-1);
      }
}

/////////////////////////////////////////////////////////////////////////////////////
void PdfMassShift::endJob(){

      if (originalEvents_==0) {
            edm::LogVerbatim("PDFAnalysis") << "NO EVENTS => NO RESULTS";
            return;
      }
      if (selectedEvents_==0) {
            edm::LogVerbatim("PDFAnalysis") << "NO SELECTED EVENTS => NO RESULTS";
            return;
      }

      edm::LogVerbatim("PDFAnalysis") << "\n>>>> Begin of PDF weight systematics summary >>>>";
      edm::LogVerbatim("PDFAnalysis") << "Total number of analyzed data: " << originalEvents_ << " [events]";
      double originalAcceptance = double(selectedEvents_)/originalEvents_;
      edm::LogVerbatim("PDFAnalysis") << "Total number of selected data: " << selectedEvents_ << " [events], corresponding to acceptance: " << originalAcceptance*100 << " [%]";

      edm::LogVerbatim("PDFAnalysis") << "\n>>>>> MASS SHIFT ON SELECTED EVENTS DUE TO PDFs >>>>>>";
      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            unsigned int nmembers = weightedMassSelectedEvents_.size()-pdfStart_[i];
            if (i<pdfWeightTags_.size()-1) nmembers = pdfStart_[i+1] - pdfStart_[i];
            unsigned int npairs = (nmembers-1)/2;
            edm::LogVerbatim("PDFAnalysis") << "Results for PDF set " << pdfWeightTags_[i].instance() << " ---->";

            double massReference = massSelectedEvents_/selectedEvents_;
            double massCentral = weightedMassSelectedEvents_[pdfStart_[i]]
                        / weightedSelectedEvents_[pdfStart_[i]];
            edm::LogVerbatim("PDFAnalysis") << "\tMass shift for central PDF member: " << massCentral-massReference << " [GeV]";

            if (npairs>0) {
                  edm::LogVerbatim("PDFAnalysis") << "\tNumber of eigenvectors for uncertainty estimation: " << npairs;
              double wplus = 0.;
              double wminus = 0.;
              for (unsigned int j=0; j<npairs; ++j) {
                  double wa = weightedMassSelectedEvents_[pdfStart_[i]+2*j+1]
                            / weightedSelectedEvents_[pdfStart_[i]+2*j+1]
                        - massCentral;
                  double wb = weightedMassSelectedEvents_[pdfStart_[i]+2*j+2]
                            / weightedSelectedEvents_[pdfStart_[i]+2*j+2]
                        - massCentral;
                  if (wa>wb) {
                        if (wa<0.) wa = 0.;
                        if (wb>0.) wb = 0.;
                        wplus += wa*wa;
                        wminus += wb*wb;
                  } else {
                        if (wb<0.) wb = 0.;
                        if (wa>0.) wa = 0.;
                        wplus += wb*wb;
                        wminus += wa*wa;
                  }
              }
              if (wplus>0) wplus = sqrt(wplus);
              if (wminus>0) wminus = sqrt(wminus);
              edm::LogVerbatim("PDFAnalysis") << "\tShift uncertainty with respect to central member: +" << std::setprecision(4) << wplus << " / -" << std::setprecision(4) << wminus << " [GeV]";
            } else {
                  edm::LogVerbatim("PDFAnalysis") << "\tNO eigenvectors for uncertainty estimation";
            }
      }

      edm::LogVerbatim("PDFAnalysis") << ">>>> End of PDF weight systematics summary >>>>";

}

/////////////////////////////////////////////////////////////////////////////////////
bool PdfMassShift::filter(edm::Event & ev, const edm::EventSetup&){
      originalEvents_++;

      bool selectedEvent = false;
      edm::Handle<edm::TriggerResults> triggerResults;
      if (!ev.getByLabel(edm::InputTag("TriggerResults"), triggerResults)) {
            edm::LogError("PDFAnalysis") << ">>> TRIGGER collection does not exist !!!";
            return false;
      }

      const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);

      unsigned int pathIndex = trigNames.triggerIndex(selectorPath_);
      bool pathFound = (pathIndex>=0 && pathIndex<trigNames.size());
      if (pathFound) {
            if (triggerResults->accept(pathIndex)) selectedEvent = true;
      }
      //edm::LogVerbatim("PDFAnalysis") << ">>>> Path Name: " << selectorPath_ << ", selected? " << selectedEvent;

      if (selectedEvent) {
            selectedEvents_++;
      } else {
            return true;
      }

      edm::Handle<reco::GenParticleCollection> genParticles;
      ev.getByLabel(genParticlesTag_, genParticles);
      unsigned int gensize = genParticles->size();

      double mass = 0.;
      const reco::GenParticle* boson = 0;
      for(unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& part = (*genParticles)[i];
            int id = part.pdgId();
            if (id!=23 && abs(id)!=24) continue;
            int status = part.status();
            if (status!=3) continue;
            int nmothers = part.numberOfMothers();
            if (nmothers!=2) continue;
            boson = &part;
            mass = boson->mass();
            break;
      }
      if (!boson) {
            edm::LogError("PDFAnalysis") << ">>> Boson not found!!";
            return false;
      }

      massSelectedEvents_ += mass;

      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            edm::Handle<std::vector<double> > weightHandle;
            if (!ev.getByLabel(pdfWeightTags_[i], weightHandle)) {
                  edm::LogError("PDFAnalysis") << ">>> Weights not found: " << pdfWeightTags_[i].encode() << " !!!";
                  return false;
            }
            std::vector<double> weights = (*weightHandle);
            unsigned int nmembers = weights.size();
            // Set up arrays the first time weights are read
            if (pdfStart_[i]<0) {
                  pdfStart_[i] = weightedSelectedEvents_.size();
                  for (unsigned int j=0; j<nmembers; ++j) {
                        weightedSelectedEvents_.push_back(0.);
                        weightedMassSelectedEvents_.push_back(0.);
                  }
            }

            for (unsigned int j=0; j<nmembers; ++j) {
                  weightedSelectedEvents_[pdfStart_[i]+j] += weights[j];
                  weightedMassSelectedEvents_[pdfStart_[i]+j] += mass*weights[j];
            }

      }

      return true;
}

DEFINE_FWK_MODULE(PdfMassShift);
