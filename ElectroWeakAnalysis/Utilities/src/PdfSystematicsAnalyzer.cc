////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/TriggerResults.h"

class PdfSystematicsAnalyzer: public edm::EDFilter {
public:
      PdfSystematicsAnalyzer(const edm::ParameterSet& pset);
      virtual ~PdfSystematicsAnalyzer();
      virtual bool filter(edm::Event &, const edm::EventSetup&) override;
      virtual void beginJob() override ;
      virtual void endJob() override ;
private:
      std::string selectorPath_;
      std::vector<edm::InputTag> pdfWeightTags_;
      std::vector<edm::EDGetTokenT<std::vector<double> > > pdfWeightTokens_;
      edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
      unsigned int originalEvents_;
      unsigned int selectedEvents_;
      std::vector<int> pdfStart_;
      std::vector<double> weightedSelectedEvents_;
      std::vector<double> weighted2SelectedEvents_;
      std::vector<double> weightedEvents_;
};

////////// Source code ////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "FWCore/Utilities/interface/transform.h"

/////////////////////////////////////////////////////////////////////////////////////
PdfSystematicsAnalyzer::PdfSystematicsAnalyzer(const edm::ParameterSet& pset) :
  selectorPath_(pset.getUntrackedParameter<std::string> ("SelectorPath","")),
  pdfWeightTags_(pset.getUntrackedParameter<std::vector<edm::InputTag> > ("PdfWeightTags")),
  pdfWeightTokens_(edm::vector_transform(pdfWeightTags_, [this](edm::InputTag const & tag){return consumes<std::vector<double> >(tag);})),
  triggerResultsToken_(consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"))) {
}

/////////////////////////////////////////////////////////////////////////////////////
PdfSystematicsAnalyzer::~PdfSystematicsAnalyzer(){}

/////////////////////////////////////////////////////////////////////////////////////
void PdfSystematicsAnalyzer::beginJob(){
      originalEvents_ = 0;
      selectedEvents_ = 0;
      edm::LogVerbatim("PDFAnalysis") << "PDF uncertainties will be determined for the following sets: ";
      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            edm::LogVerbatim("PDFAnalysis") << "\t" << pdfWeightTags_[i].instance();
            pdfStart_.push_back(-1);
      }
}

/////////////////////////////////////////////////////////////////////////////////////
void PdfSystematicsAnalyzer::endJob(){

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
      edm::LogVerbatim("PDFAnalysis") << "Total number of selected data: " << selectedEvents_ << " [events], corresponding to acceptance: [" << originalAcceptance*100 << " +- " << 100*sqrt( originalAcceptance*(1.-originalAcceptance)/originalEvents_) << "] %";

      edm::LogVerbatim("PDFAnalysis") << "\n>>>>> PDF UNCERTAINTIES ON RATE >>>>>>";
      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            bool nnpdfFlag = (pdfWeightTags_[i].instance().substr(0,5)=="NNPDF");
            unsigned int nmembers = weightedSelectedEvents_.size()-pdfStart_[i];
            if (i<pdfWeightTags_.size()-1) nmembers = pdfStart_[i+1] - pdfStart_[i];
            unsigned int npairs = (nmembers-1)/2;
            edm::LogVerbatim("PDFAnalysis") << "RATE Results for PDF set " << pdfWeightTags_[i].instance() << " ---->";

            double events_central = weightedSelectedEvents_[pdfStart_[i]];
            edm::LogVerbatim("PDFAnalysis") << "\tEstimate for central PDF member: " << int(events_central) << " [events]";
            double events2_central = weighted2SelectedEvents_[pdfStart_[i]];
            edm::LogVerbatim("PDFAnalysis") << "\ti.e. [" << std::setprecision(4) << 100*(events_central-selectedEvents_)/selectedEvents_ << " +- " <<
                100*sqrt(events2_central-events_central+selectedEvents_*(1-originalAcceptance))/selectedEvents_
            << "] % relative variation with respect to original PDF";

            if (npairs>0) {
                  edm::LogVerbatim("PDFAnalysis") << "\tNumber of eigenvectors for uncertainty estimation: " << npairs;
              double wplus = 0.;
              double wminus = 0.;
              unsigned int nplus = 0;
              unsigned int nminus = 0;
              for (unsigned int j=0; j<npairs; ++j) {
                  double wa = weightedSelectedEvents_[pdfStart_[i]+2*j+1]/events_central-1.;
                  double wb = weightedSelectedEvents_[pdfStart_[i]+2*j+2]/events_central-1.;
                  if (nnpdfFlag) {
                        if (wa>0.) {
                              wplus += wa*wa;
                              nplus++;
                        } else {
                              wminus += wa*wa;
                              nminus++;
                        }
                        if (wb>0.) {
                              wplus += wb*wb;
                              nplus++;
                        } else {
                              wminus += wb*wb;
                              nminus++;
                        }
                  } else {
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
              }
              if (wplus>0) wplus = sqrt(wplus);
              if (wminus>0) wminus = sqrt(wminus);
              if (nnpdfFlag) {
                  if (nplus>0) wplus /= sqrt(nplus);
                  if (nminus>0) wminus /= sqrt(nminus);
              }
              edm::LogVerbatim("PDFAnalysis") << "\tRelative uncertainty with respect to central member: +" << std::setprecision(4) << 100.*wplus << " / -" << std::setprecision(4) << 100.*wminus << " [%]";
            } else {
                  edm::LogVerbatim("PDFAnalysis") << "\tNO eigenvectors for uncertainty estimation";
            }
      }

      edm::LogVerbatim("PDFAnalysis") << "\n>>>>> PDF UNCERTAINTIES ON ACCEPTANCE >>>>>>";
      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            bool nnpdfFlag = (pdfWeightTags_[i].instance().substr(0,5)=="NNPDF");
            unsigned int nmembers = weightedEvents_.size()-pdfStart_[i];
            if (i<pdfWeightTags_.size()-1) nmembers = pdfStart_[i+1] - pdfStart_[i];
            unsigned int npairs = (nmembers-1)/2;
            edm::LogVerbatim("PDFAnalysis") << "ACCEPTANCE Results for PDF set " << pdfWeightTags_[i].instance() << " ---->";

            double acc_central = 0.;
            double acc2_central = 0.;
            if (weightedEvents_[pdfStart_[i]]>0) {
                  acc_central = weightedSelectedEvents_[pdfStart_[i]]/weightedEvents_[pdfStart_[i]];
                  acc2_central = weighted2SelectedEvents_[pdfStart_[i]]/weightedEvents_[pdfStart_[i]];
            }
            double waverage = weightedEvents_[pdfStart_[i]]/originalEvents_;
            edm::LogVerbatim("PDFAnalysis") << "\tEstimate for central PDF member acceptance: [" << acc_central*100 << " +- " <<
            100*sqrt((acc2_central/waverage-acc_central*acc_central)/originalEvents_)
            << "] %";
            double xi = acc_central-originalAcceptance;
            double deltaxi = (acc2_central-(originalAcceptance+2*xi+xi*xi))/originalEvents_;
            if (deltaxi>0) deltaxi = sqrt(deltaxi); //else deltaxi = 0.;
            edm::LogVerbatim("PDFAnalysis") << "\ti.e. [" << std::setprecision(4) << 100*xi/originalAcceptance << " +- " << std::setprecision(4) << 100*deltaxi/originalAcceptance << "] % relative variation with respect to the original PDF";

            if (npairs>0) {
                  edm::LogVerbatim("PDFAnalysis") << "\tNumber of eigenvectors for uncertainty estimation: " << npairs;
              double wplus = 0.;
              double wminus = 0.;
              unsigned int nplus = 0;
              unsigned int nminus = 0;
              for (unsigned int j=0; j<npairs; ++j) {
                  double wa = 0.;
                  if (weightedEvents_[pdfStart_[i]+2*j+1]>0) wa = (weightedSelectedEvents_[pdfStart_[i]+2*j+1]/weightedEvents_[pdfStart_[i]+2*j+1])/acc_central-1.;
                  double wb = 0.;
                  if (weightedEvents_[pdfStart_[i]+2*j+2]>0) wb = (weightedSelectedEvents_[pdfStart_[i]+2*j+2]/weightedEvents_[pdfStart_[i]+2*j+2])/acc_central-1.;
                  if (nnpdfFlag) {
                        if (wa>0.) {
                              wplus += wa*wa;
                              nplus++;
                        } else {
                              wminus += wa*wa;
                              nminus++;
                        }
                        if (wb>0.) {
                              wplus += wb*wb;
                              nplus++;
                        } else {
                              wminus += wb*wb;
                              nminus++;
                        }
                  } else {
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
              }
              if (wplus>0) wplus = sqrt(wplus);
              if (wminus>0) wminus = sqrt(wminus);
              if (nnpdfFlag) {
                  if (nplus>0) wplus /= sqrt(nplus);
                  if (nminus>0) wminus /= sqrt(nminus);
              }
              edm::LogVerbatim("PDFAnalysis") << "\tRelative uncertainty with respect to central member: +" << std::setprecision(4) << 100.*wplus << " / -" << std::setprecision(4) << 100.*wminus << " [%]";
            } else {
                  edm::LogVerbatim("PDFAnalysis") << "\tNO eigenvectors for uncertainty estimation";
            }
      }
      edm::LogVerbatim("PDFAnalysis") << ">>>> End of PDF weight systematics summary >>>>";

}

/////////////////////////////////////////////////////////////////////////////////////
bool PdfSystematicsAnalyzer::filter(edm::Event & ev, const edm::EventSetup&){

      edm::Handle<std::vector<double> > weightHandle;
      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            if (!ev.getByToken(pdfWeightTokens_[i], weightHandle)) {
                  if (originalEvents_==0) {
                        edm::LogError("PDFAnalysis") << ">>> WARNING: some weights not found!";
                        edm::LogError("PDFAnalysis") << ">>> But maybe OK, if you are prefiltering!";
                        edm::LogError("PDFAnalysis") << ">>> If things are OK, this warning should disappear after a while!";
                  }
                  return false;
            }
      }

      originalEvents_++;

      bool selectedEvent = false;
      edm::Handle<edm::TriggerResults> triggerResults;
      if (!ev.getByToken(triggerResultsToken_, triggerResults)) {
            edm::LogError("PDFAnalysis") << ">>> TRIGGER collection does not exist !!!";
            return false;
      }


      const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);
      unsigned int pathIndex = trigNames.triggerIndex(selectorPath_);
      bool pathFound = (pathIndex<trigNames.size()); // pathIndex >= 0, since pathIndex is unsigned
      if (pathFound) {
            if (triggerResults->accept(pathIndex)) selectedEvent = true;
      }
      //edm::LogVerbatim("PDFAnalysis") << ">>>> Path Name: " << selectorPath_ << ", selected? " << selectedEvent;

      if (selectedEvent) selectedEvents_++;

      for (unsigned int i=0; i<pdfWeightTags_.size(); ++i) {
            if (!ev.getByToken(pdfWeightTokens_[i], weightHandle)) return false;
            std::vector<double> weights = (*weightHandle);
            unsigned int nmembers = weights.size();
            // Set up arrays the first time wieghts are read
            if (pdfStart_[i]<0) {
                  pdfStart_[i] = weightedEvents_.size();
                  for (unsigned int j=0; j<nmembers; ++j) {
                        weightedEvents_.push_back(0.);
                        weightedSelectedEvents_.push_back(0.);
                        weighted2SelectedEvents_.push_back(0.);
                  }
            }

            for (unsigned int j=0; j<nmembers; ++j) {
                  weightedEvents_[pdfStart_[i]+j] += weights[j];
                  if (selectedEvent) {
                        weightedSelectedEvents_[pdfStart_[i]+j] += weights[j];
                        weighted2SelectedEvents_[pdfStart_[i]+j] += weights[j]*weights[j];
                  }
            }

            /*
            printf("\n>>>>>>>>> Run %8d Event %d, members %3d PDF set %s : Weights >>>> \n", ev.id().run(), ev.id().event(), nmembers, pdfWeightTags_[i].instance().data());
            for (unsigned int i=0; i<nmembers; i+=5) {
                  for (unsigned int j=0; ((j<5)&&(i+j<nmembers)); ++j) {
                        printf(" %2d: %7.4f", i+j, weights[i+j]);
                  }
                  safe_printf("\n");
            }
            */

      }

      return true;
}

DEFINE_FWK_MODULE(PdfSystematicsAnalyzer);
