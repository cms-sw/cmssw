////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/TriggerResults.h"

class SimpleSystematicsAnalyzer: public edm::EDFilter {
public:
      SimpleSystematicsAnalyzer(const edm::ParameterSet& pset);
      virtual ~SimpleSystematicsAnalyzer();
      virtual bool filter(edm::Event &, const edm::EventSetup&) override;
      virtual void beginJob() override ;
      virtual void endJob() override ;
private:
      std::string selectorPath_;
      std::vector<edm::InputTag> weightTags_;
      std::vector<edm::EDGetTokenT<double> > weightTokens_;
      edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
      unsigned int originalEvents_;
      std::vector<double> weightedEvents_;
      unsigned int selectedEvents_;
      std::vector<double> weightedSelectedEvents_;
      std::vector<double> weighted2SelectedEvents_;
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
SimpleSystematicsAnalyzer::SimpleSystematicsAnalyzer(const edm::ParameterSet& pset) :
  selectorPath_(pset.getUntrackedParameter<std::string> ("SelectorPath","")),
  weightTags_(pset.getUntrackedParameter<std::vector<edm::InputTag> > ("WeightTags")),
  weightTokens_(edm::vector_transform(weightTags_, [this](edm::InputTag const & tag){return consumes<double>(tag);})),
  triggerResultsToken_(consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"))) {
}

/////////////////////////////////////////////////////////////////////////////////////
SimpleSystematicsAnalyzer::~SimpleSystematicsAnalyzer(){}

/////////////////////////////////////////////////////////////////////////////////////
void SimpleSystematicsAnalyzer::beginJob(){
      originalEvents_ = 0;
      selectedEvents_ = 0;
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "Uncertainties will be determined for the following tags: ";
      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\t" << weightTags_[i].encode();
            weightedEvents_.push_back(0.);
            weightedSelectedEvents_.push_back(0.);
            weighted2SelectedEvents_.push_back(0.);
      }
}

/////////////////////////////////////////////////////////////////////////////////////
void SimpleSystematicsAnalyzer::endJob(){
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
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "Total number of selected data: " << selectedEvents_ << " [events], corresponding to acceptance: [" << originalAcceptance*100 << " +- " << 100*sqrt( originalAcceptance*(1.-originalAcceptance)/originalEvents_) << "] %";

      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "Results for Weight Tag: " << weightTags_[i].encode() << " ---->";

            double acc_central = 0.;
            double acc2_central = 0.;
            if (weightedEvents_[i]>0) {
                  acc_central = weightedSelectedEvents_[i]/weightedEvents_[i];
                  acc2_central = weighted2SelectedEvents_[i]/weightedEvents_[i];
            }
            double waverage = weightedEvents_[i]/originalEvents_;
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\tTotal Events after reweighting: " << weightedEvents_[i] << " [events]";
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\tEvents selected after reweighting: " << weightedSelectedEvents_[i] << " [events]";
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\tAcceptance after reweighting: [" << acc_central*100 << " +- " <<
            100*sqrt((acc2_central/waverage-acc_central*acc_central)/originalEvents_)
            << "] %";
            double xi = acc_central-originalAcceptance;
            double deltaxi = (acc2_central-(originalAcceptance+2*xi+xi*xi))/originalEvents_;
            if (deltaxi>0) deltaxi = sqrt(deltaxi); else deltaxi = 0.;
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\ti.e. [" << std::setprecision(4) << 100*xi/originalAcceptance << " +- " << std::setprecision(4) << 100*deltaxi/originalAcceptance << "] % relative variation with respect to the original acceptance";

      }
      edm::LogVerbatim("SimpleSystematicsAnalysis") << ">>>> End of Weight systematics summary >>>>";

}

/////////////////////////////////////////////////////////////////////////////////////
bool SimpleSystematicsAnalyzer::filter(edm::Event & ev, const edm::EventSetup&){
      originalEvents_++;

      bool selectedEvent = false;
      edm::Handle<edm::TriggerResults> triggerResults;
      if (!ev.getByToken(triggerResultsToken_, triggerResults)) {
            edm::LogError("SimpleSystematicsAnalysis") << ">>> TRIGGER collection does not exist !!!";
            return false;
      }

      const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);
      unsigned int pathIndex = trigNames.triggerIndex(selectorPath_);
      bool pathFound = (pathIndex<trigNames.size()); // pathIndex >= 0, since pathIndex is unsigned
      if (pathFound) {
            if (triggerResults->accept(pathIndex)) selectedEvent = true;
      }
      //edm::LogVerbatim("SimpleSystematicsAnalysis") << ">>>> Path Name: " << selectorPath_ << ", selected? " << selectedEvent;

      if (selectedEvent) selectedEvents_++;

      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::Handle<double> weightHandle;
            ev.getByToken(weightTokens_[i], weightHandle);
            weightedEvents_[i] += (*weightHandle);
            if (selectedEvent) {
                  weightedSelectedEvents_[i] += (*weightHandle);
                  weighted2SelectedEvents_[i] += pow((*weightHandle),2);
            }
      }

      return true;
}

DEFINE_FWK_MODULE(SimpleSystematicsAnalyzer);
