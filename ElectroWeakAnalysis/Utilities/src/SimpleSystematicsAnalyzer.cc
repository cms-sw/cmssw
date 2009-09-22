////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class SimpleSystematicsAnalyzer: public edm::EDFilter {
public:
      SimpleSystematicsAnalyzer(const edm::ParameterSet& pset);
      virtual ~SimpleSystematicsAnalyzer();
      virtual bool filter(edm::Event &, const edm::EventSetup&);
      virtual void beginJob(const edm::EventSetup& eventSetup) ;
      virtual void endJob() ;
private:
      std::vector<edm::InputTag> weightTags_;
      unsigned int originalEvents_;
      std::vector<double> weightedEvents_;
};

////////// Source code ////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

/////////////////////////////////////////////////////////////////////////////////////
SimpleSystematicsAnalyzer::SimpleSystematicsAnalyzer(const edm::ParameterSet& pset) :
  weightTags_(pset.getUntrackedParameter<std::vector<edm::InputTag> > ("WeightTags")) { 
}

/////////////////////////////////////////////////////////////////////////////////////
SimpleSystematicsAnalyzer::~SimpleSystematicsAnalyzer(){}

/////////////////////////////////////////////////////////////////////////////////////
void SimpleSystematicsAnalyzer::beginJob(const edm::EventSetup& eventSetup){
      originalEvents_ = 0;
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "Uncertainties will be determined for the following tags: ";
      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\t" << weightTags_[i].encode();
            weightedEvents_.push_back(0.);
      }
}

/////////////////////////////////////////////////////////////////////////////////////
void SimpleSystematicsAnalyzer::endJob(){
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "\n>>>> Begin of Weight systematics summary >>>>";
      edm::LogVerbatim("SimpleSystematicsAnalysis") << "Analyzed data (reference): " << originalEvents_ << " [events]";
      if (originalEvents_==0) return;
      
      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "Results for Weight Tag: " << weightTags_[i].encode() << " ---->";

            double events_central = weightedEvents_[i]; 
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\tData after reweighting: " << events_central << " [events]";
            edm::LogVerbatim("SimpleSystematicsAnalysis") << "\ti.e. " << std::setprecision(4) << 100*(events_central/originalEvents_-1.) << "% variation with respect to the original sample";

      }
      edm::LogVerbatim("SimpleSystematicsAnalysis") << ">>>> End of Weight systematics summary >>>>";

}

/////////////////////////////////////////////////////////////////////////////////////
bool SimpleSystematicsAnalyzer::filter(edm::Event & ev, const edm::EventSetup&){
      originalEvents_++;

      for (unsigned int i=0; i<weightTags_.size(); ++i) {
            edm::Handle<double> weightHandle;
            ev.getByLabel(weightTags_[i], weightHandle);
            weightedEvents_[i] += (*weightHandle);
      }

      return true;
}

DEFINE_FWK_MODULE(SimpleSystematicsAnalyzer);
