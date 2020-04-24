
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Parse.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <fstream>

class MultiEventFilter : public edm::EDFilter {

  class Event {
    public:
      Event(edm::RunNumber_t r, edm::LuminosityBlockNumber_t l, edm::EventNumber_t e) : run(r), lumi(l), event(e) {}
      edm::RunNumber_t run;
      edm::LuminosityBlockNumber_t lumi;
      edm::EventNumber_t event;
  };

  public:

    explicit MultiEventFilter(const edm::ParameterSet & iConfig);
    ~MultiEventFilter() override {}

  private:

    bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup) override;
    
    std::vector<Event> events_;
    const std::vector<std::string> eventList_;

    const bool taggingMode_;

};


MultiEventFilter::MultiEventFilter(const edm::ParameterSet & iConfig)
  : eventList_ (iConfig.getParameter<std::vector<std::string> >("EventList") )
  , taggingMode_ (iConfig.getParameter<bool>("taggingMode") )
{
  edm::FileInPath fp = iConfig.getParameter<edm::FileInPath>("file");
  std::string fFile = fp.fullPath();
  std::ifstream inStream(fFile.c_str());

  for (unsigned int i = 0; i < eventList_.size(); ++i) {
    std::vector<std::string> tokens = edm::tokenize(eventList_[i], ":");
    if(tokens.size() != 3) {
      throw edm::Exception(edm::errors::Configuration) << "Incorrect event specification";
      continue;
    }
    events_.push_back(Event(atoi(tokens[0].c_str()), atoi(tokens[1].c_str()), atoi(tokens[2].c_str())));
  }

  std::string line;
  while( getline(inStream, line) ){
     std::vector<std::string> tokens = edm::tokenize(line, ":");
     if(tokens.size() != 3) {
      throw edm::Exception(edm::errors::Configuration) << "Incorrect event specification";
      continue;
    }
    events_.push_back(Event(atoi(tokens[0].c_str()), atoi(tokens[1].c_str()), atoi(tokens[2].c_str())));
  }

  produces<bool>();
}


bool MultiEventFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  bool pass = true;

  for (unsigned int i = 0; i < events_.size(); ++i) {
    if (events_[i].event == iEvent.id().event() &&
        events_[i].run == iEvent.id().run() &&
        events_[i].lumi == iEvent.id().luminosityBlock()) pass = false; 
  }

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass;

}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MultiEventFilter);
