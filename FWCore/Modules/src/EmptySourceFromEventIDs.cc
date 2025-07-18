#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/IDGeneratorSourceBase.h"

/*
 * IDGeneratorSourceBase implements the logic to generate run, lumi, and event numbers, and event timestamps.
 * These will actually be overwritten by this source, but it's easier to do that than to write a new source base
 * type from scratch.
 */

class EmptySourceFromEventIDs : public edm::IDGeneratorSourceBase<edm::InputSource> {
public:
  explicit EmptySourceFromEventIDs(edm::ParameterSet const&, edm::InputSourceDescription const&);
  ~EmptySourceFromEventIDs() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time, edm::EventAuxiliary::ExperimentType& type) override;
  void readEvent_(edm::EventPrincipal& e) override;

  std::vector<edm::EventID> events_;
};

// Note that almost all configuration parameters passed to IDGeneratorSourceBase will effectively be ignored, because
// the EmptySourceFromEventIDs will explicitly set the run, lumi, and event numbers, the timestamp, and the event type.
EmptySourceFromEventIDs::EmptySourceFromEventIDs(edm::ParameterSet const& config,
                                                 edm::InputSourceDescription const& desc)
    : IDGeneratorSourceBase<InputSource>(config, desc, false),
      events_{config.getUntrackedParameter<std::vector<edm::EventID>>("events")}  // List of event ids to create
{
  // Invert the order of the events so they can efficiently be popped off the back of the vector
  std::reverse(events_.begin(), events_.end());
}

bool EmptySourceFromEventIDs::setRunAndEventInfo(edm::EventID& event,
                                                 edm::TimeValue_t& time,
                                                 edm::EventAuxiliary::ExperimentType& type) {
  if (events_.empty()) {
    return false;
  }

  event = events_.back();
  events_.pop_back();
  return true;
}

void EmptySourceFromEventIDs::readEvent_(edm::EventPrincipal& e) {
  doReadEvent(e, [](auto const&) {});
}

void EmptySourceFromEventIDs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Creates runs, lumis and events (containing no products) based on the provided list of event ids.");
  edm::IDGeneratorSourceBase<edm::InputSource>::fillDescription(desc);

  desc.addUntracked<std::vector<edm::EventID>>("events", {});
  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_FWK_INPUT_SOURCE(EmptySourceFromEventIDs);
