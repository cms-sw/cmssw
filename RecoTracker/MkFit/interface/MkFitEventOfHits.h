#ifndef RecoTracker_MkFit_MkFitEventOfHits_h
#define RecoTracker_MkFit_MkFitEventOfHits_h

#include <memory>

namespace mkfit {
  class EventOfHits;
}

/**
 * The mkfit::EventOfHits is a container of mkfit::LayerOfHits
 * structures that mkFit uses to group (and index) hits. Having them
 * grouped together allows mkFit to pass them easily around top-level
 * steering functions.
 *
 * It has some conceptual similarities to MeasurementTrackerEvent.
 */
class MkFitEventOfHits {
public:
  MkFitEventOfHits();
  MkFitEventOfHits(std::unique_ptr<mkfit::EventOfHits>);
  ~MkFitEventOfHits();

  MkFitEventOfHits(MkFitEventOfHits const&) = delete;
  MkFitEventOfHits& operator=(MkFitEventOfHits const&) = delete;
  MkFitEventOfHits(MkFitEventOfHits&&);
  MkFitEventOfHits& operator=(MkFitEventOfHits&&);

  mkfit::EventOfHits& get() { return *eventOfHits_; }
  mkfit::EventOfHits const& get() const { return *eventOfHits_; }

private:
  std::unique_ptr<mkfit::EventOfHits> eventOfHits_;
};

#endif
