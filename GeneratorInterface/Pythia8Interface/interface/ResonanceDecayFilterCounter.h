#ifndef RESONANCE_DECAY_FILTER_COUNTER_H
#define RESONANCE_DECAY_FILTER_COUNTER_H

class ResonanceDecayFilterCounter {
public:
  static ResonanceDecayFilterCounter& getInstance() {
    static ResonanceDecayFilterCounter instance;
    return instance;
  }

  void setEventCounter(int eventCounter) {
    eventCounter_ = eventCounter;
  }

  void setTotalCounter(int totalCounter) {
    totalCounter_ = totalCounter;
  }

  void setFilterBool(bool filterBool) {
    filterBool_ = filterBool;
  }

  int getEventCounter() const {
    return eventCounter_;
  }

  int getTotalCounter() const {
    return totalCounter_;
  }

  bool getFilterBool() const {
    return filterBool_;
  }

private:
  ResonanceDecayFilterCounter() : eventCounter_(0), totalCounter_(0), filterBool_(false) {}
  int eventCounter_;
  int totalCounter_;
  bool filterBool_;
};

#endif // RESONANCE_DECAY_FILTER_COUNTER_H