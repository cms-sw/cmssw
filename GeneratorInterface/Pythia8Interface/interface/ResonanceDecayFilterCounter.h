#ifndef GeneratorInterface_Pythia8Interface_ResonanceDecayFilterCounter_h
#define GeneratorInterface_Pythia8Interface_ResonanceDecayFilterCounter_h

class ResonanceDecayFilterCounter {
public:
  static ResonanceDecayFilterCounter& getInstance() {
    static ResonanceDecayFilterCounter instance;
    return instance;
  }

  void setEventCounter(int eventCounter) { eventCounter_ = eventCounter; }

  void setFilterBool(bool filterBool) { filterBool_ = filterBool; }

  int getEventCounter() const { return eventCounter_; }

  bool getFilterBool() const { return filterBool_; }

private:
  ResonanceDecayFilterCounter() : eventCounter_(0), filterBool_(false) {}
  int eventCounter_;
  bool filterBool_;
};

#endif  // GeneratorInterface_Pythia8Interface_ResonanceDecayFilterCounter_h