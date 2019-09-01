#ifndef Edm_Module_Timing_h
#define Edm_Module_Timing_h

#include <vector>
#include <string>

/* Original version: July 2006, Christos Leonidopoulos */

namespace edm {

  // structure holding the processing time (per event) and name of a module
  class ModuleTime {
  public:
    ModuleTime() : name_(""), time_(-1) {}
    ModuleTime(std::string Name, double Time) : name_(Name), time_(Time) {}
    ~ModuleTime() {}

    std::string name() const { return name_; }  // module name ("label")
    double time() const { return time_; }       // processing time for event (secs)
  private:
    std::string name_;
    double time_;
  };

  // structure holding processing info for all modules in event (+total time)
  class EventTime {
  private:
    std::vector<ModuleTime> moduleSet;
    double tot_time_;  // total time in event for all modules (in secs)

  public:
    EventTime() { reset(); }
    ~EventTime() {}

    // # of modules contained in event
    unsigned size() const { return moduleSet.size(); }
    // get hold of ModuleTime structure for module #i, where 0 <= i < size()
    const ModuleTime& moduleTime(unsigned i) { return moduleSet.at(i); }
    // get total processing time for event (secs)
    double tot_time() const { return tot_time_; }
    // get name for module #i, where 0 <= i < size()
    std::string name(unsigned i) const { return moduleSet.at(i).name(); }
    // get processing time for module #i (secs), where 0 <= i < size()
    double time(unsigned i) const { return moduleSet.at(i).time(); }
    // add module structure to event
    void addModuleTime(const ModuleTime& m) {
      moduleSet.push_back(m);
      tot_time_ += m.time();
    }

    // reset all info (ie. from previous event)
    void reset() {
      moduleSet.clear();
      tot_time_ = 0;
    }
  };

}  // namespace edm

#endif  // #define Edm_Module_Timing_h
