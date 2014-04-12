#ifndef PhysicsTools_HepMCCandAlgos_PdgEntryReplacer_h
#define PhysicsTools_HepMCCandAlgos_PdgEntryReplacer_h
#include <string>

namespace edm {
  class EventSetup;
}

class PdgEntryReplacer {
public:
  explicit PdgEntryReplacer(const edm::EventSetup & es) : 
    es_(& es) { }
  std::string replace(const std::string&) const;
private:
  const edm::EventSetup * es_;
};


#endif
