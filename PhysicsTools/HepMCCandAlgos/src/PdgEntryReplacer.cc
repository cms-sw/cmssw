#include "PhysicsTools/HepMCCandAlgos/interface/PdgEntryReplacer.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include <sstream>
using namespace std;

string PdgEntryReplacer::replace(const string& in) const {
  string out = in;
  for(;;) {
    size_t p1 = out.find_first_of('{');
    if(p1 == string::npos) break;
    size_t p2 = out.find_first_of('}', p1 + 1);
    if(p2 == string::npos) break;
    size_t n = p2 - p1 - 1;
    string name(out, p1 + 1, n);
    PdtEntry particle(name);
    particle.setup(*es_);
    ostringstream o;
    o << particle.pdgId();
    string s = o.str();
    out.replace(p1, n + 2, s);
  }
  return out;
}
