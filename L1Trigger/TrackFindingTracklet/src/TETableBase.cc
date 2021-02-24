#include "L1Trigger/TrackFindingTracklet/interface/TETableBase.h"

using namespace std;
using namespace trklet;

TETableBase::TETableBase(Settings const& settings) : settings_(settings) {}

void TETableBase::writeVMTable(std::string name, bool positive) {
  // Write LUT table.

  ofstream out(name);
  if (out.fail())
    throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << name;

  out << "{" << endl;
  for (unsigned int i = 0; i < table_.size(); i++) {
    if (i != 0) {
      out << "," << endl;
    }

    assert(nbits_ > 0);

    int itable = table_[i];
    if (positive) {
      if (table_[i] < 0) {
        itable = (1 << nbits_) - 1;
      }
    }

    out << itable;
  }
  out << endl << "};" << endl;
  out.close();
}
