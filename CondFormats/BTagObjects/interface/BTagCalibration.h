#ifndef BTagCalibration_H
#define BTagCalibration_H

/**
 * BTagCalibration
 *
 * The 'hierarchy' of stored information is this:
 * - by tagger (BTagCalibration)
 *   - by operating point or reshape bin
 *     - by jet parton flavor
 *       - by type of measurement
 *         - by systematic
 *           - by eta bin
 *             - as 1D-function dependent of pt
 *
 ************************************************************/

#include <map>
#include <vector>
#include <string>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/BTagObjects/interface/BTagEntry.h"

class BTagCalibration
{
public:
  BTagCalibration() {}
  BTagCalibration(std::string tagger): tagger_(tagger) {}
  ~BTagCalibration() {}

  void addEntry(BTagEntry entry);
  const std::vector<BTagEntry>& getEntries(BTagEntry::Parameters par) const;

protected:
  std::string tagger_;
  std::map<std::string, std::vector<BTagEntry> > data_;

  COND_SERIALIZABLE;
};

#endif  // BTagCalibration_H