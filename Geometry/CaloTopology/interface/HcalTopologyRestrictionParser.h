#ifndef GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGYRESTRICTIONPARSER_H
#define GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGYRESTRICTIONPARSER_H 1

#include "Geometry/CaloTopology/interface/HcalTopology.h"

/** \class HcalTopologyRestrictionParser
  *  
  *  This utility class is intended to provide a standard way to set restrictions on the 
  *  HcalTopology to exclude cells for testbeam and other purposes.  It functions as a 
  *  parser to convert textually-encoded restrictions into calls to HcalTopology::exclude().
  *
  *  The grammer is (using a rough notation)
  *
  *  line = rule >> *(; >> rule )
  *  rule = region | subdetector
  *  region = subdetname ieta1 ieta2 iphi1 iphi2 [depth1 depth2]
  *  subdetector = subdetname
  *  subdetname = "HB" | "HE" | "HO" | "HF"
  * $Date: 2005/11/30 19:55:34 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalTopologyRestrictionParser {
public:
  HcalTopologyRestrictionParser(HcalTopology& target);
  /** Parses a line of restrictions and returns an error string if there is a problem.
      The line must be formated as described in the class description.
  */
  std::string parse(const std::string& line);
private:
  HcalTopology& target_;
};

#endif
