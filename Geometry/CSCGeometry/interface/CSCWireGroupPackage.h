#ifndef CSC_WIREGROUP_PACKAGE_H
#define CSC_WIREGROUP_PACKAGE_H

// This is CSCWireGroupPackage.h

/**
 * \class CSCWireGroupPackage
 * Bundle wire group info from DDD into one package to simplify
 * passing it around. No functionality other than as an
 * encapsulation of related data.
 *
 * \author Tim Cox
 *
 */

#include <vector>

class CSCWireGroupPackage {
public:
  typedef std::vector<int> Container;

  CSCWireGroupPackage()
      : numberOfGroups(0),
        wireSpacing(0.),
        alignmentPinToFirstWire(0.),
        narrowWidthOfWirePlane(0.),
        wideWidthOfWirePlane(0.),
        lengthOfWirePlane(0.){};

  Container consecutiveGroups;
  Container wiresInEachGroup;
  int numberOfGroups;
  double wireSpacing;
  double alignmentPinToFirstWire;
  double narrowWidthOfWirePlane;
  double wideWidthOfWirePlane;
  double lengthOfWirePlane;
};
#endif
