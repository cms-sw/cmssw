#ifndef L1T_OmtfP1_OMTFConfigMaker_H
#define L1T_OmtfP1_OMTFConfigMaker_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"

#include <map>

class XMLConfigReader;
class OMTFinput;

namespace edm {
  class ParameterSet;
}

class OMTFConfigMaker {
public:
  OMTFConfigMaker(OMTFConfiguration* omtf_config);

  ~OMTFConfigMaker();

  ///Fill counts in GoldenPattern pdf bins
  ///Normalised counts will make a pdf for given GP.
  void fillCounts(unsigned int iProcessor, const OMTFinput& aInput);

  ///Fill histograms used for making the connections
  ///maps
  void makeConnetionsMap(unsigned int iProcessor, const OMTFinput& aInput);

  ///Print starting iPhi for each reference layer
  ///in each processor
  void printPhiMap(std::ostream& out);

  ///Print connections map for given logic cone
  ///in given processro. Connection map
  ///shows counts on each input.
  void printConnections(std::ostream& out, unsigned int iProcessor, unsigned int iCone);

  ///Fill vector with minimal phi in each reference
  ///layer for given processor.
  void fillPhiMaps(unsigned int iProcessor, const OMTFinput& aInput);

private:
  ///Fill map of used inputs.
  ///FIXME: using hack from OMTFConfiguration
  void fillInputRange(unsigned int iConfigMaker, unsigned int iCone, const OMTFinput& aInput);

  void fillInputRange(unsigned int iConfigMaker, unsigned int iCone, unsigned int iRefLayer, unsigned int iInput);

  ///Map of phi starting and ending points
  ///for each logic region.
  ///First index: reference layer number
  ///Second index: logic region number
  std::vector<std::vector<int> > minRefPhi2D;
  std::vector<std::vector<int> > maxRefPhi2D;

  OMTFConfiguration* myOmtfConfig;
};

#endif
