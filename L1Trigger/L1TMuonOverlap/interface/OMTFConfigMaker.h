#ifndef OMTF_OMTFConfigMaker_H
#define OMTF_OMTFConfigMaker_H

#include <map>

#include "L1Trigger/L1TMuonOverlap/interface/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFResult.h"

class XMLConfigReader;
class OMTFinput;

namespace edm{
class ParameterSet;
}

class OMTFConfigMaker{

 public:

  OMTFConfigMaker(const edm::ParameterSet & cfg);

  ~OMTFConfigMaker();

  ///Fill counts in GoldenPattern pdf bins
  ///Normalised counts will make a pdf for given GP.
  void fillCounts(unsigned int iProcessor,
		  const OMTFinput & aInput);
  
  ///Fill histograms used for making the connections
  ///maps
  void makeConnetionsMap(unsigned int iProcessor,
			 const OMTFinput & aInput);

  ///Print starting iPhi for each reference layer
  ///in each processor
  void printPhiMap(std::ostream & out);

  ///Print connections map for given logic cone
  ///in given processro. Connection map
  ///shows counts on each input.
  void printConnections(std::ostream & out,
			unsigned int iProcessor,
			unsigned int iCone);

  ///Fill vector with minimal phi in each reference
  ///layer for given processor.
  void fillPhiMaps(unsigned int iProcessor,
		   const OMTFinput & aInput);
  
 private:

  ///Fill map of used inputs.
  ///FIXME: using hack from OMTFConfiguration
  void fillInputRange(unsigned int iConfigMaker,
		      unsigned int iCone,
		      const OMTFinput & aInput);

  void fillInputRange(unsigned int iConfigMaker,
		      unsigned int iCone,
		      unsigned int iRefLayer,
		      unsigned int iHit);

  ///Map of phi starting points
  ///First index: reference layer number
  ///Second index: processor number
  std::vector<std::vector<int> > minRefPhi2D;
    
};


#endif
