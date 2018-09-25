#ifndef OMTFinputMaker_H
#define OMTFinputMaker_H

#include <vector>
#include <cstdint>
#include <memory>

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "L1Trigger/L1TMuonOverlap/interface/AngleConverter.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"


class OMTFConfiguration;

namespace edm {
  class EventSetup;
}

class OMTFinputMaker {

 public:

  OMTFinputMaker();

  ~OMTFinputMaker();

  void initialize(const edm::EventSetup& es, const OMTFConfiguration*);

  ///Method translating trigger digis into input matrix with global phi coordinates
  OMTFinput buildInputForProcessor(const L1MuDTChambPhContainer *dtPhDigis,
				   const L1MuDTChambThContainer *dtThDigis,
				   const CSCCorrelatedLCTDigiCollection *cscDigis,
				   const RPCDigiCollection *rpcDigis,
				   unsigned int iProcessor,
				   l1t::tftype type=l1t::tftype::omtf_pos,
                           int bx=0);
  

 void setFlag(int aFlag) {flag = aFlag; }
 int getFlag() const { return flag;}

 private:

  ///Take the DT digis, select chambers connected to given
  ///processor, convers logal angles to global scale.
  ///For DT take also the bending angle.
  OMTFinput processDT(const L1MuDTChambPhContainer *dtPhDigis,
		 const L1MuDTChambThContainer *dtThDigis,
		 unsigned int iProcessor,
		 l1t::tftype type, int bx);

  ///Take the CSC digis, select chambers connected to given
  ///processor, convers logal angles to global scale.
  ///For CSC do NOT take the bending angle.
  OMTFinput processCSC(const CSCCorrelatedLCTDigiCollection *cscDigis,
		  unsigned int iProcessor,
		  l1t::tftype type, int bx);

  ///Decluster nearby hits in single chamber, by taking
  ///average cluster position, expressed in half RPC strip:
  ///pos = (cluster_begin + cluster_end)
  OMTFinput processRPC(const RPCDigiCollection *rpcDigis,
		  unsigned int iProcessor,
		  l1t::tftype type, int bx);

  ///Check if digis are within a give processor input.
  ///Simply checks sectors range.
  bool acceptDigi(uint32_t rawId,
		  unsigned int iProcessor,
		  l1t::tftype type);

  ///Give input number for givedn processor, using
  ///the chamber sector number.
  ///Result is modulo allowed number of hits per chamber
  unsigned int getInputNumber(unsigned int rawId,
			      unsigned int iProcessor,
			      l1t::tftype type);

  AngleConverter myAngleConverter;

  const OMTFConfiguration* myOmtfConfig;

  int flag;

};

#endif
