#ifndef OMTFinputMaker_H
#define OMTFinputMaker_H

#include <vector>
#include <stdint.h>
#include <memory>

#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "FWCore/Framework/interface/EventSetup.h"

class AngleConverter;
class OMTFinput;

class OMTFinputMaker{

 public:

  OMTFinputMaker();

  ~OMTFinputMaker();

  void initialize(const edm::EventSetup& es);

  ///Method translating trigger digis into input matrix with global phi coordinates
  const OMTFinput * buildInputForProcessor(const L1MuDTChambPhContainer *dtPhDigis,
					   const L1MuDTChambThContainer *dtThDigis,
					   const CSCCorrelatedLCTDigiCollection *cscDigis,
					   const RPCDigiCollection *rpcDigis,
					   unsigned int iProcessor,
					   l1t::tftype type=l1t::tftype::omtf_pos);
  

 private:

  ///Take the DT digis, select chambers connected to given
  ///processor, convers logal angles to global scale.
  ///For DT take also the bending angle.
  void processDT(const L1MuDTChambPhContainer *dtPhDigis,
		 const L1MuDTChambThContainer *dtThDigis,
		 unsigned int iProcessor,
		 l1t::tftype type);

  ///Take the CSC digis, select chambers connected to given
  ///processor, convers logal angles to global scale.
  ///For CSC do NOT take the bending angle.
  void processCSC(const CSCCorrelatedLCTDigiCollection *cscDigis,
		  unsigned int iProcessor,
		  l1t::tftype type);

  ///Take the CSC digis, select chambers connected to given
  ///processor, convers logal angles to global scale.
  ///Decluster nearby hits in single chamber, by taking
  ///average clister position, expressed in half RPC strip:
  ///pos = cluster_begin + cluster_end)/2
  void processRPC(const RPCDigiCollection *rpcDigis,
		  unsigned int iProcessor,
		  l1t::tftype type);

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

  ///Output object
  OMTFinput *myInput;

  std::unique_ptr<AngleConverter> katownik;

};

#endif
