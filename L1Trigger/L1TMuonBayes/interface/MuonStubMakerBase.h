#ifndef L1TMUONBAYES_MUONSTUBMAKERBASE_H
#define L1TMUONBAYES_MUONSTUBMAKERBASE_H

#include "L1Trigger/L1TMuonBayes/interface/Omtf/OmtfAngleConverter.h"
#include "L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinput.h"
#include <vector>
#include <cstdint>
#include <memory>

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include <L1Trigger/L1TMuonBayes/interface/RpcClusterization.h>


class OMTFConfiguration;

namespace edm {
  class EventSetup;
}

class MuonStubMakerBase {

 public:

  MuonStubMakerBase();

  virtual ~MuonStubMakerBase();

  void initialize(const edm::EventSetup& es, const OMTFConfiguration*);

  ///Method translating trigger digis into input matrix with global phi coordinates
/*  OMTFinput buildInputForProcessor(const L1MuDTChambPhContainer *dtPhDigis,
				   const L1MuDTChambThContainer *dtThDigis,
				   const CSCCorrelatedLCTDigiCollection *cscDigis,
				   const RPCDigiCollection *rpcDigis,
				   unsigned int iProcessor,
				   l1t::tftype procType = l1t::tftype::omtf_pos,
				   int bxFrom = 0, int bxTo = 0);*/
  

 ///iProcessor - from 0 to 5
 ///returns the global phi in hardware scale (myOmtfConfig->nPhiBins() ) at which the scale starts for give processor
 //virtual int getProcessorPhiZero(unsigned int iProcessor) = 0;

protected:
  ///Check if digis are within a give processor input.
  ///Simply checks sectors range.
  virtual bool acceptDigi(uint32_t rawId,
		  unsigned int iProcessor,
		  l1t::tftype procType) = 0;


  //dtThDigis is provided as argument, because in the OMTF implementation the phi and eta digis are merged (even thought it is artificial)
  virtual void addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambPhDigi& digi,
     const L1MuDTChambThContainer *dtThDigis,
     unsigned int iProcessor,
     l1t::tftype procTyp) = 0;

  virtual void addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambThDigi& thetaDigi,
     unsigned int iProcessor, l1t::tftype procTyp) = 0;

//can add both phi and eta stubs
  virtual void addCSCstubs(MuonStubPtrs2D& muonStubsInLayers, unsigned int rawid, const CSCCorrelatedLCTDigi& digi,
     unsigned int iProcessor, l1t::tftype procTyp) = 0;

  virtual void addRPCstub(MuonStubPtrs2D& muonStubsInLayers, const RPCDetId& roll, const RpcCluster& cluster,
     unsigned int iProcessor, l1t::tftype procTyp) = 0;

  ///Take the DT digis, select chambers connected to given
  ///processor, convers logal angles to global scale.
  ///For DT take also the bending angle.
  virtual void processDT(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambPhContainer *dtPhDigis,
		 const L1MuDTChambThContainer *dtThDigis,
		 unsigned int iProcessor,
		 l1t::tftype procType, bool mergePhiAndTheta, int bxFrom = 0, int bxTo = 0);

  ///Take the CSC digis, select chambers connected to given
  ///processor, convers logal angles to global scale.
  ///For CSC do NOT take the bending angle.
  virtual void processCSC(MuonStubPtrs2D& muonStubsInLayers, const CSCCorrelatedLCTDigiCollection *cscDigis,
		  unsigned int iProcessor,
		  l1t::tftype procType, int bxFrom = 0, int bxTo = 0);

  ///Decluster nearby hits in single chamber, by taking
  ///average cluster position, expressed in half RPC strip:
  ///pos = (cluster_begin + cluster_end)
  virtual void processRPC(MuonStubPtrs2D& muonStubsInLayers, const RPCDigiCollection *rpcDigis,
		  unsigned int iProcessor,
		  l1t::tftype procType, int bxFrom = 0, int bxTo = 0);


  ///Give input number for givedn processor, using
  ///the chamber sector number.
  ///Result is modulo allowed number of hits per chamber
  /*virtual unsigned int getInputNumber(unsigned int rawId,
			      unsigned int iProcessor,
			      l1t::tftype type);*/

  RpcClusterization rpcClusterization;
  //OmtfAngleConverter angleConverter;

  std::unique_ptr<AngleConverterBase> angleConverter;

  const OMTFConfiguration* config;

  int flag;

};

#endif
