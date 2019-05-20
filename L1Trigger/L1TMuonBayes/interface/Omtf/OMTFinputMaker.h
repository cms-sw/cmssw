#ifndef OMTFinputMaker_H
#define OMTFinputMaker_H

#include "L1Trigger/L1TMuonBayes/interface/MuonStubMakerBase.h"

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

class OMTFinputMaker : public MuonStubMakerBase {
public:

  OMTFinputMaker();

  virtual ~OMTFinputMaker();

  virtual void initialize(const edm::ParameterSet& edmCfg, const edm::EventSetup& es, const OMTFConfiguration* procConf, MuStubsInputTokens& muStubsInputTokens);

  ///Method translating trigger digis into input matrix with global phi coordinates
/*  const OMTFinput buildInputForProcessor(const L1MuDTChambPhContainer *dtPhDigis,
				   const L1MuDTChambThContainer *dtThDigis,
				   const CSCCorrelatedLCTDigiCollection *cscDigis,
				   const RPCDigiCollection *rpcDigis,
				   unsigned int iProcessor,
				   l1t::tftype procTyp = l1t::tftype::omtf_pos,
                           int bx=0);
  */

 void setFlag(int aFlag) {flag = aFlag; }
 int getFlag() const { return flag;}

 ///iProcessor - from 0 to 5
 ///returns the global phi in hardware scale (myOmtfConfig->nPhiBins() ) at which the scale starts for give processor
 virtual int getProcessorPhiZero(unsigned int iProcessor);

private:
  ///Check if digis are within a give processor input.
  ///Simply checks sectors range.
 virtual bool acceptDigi(uint32_t rawId,
		  unsigned int iProcessor,
		  l1t::tftype type);

  ///Give input number for givedn processor, using
  ///the chamber sector number.
  ///Result is modulo allowed number of hits per chamber
 virtual unsigned int getInputNumber(unsigned int rawId,
			      unsigned int iProcessor,
			      l1t::tftype type);

 //the phi and eta digis are merged (even thought it is artificial)
 virtual void addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambPhDigi& digi,
    const L1MuDTChambThContainer *dtThDigis,
    unsigned int iProcessor,
    l1t::tftype procTyp);

 virtual void addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambThDigi& thetaDigi,
      unsigned int iProcessor, l1t::tftype procTyp) {
   //this function is not needed here, in OMTF the phi and theta segments are merged into one - TODO - implement better if has sense
 }

 virtual void addCSCstubs(MuonStubPtrs2D& muonStubsInLayers,  unsigned int rawid, const CSCCorrelatedLCTDigi& digi,
    unsigned int iProcessor, l1t::tftype procTyp);

 virtual void addRPCstub(MuonStubPtrs2D& muonStubsInLayers, const RPCDetId& roll, const RpcCluster& cluster,
    unsigned int iProcessor, l1t::tftype procTyp);

 virtual void addStub(MuonStubPtrs2D& muonStubsInLayers, unsigned int iLayer, unsigned int iInput, MuonStub& stub);

  RpcClusterization rpcClusterization;
  //OmtfAngleConverter angleConverter;

  OmtfAngleConverter angleConverter;

  const OMTFConfiguration* config =  nullptr;

  int flag;

};

#endif
