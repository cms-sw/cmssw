// Emulator that takes RPC hits and produces CPPFDigis to send to EMTF
// Author Alejandro Segura -- Universidad de los Andes

#ifndef L1Trigger_L1TMuonCPPF_L1TMuonCPPFDigiProducer_h
#define L1Trigger_L1TMuonCPPF_L1TMuonCPPFDigiProducer_h

#include "L1Trigger/L1TMuonCPPF/interface/EmulateCPPF.h"

// System include files
#include <memory>

// User include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Other includes (all needed? - AWB 27.07.17)
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConversion.h"
#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "TVector3.h"
#include <cassert>
#include <fstream>
#include <string>

// Class declaration
class L1TMuonCPPFDigiProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TMuonCPPFDigiProducer(const edm::ParameterSet &);
  ~L1TMuonCPPFDigiProducer() override;

private:
  void beginStream(edm::StreamID) override;
  void endStream() override;
  void produce(edm::Event &event, const edm::EventSetup &setup) override;

private:
  std::unique_ptr<EmulateCPPF> cppf_emulator_;
};

#endif /* #define L1Trigger_L1TMuonCPPF_L1TMuonCPPFDigiProducer_h */
