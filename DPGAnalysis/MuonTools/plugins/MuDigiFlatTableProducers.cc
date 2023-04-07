/** \class MuDigiFlatTableProducers.cc MuDigiFlatTableProducers.cc DPGAnalysis/MuonTools/src/MuDigiFlatTableProducers.cc
 *  
 * EDProducers : the flat table producers for CSC, DT, GEM and RPC digis
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/interface/MuDigiBaseProducer.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
using CSCWireDigiFlatTableProducer = MuDigiBaseProducer<CSCDetId, CSCWireDigi>;

#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
using CSCAlctDigiFlatTableProducer = MuDigiBaseProducer<CSCDetId, CSCALCTDigi>;

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
using DTDigiFlatTableProducer = MuDigiBaseProducer<DTLayerId, DTDigi>;

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
using RPCDigiFlatTableProducer = MuDigiBaseProducer<RPCDetId, RPCDigi>;

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
using GEMDigiFlatTableProducer = MuDigiBaseProducer<GEMDetId, GEMDigi>;

#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
using GEMOHStatusFlatTableProducer = MuDigiBaseProducer<GEMDetId, GEMOHStatus>;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CSCWireDigiFlatTableProducer);
DEFINE_FWK_MODULE(CSCAlctDigiFlatTableProducer);
DEFINE_FWK_MODULE(DTDigiFlatTableProducer);
DEFINE_FWK_MODULE(RPCDigiFlatTableProducer);
DEFINE_FWK_MODULE(GEMDigiFlatTableProducer);
DEFINE_FWK_MODULE(GEMOHStatusFlatTableProducer);
