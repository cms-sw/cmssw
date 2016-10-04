#include "FWCore/Framework/interface/stream/EDProducerBase.h"

#include "implementations_stage1/CaloSetup.h"
#include "implementations_stage1/CaloSpareHFPacker.h"
#include "implementations_stage1/CaloSpareHFUnpacker.h"
#include "implementations_stage1/EtSumPacker.h"
#include "implementations_stage1/EtSumUnpacker.h"
#include "implementations_stage1/HFRingPacker.h"
#include "implementations_stage1/HFRingUnpacker.h"
#include "implementations_stage1/LegacyEtSumUnpacker.h"
#include "implementations_stage1/LegacyHFRingUnpacker.h"
#include "implementations_stage1/LegacyPhysCandUnpacker.h"
#include "implementations_stage1/MissEtPacker.h"
#include "implementations_stage1/MissEtUnpacker.h"
#include "implementations_stage1/MissHtPacker.h"
#include "implementations_stage1/MissHtUnpacker.h"
#include "implementations_stage1/PhysCandPacker.h"
#include "implementations_stage1/PhysCandUnpacker.h"
#include "implementations_stage1/RCTEmRegionPacker.h"
#include "implementations_stage1/RCTEmRegionUnpacker.h"

#include "implementations_stage2/BMTFSetup.h"
#include "implementations_stage2/BMTFUnpackerInputs.h"
#include "implementations_stage2/BMTFUnpackerOutput.h"
#include "implementations_stage2/CaloSetup.h"
#include "implementations_stage2/CaloTowerPacker.h"
#include "implementations_stage2/CaloTowerUnpacker.h"
#include "implementations_stage2/EGammaPacker.h"
#include "implementations_stage2/EGammaUnpacker.h"
#include "implementations_stage2/EMTFBlockCounters.h"
#include "implementations_stage2/EMTFBlockHeaders.h"
#include "implementations_stage2/EMTFBlockME.h"
#include "implementations_stage2/EMTFBlockRPC.h"
#include "implementations_stage2/EMTFBlockSP.h"
#include "implementations_stage2/EMTFBlockTrailers.h"
#include "implementations_stage2/EMTFSetup.h"
#include "implementations_stage2/EtSumPacker.h"
#include "implementations_stage2/EtSumUnpacker.h"
#include "implementations_stage2/GMTSetup.h"
#include "implementations_stage2/GTSetup.h"
#include "implementations_stage2/GlobalAlgBlkPacker.h"
#include "implementations_stage2/GlobalAlgBlkUnpacker.h"
#include "implementations_stage2/GlobalExtBlkPacker.h"
#include "implementations_stage2/GlobalExtBlkUnpacker.h"
#include "implementations_stage2/JetPacker.h"
#include "implementations_stage2/JetUnpacker.h"
#include "implementations_stage2/MPUnpacker.h"
#include "implementations_stage2/MPUnpacker_0x1001000b.h"
#include "implementations_stage2/MPUnpacker_0x10010010.h"
#include "implementations_stage2/MuonPacker.h"
#include "implementations_stage2/MuonUnpacker.h"
#include "implementations_stage2/RegionalMuonGMTPacker.h"
#include "implementations_stage2/RegionalMuonGMTUnpacker.h"
#include "implementations_stage2/TauPacker.h"
#include "implementations_stage2/TauUnpacker.h"

DEFINE_L1T_PACKER(l1t::stage1::CaloSpareHFPacker);
DEFINE_L1T_PACKER(l1t::stage1::CentralJetPacker);
DEFINE_L1T_PACKER(l1t::stage1::EtSumPacker);
DEFINE_L1T_PACKER(l1t::stage1::ForwardJetPacker);
DEFINE_L1T_PACKER(l1t::stage1::HFRingPacker);
DEFINE_L1T_PACKER(l1t::stage1::IsoEGammaPacker);
DEFINE_L1T_PACKER(l1t::stage1::IsoTauPacker);
DEFINE_L1T_PACKER(l1t::stage1::MissEtPacker);
DEFINE_L1T_PACKER(l1t::stage1::MissHtPacker);
DEFINE_L1T_PACKER(l1t::stage1::NonIsoEGammaPacker);
DEFINE_L1T_PACKER(l1t::stage1::RCTEmRegionPacker);
DEFINE_L1T_PACKER(l1t::stage1::TauPacker);
DEFINE_L1T_PACKER(l1t::stage2::CaloEGammaPacker);
DEFINE_L1T_PACKER(l1t::stage2::CaloEtSumPacker);
DEFINE_L1T_PACKER(l1t::stage2::CaloJetPacker);
DEFINE_L1T_PACKER(l1t::stage2::CaloTauPacker);
DEFINE_L1T_PACKER(l1t::stage2::CaloTowerPacker);
DEFINE_L1T_PACKER(l1t::stage2::GMTMuonPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTEGammaPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTEtSumPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTJetPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTMuonPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTTauPacker);
DEFINE_L1T_PACKER(l1t::stage2::GlobalAlgBlkPacker);
DEFINE_L1T_PACKER(l1t::stage2::GlobalExtBlkPacker);
DEFINE_L1T_PACKER(l1t::stage2::RegionalMuonGMTPacker);
DEFINE_L1T_PACKING_SETUP(l1t::stage1::CaloSetup);
DEFINE_L1T_PACKING_SETUP(l1t::stage2::BMTFSetup);
DEFINE_L1T_PACKING_SETUP(l1t::stage2::CaloSetup);
DEFINE_L1T_PACKING_SETUP(l1t::stage2::EMTFSetup);
DEFINE_L1T_PACKING_SETUP(l1t::stage2::GMTSetup);
DEFINE_L1T_PACKING_SETUP(l1t::stage2::GTSetup);
DEFINE_L1T_UNPACKER(l1t::stage1::CaloSpareHFUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::CentralJetUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::CentralJetUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::EtSumUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::ForwardJetUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::ForwardJetUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::HFRingUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoEGammaUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoEGammaUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoTauUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoTauUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::MissEtUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::MissHtUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::NonIsoEGammaUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::NonIsoEGammaUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::RCTEmRegionUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::TauUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::TauUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::CentralJetUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::EtSumUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::ForwardJetUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::HFRingUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::IsoEGammaUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::IsoTauUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::NonIsoEGammaUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::legacy::TauUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::BMTFUnpackerInputs);
DEFINE_L1T_UNPACKER(l1t::stage2::BMTFUnpackerOutput);
DEFINE_L1T_UNPACKER(l1t::stage2::CaloTowerUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::EGammaUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::EtSumUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::GlobalAlgBlkUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::GlobalExtBlkUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::JetUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker_0x1001000b);
DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker_0x10010010);
DEFINE_L1T_UNPACKER(l1t::stage2::MuonUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::RegionalMuonGMTUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::TauUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::emtf::CountersBlockUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::emtf::HeadersBlockUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::emtf::MEBlockUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::emtf::RPCBlockUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::emtf::SPBlockUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage2::emtf::TrailersBlockUnpacker);
