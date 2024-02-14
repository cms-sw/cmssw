// Build a working release including the desired class versions in the
// class definitions and classes_def.xml files. You might need to add or
// remove lines below to get the StreamerInfo objects for the desired classes.
// Then run the following to execute this script:
//
//   root -l -b -q makeFileContainingStreamerInfos.C
//
// Then rename the output file as appropriate. If it is of general use
// you might want to reposit the file in the IOPool/Input data repository.
// This output file can be used by the service named FixMissingStreamerInfos.

#include <iostream>

void makeFileContainingStreamerInfos() {
  std::cout << "Executing makeFileContainingStreamerInfos()" << std::endl;
  auto f = TFile::Open("fileContainingStreamerInfos.root", "NEW");

  TClass::GetClass("BeamSpotOnline")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("CTPPSLocalTrackLite")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("CTPPSPixelDataError")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("CTPPSPixelDigi")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("CorrMETData")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("DcsStatus")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("EcalTriggerPrimitiveSample")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("HaloTowerStrip")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("HcalElectronicsId")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1AcceptBunchCrossing")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctEmCand")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctEtHad")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctEtMiss")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctEtTotal")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctHFBitCounts")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctHFRingEtSums")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctHtMiss")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctJetCand")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GctJetCounts")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GtFdlWord")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1GtPsbWord")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1MuGMTReadoutRecord")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("L1TriggerScalers")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("Level1TriggerScalers")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("LumiScalers")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("Measurement1DFloat")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("PixelFEDChannel")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("Run3ScoutingParticle")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("Run3ScoutingTrack")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("Run3ScoutingVertex")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("TotemFEDInfo")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("ZDCDataFrame")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("ZDCRecHit")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<CTPPSDiamondDigi>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<CTPPSDiamondLocalTrack>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<CTPPSDiamondRecHit>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<CTPPSPixelLocalTrack>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemRPCluster>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemRPDigi>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemRPLocalTrack>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemRPRecHit>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemRPUVPattern>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemTimingDigi>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemTimingLocalTrack>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::DetSet<TotemTimingRecHit>")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::HLTPathStatus")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::IndexIntoFile::RunOrLumiEntry")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::RefCoreWithIndex")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::StoredMergeableRunProductMetadata::SingleRunEntry")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::StoredMergeableRunProductMetadata::SingleRunEntryAndProcess")
      ->GetStreamerInfo()
      ->ForceWriteInfo(f);
  TClass::GetClass("edm::StoredProductProvenance")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("edm::ThinnedAssociationBranches")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("l1t::CaloTower")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("l1t::RegionalMuonShower")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::DeDxData")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::ElectronSeed::PMVars")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::ForwardProton")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::JetID")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::MuonCosmicCompatibility")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::MuonGEMHitMatch")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::MuonMETCorrectionData")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::MuonRPCHitMatch")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::MuonTimeExtra")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::PhiWedge")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("reco::RecoEcalCandidate")->GetStreamerInfo()->ForceWriteInfo(f);
  TClass::GetClass("trigger::TriggerObject")->GetStreamerInfo()->ForceWriteInfo(f);

  TClass::GetClass("l1t::MuonShower")->GetStreamerInfo()->ForceWriteInfo(f);

  delete f;
}
