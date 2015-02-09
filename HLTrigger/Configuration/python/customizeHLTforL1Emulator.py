import FWCore.ParameterSet.Config as cms
import sys, string

def switchToL1Emulator(process,
  # default settings given are such that only the GT is re-emulated
  newGmtSetting    = False,
  MergeMethodPtBrl = 'minPt',
  MergeMethodPtFwd = 'minPt',
  newCSCTFLUTs     = False,
  newGctSetting    = False,
  newECALLUTs      = False,
  newHCALLUTs      = False):
  """patch the process to run the RawToDigi and SimL1Emulator sequences instead of unpacking the hltGctDigis and hltGtDigis"""

  # redefine the HLTL1UnpackerSequence
  HLTL1UnpackerSequence = cms.Sequence( process.RawToDigi + process.SimL1Emulator + process.hltL1GtObjectMap + process.hltL1extraParticles )

  for iterable in process.sequences.itervalues():
      iterable.replace( process.HLTL1UnpackerSequence, HLTL1UnpackerSequence)

  for iterable in process.paths.itervalues():
      iterable.replace( process.HLTL1UnpackerSequence, HLTL1UnpackerSequence)

  for iterable in process.endpaths.itervalues():
      iterable.replace( process.HLTL1UnpackerSequence, HLTL1UnpackerSequence)

  process.HLTL1UnpackerSequence = HLTL1UnpackerSequence

  # redefine the single hltGtDigis module, for paths that do not use the HLTL1UnpackerSequence
  process.HLTL1GtDigisSequence = cms.Sequence( process.RawToDigi + process.SimL1Emulator )

  for iterable in process.sequences.itervalues():
      iterable.replace( process.hltGtDigis, process.HLTL1GtDigisSequence)

  for iterable in process.paths.itervalues():
      iterable.replace( process.hltGtDigis, process.HLTL1GtDigisSequence)

  for iterable in process.endpaths.itervalues():
      iterable.replace( process.hltGtDigis, process.HLTL1GtDigisSequence)

  # GMT re-emulation
  if newGmtSetting:
    process.load('L1TriggerConfig.GMTConfigProducers.L1MuGMTParameters_cfi')

    # configure muon rank algo for GMT re-emulation
    process.L1MuGMTParameters.MergeMethodPtBrl = cms.string(MergeMethodPtBrl)
    process.L1MuGMTParameters.MergeMethodPtFwd = cms.string(MergeMethodPtFwd)

    process.L1MuGMTParameters.VersionSortRankEtaQLUT = cms.uint32(275)

    import L1Trigger.CSCTrackFinder.csctfDigis_cfi as csctfDigisGMT

    process.csctfReEmulDigis =  csctfDigisGMT.csctfDigis.clone()
    process.csctfReEmulDigis.CSCTrackProducer = cms.untracked.InputTag("csctfReEmulTracks")

    import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi as csctfTrackDigis

    process.csctfReEmulTracks = csctfTrackDigis.csctfTrackDigis.clone()
    process.csctfReEmulTracks.readDtDirect                       = True
    process.csctfReEmulTracks.SectorReceiverInput                = cms.untracked.InputTag("csctfDigis")
    process.csctfReEmulTracks.DtDirectProd                       = cms.untracked.InputTag("csctfDigis", "DT")
    process.csctfReEmulTracks.SectorProcessor.initializeFromPSet = True

    process.load("L1Trigger.RPCTrigger.rpcTriggerDigis_cfi")
    process.rpcReEmulDigis = process.rpcTriggerDigis.clone()

    process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cfi")
    process.gmtReEmulDigis = process.gmtDigis.clone()
    process.gmtReEmulDigis.DTCandidates = cms.InputTag("gtDigis","DT")
    process.gmtReEmulDigis.RPCbCandidates = cms.InputTag("gtDigis","RPCb")

    # switch GMT input to use new re-emulated CSCTF input
    if newCSCTFLUTs:
      process.gmtReEmulDigis.CSCCandidates = cms.InputTag("csctfReEmulDigis","CSC")
    else:
      process.gmtReEmulDigis.CSCCandidates = cms.InputTag("gtDigis","CSC")

    process.gmtReEmulDigis.RPCfCandidates = cms.InputTag("gtDigis","RPCf")
    process.gmtReEmulDigis.MipIsoData = cms.InputTag("none")

    HLTL1MuonTriggerSequence = cms.Sequence( process.csctfReEmulTracks + process.csctfReEmulDigis + process.gmtReEmulDigis )

    # configure GT re-emulation to use new re-emulated GMT input
    process.simGtDigis.GmtInputTag = 'gmtReEmulDigis'
    process.HLTL1MuonTriggerSequence = HLTL1MuonTriggerSequence
    process.HLTL1UnpackerSequence.replace( process.simGtDigis, process.HLTL1MuonTriggerSequence + process.simGtDigis)

  # GCT re-emulation
  if newGctSetting:
    process.load('SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff')
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    # settings for using new ECAL LUTs
    if newECALLUTs:
      process.ecalReEmulDigis = process.simEcalTriggerPrimitiveDigis.clone()
      process.ecalReEmulDigis.Label = 'ecalDigis'
      process.ecalReEmulDigis.InstanceEB = 'ebDigis'
      process.ecalReEmulDigis.InstanceEE = 'eeDigis'
      process.ecalReEmulDigis.BarrelOnly = False

    # settings for using new HCAL LUTs
    if newHCALLUTs:
      process.hcalReEmulDigis = process.simHcalTriggerPrimitiveDigis.clone()
      process.hcalReEmulDigis.inputLabel = cms.VInputTag(cms.InputTag('hcalDigis'), cms.InputTag('hcalDigis'))
      process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)

    # configure RCT re-emulation
    import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
    process.rctReEmulDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()

    if newECALLUTs:
      process.rctReEmulDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalReEmulDigis' ) )
    else:
      process.rctReEmulDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )

    if newHCALLUTs:
      process.rctReEmulDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'hcalReEmulDigis' ) )
    else:
      process.rctReEmulDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'hcalDigis' ) )

    # configure GCT re-emulation
    import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
    process.gctReEmulDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
    process.gctReEmulDigis.inputLabel = 'rctReEmulDigis'

    if newECALLUTs and newHCALLUTs:
      HLTL1CaloTriggerSequence = cms.Sequence( process.ecalReEmulDigis + process.hcalReEmulDigis + process.rctReEmulDigis + process.gctReEmulDigis )
    elif newECALLUTs:
      HLTL1CaloTriggerSequence = cms.Sequence( process.ecalReEmulDigis + process.rctReEmulDigis + process.gctReEmulDigis )
    elif newHCALLUTs:
      HLTL1CaloTriggerSequence = cms.Sequence( process.hcalReEmulDigis + process.rctReEmulDigis + process.gctReEmulDigis )
    else:
      HLTL1CaloTriggerSequence = cms.Sequence( process.rctReEmulDigis + process.gctReEmulDigis )

    # configure GT re-emulation to use new re-emulated GCT input
    process.simGtDigis.GctInputTag = 'gctReEmulDigis'
    process.HLTL1CaloTriggerSequence = HLTL1CaloTriggerSequence
    process.HLTL1UnpackerSequence.replace( process.simGtDigis, process.HLTL1CaloTriggerSequence + process.simGtDigis)

  return process


def switchToCustomL1Digis(process, customGmt, customGct, customGt):
  """patch the process to use custom GMT, GCT and GT results"""

  # explicit replacements to use "simGtDigis", "simGmtDigis" and "simGctDigis" instead of "hltGtDigis" or "hltGctDigis"
  if 'hltL1GtObjectMap' in process.__dict__:
    process.hltL1GtObjectMap.GmtInputTag = cms.InputTag( customGmt )
    process.hltL1GtObjectMap.GctInputTag = cms.InputTag( customGct )
  if 'hltL1extraParticles' in process.__dict__:
    process.hltL1extraParticles.muonSource            = cms.InputTag( customGmt )
    process.hltL1extraParticles.isolatedEmSource      = cms.InputTag( customGct, 'isoEm' )
    process.hltL1extraParticles.nonIsolatedEmSource   = cms.InputTag( customGct, 'nonIsoEm' )
    process.hltL1extraParticles.centralJetSource      = cms.InputTag( customGct, 'cenJets' )
    process.hltL1extraParticles.forwardJetSource      = cms.InputTag( customGct, 'forJets' )
    process.hltL1extraParticles.tauJetSource          = cms.InputTag( customGct, 'tauJets' )
    process.hltL1extraParticles.isoTauJetSource       = cms.InputTag( customGct, 'isoTauJets' )
    process.hltL1extraParticles.etTotalSource         = cms.InputTag( customGct )
    process.hltL1extraParticles.etHadSource           = cms.InputTag( customGct )
    process.hltL1extraParticles.etMissSource          = cms.InputTag( customGct )
    process.hltL1extraParticles.htMissSource          = cms.InputTag( customGct )
    process.hltL1extraParticles.hfRingEtSumsSource    = cms.InputTag( customGct )
    process.hltL1extraParticles.hfRingBitCountsSource = cms.InputTag( customGct )
  if 'hltL2MuonSeeds' in process.__dict__:
    process.hltL2MuonSeeds.GMTReadoutCollection = cms.InputTag( customGmt )
  if 'hltL2CosmicMuonSeeds' in process.__dict__:
    process.hltL2CosmicMuonSeeds.GMTReadoutCollection = cms.InputTag( customGmt )

  # automatic replacements to use "simGtDigis" and "simGctDigis" instead of "hltGtDigis" or "hltGctDigis"
  for module in process.__dict__.itervalues():
    if isinstance(module, cms._Module):
      for parameter in module.__dict__.itervalues():
        if isinstance(parameter, cms.InputTag):
          if parameter.moduleLabel == 'hltGtDigis':
            parameter.moduleLabel = customGt
          elif parameter.moduleLabel == 'hltGctDigis':
            parameter.moduleLabel = customGct

  # check if "hltGtDigis" and "hltGctDigis" are defined
  hasGtDigis  = 'hltGtDigis'  in process.producers
  hasGctDigis = 'hltGctDigis' in process.producers

  # remove "hltGtDigis" and "hltGctDigis" from all paths, endpaths and sequences
  for iterable in process.sequences.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  for iterable in process.paths.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  for iterable in process.endpaths.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  return process


def switchToSimGtDigis(process):
  """patch the process to use newly emulated GT results"""
  return switchToCustomL1Digis(process, 'gtDigis', 'gctDigis', 'simGtDigis')

def switchToSimGmtGctGtDigis(process):
  """patch the process to use newly emulated GMT, GCT and GT results"""
  return switchToCustomL1Digis(process, 'simGmtDigis', 'simGctDigis', 'simGtDigis')

def switchToSimStage1Digis(process):
  """patch the process to use newly emulated GMT, GCT and GT results"""
  return switchToCustomL1Digis(process, 'gmtReEmulDigis', 'simCaloStage1LegacyFormatDigis', 'simGtDigis')

def switchToSimGctGtDigis(process):
  """patch the process to use gtDigis for GMT results, and newly emulated GCT and GT results"""
  return switchToCustomL1Digis(process, 'gtDigis', 'simGctDigis', 'simGtDigis')

def switchToSimGmtGtDigis(process):
  """patch the process to use gctDigis for GCT results, and newly emulated GMT and GT results"""
  return switchToCustomL1Digis(process, 'simGmtDigis', 'gctDigis', 'simGtDigis')

def switchToSimGtReEmulGmtGctDigis(process):
  """patch the process to use newly emulated GMT, GCT and GT results starting from new Muon and Calo LUTs (eventually)"""
  return switchToCustomL1Digis(process, 'gmtReEmulDigis', 'gctReEmulDigis', 'simGtDigis')

def switchToSimGtReEmulGmtDigis(process):
  """patch the process to use newly emulated GMT and GT results starting from new Muon LUTs (eventually)"""
  return switchToCustomL1Digis(process, 'gmtReEmulDigis', 'gctDigis', 'simGtDigis')

def switchToSimGtReEmulGctDigis(process):
  """patch the process to use newly emulated GCT and GT results starting from new Calo LUTs (eventually)"""
  return switchToCustomL1Digis(process, 'gtDigis', 'gctReEmulDigis', 'simGtDigis')

