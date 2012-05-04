import FWCore.ParameterSet.Config as cms
import sys, string

def switchToL1Emulator(*arglist):
  """patch the process to run the RawToDigi and SimL1Emulator sequences instead of unpacking the hltGctDigis and hltGtDigis"""

  # default settings given are such that only the GT is re-emulated
  newGmtSetting    = 'False'
  MergeMethodPtBrl = 'minPt'
  MergeMethodPtFwd = 'minPt'
  newCSCTFLUts     = 'False'
  newGctSetting    = 'False'
  caloPrimitives   = 'CALOPRIMITIVES'
  JetFinderCentralJetSeed = '0'
  JetFinderForwardJetSeed = '0'

  if not arglist:
    print "ERROR: no arglist given ...\n"
    sys.exit(1) 
  else:
    process = arglist[0]
    if len(arglist) > 1:
      newGmtSetting = arglist[1]
      print '\nCustomized settings:' 
      print '  newGmtSetting             : "%s"' % newGmtSetting
    if len(arglist) > 2:
      MergeMethodPtBrl = arglist[2] 
      print '  MergeMethodPtBrl          : "%s"' % MergeMethodPtBrl
    if len(arglist) > 3:
      MergeMethodPtFwd = arglist[3] 
      print '  MergeMethodPtFwd          : "%s"' % MergeMethodPtFwd
    if len(arglist) > 4:
      newCSCTFLUts = arglist[4] 
      print '  newCSCTFLUts              : "%s"' % newCSCTFLUts
    if len(arglist) > 5:
      newGctSetting = arglist[5] 
      print '  newGctSetting             : "%s"' % newGctSetting
    if len(arglist) > 6:
      caloPrimitives = arglist[6] 
      print '  caloPrimitives            : "%s"' % caloPrimitives
    if len(arglist) > 7:
      JetFinderCentralJetSeed = arglist[7] 
      print '  JetFinderCentralJetSeed   : "%s"' % JetFinderCentralJetSeed
    if len(arglist) > 8:
      JetFinderForwardJetSeed = arglist[8] 
      print '  JetFinderForwardJetSeed   : "%s"' % JetFinderForwardJetSeed

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


  if newGmtSetting=='True':
    process.load('L1TriggerConfig.GMTConfigProducers.L1MuGMTParameters_cfi')
    
    process.L1MuGMTParameters.MergeMethodPtBrl=cms.string(MergeMethodPtBrl)
    process.L1MuGMTParameters.MergeMethodPtFwd=cms.string(MergeMethodPtFwd)
          
    process.L1MuGMTParameters.VersionSortRankEtaQLUT = cms.uint32(275)

    import L1Trigger.CSCTrackFinder.csctfDigis_cfi as csctfDigisGMT
      
    process.csctfReEmulDigis =  csctfDigisGMT.csctfDigis.clone()
    process.csctfReEmulDigis.CSCTrackProducer = cms.untracked.InputTag("csctfReEmulTracks")

    import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi as csctfTrackDigis

    process.csctfReEmulTracks = csctfTrackDigis.csctfTrackDigis.clone()
    process.csctfReEmulTracks.readDtDirect                       = True
    process.csctfReEmulTracks.SectorReceiverInput                = cms.untracked.InputTag("csctfDigis")
    process.csctfReEmulTracks.DtDirectProd                       = cms.untracked.InputTag("csctfDigis","DT")
    process.csctfReEmulTracks.SectorProcessor.initializeFromPSet = True

    process.load("L1Trigger.RPCTrigger.rpcTriggerDigis_cfi")
    process.rpcReEmulDigis = process.rpcTriggerDigis.clone()

    process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cfi")
    process.gmtReEmulDigis = process.gmtDigis.clone()
    process.gmtReEmulDigis.DTCandidates = cms.InputTag("gtDigis","DT")
    process.gmtReEmulDigis.RPCbCandidates = cms.InputTag("gtDigis","RPCb")
    if newCSCTFLUts=='True':
      process.gmtReEmulDigis.CSCCandidates = cms.InputTag("csctfReEmulDigis","CSC")
    else:
      process.gmtReEmulDigis.CSCCandidates = cms.InputTag("gtDigis","CSC")
    process.gmtReEmulDigis.RPCfCandidates = cms.InputTag("gtDigis","RPCf")
    process.gmtReEmulDigis.MipIsoData = cms.InputTag("none")
    
    HLTL1MuonTriggerSequence= cms.Sequence( process.csctfReEmulTracks + process.csctfReEmulDigis + process.gmtReEmulDigis )

    process.HLTL1MuonTriggerSequence = HLTL1MuonTriggerSequence

  if newGctSetting=='True':

    process.load('SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff')
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    # set the new input tags after RawToDigi
    if caloPrimitives.find("ECAL") != -1 :
      process.ecalReEmulDigis = process.simEcalTriggerPrimitiveDigis.clone()
      process.ecalReEmulDigis.Label = 'ecalDigis'     
      process.ecalReEmulDigis.InstanceEB = 'ebDigis'
      process.ecalReEmulDigis.InstanceEE = 'eeDigis'
      process.ecalReEmulDigis.BarrelOnly = False
      
    if caloPrimitives.find("HCAL") != -1 :
      process.hcalReEmulDigis = process.simHcalTriggerPrimitiveDigis.clone()
      process.hcalReEmulDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
        )
      process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)

      # CB get them from DB
      #process.HcalTPGCoderULUT.read_XML_LUTs = cms.bool(True)
      #process.HcalTPGCoderULUT.inputLUTs = cms.FileInPath("Physics2012v1a.xml")
      
    import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
    process.rctReEmulDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()
    if caloPrimitives.find("ECAL") != -1 :
      process.rctReEmulDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalReEmulDigis' ) )
    else :
      process.rctReEmulDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
    if caloPrimitives.find("HCAL") != -1 :
      process.rctReEmulDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'hcalReEmulDigis' ) )
    else :
      process.rctReEmulDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'hcalDigis' ) )

    import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
    process.gctReEmulDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
    process.gctReEmulDigis.inputLabel = 'rctReEmulDigis'

    if caloPrimitives.find("ECAL") != -1 and caloPrimitives.find("HCAL") != -1 :
      HLTL1CaloTriggerSequence= cms.Sequence( process.ecalReEmulDigis + process.hcalReEmulDigis + process.rctReEmulDigis + process.gctReEmulDigis )
    elif caloPrimitives.find("ECAL") != -1  :
      HLTL1CaloTriggerSequence= cms.Sequence( process.ecalReEmulDigis + process.rctReEmulDigis + process.gctReEmulDigis )
    elif caloPrimitives.find("HCAL") != -1 :
      HLTL1CaloTriggerSequence= cms.Sequence( process.hcalReEmulDigis + process.rctReEmulDigis + process.gctReEmulDigis )
    else :
      HLTL1CaloTriggerSequence= cms.Sequence( process.rctReEmulDigis + process.gctReEmulDigis )

    process.HLTL1CaloTriggerSequence = HLTL1CaloTriggerSequence
 
  return process

def switchToCustomL1Digis(process, customGmt, customGct, customGt):
  """patch the process to use custom GMT, GCT and GT results"""

  # explicit replacements to use "simGtDigis", "simGmtDigis" and "simGctDigis" instead of "hltGtDigis" or "hltGctDigis"
  if 'hltL1GtObjectMap' in process.__dict__:
    process.hltL1GtObjectMap.GmtInputTag = cms.InputTag( customGmt )
    process.hltL1GtObjectMap.GctInputTag = cms.InputTag( customGct )
  if 'hltL1extraParticles' in process.__dict__:
    process.hltL1extraParticles.muonSource          = cms.InputTag( customGmt )
    process.hltL1extraParticles.isolatedEmSource    = cms.InputTag( customGct, 'isoEm' )
    process.hltL1extraParticles.nonIsolatedEmSource = cms.InputTag( customGct, 'nonIsoEm' )
    process.hltL1extraParticles.centralJetSource    = cms.InputTag( customGct, 'cenJets' )
    process.hltL1extraParticles.forwardJetSource    = cms.InputTag( customGct, 'forJets' )
    process.hltL1extraParticles.tauJetSource        = cms.InputTag( customGct, 'tauJets' )
    process.hltL1extraParticles.etTotalSource       = cms.InputTag( customGct )
    process.hltL1extraParticles.etHadSource         = cms.InputTag( customGct )
    process.hltL1extraParticles.etMissSource        = cms.InputTag( customGct )
  if 'hltL2MuonSeeds' in process.__dict__:
    process.hltL2MuonSeeds.GMTReadoutCollection = cms.InputTag( customGmt )

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
  """patching the process to use newly emulated GT results"""
  return switchToCustomL1Digis(process, 'gtDigis', 'gctDigis', 'simGtDigis')

def switchToSimGmtGctGtDigis(process):
  """patching the process to use newly re-emulated Gmt, GCT followed by GT result re-emulation"""
  return switchToCustomL1Digis(process, 'simGmtDigis', 'simGctDigis', 'simGtDigis')

def switchToSimGctGtDigis(process):
  """patching the process to use newly re-emulated simGctDigis for GT re-emulation"""
  return switchToCustomL1Digis(process, 'gtDigis', 'simGctDigis', 'simGtDigis')

def switchToSimGmtGtDigis(process):
  """patching the process to use newly re-emulated simGmtDigis for GT re-emulation"""
  return switchToCustomL1Digis(process, 'simGmtDigis', 'gctDigis', 'simGtDigis')

def switchToSimGtReEmulGmtGctDigis(process):
  """patching the process to use newly emulated Gmt, GCT and GT results incl. re-emulation starting from new Muon and Calo LUTs"""
  return switchToCustomL1Digis(process, 'GmtReEmulDigis', 'gctReEmulDigis', 'simGtDigis')

def switchToSimGtReEmulGmtDigis(process):
  """patching the process to use newly emulated Gmt payload incl. re-emulation starting from new Muon LUTs followed by GT result re-emulation"""
  return switchToCustomL1Digis(process, 'GmtReEmulDigis', 'gctDigis', 'simGtDigis')

def switchToSimGtReEmulGctDigis(process):
  """patching the process to use newly emulated Gct payload incl. re-emulation starting from new Muon LUTs followed by GT result re-emulation"""
  return switchToCustomL1Digis(process, 'gtDigis', 'gctReEmulDigis', 'simGtDigis')

