import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonSysAging.DTChamberMasker_cfi import DTChamberMasker

def appendChamberMaskerAtUnpacking(process, doDigis, doTrigger, chambRegEx):

    if doDigis and hasattr(process,'muonDTDigis') :

        print "[appendChamberMasker] : Found muonDTDigis, applying filter"

        process.preDtDigis = process.muonDTDigis.clone()
        process.muonDTDigis = DTChamberMasker.clone()
        if len(chambRegEx) > 0 :
            process.muonDTDigis.maskedChRegEx = chambRegEx
        process.muonDTDigis.digiTag = cms.InputTag('preDtDigis') 
        process.muonDTDigis.triggerPrimPhTag = cms.InputTag('')
        process.muonDTDigis.triggerPrimThTag = cms.InputTag('')
        process.filteredDigiSequence = cms.Sequence(process.preDtDigis + process.muonDTDigis)
        process.RawToDigi.replace(process.muonDTDigis, process.filteredDigiSequence)

    if doDigis and hasattr(process,'dttfDigis') :

        print "[appendChamberMasker] : Found dttfDigis, applying filter"

        process.preDttfDigis = process.dttfDigis.clone()
        process.dttfDigis = DTChamberMasker.clone()
        if len(chambRegEx) > 0 :
            process.dttfDigis.maskedChRegEx = chambRegEx
      #  process.dttfDigis.digiTag = cms.InputTag('')
        process.dttfDigis.digiTag = cms.InputTag('')
        process.dttfDigis.triggerPrimPhTag = cms.InputTag('preDttfDigis')
        process.dttfDigis.triggerPrimThTag = cms.InputTag('preDttfDigis')

        process.filteredTrigSequence = cms.Sequence(process.preDttfDigis + process.dttfDigis)
        process.RawToDigi.replace(process.dttfDigis, process.filteredTrigSequence)
    return process
        
def appendChamberMaskerAtHLT(process, doDigis, doTrigger, chambRegEx):
    if doDigis and hasattr(process,'hltMuonDTDigis') :

        print "[appendChamberMasker] : Found hltMuonDTDigis, applying filter"

        process.preHltDtDigis = process.hltMuonDTDigis.clone()
        process.hltMuonDTDigis = DTChamberMasker.clone()
        if len(chambRegEx) > 0 :
            process.hltMuonDTDigis.maskedChRegEx = chambRegEx
            process.hltMuonDTDigis.digiTag = "preHltDtDigis"
        process.hltMuonDTDigis.triggerPrimPhTag = cms.InputTag('simDtTriggerPrimitiveDigis')
        process.hltMuonDTDigis.triggerPrimThTag = cms.InputTag('simDtTriggerPrimitiveDigis')
        process.filteredHltDtDigiSequence = cms.Sequence(process.preHltDtDigis + process.hltMuonDTDigis)
        process.HLTMuonLocalRecoSequence.replace(process.hltMuonDTDigis, process.filteredHltDtDigiSequence)
    return process

def reRunDttf( process ):

    #process.dttfDigisMasked = process.valDttfDigis.clone()
    #process.dttfDigisMasked.DTDigi_Source = cms.InputTag("muonDTDigis", "DT")
    #from L1Trigger.GlobalMuonTrigger.gmtDigis_cfi import gmtDigis
    from L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi import csctfTrackDigis
    process.load ( "L1Trigger.GlobalMuonTrigger.gmtDigis_cfi")
    process.load ( "L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi")
    process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
    process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        cscTriggerPrimitiveDigis = cms.PSet(
            initialSeed = cms.untracked.uint32(789342)

        ),
        csc2DRecHitsOverload = cms.PSet(
            initialSeed = cms.untracked.uint32(81)
        ),
    )
    from L1Trigger.DTTrackFinder.dttfDigis_cfi import dttfDigis as dttfTrackFinder
    process.dttfTrackDigis = dttfTrackFinder.clone()
    process.dttfTrackDigis.DTDigi_Source = cms.InputTag("dttfDigis")
    process.dttfTrackDigis.CSCStub_Source = cms.InputTag("csctfTrackDigis")

    if hasattr(process.dttfTrackDigis, 'DTDigi_Theta_Source'):
        process.dttfTrackDigis.DTDigi_Theta_Source = cms.InputTag("dttfDigis") 
    #process.csctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("muonCSCDigis")
    process.csctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("cscTriggerPrimitiveDigis")
    process.csctfTrackDigis.DTproducer = cms.untracked.InputTag("dttfDigis")
    process.gmtDigis.DTCandidates = cms.InputTag("dttfTrackDigis","DT")
    process.gmtDigis.RPCbCandidates = cms.InputTag("gtDigis","RPCb")
    process.gmtDigis.RPCfCandidates = cms.InputTag("gtDigis","RPCf")
    process.gmtDigis.CSCCandidates  = cms.InputTag("gtDigis","CSC")
    process.RawToDigiL1MuonEmulator = cms.Sequence( process.RawToDigi
                                                    * process.cscTriggerPrimitiveDigis
                                                    * process.csctfTrackDigis 
                                                    * process.dttfTrackDigis
                                                    * process.gmtDigis )
    process.raw2digi_step.replace( process.RawToDigi, process.RawToDigiL1MuonEmulator )

    # process.ValL1MuTrackFinders.replace(process.valGmtDigis, process.gmtDigis)
    # process.ValL1Emulator.replace(process.valGmtDigis, process.gmtDigis)
    # process.gmtDigis.DTCandidates = cms.InputTag("dttfDigis", "DT")
    # process.l1compare.GMTsourceEmul = cms.InputTag("gmtDigis")
    # process.deGmt.replace(process.valGmtDigis, process.gmtDigis)
    if hasattr(process,'RECOSIMoutput'):
        process.RECOSIMoutput.outputCommands.extend( ['keep *_dttfDigis_*_*', 
                                                      'keep *_dttfTrackDigis_*_*', 
                                                      'keep *_gmtDigis_*_*'] )
    if hasattr(process,'AODSIMoutput'):
        process.AODSIMoutput.outputCommands.extend( ['keep *_dttfDigis_*_*',
                                                     'keep *_dttfTrackDigis_*_*',
                                                     'keep *_gmtDigis_*_*'] )
    return process
