import FWCore.ParameterSet.Config as cms

def TkOnlyDigi(process):
    print "!!! Special version of the digitization for tracker only !!!"
    if hasattr(process,'digitisationTkOnly_step'):
        process=customise_DigiTkOnly(process)

    return process

def customise_DigiTkOnly(process):
    process.load('Configuration.StandardSequences.Digi_cff')
    process.doAllDigi = cms.Sequence()
    process.load('SimGeneral.MixingModule.mixObjects_cfi')
    process.digitisationTkOnly_step.remove(process.mix.mixObjects.mixCH)
    del process.simCastorDigis
    del process.simEcalUnsuppressedDigis
    del process.simHcalUnsuppressedDigis
    process.mix.digitizers = cms.PSet(process.theDigitizersValid)
    del process.mix.digitizers.ecal
    del process.mix.digitizers.hcal
    del process.mix.digitizers.castor
    process.digitisationTkOnly_step.remove(process.mix.digitizers.pixel)
    process.load('SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi')
    process.mix.digitizers.pixel=process.phase2TrackerDigitizer
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    #Check if mergedtruth is in the sequence first, could be taken out depending on cmsDriver options
    if hasattr(process.mix.digitizers,"mergedtruth") :
        process.mix.digitizers.mergedtruth.simHitCollections.muon = cms.VInputTag( )
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"))

    # keep new digis
    alist=['FEVTDEBUG','FEVTDEBUGHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep Phase2TrackerDigiedmDetSetVector_*_*_*')
    return process

