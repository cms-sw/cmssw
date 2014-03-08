import FWCore.ParameterSet.Config as cms

def customise(process):
    #if hasattr(process,'famosMuonSequence'):
    #if hasattr(process,'globalMuons'):
    if hasattr(process,'famosWithEverything'):
        process=customise_AssumeOneCustomiseStep(process)
    return process

def customise_AssumeOneCustomiseStep(process):
    #from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
    
    #process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

    process.load('FastSimulation.Muons.me0SegmentProducer_cfi')
    process.load('RecoMuon.MuonIdentification.me0SegmentMatcher_cfi')
    process.load('RecoMuon.MuonIdentification.me0MuonConverter_cfi')

    #process.famosMuonSequence += process.me0SegmentProducer
    #process.famosMuonSequence += process.me0SegmentMatcher
    #process.famosMuonSequence += process.me0MuonConverter

    process.reconstructionWithFamos += process.me0SegmentProducer
    process.reconstructionWithFamos += process.me0SegmentMatcher
    process.reconstructionWithFamos += process.me0MuonConverter
    process=outputCustoms(process)
    


    return process


def outputCustoms(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        #Not entirely sure what the purpose of this is, but maybe I should include the processes that were put in above?
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_me0SegmentProducer_*_*')
            getattr(process,b).outputCommands.append('keep *_me0SegmentMatcher_*_*')
            getattr(process,b).outputCommands.append('keep *_me0MuonConverter_*_*')

    return process
