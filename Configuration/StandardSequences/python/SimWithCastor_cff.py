import FWCore.ParameterSet.Config as cms

def customise(process):

  if hasattr(process,'g4SimHits'):
    # time window 10 millisecond
    process.common_maximum_time.DeadRegions = cms.vstring('InterimRegion')
    # Eta cut
    process.g4SimHits.Generator.MinEtaCut = cms.double(-7.0)
    process.g4SimHits.Generator.MaxEtaCut = cms.double(5.5)
    # stacking action
    process.g4SimHits.StackingAction.DeadRegions = cms.vstring('InterimRegion')
    # stepping action
    process.g4SimHits.SteppingAction.DeadRegions = cms.vstring('InterimRegion')
    # castor shower library
    process.g4SimHits.CastorSD.useShowerLibrary = cms.bool(True)

    return(process)
