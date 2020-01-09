import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")


process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')


process.load('Configuration.StandardSequences.MagneticField_cff')

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load( "DQMServices/Core/DQMStore_cfg" )



process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }


)

# Configurations for MuonTrackValidators


# Configurations for RecoMuonValidators

#import SimGeneral.MixingModule.mixNoPU_cfi
#process.TrackAssociatorByChi2ESProducer = Validation.RecoMuon.associators_cff.TrackAssociatorByChi2ESProducer.clone(chi2cut = 100.0,ComponentName = 'TrackAssociatorByChi2')


import SimMuon.MCTruth.muonAssociatorByHitsHelper_cfi

process.muonAssociatorByHits = SimMuon.MCTruth.muonAssociatorByHitsHelper_cfi.muonAssociatorByHitsHelper.clone(#ComponentName = "muonAssociatorByHits",
 #tpTag = 'mix:MergedTrackTruth',
 UseTracker = True,
 UseMuon = False,
 EfficiencyCut_track = cms.double(0.0),
 PurityCut_track = cms.double(0.0)
 )


from CommonTools.RecoAlgos.me0Associator import *

#process.me0MuonSel = cms.Sequence(
#    me0muonColl_seq
#    )

process.me0MuonSel = me0muon

process.Test = cms.EDAnalyzer("ME0MuonAnalyzer",
                              

                              HistoFolder = cms.string('OUTPUTTEMPLATE'),
                              HistoFile = cms.string('OUTPUTTEMPLATE.root'),

                              ME0MuonSelectionType = cms.string('Loose'),
                              FakeRatePtCut = cms.double(3.0),
                              MatchingWindowDelR = cms.double (.15),
                              RejectEndcapMuons = cms.bool(False),
                              UseAssociators = cms.bool(False),

                              associators = cms.vstring('muonAssociatorByHits'),

                              label = cms.VInputTag('me0muon'),
                              
                              
)

process.p = cms.Path(process.me0MuonSel*process.Test)



process.PoolSource.fileNames = [

'file:/afs/cern.ch/work/d/dnash/ME0Segments/ForRealSegmentsOnly/ReCommit75X/CMSSW_7_5_X_2015-06-29-2300/src/out_digi_test.root'

]
