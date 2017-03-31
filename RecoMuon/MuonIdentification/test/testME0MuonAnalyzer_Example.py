import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")


process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')


process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load( "DQMServices/Core/DQMStore_cfg" )

process.load('Validation.RecoMuon.associators_cff')
process.load('Validation.RecoMuon.selectors_cff')
process.load('Validation.RecoMuon.MuonTrackValidator_cfi')
process.load('Validation.RecoMuon.RecoMuonValidator_cfi')



process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }


)

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *
# Configurations for MuonTrackValidators
import Validation.RecoMuon.MuonTrackValidator_cfi


# Configurations for RecoMuonValidators
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

#import SimGeneral.MixingModule.mixNoPU_cfi
from SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi import *
from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters

#process.TrackAssociatorByChi2ESProducer = Validation.RecoMuon.associators_cff.TrackAssociatorByChi2ESProducer.clone(chi2cut = 100.0,ComponentName = 'TrackAssociatorByChi2')

import SimMuon.MCTruth.MuonAssociatorByHitsESProducer_cfi

process.muonAssociatorByHits = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_cfi.muonAssociatorByHitsESProducer.clone(ComponentName = 'muonAssociatorByHits',
 #tpTag = 'mix:MergedTrackTruth',
 UseTracker = True,
 UseMuon = False,
 EfficiencyCut_track = cms.double(0.0),
 PurityCut_track = cms.double(0.0)
 )

process.recoMuonValidation = cms.Sequence(#probeTracks_seq*
    #(selectedVertices * selectedFirstPrimaryVertex) * 
    #bestMuonTuneP_seq*
    #muonColl_seq*trackColl_seq*extractedMuonTracks_seq*bestMuon_seq*trackerMuon_seq*
    me0muonColl_seq
    #((process.muonValidation_seq))
    )

process.Test = cms.EDAnalyzer("ME0MuonAnalyzer",
                              

                              HistoFolder = cms.string('OUTPUTTEMPLATE'),
                              HistoFile = cms.string('OUTPUTTEMPLATE.root'),

                              ME0MuonSelectionType = cms.string('Loose'),
                              FakeRatePtCut = cms.double(3.0),
                              MatchingWindowDelR = cms.double (.15),
                              RejectEndcapMuons = cms.bool(False),
                              UseAssociators = cms.bool(True),

                              associators = cms.vstring('muonAssociatorByHits'),

                              label = cms.VInputTag('me0muon'),
                              
                              
)

process.p = cms.Path(process.recoMuonValidation*process.Test)



process.PoolSource.fileNames = [

FILETEMPLATE

]
