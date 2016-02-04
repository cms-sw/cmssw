## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## add track candidates
from PhysicsTools.PatAlgos.tools.trackTools import *

makeTrackCandidates(process,
    label        = 'TrackCands',                  
    tracks       = cms.InputTag('generalTracks'), 
    particleType = 'pi+',                         
    preselection = 'pt > 10',                     
    selection    = 'pt > 10',                     
    isolation    = {'tracker':0.3, 'ecalTowers':0.3, 'hcalTowers':0.3},                            
    isoDeposits  = [],                            
    mcAs         = 'muon'           
)                                   

## select best vertex
process.bestVertex = cms.EDFilter(
    "PATSingleVertexSelector",
    mode      = cms.string("nearestToCandidate"),
    fallbacks = cms.vstring("fromCandidate", "beamSpot"),
    vertices              = cms.InputTag("offlinePrimaryVerticesWithBS"),
    vertexPreselection    = cms.vstring("(chi2prob(chi2,ndf) > 0.01) && (trackSize >= 3)"),
    candidates            = cms.VInputTag(cms.InputTag('gsfElectrons'), cms.InputTag('muons')),
    candidatePreselection = cms.string("pt > 5"),
    beamSpot              = cms.InputTag('offlineBeamSpot'),
)

## produce vertex associations
process.patTrackVertexInfo = cms.EDProducer(
    "PATVertexAssociationProducer",
    candidates = cms.VInputTag( 
    cms.InputTag('gsfElectrons'),
    cms.InputTag('muons'),
    cms.InputTag('patAODTrackCands'),
    ),
    useTracks = cms.bool(True), 
    vertices  = cms.InputTag('bestVertex'),
)

## add modules to the default sequence right after the patAODTrackCands 
process.patDefaultSequence.replace(process.patAODTrackCands,
                                   process.patAODTrackCands *
                                   process.bestVertex *
                                   process.patTrackVertexInfo
                                   )

## add it to the track candidates
process.patTrackCands.vertexing = cms.PSet(
    vertexAssociations = cms.InputTag("patTrackVertexInfo"),
)

## add generic tracks to the output file
process.out.outputCommands.append('keep *_selectedPatTrackCands_*_*')
process.out.outputCommands.append('keep *_patTrackVertexInfo_*_*')
process.out.outputCommands.append('keep *_bestVertex_*_*')

## let it run
process.p = cms.Path(
        process.patDefaultSequence 
)



## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#   process.source.fileNames = [          ##
#    '/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/14920B0A-0DE8-DE11-B138-002618943926.root'
#   ]                                     ##  (e.g. 'file:AOD.root')
#                                         ##
#   process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
#   process.out.fileName = ...            ##  (e.g. 'myTuple.root')
#                                         ##
#   process.options.wantSummary = True    ##  (to suppress the long output at the end of the job)    
