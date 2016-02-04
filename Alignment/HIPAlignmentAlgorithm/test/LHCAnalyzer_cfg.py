import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
##process.load("AuxCode.CheckTkCollection.Run123151_RECO_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
                            cout = cms.untracked.PSet(
                                   threshold = cms.untracked.string('WARNING')
                                   ),
                            destinations = cms.untracked.vstring('cout')
                            )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR09_R_34X_V2::All'
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_123615.root','rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_124009.root','rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_124020.root','rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_124022.root')


#

#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_124024.root','rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_124030.root','rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_124230.root'


#,'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/user/emiglior/ALCARECO/08Jan10/TkAlMinBias_124120.root' #2.36TeV run

)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.LhcTrackAnalyzer = cms.EDAnalyzer("LhcTrackAnalyzer",
#                                          TrackCollectionTag = cms.InputTag("generalTracks"),
                                          TrackCollectionTag = cms.InputTag("ALCARECOTkAlMinBias"),
                                          PVtxCollectionTag = cms.InputTag("offlinePrimaryVertices"),
                                          OutputFileName = cms.string("AnalyzerOutput_1.root"),
                                          Debug = cms.bool(False)
                                          )

process.p = cms.Path(process.LhcTrackAnalyzer)
