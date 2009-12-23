
import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEstep1_161209.root')
)

process.MessageLogger = cms.Service("MessageLogger",
   
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
inputCommands = cms.untracked.vstring("keep *", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap_*_HLT"),
                             duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
fileNames = cms.untracked.vstring(
#Run123596 newBS
#'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-123596-stdReco-fitBS123592v2-lumi1-68.root',    
#'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-123596-stdReco-fitBS123592v2-lumi69-129.root',
#'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-123596-stdReco-fitBS123592v2-lumi130-143.root'
#MC con vertice come i dati
#'rfio:/castor/cern.ch/user/l/lucaroni/PerLuca/MCwithVertexDATA.root'
#MinimumBias_BeamCommissioning09-Dec9thReReco_BSCNOBEAMHALO-v1
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/FCEF8EA6-22E5-DE11-82BE-0026189438F8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/F66B5C9A-22E5-DE11-88D1-0026189438B5.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/ECE9B13A-22E5-DE11-B670-00261894386C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/C4F58276-22E5-DE11-96BC-00261894386D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/745F3357-22E5-DE11-A666-0026189438B5.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/6C216467-22E5-DE11-9CCB-0026189438B5.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/68F3B885-22E5-DE11-88B7-00261894386C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/687B33A1-22E5-DE11-9674-002618943896.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/60463596-22E5-DE11-A9F4-002618943920.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/5EEE0F8E-22E5-DE11-9759-002618943920.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/5C1FE043-22E5-DE11-8ACB-002618943920.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/5A8A4A57-22E5-DE11-81B0-0026189438B5.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/4EF5D17A-22E5-DE11-8405-00261894386D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/289FCB66-22E5-DE11-89DC-002618FDA265.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/1A202DCE-1DE5-DE11-A176-002618943985.root'
)
             
                            )

process.p1 = cms.Path(process.UEAnalysisTracks*process.UEAnalysisJetsOnlyReco*process.UEAnalysis)

process.UEAnalysisRootple.OnlyRECO     = True
process.UEAnalysisRootple500.OnlyRECO  = True
process.UEAnalysisRootple1500.OnlyRECO = True
process.UEAnalysisRootple700.OnlyRECO  = True
process.UEAnalysisRootple1100.OnlyRECO = True

process.UEAnalysisRootple.GenJetCollectionName      = 'ueSisCone5GenJet'
process.UEAnalysisRootple.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet'
process.UEAnalysisRootple.TracksJetCollectionName   = 'ueSisCone5TracksJet'
process.UEAnalysisRootple.RecoCaloJetCollectionName = 'sisCone5CaloJets'
process.UEAnalysisRootple500.GenJetCollectionName      = 'ueSisCone5GenJet500'
process.UEAnalysisRootple500.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet500'
process.UEAnalysisRootple500.TracksJetCollectionName   = 'ueSisCone5TracksJet500'
process.UEAnalysisRootple500.RecoCaloJetCollectionName = 'sisCone5CaloJets'
process.UEAnalysisRootple1500.GenJetCollectionName      = 'ueSisCone5GenJet1500'
process.UEAnalysisRootple1500.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet1500'
process.UEAnalysisRootple1500.TracksJetCollectionName   = 'ueSisCone5TracksJet1500'
process.UEAnalysisRootple1500.RecoCaloJetCollectionName = 'sisCone5CaloJets'
process.UEAnalysisRootple700.GenJetCollectionName      = 'ueSisCone5GenJet700'
process.UEAnalysisRootple700.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet700'
process.UEAnalysisRootple700.TracksJetCollectionName   = 'ueSisCone5TracksJet700'
process.UEAnalysisRootple700.RecoCaloJetCollectionName = 'sisCone5CaloJets'
process.UEAnalysisRootple1100.GenJetCollectionName      = 'ueSisCone5GenJet1100'
process.UEAnalysisRootple1100.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet1100'
process.UEAnalysisRootple1100.TracksJetCollectionName   = 'ueSisCone5TracksJet1100'
process.UEAnalysisRootple1100.RecoCaloJetCollectionName = 'sisCone5CaloJets'
