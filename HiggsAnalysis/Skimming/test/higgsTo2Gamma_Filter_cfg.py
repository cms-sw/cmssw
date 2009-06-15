import FWCore.ParameterSet.Config as cms

process = cms.Process('SkimH2Gam')

# Complete Preselection Sequence for 2e2mu analysis

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_31X::All'

# Complete Skim analysis
process.load('HiggsAnalysis/Skimming/higgsTo2Gamma_Sequences_cff')

process.eca = cms.EDAnalyzer("EventContentAnalyzer")

process.hTo2GammaSkimPath = cms.Path(process.higgsTo2GammaSequence)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hTo2Gamma_Skim.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsTo2Gamma_Sequence')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('hTo2GammaSkimPath')
    )
                               
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0004/1411F38E-E641-DE11-A3FB-001D09F2B2CF.root',
    '/store/relval/CMSSW_3_1_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0003/B252A5A3-7741-DE11-B4A2-001D09F2983F.root',
    '/store/relval/CMSSW_3_1_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0003/40359B99-7541-DE11-B614-001D09F28C1E.root',
    '/store/relval/CMSSW_3_1_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0003/0CC3E01B-7741-DE11-8A5F-001D09F2841C.root'
                    )
                            
                            )

#Loose cuts (this is the default)
#process.higgsTo2GammaFilter.photonLooseMinPt = cms.double(15.0)
#process.higgsTo2GammaFilter.photonTightMinPt = cms.double(25.0)
#process.higgsTo2GammaFilter.photonLooseMaxEta = cms.double(3.1)
#process.higgsTo2GammaFilter.photonTightMaxEta = cms.double(2.6)
#process.higgsTo2GammaFilter.photonLooseMaxHoE = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHoE = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonLooseMaxHIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHIsol = cms.double(20.)
#process.higgsTo2GammaFilter.photonLooseMaxEIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxEIsol = cms.double(20.)
#process.higgsTo2GammaFilter.photonLooseMaxTIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxTIsol = cms.double(10.)
#process.higgsTo2GammaFilter.nPhotonLooseMin = cms.int32(2)
#process.higgsTo2GammaFilter.nPhotonTightMin = cms.int32(1)
#process.higgsTo2GammaFilter.DebugHiggsTo2GammaSkim = cms.bool(False)

#Tight cuts (uncomment the following) 
#process.higgsTo2GammaFilter.photonLooseMinPt = cms.double(20.0)
#process.higgsTo2GammaFilter.photonTightMinPt = cms.double(30.0)
#process.higgsTo2GammaFilter.photonLooseMaxEta = cms.double(3.1)
#process.higgsTo2GammaFilter.photonTightMaxEta = cms.double(2.6)
#process.higgsTo2GammaFilter.photonLooseMaxHoE = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHoE = cms.double(0.2)
#process.higgsTo2GammaFilter.photonLooseMaxHIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHIsol = cms.double(15.)
#process.higgsTo2GammaFilter.photonLooseMaxEIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxEIsol = cms.double(10.)
#process.higgsTo2GammaFilter.photonLooseMaxTIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxTIsol = cms.double(5.)
#process.higgsTo2GammaFilter.nPhotonLooseMin = cms.int32(2)
#process.higgsTo2GammaFilter.nPhotonTightMin = cms.int32(1)
#process.higgsTo2GammaFilter.DebugHiggsTo2GammaSkim = cms.bool(False)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Endpath
process.o = cms.EndPath ( process.output )



