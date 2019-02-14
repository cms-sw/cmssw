import FWCore.ParameterSet.Config as cms
import sys 
process = cms.Process("L1TrackJets")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:/mnt/hadoop/store/user/rish/MinBiasPU200/reprocessMinBias%s.root' %sys.argv[2]
        #'file:/mnt/hadoop/store/user/rish/TTBarPU200_10X/reprocessTTJets%s.root' %sys.argv[2]
    )
)
#from  CfiFile_cfi.py import  TwoLayerJets 
process.TwoLayerJets = cms.EDProducer('TwoLayerJets',
                L1TrackInputTag= cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
                ZMAX = cms.double ( 15. ) ,
                PTMAX = cms.double( 200. ),
                Etabins=cms.int32(24),
                Phibins=cms.int32(27),
                Zbins=cms.int32(60),
                TRK_PTMIN = cms.double(2.0),        # minimum track pt [GeV]
                TRK_ETAMAX = cms.double(2.5),       # maximum track eta
		CHI2_MAX=cms.double(50.),
		PromptBendConsistency=cms.double(1.75),
		D0_Cut=cms.double(0.1),
		NStubs4Chi2_rz_Loose=cms.double(0.5),
		NStubs4Chi2_rphi_Loose=cms.double(0.5),
		NStubs4Displacedbend_Loose=cms.double(1.25),	
		NStubs5Chi2_rz_Loose=cms.double(2.5),
		NStubs5Chi2_rphi_Loose=cms.double(5.0),
		NStubs5Displacedbend_Loose=cms.double(5.0),
		NStubs5Chi2_rz_Tight=cms.double(2.0),
		NStubs5Chi2_rphi_Tight=cms.double(3.5),
		NStubs5Displacedbend_Tight=cms.double(4.0)

)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/nfs/data39/cms/rish/Production93X/myOutputFile%s.root' %sys.argv[2]),
outputCommands = cms.untracked.vstring('drop *',
					 'keep *_TwoLayerJets_*_*',
							
	)
)
  
process.p = cms.Path(process.TwoLayerJets)

process.e = cms.EndPath(process.out)
