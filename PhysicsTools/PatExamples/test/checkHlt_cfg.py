import FWCore.ParameterSet.Config as cms

process = cms.Process( "HLTPROV" )

# source
process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/06FC3959-4DFC-DD11-B504-00E08178C091.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/08126A32-C2FC-DD11-BFF3-00E08178C091.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/089F9442-28FC-DD11-803B-0015170AD174.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/0A1B6F50-B7FC-DD11-9744-00E08178C091.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/0A3AFC80-30FC-DD11-A41B-00E081791807.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/0E61259E-32FC-DD11-BECB-00E08178C0EF.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/0ED486E9-44FC-DD11-83ED-00E08178C091.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/109E30E1-C2FC-DD11-9459-00E08178C091.root',
        'dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/mc/Fall08/TTJets-madgraph/GEN-SIM-RECO/IDEAL_V11_redigi_v10/0000/121F3706-C4FC-DD11-A0DD-00E08178C091.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

# HLT analyzers
process.load( "HLTrigger.HLTcore.hltEventAnalyzerAOD_cfi" )
process.hltEventAnalyzerAOD.triggerName = cms.string( '@' )
process.load( "HLTrigger.HLTcore.triggerSummaryAnalyzerAOD_cfi" )
                                                                       
process.p = cms.Path(
    process.hltEventAnalyzerAOD       +
    process.triggerSummaryAnalyzerAOD
)
