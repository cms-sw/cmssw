#
import FWCore.ParameterSet.Config as cms

process = cms.Process("RawDumper")

# process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dumper'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

#process.MessageLogger.cerr.FwkReport.reportEvery = 1
#process.MessageLogger.cerr.threshold = 'Debug'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/COSMIC/RAW/002C2E77-8CD5-DE11-9533-000423D944FC.root')
#   fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/ONE_BEAM/RAW/ZeroBias/0EB6E8E2-FCDD-DE11-9191-001D09F24DDF.root')
# Min Bias
#   fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/ONE_BEAM/RAW/MinBias/262EAE54-F9DD-DE11-B1DF-003048D37580.root')
#   fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/ONE_BEAM/RAW/MinBias/00F05467-F4DD-DE11-857C-003048D2C0F4.root')
#   fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/ONE_BEAM/RAW/MinBias/6C8F0233-FCDD-DE11-BF8E-001D09F297EF.root')
# 130269 - cosmics
    fileNames = cms.untracked.vstring(                          
    "/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/0254F0EC-8529-DF11-9E85-001D09F23A34.root" 
#06A61340-6429-DF11-AB5C-0030487C6090.root
#0ED7BC94-7829-DF11-BB00-001D09F29538.root
#105B729E-6529-DF11-A924-000423D6B358.root
#186DB116-6229-DF11-B151-000423D99160.root
#28856A31-7029-DF11-9D76-001617DC1F70.root
#2E4469B2-6729-DF11-B2F7-000423D94E1C.root
#3A37DB7D-7629-DF11-9F33-0030487CD812.root
#3A6CE564-8729-DF11-9AAF-001D09F2423B.root
#3CF88999-7F29-DF11-95C8-001D09F24664.root
#4060F389-7D29-DF11-954C-00304879EDEA.root
#40DC63E4-7729-DF11-A2BF-0030487C90EE.root
#4AF113C0-8129-DF11-B883-0030487A17B8.root
#5AC8F814-7529-DF11-9A1F-0030487CD6D2.root
#5C32211F-7C29-DF11-B93A-0030487CD7E0.root
#6425A86A-9529-DF11-BCBF-0019B9F581C9.root
#6A88F9F1-8529-DF11-B43E-001D09F250AF.root
#6CFCCA08-6E29-DF11-9F8E-001617C3B706.root
#6EB6C7EC-8529-DF11-B619-000423D99614.root
#7270F464-8729-DF11-86DF-001D09F29146.root
#7407BB64-8729-DF11-89EE-001D09F2AF96.root
#74A70CDB-8329-DF11-92CA-0030487A18D8.root
#7A75AEFA-6B29-DF11-A618-00304879FA4C.root
#86514DF4-7229-DF11-94AA-001D09F24FEC.root
#8814400C-6729-DF11-9BA4-0030486780B8.root
#8CE8F270-6829-DF11-8E95-001D09F2527B.root
#8E645094-7829-DF11-945A-001D09F252E9.root
#8E91CCA6-6C29-DF11-B852-001D09F29619.root
#9618F88A-7D29-DF11-894A-0030487C90C2.root
#9638F115-7529-DF11-B567-0030487CD77E.root
#9C4A835C-6129-DF11-B326-001617C3B6E2.root
#A298D2D8-6929-DF11-A962-001D09F25109.root
#A8E2A844-6429-DF11-952B-0030487CD17C.root
#AC3CA50A-6729-DF11-AA54-0030486780A8.root
#AE87C3C0-8129-DF11-97DC-0030487A18F2.root
#B0F975DC-6229-DF11-A577-0030487CD710.root
#B48D5BFB-7029-DF11-B0CF-000423D99E46.root
#BE21BBDB-8329-DF11-ACDC-0030487C608C.root
#C4DA26E0-7729-DF11-B5C6-0030487C8CB6.root
#CAB036EF-7229-DF11-866E-0019B9F72BAA.root
#D6BE6A60-6D29-DF11-A511-000423D6A6F4.root
#DA1B28AD-7A29-DF11-83BA-000423D6B48C.root
#DE4E1F39-6B29-DF11-9BE6-0030487A3C92.root
#E2454CE9-8329-DF11-9C51-0030487A3C92.root
#E2A23CE8-6429-DF11-B5EB-001617E30F50.root
#E81E4D36-6B29-DF11-9110-001617C3B6E2.root
#EC5B65D9-6929-DF11-987B-0030487CD180.root
#EE1A5B72-6F29-DF11-938F-001D09F24DA8.root
#EE79F489-7D29-DF11-836E-0030487CD7B4.root
#F0C5F51D-7C29-DF11-B12C-0030487CD704.root
#FA924CE7-7029-DF11-868A-0019DB2F3F9A.root
#FCF75699-7F29-DF11-802D-001D09F291D7.root
#FEEB9DB8-7329-DF11-BB49-001D09F253FC.root


    )

)

#process.out = cms.OutputModule("PoolOutputModule",
#    fileName =  cms.untracked.string('file:histos.root')
#)

process.dumper = cms.EDAnalyzer("SiPixelRawDumper", 
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('source'),
#    InputLabel = cms.untracked.string('siPixelRawData'),
    CheckPixelOrder = cms.untracked.bool(False)
)

# process.s = cms.Sequence(process.dumper)

process.p = cms.Path(process.dumper)

# process.ep = cms.EndPath(process.out)


