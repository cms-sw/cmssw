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
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.source = cms.Source("PoolSource",
# 157884 - cosmics
    fileNames = cms.untracked.vstring(                          
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/F22D81ED-BF38-E011-AFA2-0019B9F70468.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/DE96271E-D538-E011-9EFD-001D09F24682.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/DCC2B27F-BE38-E011-892B-001D09F2932B.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/D8D8037B-D838-E011-9DF6-001D09F24259.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/D4F50EC5-C938-E011-9E9C-0030487C608C.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/BC14A85C-DD38-E011-88C0-003048F118D4.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/AC95A605-BB38-E011-A39F-0030487D05B0.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/A0FA70E0-CB38-E011-94C3-003048F1110E.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/92501BF6-E638-E011-B461-003048F1C832.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/82E99EEA-B838-E011-8017-0030487CD184.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/8034D960-B538-E011-B121-003048F11DE2.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/7E239292-D338-E011-BB32-0030487C5CFA.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/76AC3661-EE38-E011-BFA8-0030487CD6DA.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/5E14751A-DE38-E011-B9E3-001D09F28D4A.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/589162EF-E038-E011-9907-001D09F244BB.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/528FFA66-CF38-E011-BA76-001D09F25479.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/4E462F65-BC38-E011-B9FA-001D09F23F2A.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/4043F509-C238-E011-A52F-001D09F34488.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/3ED25535-B838-E011-9593-0030487CD7B4.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/3AADB84F-CD38-E011-A34C-0019B9F709A4.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/3A2D4ED8-DE38-E011-AF77-0019B9F72BAA.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/3469F1E0-C538-E011-A153-0030487C7E18.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/2AE84352-E238-E011-8025-0030487C912E.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/28D1741D-D738-E011-B9DF-0030487D1BCC.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/22403963-C838-E011-AA86-001D09F24F65.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/1ADF7E78-D138-E011-B88F-003048F118D4.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/0CE94E23-C438-E011-B2CF-0030487CD6DA.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/0A600A8C-DF38-E011-87FB-001D09F232B9.root",
 "/store/data/Commissioning11/CommissioningNoBeam/RAW/v1/000/157/884/02E8D4AD-DC38-E011-9046-001617C3B6CC.root"
 

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


