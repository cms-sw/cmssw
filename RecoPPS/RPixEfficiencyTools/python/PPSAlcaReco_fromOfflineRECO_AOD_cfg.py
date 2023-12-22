import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing

import sys

process = cms.Process('MANUALALCARECO', eras.Run3)

#SETUP PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    )
options = VarParsing.VarParsing ('analysis')
options.register('outputFileName',
                'PPS_ALCARECO.root',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('jsonFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "JSON file list name")
options.register('alignmentXMLName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "Alignment XML file name")
options.register('alignmentDBName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "Alignmend DB file name")

options.register('globalTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "GT to use")
options.parseArguments()

#SETUP LOGGER
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( 
        optionalPSet = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(100),
            limit = cms.untracked.int32(50000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring(
        "FwkReport"
    ),
)


#CONFIGURE PROCESS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#SETUP GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
if options.globalTag != '':
    gt = options.globalTag
else:
    gt = 'auto:run3_data_prompt'


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
print('Using GT:',gt)
process.GlobalTag = GlobalTag(process.GlobalTag, gt)

# Patch for LHCInfo not in GT
# process.GlobalTag.toGet.append(
#     cms.PSet(
#     record = cms.string("LHCInfoPerFillRcd"),
#     tag = cms.string("LHCInfoPerFill_endFill_Run3_v1"),
#     label = cms.untracked.string(""),
#     connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#     )
# )
# process.GlobalTag.toGet.append(
#     cms.PSet(
#     record = cms.string("LHCInfoPerLSRcd"),
#     tag = cms.string("LHCInfoPerLS_endFill_Run3_v2"),
#     label = cms.untracked.string(""),
#     connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#     )
# )
# End of patch

# Handle alignment inputs
if options.alignmentXMLName and options.alignmentDBName:
    print('ERROR: Both alignment XML and DB files specified. Please specify only one.')
    sys.exit(1)

if options.alignmentXMLName:
    # Load alignments from XML  
    print('Loading alignment from XML file:', options.alignmentXMLName)
    process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
    process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(options.alignmentXMLName)
    process.esPreferLocalAlignment = cms.ESPrefer("CTPPSRPAlignmentCorrectionsDataESSourceXML", "ctppsRPAlignmentCorrectionsDataESSourceXML")
elif options.alignmentDBName:
    # Load alignments from DB file
    print('Loading alignment from DB file:', options.alignmentDBName)
    from CondCore.CondDB.CondDB_cfi import *
    ppsAlignmentDB = CondDB.clone()
    ppsAlignmentDB.connect = cms.string("sqlite_file:"+options.alignmentDBName)
    process.ppsAlignment = cms.ESSource("PoolDBESSource",ppsAlignmentDB,
        toGet = cms.VPSet(
            cms.PSet(
            record = cms.string("RPRealAlignmentRecord"),
            tag = cms.string("CTPPSRPAlignment_real"),
            label = cms.untracked.string("")
            )
        )
    )
    process.es_prefer_ppsAlignment = cms.ESPrefer("PoolDBESSource","ppsAlignment")
else: 
    print('Using alignment from GT.')

if len(options.inputFiles) != 0:
    # Add 'file:' in front of file names
    inputFiles = []
    for file_name in options.inputFiles:
        inputFiles.append('file:'+file_name)
    inputFiles = cms.untracked.vstring(inputFiles) 
else:
    inputFiles = cms.untracked.vstring('/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/3581759b-c29a-4422-ac31-7a14c172846f.root')
print('Input files:\n',inputFiles, sep='')

#SETUP INPUT
process.source = cms.Source("PoolSource",
    fileNames = inputFiles,
    # Drop everything from the prompt alcareco besides the digis at input
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_hltGtStage2Digis_*_*',
        'keep *_gtStage2Digis_*_*',
        'keep *_*Digi*_*_RECO'
    )
)

if options.jsonFileName:
    import FWCore.PythonUtilities.LumiList as LumiList
    jsonFileName = options.jsonFileName
    print("Using JSON file:",jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(options.outputFileName),
    # Keep only the new products
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_hltGtStage2Digis_*_*',
        'keep *_gtStage2Digis_*_*',
    )
)

# Load the ALCARECO reco step from DIGI
process.load("Calibration.PPSAlCaRecoProducer.ALCARECOPPSCalMaxTracks_cff")
# Remove sampic reco
process.recoPPSSequenceAlCaRecoProducer.remove(process.diamondSampicLocalReconstructionTaskAlCaRecoProducer)

# Adapt ALCARECO InputTags to Offline products
process.ctppsPixelClustersAlCaRecoProducer.tag = cms.InputTag("ctppsPixelDigis","","RECO")
process.ctppsDiamondRecHitsAlCaRecoProducer.digiTag = cms.InputTag("ctppsDiamondRawToDigi","TimingDiamond","RECO")

# processing sequences
process.path = cms.Path(
    process.recoPPSSequenceAlCaRecoProducer
)

process.end_path = cms.EndPath(
  process.output
)