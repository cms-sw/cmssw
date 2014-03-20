# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: test_11_a_1 -s RAW2DIGI,RECO,DQM -n 100 --eventcontent DQM --conditions auto:com10 --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_11_a_1.py
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process('RECO')

options = VarParsing.VarParsing ('analysis')

options.register('runNumber',
                 100, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,          # string, int, or float
                 "Run Number")

options.register('inputDir',
                 '/afs/cern.ch/user/b/borrell/public/OnlineDQM/', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,          # string, int, or float
                 "Directory where input file for DQM online application will appear")

options.parseArguments()

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200),
)


# Input source
process.source = cms.Source("NewDQMStreamerFileReader",
                            runNumber = cms.untracked.uint32(options.runNumber),
                            dqmInputDir = cms.untracked.string(options.inputDir)
                            )

#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring(
#    'file:/tmp/borrell/run000100_ls0001_streamA_fullformat.dat'
##    'file:/tmp/borrell/Data.00190389.0001.A.storageManager.00.0000.dat'
#    )
#)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('test_11_a_1 nevts:100'),
    name = cms.untracked.string('Applications')
)

process.MessageLogger = cms.Service("MessageLogger",
#    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )),
#    cout = cms.untracked.PSet(threshold = cms.untracked.string( "ERROR" )),
#    destinations = cms.untracked.vstring( 'cout' )
    threshold = cms.untracked.string('ERROR'),              
    destinations = cms.untracked.vstring( 'info.txt' )
    )

process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )

# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('test_11_a_1_RAW2DIGI_RECO_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
#process.dqmoffline_step = cms.Path(process.DQMOffline)
process.dqmoffline_step = cms.Path(process.dqmDcsInfo)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.dqmoffline_step,process.endjob_step,process.DQMoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from DQMServices.Components.test.customDQM
#from DQMServices.Components.test.customDQM import customise 

#call to customisation function customise imported from DQMServices.Components.test.customDQM
#process = customise(process)

# End of customisation functions

