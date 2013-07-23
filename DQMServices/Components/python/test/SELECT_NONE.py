# Auto generated configuration file
# using: 
# Revision: 1.372.2.1 
# Source: /local/reps/CMSSW.admin/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: SELECT -s NONE --filein=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root --fileout=Run1Run2.root --no_exec --conditions auto:com10 --data --eventcontent=RAW

# edmFileUtil -e  /store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/A8FCA637-8E69-E211-9B26-002590596468.root | egrep -e "^[[:space:]]+([1-9]+[[:space:]]+){3}" | grep 196452 | head -n 100 | gawk '{print $1 " " $2 " " $3}' | sort --key=1 -n | gawk '{print "\x27" $1 "\x27, \x27" $2 "\x27, \x27" $3 "\x27"}'

import FWCore.ParameterSet.Config as cms

process = cms.Process('NONE')

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
#myFirstRunLSEvt = ['172791', '1015', '1414286969']
#myLastRunLSEvt  = ['172791', '1226', '1693205411']
#myFirstRunLSEvt = ['172791',             '67',       '35621558']
#myLastRunLSEvt = ['172791',             '81',       '58985871']
#myFirstRunLSEvt = ['172819',             '58',       '30852858']
#myLastRunLSEvt = ['172819',             '60',       '34303685']
#myFirstRunLSEvt = ['173241', '44', '62788265']
#myLastRunLSEvt = ['173241', '48', '67794693']

#myFirstRunLSEvt = ['196452', '829', '1114173926', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/A8FCA637-8E69-E211-9B26-002590596468.root']
#myLastRunLSEvt  = ['196452', '829', '1114287699', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/A8FCA637-8E69-E211-9B26-002590596468.root']

#myFirstRunLSEvt  = ['196452', '829', '1113629261', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/A8FCA637-8E69-E211-9B26-002590596468.root']
#myLastRunLSEvt   = ['196452', '829', '1113689764', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/A8FCA637-8E69-E211-9B26-002590596468.root']

#myFirstRunLSEvt = ['196452', '174', '196826855', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/A8FCA637-8E69-E211-9B26-002590596468.root']
#myLastRunLSEvt = ['196452', '174', '197239827', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/A8FCA637-8E69-E211-9B26-002590596468.root']

#myFirstRunLSEvt = ['196453', '814', '692514946', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/002711B7-D169-E211-BF5D-0026189438EA.root']
#myLastRunLSEvt  = ['196453', '814', '692526334', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/002711B7-D169-E211-BF5D-0026189438EA.root']

myFirstRunLSEvt = ['196437', '165', '155564199', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/6483FF2F-9B69-E211-A075-0025905964A6.root']
myLastRunLSEvt  = ['196437', '165', '156191367', '/store/data/Run2012B/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/20000/6483FF2F-9B69-E211-A075-0025905964A6.root']

process.source = cms.Source("PoolSource",
                            firstRun = cms.untracked.uint32(int(myFirstRunLSEvt[0])),
                            eventsToProcess = cms.untracked.VEventRange(
    ':'.join(myFirstRunLSEvt[0:3])+'-'+':'.join(myLastRunLSEvt[0:3])
    ),
                            secondaryFileNames = cms.untracked.vstring(),
                            fileNames = cms.untracked.vstring('%s' % myFirstRunLSEvt[3])
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('SELECT nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RAWoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWEventContent.outputCommands,
    fileName = cms.untracked.string('skim%s-%s-%s_%s-%s-%s.root' % tuple(myFirstRunLSEvt[0:3]+myLastRunLSEvt[0:3])),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'GR_R_52_V4::All'

# Path and EndPath definitions
process.RAWoutput_step = cms.EndPath(process.RAWoutput)

# Schedule definition
process.schedule = cms.Schedule(process.RAWoutput_step)

