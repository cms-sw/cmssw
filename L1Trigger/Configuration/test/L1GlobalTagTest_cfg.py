# cfg file to print the L1 content of a global tag (with IoV infinity)

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('L1GlobalTagTest')

###################### user choices ######################

# choose the global tag type
# 310 ->

#useGlobalTag = 'MC_31X_V1'
#useGlobalTag = 'STARTUP31X_V1'

useGlobalTag = 'GR09_31X_V1P'
#useGlobalTag = 'GR09_31X_V1H'

# include printing for subsystems
printL1Rct = False
printL1Gct = False

printL1DtTPG = False
printL1DtTF = False

printL1CscTF= False

printL1Rpc = False

printL1Gmt = False

printL1Gt = True

# infinite IoV (use empty source) or run over given event sample (POOL source - default)
#cmsSource = 'EmptySource'
cmsSource = 'PoolSource'


###################### end user choices ###################

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

if cmsSource == 'EmptySource' :
    process.source = cms.Source('EmptySource')
else :
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)
    
    dataset = cms.untracked.vstring('/Cosmics/Commissioning09-v3/RAW')
    readFiles.extend( [
            '/store/data/Commissioning09/Cosmics/RAW/v3/000/105/847/6A699BB9-2072-DE11-995B-001D09F34488.root'
        
        ] );

   

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = useGlobalTag + '::All'

# 
process.load('L1Trigger.Configuration.L1GlobalTagTest_cff')

# Path definitions
process.printGTagL1Rct = cms.Path(process.printGlobalTagL1Rct)
process.printGTagL1Gct = cms.Path(process.printGlobalTagL1Gct)

#process.printGTagL1DtTPG = cms.Path(process.printGlobalTagL1DtTPG)
process.printGTagL1DtTF = cms.Path(process.printGlobalTagL1DtTF)

#process.printGTagL1CscTF = cms.Path(process.printGlobalTagL1CscTF)

process.printGTagL1Rpc = cms.Path(process.printGlobalTagL1Rpc)

process.printGTagL1Gmt = cms.Path(process.printGlobalTagL1Gmt)
process.printGTagL1Gt = cms.Path(process.printGlobalTagL1Gt)

# Schedule definition
process.schedule = cms.Schedule()

print ''

if printL1Rct == True :
    process.schedule.extend([process.printGTagL1Rct])
    print "Printing L1 RCT content of global tag ", useGlobalTag
else :
    print "L1 RCT content of global tag ", useGlobalTag, " not requested to be printed"

#
if printL1Gct == True :
    process.schedule.extend([process.printGTagL1Gct])
    print "Printing L1 GCT content of global tag ", useGlobalTag
else :
    print "L1 GCT content of global tag ", useGlobalTag, " not requested to be printed"

#

if printL1DtTPG == True :
    #process.schedule.extend([process.printGTagL1DtTPG])
    print "Printing L1 DtTPG content of global tag ", useGlobalTag, ": MISSING"
else :
    print "L1 DtTPG content of global tag ", useGlobalTag, " not requested to be printed"


if printL1DtTF == True :
    process.schedule.extend([process.printGTagL1DtTF])
    print "Printing L1 DtTF content of global tag ", useGlobalTag
else :
    print "L1 DtTF content of global tag ", useGlobalTag, " not requested to be printed"

#

if printL1CscTF == True :
    #process.schedule.extend([process.printGTagL1CscTF])
    print "Printing L1 CscTF content of global tag ", useGlobalTag, ": MISSING"
else :
    print "L1 CscTF content of global tag ", useGlobalTag, " not requested to be printed"

#

if printL1Rpc == True :
    process.schedule.extend([process.printGTagL1Rpc])
    print "Printing L1 RPC content of global tag ", useGlobalTag
else :
    print "L1 RPC content of global tag ", useGlobalTag, " not requested to be printed"

#

if printL1Gmt == True :
    process.schedule.extend([process.printGTagL1Gmt])
    print "Printing L1 GMT content of global tag ", useGlobalTag
else :
    print "L1 GMT content of global tag ", useGlobalTag, " not requested to be printed"

#

if printL1Gt == True :
    process.schedule.extend([process.printGTagL1Gt])
    print "Printing L1 GT content of global tag ", useGlobalTag
else :
    print "L1 GT content of global tag ", useGlobalTag, " not requested to be printed"
 
print ''


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['*']

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) )

