from __future__ import print_function
########################################################################################
###
###  Read and write APEs to and from database and ASCII files
###
###  The ASCII file contains one row per module, where the first column
###  lists the module id and the following 21 columns the diagonal and
###  lower elements x11,x21,x22,x31,x32,x33,0... of the 6x6 covariance
###  matrix, where the upper 3x3 sub-matrix contains the position APEs.
###  The elements are stored in units of cm^2.
###
########################################################################################



###### Steering parameters #############################################################

### specify the input APE
#
GT = "auto:phase1_2017_design"
#
# read the APE from database or from ASCII
# True  : from database
# False : from ASCII
readAPEFromDB = False

### specify APE input from database (only relevant if 'readAPEFromDB=True')
#
# specify run (to get proper IOV in IOV dependent databases)
# for data payload only, "1" for MC
readDBRun = 1
#
# False : APE from GT,
# True  : APE from es_prefer statement
readDBOverwriteGT = False
#
# info for es_prefer to overwrite APE info in GT
# (only relevant if 'readDBOverwriteGT=True')
readDBConnect = "frontier://FrontierProd/CMS_CONDITIONS"
readDBTag     = "TrackerAlignmentErrorsExtended_Upgrade2017_design_v0"

### specify APE input from ASCII (only relevant if 'readAPEFromDB=False')
#
# file name (relative to $CMSSW_BASE/src)
readASCIIFile = "Alignment/APEEstimation/macros/ape.txt"
#

### specify APE output to ASCII file
#
saveAPEtoASCII = True
saveASCIIFile = "apeDump.txt"

### specify APE output to database file
#
saveAPEtoDB = True
saveAPEFile = "APE_BPX-L1-Scenario_v0.db"
saveAPETag  = "APEs"



###### Main script #####################################################################

import FWCore.ParameterSet.Config as cms
process = cms.Process("APEtoASCIIDump")

# Load the conditions
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,GT)
print("Using Global Tag:", process.GlobalTag.globaltag._value)
if readAPEFromDB and readDBOverwriteGT:
    print("Overwriting APE payload with "+readDBTag)
    process.GlobalTag.toGet.append(
        cms.PSet(
            record = cms.string("TrackerAlignmentErrorExtendedRcd"),
            tag = cms.string(readDBTag)
            )
        )


### setup the alignmnet producer to read the APEs and dump them
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
from Alignment.CommonAlignmentAlgorithm.ApeSettingAlgorithm_cfi import ApeSettingAlgorithm
#
# general settings
process.AlignmentProducer.algoConfig = ApeSettingAlgorithm.clone()
process.AlignmentProducer.algoConfig.setComposites       = False
process.AlignmentProducer.algoConfig.saveComposites      = False
process.AlignmentProducer.algoConfig.readLocalNotGlobal  = False
process.AlignmentProducer.algoConfig.readFullLocalMatrix = True
process.AlignmentProducer.algoConfig.saveLocalNotGlobal  = False
#
# define how APEs are read: either from DB or from ASCII
process.AlignmentProducer.applyDbAlignment            = readAPEFromDB
process.AlignmentProducer.checkDbAlignmentValidity    = False # enable reading from tags with several IOVs
process.AlignmentProducer.algoConfig.readApeFromASCII = not readAPEFromDB
process.AlignmentProducer.algoConfig.apeASCIIReadFile = cms.FileInPath(readASCIIFile)
#
# define how APEs are written
process.AlignmentProducer.saveApeToDB                 = saveAPEtoDB
process.AlignmentProducer.algoConfig.saveApeToASCII   = saveAPEtoASCII
process.AlignmentProducer.algoConfig.apeASCIISaveFile = saveASCIIFile


### specify the output database file
from CondCore.CondDB.CondDB_cfi import CondDB
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondDB.clone(connect = cms.string("sqlite_file:"+saveAPEFile)),
    timetype = cms.untracked.string("runnumber"),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string("TrackerAlignmentErrorExtendedRcd"),
            tag = cms.string(saveAPETag)
            ),
        )
    )

     
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    files = cms.untracked.PSet(
        alignment = cms.untracked.PSet(
            Alignment = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            enableStatistics = cms.untracked.bool(True),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('INFO')
        )
    )
)


### speficy the source
# 
# process an empty source
process.source = cms.Source(
    "EmptySource",
    firstRun = cms.untracked.uint32(readDBRun)
    )
#
# need to run over 1 event
# NB: will print an "MSG-e" saying no events to process. This can be ignored.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

# We do not even need a path - producer is called anyway...
