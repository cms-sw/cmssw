#   **********************************************************************************************************************************************************
#
#   Author(s):      Stilianos Kesisoglou - Institute of Nuclear Physics NCSR Demokritos (current author)
#                   Nikolaos Rompotis    - Imperial College London                      (original author) 
#                 
#   Contact:        Stilianos.Kesisoglou@cern.ch
# 
#   Description:    <one line class summary>
# 
#   Implementation:
# 
#   Changes Log:
# 
#   09 Dec 2009     Option to ignore trigger
#  
#   25 Feb 2010     Added options to use Conversion Rejection, Expected missing hits and valid hit at first PXB
#  
#                   Added option to calculate these criteria and store them in the pat electron object this is done by setting in the configuration the flags
#                 
#                       calculateValidFirstPXBHit = true
#                       calculateConversionRejection = true
#                       calculateExpectedMissinghits = true
#                     
#   28 May 2010     Implementation of Spring10 selections
# 
#   25 Jun 2010     Author change (Nikolaos Rompotis -> Stilianos Kesisoglou)
#                   Preparation of the code for the ICHEP 2010 presentation of the ElectroWeak results.
#                 
#   04 Nov 2010     Preparation of the code for the Moriond 2011 presentation of the ElectroWeak results.
#                   Code modification to apply common or separate electron criteria.
#                 
#   23 Jan 2011     Modify the code to auto-create input/output filenames, and make inventory.
#  
#   30 Jan 2011     Modify the trigger part to make it easier to add/remove/disable specific Trigger Paths.
#
#   01 Feb 2011     Added switch to load ROOT files to process either from internal or externaly provided list.
#                   If switch is activated check that the external file exists and is non-empty.
#
#                   Added switch to control usage or not of a JSON file.
#                   If switch is activated check that the JSON file exists.
#
#   08 Feb 2011     Modifications to allow for a "vector-like" treatment of all triggers
#
#   13 Feb 2011     Script modification to allow for easier trigger definition.
#                   Multiple changes to synchronize the script with changes in "ZeeCandidateFilter" and "ZeePlots"
#
#   **********************************************************************************************************************************************************



#   INITIALIZATIONS
#   ---------------

#   Define the name of the Python script itself
#   Define the name of the list file (list of input ROOT files - Default is <Python script name>.list)
#
import sys
import os.path

for inArg in sys.argv:
    if inArg.find('.py') == -1:
        continue
    else:
        pyBaseName=inArg.replace('.py','')

pyFile = "%s.py" % (pyBaseName)

#   For CRAB/GRID submition some options have to be disabled (comparing to local run)
#   Setting the flag to "False" it sets-up the script for CRAB/GRID run
#
isLocalRun = True

#   Externaly provided input-files list to be used and activation toggle.
#   Check that the externaly provided input-files list exists and is non-empty (if activated)
#   (empty lists evaluate to FALSE in Boolean contexts [such as if some_list:])
#
uselstFile = True ; lstFile = "%s.list" % (pyBaseName)

#   DO NOT EDIT THIS LINE
uselstFile = uselstFile and isLocalRun

if uselstFile:
    if not os.path.exists(lstFile):
        sys.exit("Input Filelist does not exist")
    else:
        import FWCore.Utilities.FileUtils as FileUtils
        filelist = FileUtils.loadListFromFile(lstFile) 
        if not filelist:
            sys.exit("Input Filelist is empty")


#   ROOT output files made by this Python script (Default is <Python script name><some user string>.root)
#
outFiles = []
outFiles.append("%s_histos.root" % (pyBaseName))
outFiles.append("%s_ZEE_VBTFselection.root" % (pyBaseName))
outFiles.append("%s_ZEE_VBTFpreselection.root" % (pyBaseName))


#   Auxiliary files produced by cmsRun.
#
xmlFile = "%s.xml" % (pyBaseName)           #   XML file produced from cmsRun   - Default is <Python script name>.xml
scrFile = "%s.scr" % (pyBaseName)           #   File containing screen messages - Default is <Python script name>.scr

auxFiles = []
auxFiles.append(xmlFile)
auxFiles.append(scrFile)


#   Inventory list (used from the LxBatch script to auto-return results via SCP)
#
invFile = "%s.inv" % (pyBaseName)           #   Inventory file - Default is <Python script name>.inv

invList = []
invList.extend(outFiles)
invList.extend(auxFiles)

#   Create the inventory file only if running locally and not on GRID
#
if isLocalRun:
    invHandle = open (invFile,'w')

    for item in invList:
        invHandle.write(item)
        invHandle.write('\n')

    invHandle.close()



#   CMS RELATED CODE STARTS HERE
#   ----------------------------

import FWCore.ParameterSet.Config as cms

process = cms.Process("ZEEPLOTS")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

#process.MessageLogger = cms.Service(
#        "MessageLogger",
#            categories = cms.untracked.vstring('info', 'debug','cout')
#            )


process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")

process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000000),
    limit = cms.untracked.int32(10000000)
       )
       
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
      reportEvery = cms.untracked.int32(100000),
         limit = cms.untracked.int32(10000000)
      )
      
process.options = cms.untracked.PSet(
       wantSummary = cms.untracked.bool(True)
       )


#   source - External or Local loading of input files
#
if uselstFile:
    process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(*filelist))
else:
    process.source = cms.Source('PoolSource',
        fileNames = cms.untracked.vstring(
        #   Comma terminated list of input files. Comment out to disable files selectively.
            'rfio:/tmp/ikesisog/TestFiles/00000000-0000-0000-0000-000000000000.root',
            'rfio:/tmp/ikesisog/TestFiles/00000000-0000-0000-0000-000000000000.root',
            'rfio:/tmp/ikesisog/TestFiles/00000000-0000-0000-0000-000000000000.root',
        )
    )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


#   Global Tag:
#
process.GlobalTag.globaltag = cms.string('GR_R_39X_V5::All')

process.load("Configuration.StandardSequences.MagneticField_cff")


#   Z Selection that you prefer
#
from ElectroWeakAnalysis.ZEE.simpleCutBasedSpring10SelectionBlocks_cfi import *

selection_inverse = cms.PSet (
    deta_EB_inv = cms.untracked.bool(True),
    deta_EE_inv = cms.untracked.bool(True)
    )


rules_Plotter = cms.PSet (
    #   treat or not the elecrons with same criteria. if yes then rules_Plotter_Elec1/2 are neglected
    useSameSelectionOnBothElectrons = cms.untracked.bool(True),    
    # some extra information on the ntuple production:
    sihih0_EB = cms.untracked.double(1000.0),
    dphi0_EB  = cms.untracked.double(1000.0),
    deta0_EB  = cms.untracked.double(1000.0),
    hoe0_EB   = cms.untracked.double(1000.0),
    sihih0_EE = cms.untracked.double(1000.0),
    dphi0_EE  = cms.untracked.double(1000.0),
    deta0_EE  = cms.untracked.double(1000.0),
    hoe0_EE   = cms.untracked.double(1000.0),
    includeJetInformationInNtuples = cms.untracked.bool(True),
    caloJetCollectionTag = cms.untracked.InputTag('ak5CaloJetsL2L3'),
    pfJetCollectionTag = cms.untracked.InputTag('ak5PFJetsL2L3'),
    DRJetFromElectron = cms.untracked.double(0.3),
    #
    PrimaryVerticesCollection = cms.untracked.InputTag("offlinePrimaryVertices"),
    PrimaryVerticesCollectionBS = cms.untracked.InputTag("offlinePrimaryVerticesWithBS"),
    #
    zeeCollectionTag = cms.untracked.InputTag("zeeFilter","selectedZeeCandidates","PAT"),
    vZeeOutputFileNames = cms.untracked.vstring(*outFiles),
    DatasetTag =  cms.untracked.int32(100),
    storeExtraInformation = cms.untracked.bool(True)
)

rules_Plotter_Elec0 = cms.PSet (
    # The selection to be used here:
    usePrecalcID0                   = cms.untracked.bool(True),
    usePrecalcIDType0               = cms.untracked.string('simpleEleId80relIso'),
    usePrecalcIDSign0               = cms.untracked.string('='),
    usePrecalcIDValue0              = cms.untracked.double(7),    
    ## preselection criteria are independent of useSameSelectionOnBothElectrons
    #  set them to False if you don't want them
    useValidFirstPXBHit0            = cms.untracked.bool(False),
    useConversionRejection0         = cms.untracked.bool(True),
    useExpectedMissingHits0         = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits0 = cms.untracked.int32(0),    
    ##
    calculateValidFirstPXBHit0      = cms.untracked.bool(False),
    calculateExpectedMissingHits0   = cms.untracked.bool(False)
)

rules_Plotter_Elec1 = cms.PSet (
    # The selection to be used here:
    usePrecalcID1                   = cms.untracked.bool(True),
    usePrecalcIDType1               = cms.untracked.string('simpleEleId80relIso'),
    usePrecalcIDSign1               = cms.untracked.string('='),
    usePrecalcIDValue1              = cms.untracked.double(7),    
    ## preselection criteria are independent of useSameSelectionOnBothElectrons
    #  set them to False if you don't want them
    useValidFirstPXBHit1            = cms.untracked.bool(False),
    useConversionRejection1         = cms.untracked.bool(True),
    useExpectedMissingHits1         = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits1 = cms.untracked.int32(0),    
    ##
    calculateValidFirstPXBHit1      = cms.untracked.bool(False),
    calculateExpectedMissingHits1   = cms.untracked.bool(False)
)

rules_Plotter_Elec2 = cms.PSet (
    # The selection to be used here:
    usePrecalcID2                   = cms.untracked.bool(True),
    usePrecalcIDType2               = cms.untracked.string('simpleEleId80relIso'),
    usePrecalcIDSign2               = cms.untracked.string('='),
    usePrecalcIDValue2              = cms.untracked.double(7),    
    ## preselection criteria are independent of useSameSelectionOnBothElectrons
    #  set them to False if you don't want them
    useValidFirstPXBHit2            = cms.untracked.bool(False),
    useConversionRejection2         = cms.untracked.bool(True),
    useExpectedMissingHits2         = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits2 = cms.untracked.int32(0),    
    ##
    calculateValidFirstPXBHit2      = cms.untracked.bool(False),
    calculateExpectedMissingHits2   = cms.untracked.bool(False)
)


# Storage of Jet information is nneded (hence we have to produce the jets)
#
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")

process.jetSequence = cms.Sequence( process.ak5CaloJetsL2L3  )

process.pfjetAK5Sequence = cms.Sequence( process.ak5PFJetsL2L3 )

process.ourJetSequence = cms.Sequence( process.jetSequence * process.pfjetAK5Sequence )


process.plotter = cms.EDAnalyzer('ZeePlots',
    rules_Plotter,
    rules_Plotter_Elec0,
    rules_Plotter_Elec1,
    rules_Plotter_Elec2   
)


# For DATA, 397 electrons are fixed (misalignment corrections included)
#
process.p = cms.Path( process.ourJetSequence*process.plotter)

