# Auto generated configuration file
# using: 
# Revision: 1.381.2.17 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: exampleFilters -s RAW2DIGI,FILTER:hcallaserhbhehffilter2012_cff.hcallLaser2012Filter --data --python hcal_test_exampleFilters.py --no_exec --conditions auto:com10 --eventcontent RAW --datatier RAW -n 10 --no_exec --filein /store/data/Run2012B/SingleElectron/RAW-RECO/laserHCALskim_534p2_24Oct2012-v1/00000/541454BA-4E1F-E211-AC99-003048678B30.root


# example of usage of the filter ** HcalLaserEventFilter2012 **
# which is based on a list of laser-like events from 2012A,B,C :
#
#                    eventFileName = cms.string(os.getenv('CMSSW_BASE')+"/src/EventFilter/HcalRawToDigi/data/HCALLaser2012AllDatasets.txt.gz"),
#
# obtained from all datasets.
# Such list has been defined using another filter HcalLaserHBHEHFFilter2012 which implements the actual laser candidate event definition
# which is illustrated in this talk by Jeff Temple https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=169318
# PLUS the fix explained on slide 6 of G.F. https://indico.cern.ch/getFile.py/access?contribId=3&resId=0&materialId=slides&confId=169321

# ** HcalLaserEventFilter2012  ** is meant to be used by all analyzes which use prompt 2012 datasets 
# ** HcalLaserHBHEHFFilter2012 ** is also run in this configuration, for comparison

import FWCore.ParameterSet.Config as cms

process = cms.Process('exampleFilters')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),

    # FILTER and RECO were the names of the
    # process and subprocess of the first step of the skimming previous processing (534)                            
    inputCommands = cms.untracked.vstring('keep *',
                                          'drop *_*_*_FILTER',
                                          'drop *_*_*_RECO'
                                          ),

    # this file belong to the SKIM OF LASER_LIKE candidates => by construction filters return FALSE on all these events 
    # both HcalLaserHBHEHFFilter2012 and HcalLaserEventFilter2012 return TRUE / FALSE  for events which are    good-for-physics / laser-like 
    fileNames = cms.untracked.vstring('/store/data/Run2012A/Photon/RAW-RECO/20Nov2012-v2/00000/06538BB7-9B34-E211-93A1-003048678CA2.root')

)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('exampleFilters nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RAWoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWEventContent.outputCommands,
    fileName = cms.untracked.string('exampleFilters_RAW2DIGI_FILTER.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('RAW')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('filtering_step')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '')



#################################################
# this is the filter which uses the .txt.gz list of events to perform the rejection of laser-like candidates
process.load('EventFilter.HcalRawToDigi.hcallasereventfilter2012_cfi')
hcallasereventfilter2012=cms.EDFilter("HcalLaserEventFilter2012")
#hcallasereventfilter2012=cms.EDFilter("HcalLaserEventFilter2012",
#                                      # Specify laser events to remove in gziped file
#                                      eventFileName = cms.string("EventFilter/HcalRawToDigi/data/HCALLaser2012AllDatasets.txt.gz"),
#                                      # if verbose==true, run:ls:event for any event failing filter will be printed to cout
#                                      verbose   = cms.untracked.bool(False),
#                                      # Select a prefix to appear before run:ls:event when run info dumped to cout.  This makes searching for listed events a bit easier
#                                      prefix    = cms.untracked.string(""),
#                                      # If minrun or maxrun are >-1, then only a subsection of EventList corresponding to the given [minrun,maxrun] range are searched when looking to reject bad events.  This can speed up the code a bit when looking over a small section of data, since the bad EventList can be shortened considerably.  
#                                      minrun    = cms.untracked.int32(-1),
#                                      maxrun    = cms.untracked.int32(-1),
#                                      WriteBadToFile = cms.untracked.bool(False), # if set to 'True', then the list of events failing the filter cut will be written to a text file 'badHcalLaserList_eventfilter.txt'.  Events in the file will not have any prefix added, but will be a simple list of run:ls:event.
#                                      forceFilterTrue=cms.untracked.bool(False) # if specified, filter will always return 'True'.  You could use this along with the 'verbose' or 'WriteBadToFile' booleans in order to dump out bad event numbers without actually filtering them
#                                      )



#################################################
# this is the filter from which uses RAW in input
# it's this one which has been used to _determine_ the the .txt.gz list of events in the first place 
process.load('EventFilter.HcalRawToDigi.hcallaserhbhehffilter2012_cfi')
hcallaserhbhehffilter2012=cms.EDFilter("HcalLaserHBHEHFFilter2012")
#process.hcallaserhbhehffilter2012 = cms.EDFilter("HcalLaserHBHEHFFilter2012",
#    forceFilterTrue = cms.untracked.bool(False),
#    filterHF = cms.bool(True),
#    HBHEcalibThreshold = cms.double(15.0),
#    verbose = cms.untracked.bool(False),
#    minCalibChannelsHBHELaser = cms.int32(20),
#    minCalibChannelsHFLaser = cms.int32(10),
#    WriteBadToFile = cms.untracked.bool(False),
#    digiLabel = cms.InputTag("hcalDigis"),
#    prefix = cms.untracked.string(''),
#    filterHBHE = cms.bool(True),
#    CalibTS = cms.vint32(3, 4, 5, 6),
#    minFracDiffHBHELaser = cms.double(0.3)
#)







#################################################
# This is an example of my dummy analyzers
#process.myHelloWorldAnalyzerONE = cms.EDAnalyzer('HelloWorldAnalyzer',
#					     whatToSay   = cms.string('happyHappyONE')
#					     )
#
#process.myHelloWorldAnalyzerTWO = cms.EDAnalyzer('HelloWorldAnalyzer',
#					     whatToSay   = cms.string('happyHappyTWO')
#					     )




# this is the filter from which uses RAW in input
# filter returns TRUE if event is good for physics; FALSE if it's a laser-like candidate
process.hcallLaser2012Filter = cms.Sequence(process.hcallaserhbhehffilter2012)
process.NOThcallLaser2012Filter = cms.Sequence(~process.hcallaserhbhehffilter2012)

# this is the filter which uses the .txt.gz list of events to perform the rejection of laser-like candidates
# filter returns TRUE if event is good for physics; FALSE if it's a laser-like candidate
process.hcallEvent2012Filter = cms.Sequence(process.hcallasereventfilter2012)
process.NOThcallEvent2012Filter = cms.Sequence(~process.hcallasereventfilter2012)

# templates of your analysis code: whatever needs to be protected running after the HCAL laser filtering
# process.myAnalysisONE        = cms.Sequence(process.myHelloWorldAnalyzerONE) 
# process.myAnalysisTWO        = cms.Sequence(process.myHelloWorldAnalyzerTWO) 
# process.myAnalysisTHREE      = cms.Sequence(process.myHelloWorldAnalyzerTHREE) 
# process.myAnalysisFOUR       = cms.Sequence(process.myHelloWorldAnalyzerFOUR) 



#################################################
# Path and EndPath definitions
process.raw2digi_step      = cms.Path(process.RawToDigi)
# filtering_step determines the RAW output  
process.filtering_step     = cms.Path(process.hcallEvent2012Filter)
# the steps with your analysis, protected running after the HCAL laser filtering
process.analysis_stepONE   = cms.Path(process.hcallLaser2012Filter    #* process.myAnalysisONE
                                      )
process.analysis_stepTWO   = cms.Path(process.hcallEvent2012Filter    #* process.myAnalysisTWO
                                      )
process.analysis_stepTHREE = cms.Path(process.NOThcallLaser2012Filter #* process.myAnalysisTHREE
                                      )
process.analysis_stepFOUR  = cms.Path(process.NOThcallEvent2012Filter #* process.myAnalysisFOUR
                                      )
process.endjob_step        = cms.EndPath(process.endOfProcess)
process.RAWoutput_step     = cms.EndPath(process.RAWoutput)


# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,
				process.analysis_stepONE,
				process.analysis_stepTWO,
				process.analysis_stepTHREE,
				process.analysis_stepFOUR,
				process.filtering_step,
				process.endjob_step,
				process.RAWoutput_step)

## filter all path with the production filter sequence
#for path in process.paths:
#	getattr(process,path)._seq = process.hcallLaser2012Filter * getattr(process,path)._seq 

process.options   = cms.untracked.PSet(
	                    wantSummary = cms.untracked.bool(True),
			                        SkipEvent = cms.untracked.vstring('ProductNotFound')
			    )
