import FWCore.ParameterSet.Config as cms

MAXEVENTS=25000


import os

# myfiles is list of input files
myfiles=[]
# Test -- put all .root files in the given directory dir into myfiles list
dir="/store/data/Run2012C/JetHT/RAW/v1/000/199/834/"
for i in os.listdir(dir):
    if i.endswith(".root"):
        full=os.path.join(dir,i)
        myfiles.append(full)



process = cms.Process("HcalLaser")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
#process.MessageLogger.cerr.FwkReport.reportEvery = 1

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')


process.GlobalTag.globaltag = "GR_R_52_V9D::All"


from Configuration.AlCa.autoCond import autoCond


# Set maximum number of events to MAXEVENTS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(MAXEVENTS) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(myfiles
    )
)


# Load Hcal RAW-TO-DIGI producer
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.hcalDigis.InputLabel="rawDataCollector"
process.digipath=cms.Path(process.hcalDigis)


# Load Hcal Laser Filters
process.load("EventFilter.HcalRawToDigi.hcallasereventfilter2012_cfi")
process.load("EventFilter.HcalRawToDigi.hcallaserhbhefilter2012_cfi")
process.load("EventFilter.HcalRawToDigi.hcallaserhffilter2012_cfi")

# -------------------------------------------------------------------------- 
This is an example of how to read in a txt (gzipped) file of bad events, and put in in the hcallasereventfilter2012 'EventList' vector.  Input (gzipped) files should have the structure run:LS:event

import string
import os
import gzip

#--- taking events list from the release area
#file = gzip.GzipFile(os.getenv('CMSSW_RELEASE_BASE')+"/src/EventFilter/HcalRawToDigi/data/AllBadHCALLaser.txt.gz")

#--- alternatively - taking events list from local area
file = gzip.GzipFile(os.getenv('CMSSW_BASE')+"/src/EventFilter/HcalRawToDigi/data/AllBadHCALLaser.txt.gz")

mylist=file.readlines()  # read all lines in the inputfile
for j in mylist:
        process.hcallasereventfilter2012.EventList.append(string.strip(j)) 
print "A total of %i events read"%len(process.hcallasereventfilter2012.EventList)



### Filter definitions:
# process.hcallasereventfilter2012 -- vetoes based on a user-provided list of bad events
# process.hcallaserhbhefilter2012 -- vetoes HBHE laser events (based on calib and HBHE digis)
# process.hcallaserhffilter2012 -- vetoes HF laser events (based on calib digis)

# Setting a filter's 'verbose' variable to True will cause the event:LS:run of any event failing the filter to be printed
# Setting a filter's 'prefix' variable will cause the specified prefix to be printed prior to the run:LS:event info.  (This is useful if you want to know which filter produced the failure.)
process.hcallasereventfilter2012.verbose=True
process.hcallasereventfilter2012.prefix="EVT__"
process.hcallaserhbhefilter2012.verbose=True
process.hcallaserhbhefilter2012.prefix="HBHE_"
process.hcallaserhffilter2012.verbose=True
process.hcallaserhffilter2012.prefix="HF___"

process.hcallasereventfilter2012.WriteBadToFile=True
process.hcallaserhbhefilter2012.WriteBadToFile=True
process.hcallaserhffilter2012.WriteBadToFile=True


# In general, you only need to run either the event filter, or the (hbhe*hf) filters.
process.HcalLaser = cms.Path(
#    process.hcallasereventfilter2012*
    process.hcallaserhbhefilter2012*
    process.hcallaserhffilter2012
    )



process.schedule=cms.Schedule(process.digipath,  # make digis from raw
                              process.HcalLaser
                              )
