#
# provide online L1 Trigger DQM input from file(s)
#
# V M Ghete 2010-07-09

import FWCore.ParameterSet.Config as cms

# choose the number of events and the type of the file

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

dataType = 'RAW'
#dataType = 'StreamFile'

# source according to data type
if dataType == 'StreamFile' :
    readFiles = cms.untracked.vstring()
    source = cms.Source("NewEventStreamFileReader", fileNames=readFiles)
    
    readFiles.extend( [
        'file:/lookarea_SM/Data.00139796.0001.A.storageManager.00.0000.dat'
        ] );

else :        
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)


    runNumber=137028
    readFiles.extend( [
        '/store/data/Run2010A/ZeroBias/RAW/v1/000/137/028/0C88B386-3971-DF11-A163-000423D99896.root' 
        ] );
