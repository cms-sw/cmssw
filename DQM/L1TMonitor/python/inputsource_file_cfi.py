#
# provide online L1 Trigger DQM input from file(s)
#
# V M Ghete 2010-07-09

import FWCore.ParameterSet.Config as cms

###################### user choices ######################

dataType = 'RAW'
#dataType = 'StreamFile'


# runNumber for RAW only 
runNumber = 143657
#runNumber = 137028

maxNumberEvents = 5000

###################### end user choices ###################


maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(maxNumberEvents)
)


# for RAW data, run first the RAWTODIGI 
if dataType == 'StreamFile' :
    readFiles = cms.untracked.vstring()
    source = cms.Source("NewEventStreamFileReader", fileNames=readFiles)
else :        
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)


if dataType == 'RAW' : 

    if runNumber == 137028: 
    
        readFiles.extend( [
            '/store/data/Run2010A/ZeroBias/RAW/v1/000/137/028/0C88B386-3971-DF11-A163-000423D99896.root' 
            ] );

        secFiles.extend([
            ])    
    
    elif runNumber == 143657 : 
    
        readFiles.extend( [
            '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/00FB1636-91AE-DF11-B177-001D09F248F8.root',
            '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/023EB128-51AE-DF11-96D3-001D09F24682.root'
            ] );

        secFiles.extend([
            ])    
   
elif dataType == 'StreamFile' : 

    readFiles.extend( [
        'file:/lookarea_SM/Data.00147754.0001.A.storageManager.00.0000.dat'       
        ] );

else :
    print 'Error: no such dataType:' + dataType + 'was specified'
