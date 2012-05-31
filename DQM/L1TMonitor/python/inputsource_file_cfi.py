#
# provide online L1 Trigger DQM input from file(s)
#
# V M Ghete 2010-07-09

import FWCore.ParameterSet.Config as cms

###################### user choices ######################

dataType = 'RAW'
#dataType = 'StreamFile'


# runNumber for RAW only 
runNumber = '165633-CAFDQM'
#runNumber = 163661
#runNumber = 161312
#runNumber = 143657
#runNumber = 137028

maxNumberEvents = 1000

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
   
    elif runNumber == 161312 : 
    
        readFiles.extend( [
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/FEE65985-EF55-E011-A137-001617E30F50.root',
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/F4E4E71A-E755-E011-B7BA-001617E30CC8.root',
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/F442D1A6-D755-E011-A93F-003048F01E88.root'
            ] );

        secFiles.extend([
            ])    
    elif runNumber == 163661 : 
    
        readFiles.extend( [
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/163/661/2E455CA5-2272-E011-A3A8-003048F024F6.root'
            ] );

        secFiles.extend([
            ])    
   
    elif runNumber == '165633-CAFDQM' : 
    
        readFiles.extend( [
            'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root'
            ] );

        secFiles.extend([
            ])    
   
elif dataType == 'StreamFile' : 

    readFiles.extend( [
        'file:/lookarea_SM/Data.00147754.0001.A.storageManager.00.0000.dat'       
        ] );

else :
    print 'Error: no such dataType:' + dataType + 'was specified'
