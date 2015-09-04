#
# configuration file fragment containing user choices
# for release to run, sample and data type to use in
# a cfg file
#
 
#

import FWCore.ParameterSet.Config as cms
import sys

    

###################### user choices ######################

# number of events to be run (-1 for all)
maxNumberEvents = 10
maxNumberEvents = -1

# useRelValSample: choose the type of sample used:
#   True to use MC RelVal
#   False to use data

# default value
useRelValSample = False
#
# comment/uncomment the next line to choose sample type 
# (un-commented selects MC RelVal)
#useRelValSample=True

if useRelValSample == False :
    
    # choose the global tag type 
    # WARNING: global mess in global tag management, 
    #      must be chosen per sample
    #      must be checked in the release
    #  
    globalTag = '75X_dataRun2_HLT_v2'
    
    # choose one sample identifier from the list of data samples 
    #
    sampleIdentifier = '251162'
    #sampleIdentifier = '165633-CAFDQM'
    #sampleIdentifier = '191833_RAW'
    #sampleIdentifier = '205666.A.storageManager'

else :
    # choose the global tag type 
    # WARNING: global mess in global tag management, 
    #      must be chosen per sample
    #      must be checked in the release
    #  
    globalTag = 'auto:run1_mc'
    #globalTag = 'auto:MCRUN2_72_V3A'
    #globalTag = 'auto:run1_mc_hi'
    
    # choose (pre)release used to produce the RelVal samples
    sampleFromRelease = 'CMSSW_7_5_0'

    # RelVals samples - add the "short name" of the dataset e.g. /RelValLM1_sfts/...
    #
    #dataset = 'RelValMinBias'
    dataset = 'RelValTTbar'
    
    # data type
    #
    dataType = 'RAW'
    #dataType = 'RECO'
        
# change to True to use local files
#     the type of file must be matched by hand

useLocalFiles = False 
#useLocalFiles = True 

# override default global tag - expert choice, do it only if you know what you do
overrideGlobalTag = False
#overrideGlobalTag = True

if overrideGlobalTag == True :
    myGlobalTag = 'GR_P_V28'

###################### end user choices ###################

#
errorUserOptions = False

# initialize list of files, of secondary files, list of selected events and luminosity segments
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
selectedEvents = cms.untracked.VEventRange()
selectedLumis= cms.untracked.VLuminosityBlockRange()

# type of sample used (True for RelVal, False for data)

if (useRelValSample == True) and (useLocalFiles == False) :
    
    #            

    print "   Release:   ", sampleFromRelease
    print "   Dataset:   ", dataset
    print "   Data type: ", dataType
    print 

    useDAS = True
    
    if useDAS :
        import das_client
        import os
        
        # query DAS
        
        myQuery =  'dataset release=' + sampleFromRelease
        dasClientCommand = 'das_client.py --limit=0 --format=plain --query='+'"'+myQuery+'"'
        datasets = os.popen(dasClientCommand)
        
        
        # adjust strings for dataset and sampleFromRelease to search the dataset exact name 
        dataset = '/' + dataset + '/'
        sampleFromRelease = sampleFromRelease +'-'
        
        if 'start' in globalTag :
            gTag = 'START'
        elif 'mc' in globalTag :
            gTag = 'MC'
        else :
            gTag =''  
        
        datasetName = ''
        for line in datasets.readlines() :
            if dataset in line :
              if sampleFromRelease in line :
                  if gTag in line :
                      if dataType in line :
                          if 'ALCA' not in line :
                              datasetName = line.strip(' \n')
                              # print datasetName

        # print datasetName
        
        if datasetName == '' :
            print "\n   No dataset found."
            errorUserOptions = True 

        if not errorUserOptions :
            
            myQuery = 'file dataset=' + '"' + datasetName + '"'
            dasClientCommand = 'das_client.py --limit=0 --format=plain --query=' + '"' + myQuery + '"'
            data = os.popen(dasClientCommand)
            filePaths = data.readlines()

            
            print '\n   das_client using the query'
            print '      ', myQuery
            print '   retrieved the following files\n'
        
            for line in filePaths :
                print '      ', line
           
            readFiles.extend(filePaths);
        
        
            # nothing added to secondary files by DAS 
            secFiles.extend([
                ])
    
   
    

elif (useRelValSample == False) and (useLocalFiles == False) :

    # data
    
    if sampleIdentifier == '191833_RAW' :
        runNumber = '191833'
        dataset = '/DoubleElectron/Run2012A-v1/RAW'
        dataType = 'RAW'
        useDAS = True
        selectedLumis= cms.untracked.VLuminosityBlockRange(
                                                '191833:1'
                                                )
        selectedEvents = cms.untracked.VEventRange(
                                    '191833:256674',
                                    '191833:588211'
                                    )
        
    elif sampleIdentifier == '251162' :
        runNumber = '251162'
        dataset = '/Run2015B/DoubleMuon/RAW'
        dataType = 'RAW'
        useDAS = False
        readFiles.extend( [
                '/store/data/Run2015B/DoubleMuon/RAW/v1/000/251/162/00000/9A6A3CB4-AD25-E511-84E5-02163E01264D.root'       
                ] );
        

    elif sampleIdentifier == '191833_RECO' :
        runNumber = '191833'
        dataset = '/DoubleElectron/Run2012A-PromptReco-v1/RECO'
        dataType = 'RECO'
        useDAS = True
        selectedLumis= cms.untracked.VLuminosityBlockRange(
                                                '191833:1'
                                                )
        selectedEvents = cms.untracked.VEventRange(
                                    '191833:256674',
                                    '191833:588211'
                                    )

    elif sampleIdentifier == '191833_AOD' :
        runNumber = '191833'
        dataset = '/DoubleElectron/Run2012A-PromptReco-v1/AOD'
        dataType = 'RECO'
        useDAS = True
        selectedLumis= cms.untracked.VLuminosityBlockRange(
                                                '191833:1'
                                                )
        selectedEvents = cms.untracked.VEventRange(
                                    '191833:256674',
                                    '191833:588211'
                                    )

    # splash events 2012    
    elif sampleIdentifier == '187858' :
        runNumber = '187858'
        dataset = '/Commissioning/Commissioning12-v1/RAW'
        dataType = 'RAW'
        useDAS = True
        selectedLumis= cms.untracked.VLuminosityBlockRange(
                                                '187858:394', 
                                                '187858:499'
                                                )
        selectedEvents = cms.untracked.VEventRange(
                                    '187858:3244833',
                                    '187858:4094284'
                                    )
       
    # high PU run 2011   
    elif sampleIdentifier == '179828' :
        runNumber = '179828'
        dataset =  '/ZeroBiasHPF0/Run2011B-v1/RAW'
        dataType = 'RAW'
        useDAS = True
        
        
    elif sampleIdentifier == '165633-CAFDQM' :
        runNumber = '165633'
        dataset = '/ZeroBiasHPF0/Run2011B-v1/RAW'
        dataType = 'RAW'
        useDAS = False
        readFiles.extend( [ 
                'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root'                       
                ]);    
                                                                                                           
       
    elif sampleIdentifier == '137028' :
        runNumber = '137028'
        dataset = '/Run2010A/ZeroBias/RECO-v2'
        dataType = 'RECO'
        useDAS = False
        readFiles.extend( [
                '/store/data/Run2010A/ZeroBias/RECO/v2/000/137/028/08BF857D-2471-DF11-9CEB-003048D2BBF0.root'       
                ] );
                
 
    elif sampleIdentifier == 'StreamFile_105760' :
        runNumber = '105760'
        dataset = 'A_Stream'
        dataType = 'StreamFile'
        useDAS = False
        readFiles.extend( [
                'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'       
                ] );
                
    # run with L1 GT data - emulator disagreements for the selectedEvents               
    elif sampleIdentifier == '205666.A.storageManager' :
        runNumber = '205666'
        dataset = 'A.storageManager'
        dataType = 'StreamFile'
        useDAS = False
        
        selectedEvents = cms.untracked.VEventRange(
                                    '205666:101:125069341',
                                    '205666:161:238160322',
                                    '205666:281:443371795',
                                    '205666:341:537654884',
                                    '205666:361:567121052'
                                    )
        
        readFiles.extend( [ 
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0001.A.storageManager.00.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0021.A.storageManager.01.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0041.A.storageManager.02.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0061.A.storageManager.03.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0081.A.storageManager.04.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0101.A.storageManager.05.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0121.A.storageManager.06.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0141.A.storageManager.07.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0161.A.storageManager.08.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0181.A.storageManager.09.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0201.A.storageManager.10.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0221.A.storageManager.11.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0241.A.storageManager.12.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0261.A.storageManager.13.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0281.A.storageManager.14.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0301.A.storageManager.15.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0321.A.storageManager.00.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0341.A.storageManager.01.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0361.A.storageManager.02.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0381.A.storageManager.03.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0401.A.storageManager.04.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0421.A.storageManager.05.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0441.A.storageManager.06.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0461.A.storageManager.07.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0481.A.storageManager.08.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0501.A.storageManager.09.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0521.A.storageManager.10.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0541.A.storageManager.11.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0561.A.storageManager.12.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0581.A.storageManager.13.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0601.A.storageManager.14.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0621.A.storageManager.15.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0641.A.storageManager.00.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0661.A.storageManager.01.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0681.A.storageManager.02.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0701.A.storageManager.03.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0721.A.storageManager.04.0000.dat'                     
           ]);    
                
                
    else :
        print 'Error: sample identifier ', sampleIdentifier, ' not defined.\n'
        errorUserOptions = True 
        runNumber = '0'
        dataset = 'None'
        dataType = 'None'
        useDAS = False
       
     
    #            
    # end of data samples 
    #            

    print "   Run number: ", runNumber
    print "   Dataset: ", dataset
    print "   Data type: ", dataType

    if useDAS :
        import das_client
        import os

        # query DAS
        myQuery =  'file dataset=' + dataset + ' run=' + runNumber
        dasClientCommand = 'das_client.py --limit=0 --format=plain --query='+'"'+myQuery+'"'
        data = os.popen(dasClientCommand)
        filePaths = data.readlines()
            
       
        print '\n   das_client using the query'
        print '      ', myQuery
        print '   retrieved the following files\n'
        
        for line in filePaths :
            print '      ', line
           
        readFiles.extend(filePaths);
        
        
        # nothing added to secondary files by DAS 
        secFiles.extend([
                ])

        
            
else :
    # local file(s)
    

    if sampleIdentifier == 'dummy' :
        dataType = 'RAW'
        readFiles.extend( [                        
            'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/LocalFile_source.root'
            ]);                                                                                               

        secFiles.extend([
            ])

    print 'Local file(s)', readFiles

if overrideGlobalTag == True :
    globalTag = myGlobalTag
  
if globalTag.count('auto') :
    from Configuration.AlCa.autoCond_condDBv2 import autoCond
    useGlobalTag = autoCond[globalTag.replace('auto:', '')]
else :
    useGlobalTag = globalTag   
    
print "\n   Using global tag ", useGlobalTag, "\n"
    
        