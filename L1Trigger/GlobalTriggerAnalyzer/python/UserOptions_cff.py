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
#maxNumberEvents = -1

# useRelValSample: choose the type of sample used:
#   True to use MC RelVal
#   False to use data

# default value
useRelValSample = False
#
# comment/uncomment the next line to choose sample type 
# (un-commented selects MC RelVal)
useRelValSample=True

if useRelValSample == False :
    
    # choose the global tag type  
    #  
    globalTag = 'auto:com10'        # GR_R_*
    #globalTag = 'auto:hltonline'   # GR_H_*
    
    # choose one sample identifier from the list of data samples 
    #
    sampleIdentifier = '191833_AOD'


else :

    # choose the global tag type  
    #  
    #globalTag = 'auto:mc'
    globalTag = 'auto:startup'
    #globalTag = 'auto:starthi'
    
    # choose (pre)release used to produce the RelVal samples
    #
    sampleFromRelease = 'CMSSW_5_2_3'

   # RelVals samples - add the "short name" of the dataset e.g. /RelValLM1_sfts/...
    #
    #dataset = 'RelValMinBias'
    #dataset = 'RelValTTbar'
    #dataset = 'RelValQCD_Pt_80_120'
    dataset = 'RelValLM1_sfts'
    
    # data type
    #
    #dataType = 'RAW'
    dataType = 'RECO'
        
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
    
    # end of data samples 
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
                
 
    elif sampleIdentifier == 'FileStream_105760' :
        runNumber = '105760'
        dataset = 'A_Stream'
        dataType = 'FileStream'
        useDAS = False
        readFiles.extend( [
                'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'       
                ] );
                
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
    # local file
    
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
    from Configuration.AlCa.autoCond import autoCond
    useGlobalTag = autoCond[globalTag.replace('auto:', '')]
else :
    useGlobalTag = globalTag+'::All'    
    
print "\n   Using global tag ", useGlobalTag, "\n"
        