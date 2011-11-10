import os
import sys

if len(sys.argv) != 2:
    print "Error: please, specify the configuration file"
    sys.exit()
else:
    configFile = sys.argv[1]

if os.path.isfile(configFile):
    print "Using configuration file:", configFile
else:
    print "Error: configuration file", configFile, "not found"
    sys.exit()

# Read the parameters
import HDQMDatabaseProducerConfiguration
config = HDQMDatabaseProducerConfiguration.HDQMDatabaseProducerConfiguration(configFile)
# print SourceDir

import DiscoverDQMFiles
discoverDQMFiles = DiscoverDQMFiles.DiscoverDQMFiles()

import PopulateDB
populateDB = PopulateDB.PopulateDB()

import ProducePlots
producePlots = ProducePlots.ProducePlots()


# If needed, discover all the RecoTypes and use them
if len(config.RecoTypes) == 0:
    fullList = discoverDQMFiles.filesList(config.SourceDir)
    keys = {}
    for file in fullList:
        # print file.split("__")[2]
        keys[file.split("__")[2]] = 1
    print "Found the following reco types:", keys.keys()
    config.RecoTypes = keys.keys()

# Save the list of recoTypes to file for the WebInterface
recoTypesFile = open('recoTypesFile.txt', 'w')

# import sys
# sys.exit()

for recoType in config.RecoTypes:
    # Create the list of DQM root files
    fullList = discoverDQMFiles.filesList(config.SourceDir, recoType)
    for subDetAndTag in config.SubDetsAndTags:
        FullTagName = config.TagName+"_"+subDetAndTag.SubDet+"_"+subDetAndTag.Tag
        # Take the list of already processed runs
        import DiscoverProcessedRuns
        discoverProcessedRuns = DiscoverProcessedRuns.DiscoverProcessedRuns()
    
        discoverProcessedRuns.Database =           config.Database
        discoverProcessedRuns.AuthenticationPath = config.AuthenticationPath
        discoverProcessedRuns.CMSSW_Version =      config.CMSSW_Version
        discoverProcessedRuns.TagName =            FullTagName
        
        processedRuns = discoverProcessedRuns.runsList()

        # Filter out the files of already processed runs
        filteredList = list(fullList)
        for run in processedRuns:
            for file in fullList:
                if( file.find(run) != -1 ):
                    # It should not give an exception unless a file is removed twice...
                    try:
                        filteredList.remove(file)
                    except Exception:
                        print "Error, trying to remove file:", file, "twice!"
                        pass

        # Sort with run number and remove the last RunsToSkip runs
        filteredList = sorted(filteredList, key=lambda run: run.split("R0")[1].split("__")[0])
        runsToSkip = int(config.RunsToSkip)
        if( runsToSkip > 0 ):
            for lastRun in range(runsToSkip):
                filteredList.pop()

        # Store the information in the database
        tempDir =  config.BaseDir+"/"+subDetAndTag.SubDet+"_"+config.RunType+"/"
        if not os.path.isdir(tempDir):
            os.mkdir(tempDir)

        for file in filteredList:
            runNumber = int(file.split("R0")[1].split("__")[0])

            populateDB.RunNumber = str(runNumber)
            populateDB.AuthenticationPath = config.AuthenticationPath
            populateDB.Database = config.Database
            populateDB.TagName = FullTagName
            populateDB.FileName = file
            populateDB.TemplatesDir = config.TemplatesDir
            populateDB.DetName = subDetAndTag.SubDet
            populateDB.Dir = config.BaseDir+"/"+subDetAndTag.SubDet+"_"+config.RunType+"/"
            populateDB.CMSSW_Version = config.CMSSW_Version

            populateDB.run()

        # Produce the plots if needed
        if len(filteredList) == 0:
            print "Creating plots"
            
            producePlots.Dir = config.BaseDir+"/"
            producePlots.TagName = FullTagName
            producePlots.DetName = subDetAndTag.SubDet
            producePlots.StorageDir = config.StorageDir
            producePlots.BaseDir = config.BaseDir
            producePlots.FirstRun = str(config.FirstRun)
            producePlots.CMSSW_Version = config.CMSSW_Version
            producePlots.Database = config.Database
            producePlots.Password = PASSWORD
            producePlots.RunType = config.RunType
            producePlots.Group = config.Group
            producePlots.QualityFlag = subDetAndTag.QualityFlag
            
            producePlots.makePlots()

    # All done, save the recoType to file
    recoTypesFile.write(recoType+"\n")

recoTypesFile.close()
