# idea stolen from:
# http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/
#        PhysicsTools/PatAlgos/python/tools/cmsswVersionTools.py
import das_client
import json
import os
import bisect
from FWCore.PythonUtilities.LumiList import LumiList
from TkAlExceptions import AllInOneError


class Dataset:
    def __init__( self, datasetName, dasLimit = 0 ):
        self.__name = datasetName
        self.__dasLimit = dasLimit
        self.__fileList = None
        self.__dataType = self.__getDataType()
        self.__fileInfoList = None
        self.__runList = None

    def __getData( self, dasQuery, dasLimit = 0 ):
        dasData = das_client.get_data( 'https://cmsweb.cern.ch',
                                       dasQuery, 0, dasLimit, False )
        jsondict = json.loads( dasData )
        # Check, if the DAS query fails
        if jsondict["status"] != 'ok':
            print "Status not 'ok', but:", jsondict["status"]
            raise StandardError
        return jsondict["data"]

    def __getDataType( self ):
        dasQuery_type = ( 'dataset dataset=%s | grep dataset.datatype,'
                          'dataset.name'%( self.__name ) )
        data = self.__getData( dasQuery_type )
        return data[0]["dataset"][0]["datatype"]

    def __getFileInfoList( self, dasLimit ):
        if self.__fileInfoList:
            return self.__fileInfoList

        dasQuery_files = ( 'file dataset=%s | grep file.name, file.nevents, '
                           'file.creation_time, '
                           'file.modification_time'%( self.__name ) )
        data = self.__getData( dasQuery_files, dasLimit )
        data = [ entry["file"] for entry in data ]
        fileInformationList = []
        for file in data:
            fileName = file[0]["name"]
            fileCreationTime = file[2]["creation_time"]
            for ii in range(3):
                try:
                    fileNEvents = file[ii]["nevents"]
                except KeyError:
                    continue
                break
            # select only non-empty files
            if fileNEvents == 0:
                continue
            fileDict = { "name": fileName,
                         "creation_time": fileCreationTime,
                         "nevents": fileNEvents
                         }
            fileInformationList.append( fileDict )
        fileInformationList.sort( key=lambda info: info["name"] )
        return fileInformationList

    def __getRunList( self ):
        if self.__runList:
            return self.__runList
        dasQuery_runs = ( 'run dataset=%s | grep run.run_number,'
                          'run.creation_time'%( self.__name ) )
        data = self.__getData( dasQuery_runs )
        data = [ entry["run"][0] for entry in data ]
        data.sort( key = lambda run: run["creation_time"] )
        self.__runList = data
        return data

    def dataType( self ):
        return self.__dataType

    def fileList( self ):
        if self.__fileList:
            return self.__fileList
        fileList = [ fileInfo["name"] \
                     for fileInfo in self.fileInfoList() ]
        self.__fileList = fileList  # store list of files for 'dump_cff' method
        return fileList

    __cff_template= ("import FWCore.ParameterSet.Config as cms\n"
                     "import FWCore.PythonUtilities.LumiList as LumiList\n\n"
                     "%(goodLumiSecStr)s"
                     "maxEvents = cms.untracked.PSet( "
                     "input = cms.untracked.int32(-1) )\n"
                     "readFiles = cms.untracked.vstring()\n"
                     "secFiles = cms.untracked.vstring()\n"
                     "source = cms.Source(\"PoolSource\",\n%(lumiStr)s"
                     "                    secondaryFileNames = secFiles,\n"
                     "                    fileNames = readFiles\n"
                     ")\n"
                     "%(files)s\n"
                     "%(lumiSecExtend)s\n")

    def __chunks( self, theList, n ):
        """ Yield successive n-sized chunks from theList.
        """
        for i in xrange( 0, len( theList ), n ):
            yield theList[i:i+n]

    def dump_cff( self, outName = None, jsonPath = None, begin = None,
                  end = None, firstRun = None, lastRun = None ):
        if outName == None:
            outName = "Dataset"
        packageName = os.path.join( "Alignment", "OfflineValidation" )
        if not os.path.exists( os.path.join(
            os.environ["CMSSW_BASE"], "src", packageName ) ):
            msg = ("You try to store the predefined dataset'%s'.\n"
                   "For that you need to check out the package '%s' to your "
                   "private relase area in\n"%( outName, packageName )
                   + os.environ["CMSSW_BASE"] )
            raise AllInOneError( msg )
        theMap = { "process": "",
                   "tab": "",
                   "nEvents": str( -1 ),
                   "importCms": "import FWCore.ParameterSet.Config as cms\n" }
        dataset_cff = self.__createSnippet( jsonPath = jsonPath,
                                            begin = begin,
                                            end = end,
                                            firstRun = firstRun,
                                            lastRun = lastRun,
                                            repMap = theMap)
        filePath = os.path.join( os.environ["CMSSW_BASE"], "src", packageName,
                                 "python", outName + "_cff.py" )
        if os.path.exists( filePath ):
            existMsg = "The predefined dataset '%s' already exists.\n"%( outName )
            askString = "Do you want to overwrite it? [y/n]\n"
            inputQuery = existMsg + askString
            while True:
                userInput = raw_input( inputQuery ).lower()
                if userInput == "y":
                    break
                elif userInput == "n":
                    return
                else:
                    inputQuery = askString
        print ( "The predefined dataset '%s' will be stored in the file\n"
                %( outName )
                + filePath +
                "\nFor future use you have to do 'scram b'." )
        print
        theFile = open( filePath, "w" )
        theFile.write( dataset_cff )
        theFile.close()
        return

    __source_template= ("%(importCms)s"
                        "import FWCore.PythonUtilities.LumiList as LumiList\n\n"
                        "%(goodLumiSecStr)s"
                        "maxEvents = cms.untracked.PSet( "
                        "input = cms.untracked.int32(%(nEvents)s) )\n"
                        "readFiles = cms.untracked.vstring()\n"
                        "secFiles = cms.untracked.vstring()\n"
                        "%(process)ssource = cms.Source(\"PoolSource\",\n"
                        "%(lumiStr)s"
                        "%(tab)s                    secondaryFileNames ="
                        "secFiles,\n"
                        "%(tab)s                    fileNames = readFiles\n"
                        ")\n"
                        "%(files)s\n"
                        "%(lumiSecExtend)s\n")

    def __createSnippet( self, jsonPath = None, begin = None, end = None,
                         firstRun = None, lastRun = None, repMap = None ):
        if ( begin or end ) and (firstRun or lastRun ):
            msg = ( "Don't mix time-based limits (begin/end) with run-based "
                    "limits (firstRun/lastRun)!" )
            raise AllInOneError( msg )
        if jsonPath == None:
            if ( begin or end ) or (firstRun or lastRun):
                msg = ("Specifying 'firstRun'/'lastRun' or 'begin'/'end' "
                       "without specifying a JSON file is not supported.")
                raise AllInOneError( msg )
            goodLumiSecStr = ""
            lumiStr = ""
            lumiSecExtend = ""
        else:
            if begin or end:
                runList = [ run["run_number"] for run in self.__getRunList() ]
                runTimeList = [ run["creation_time"] for run in self.__getRunList() ]
            if begin:
                runIndex = bisect.bisect_left( runTimeList, begin )
                firstRun = runList[runIndex]
                print "Wanted start time:", begin
                print "Run before:", runList[runIndex - 1], runTimeList[runIndex - 1]
                print "Run taken: ", runList[runIndex], runTimeList[runIndex]
                print "Run after: ", runList[runIndex + 1], runTimeList[runIndex + 1]
            if end:
                runIndex = bisect.bisect_left( runTimeList, begin )
                lastRun = runList[runIndex]
                print "Wanted start time:", end
                print "Run before:", runList[runIndex - 1], runTimeList[runIndex - 1]
                print "Run taken: ", runList[runIndex], runTimeList[runIndex]
                print "Run after: ", runList[runIndex + 1], runTimeList[runIndex + 1]
            if firstRun or lastRun:
                goodLumiSecStr = ( "lumiSecs = cms.untracked."
                                   "VLuminosityBlockRange()\n" )
                lumiStr = "                    lumisToProcess = lumiSecs,\n"
                theLumiList = LumiList ( filename = jsonPath )
                allRuns = theLumiList.getRuns()
                runsToRemove = []
                for run in allRuns:
                    if firstRun and int( run ) < firstRun:
                        runsToRemove.append( run )
                    if lastRun and int( run ) > lastRun:
                        runsToRemove.append( run )
                theLumiList.removeRuns( runsToRemove )
                splitLumiList = list( self.__chunks(
                    theLumiList.getCMSSWString().split(','), 255 ) )
                if len(splitLumiList[0][0]) == 0:
                    lumiSecExtend = ""
                else:
                    lumiSecStr = [ "',\n'".join( lumis ) \
                                   for lumis in splitLumiList ]
                    lumiSecStr = [ "lumiSecs.extend( [\n'" + lumis + "'\n] )" \
                                   for lumis in lumiSecStr ]
                    lumiSecExtend = "\n".join( lumiSecStr )
            else:
                goodLumiSecStr = ( "goodLumiSecs = LumiList.LumiList(filename"
                                   "= '%(json)s').getCMSSWString().split(',')\n"
                                   "lumiSecs = cms.untracked"
                                   ".VLuminosityBlockRange()\n"
                                   )
                lumiStr = "                    lumisToProcess = lumiSecs,\n"
                lumiSecExtend = "lumiSecs.extend(goodLumiSecs)\n"
        splitFileList = list( self.__chunks( self.fileList(), 255 ) )
        fileStr = [ "',\n'".join( files ) for files in splitFileList ]
        fileStr = [ "readFiles.extend( [\n'" + files + "'\n] )" \
                    for files in fileStr ]
        files = "\n".join( fileStr )
        theMap = repMap
        theMap["files"] = files
        theMap["json"] = jsonPath
        theMap["lumiStr"] = lumiStr
        theMap["goodLumiSecStr"] = goodLumiSecStr%( theMap )
        theMap["lumiSecExtend"] = lumiSecExtend
        dataset_snippet = self.__source_template%( theMap )
        return dataset_snippet

    
    def datasetSnippet( self, jsonPath = None, begin = None, end = None,
                        firstRun = None, lastRun = None, nEvents = None ):
        theMap = { "process": "process.",
                   "tab": " " * len( "process." ),
                   "nEvents": str( nEvents ),
                   "importCms": ""
                   }
        datasetSnippet = self.__createSnippet( jsonPath = jsonPath,
                                               begin = begin,
                                               end = end,
                                               firstRun = firstRun,
                                               lastRun = lastRun,
                                               repMap = theMap)
        return datasetSnippet
    
    def fileInfoList( self ):
        return self.__getFileInfoList( self.__dasLimit )

    def name( self ):
        return self.__name


if __name__ == '__main__':
    datasetName = '/MinimumBias/Run2012D-TkAlMinBias-v1/ALCARECO'
    jsonFile = ( '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/'
                 'Collisions12/8TeV/Prompt/'
                 'Cert_190456-207898_8TeV_PromptReco_Collisions12_JSON.txt' )
    dataset = Dataset( datasetName )
    dataset.dump_cff( outName = "Dataset_1", jsonPath = jsonFile, firstRun = 1 )
    dataset.dump_cff( outName = "Dataset_2", jsonPath = jsonFile )
    dataset.dump_cff( outName = "Dataset_3" )
    print dataset.datasetSnippet( nEvents = 100, jsonPath = jsonFile, firstRun = 1 )
    print dataset.datasetSnippet( nEvents = 100, jsonPath = jsonFile )
    print dataset.datasetSnippet( nEvents = 100 )
    print dataset.datasetSnippet( nEvents = 100, jsonPath = jsonFile,
                                  begin = "2012-11-27 00:00:00" )
    print dataset.datasetSnippet( nEvents = 100, jsonPath = jsonFile,
                                  begin = "2012-12-01 12:47:13" )
    print dataset.datasetSnippet( nEvents = 100, jsonPath = jsonFile,
                                  begin = "2012-10-30 12:47:13" )
    print dataset.datasetSnippet( nEvents = 100, jsonPath = jsonFile,
                                  end = "2012-11-28 00:00:00" )
    print dataset.datasetSnippet( nEvents = 100, jsonPath = jsonFile,
                                  begin = "2012-11-27 00:00:00",
                                  end = "2012-11-28 00:00:00" )
