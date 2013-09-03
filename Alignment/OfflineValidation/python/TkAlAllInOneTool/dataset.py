# idea stolen from:
# http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/
#        PhysicsTools/PatAlgos/python/tools/cmsswVersionTools.py
import das_client
import json
import os
import bisect
import re
from FWCore.PythonUtilities.LumiList import LumiList
from TkAlExceptions import AllInOneError


class Dataset:
    def __init__( self, datasetName, dasLimit = 0 ):
        self.__name = datasetName
        # check, if dataset name matches CMS dataset naming scheme
        if re.match( r'/.+/.+/.+', self.__name ):
            self.__dataType = self.__getDataType()
            self.__predefined = False
        else:
            fileName = self.__name + "_cff.py"
            searchPath1 = os.path.join( os.environ["CMSSW_BASE"], "python",
                                        "Alignment", "OfflineValidation",
                                        fileName )
            searchPath2 = os.path.join( os.environ["CMSSW_BASE"], "src",
                                        "Alignment", "OfflineValidation",
                                        "python", fileName )
            searchPath3 = os.path.join( os.environ["CMSSW_RELEASE_BASE"],
                                        "python", "Alignment",
                                        "OfflineValidation", fileName )
            if os.path.exists( searchPath1 ):
                pass
            elif os.path.exists( searchPath2 ):
                msg = ("The predefined dataset '%s' does exist in '%s', but "
                       "you need to run 'scram b' first."
                       %( self.__name, searchPath2 ))
                raise AllInOneError( msg )
            elif os.path.exists( searchPath3 ):
                pass
            else:
                msg = ("The predefined dataset '%s' does not exist. Please "
                       "create it first or check for typos."%( self.__name ))
                raise AllInOneError( msg )
            self.__dataType = "unknown"
            self.__predefined = True
        self.__dasLimit = dasLimit
        self.__fileList = None
        self.__fileInfoList = None
        self.__runList = None

    def __chunks( self, theList, n ):
        """ Yield successive n-sized chunks from theList.
        """
        for i in xrange( 0, len( theList ), n ):
            yield theList[i:i+n]

    def __createSnippet( self, jsonPath = None, begin = None, end = None,
                         firstRun = None, lastRun = None, repMap = None,
                         crab = False ):
        if firstRun:
            firstRun = int( firstRun )
        if lastRun:
            lastRun = int( lastRun )
        if ( begin and firstRun ) or ( end and lastRun ):
            msg = ( "The Usage of "
                    + "'begin' & 'firstRun' " * int( bool( begin and
                                                           firstRun ) )
                    + "and " * int( bool( ( begin and firstRun ) and
                                         ( end and lastRun ) ) )
                    + "'end' & 'lastRun' " * int( bool( end and lastRun ) )
                    + "is ambigous." )
            raise AllInOneError( msg )
        if begin or end:
            ( firstRun, lastRun ) = self.convertTimeToRun(
                begin = begin, end = end, firstRun = firstRun,
                lastRun = lastRun )
        if ( firstRun and lastRun ) and ( firstRun > lastRun ):
            msg = ( "The lower time/runrange limit ('begin'/'firstRun') "
                    "chosen is greater than the upper time/runrange limit "
                    "('end'/'lastRun').")
            raise AllInOneError( msg )
        goodLumiSecStr = ""
        lumiStr = ""
        lumiSecExtend = ""
        if firstRun or lastRun:
            goodLumiSecStr = ( "lumiSecs = cms.untracked."
                               "VLuminosityBlockRange()\n" )
            lumiStr = "                    lumisToProcess = lumiSecs,\n"
            if not jsonPath:
                selectedRunList = self.__getRunList()
                if firstRun:
                    selectedRunList = [ run for run in selectedRunList \
                                        if run["run_number"] >= firstRun ]
                if lastRun:
                    selectedRunList = [ run for run in selectedRunList \
                                        if run["run_number"] <= lastRun ]
                lumiList = [ str( run["run_number"] ) + ":1-" \
                             + str( run["run_number"] ) + ":max" \
                             for run in selectedRunList ]
                splitLumiList = list( self.__chunks( lumiList, 255 ) )
            else:
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
            if not len(splitLumiList[0][0]) == 0:
                lumiSecStr = [ "',\n'".join( lumis ) \
                               for lumis in splitLumiList ]
                lumiSecStr = [ "lumiSecs.extend( [\n'" + lumis + "'\n] )" \
                               for lumis in lumiSecStr ]
                lumiSecExtend = "\n".join( lumiSecStr )
        elif jsonPath:
                goodLumiSecStr = ( "goodLumiSecs = LumiList.LumiList(filename"
                                   "= '%(json)s').getCMSSWString().split(',')\n"
                                   "lumiSecs = cms.untracked"
                                   ".VLuminosityBlockRange()\n"
                                   )
                lumiStr = "                    lumisToProcess = lumiSecs,\n"
                lumiSecExtend = "lumiSecs.extend(goodLumiSecs)\n"
        if crab:
            files = ""
        else:
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
        if crab:
            dataset_snippet = self.__dummy_source_template%( theMap )
        else:
            dataset_snippet = self.__source_template%( theMap )
        return dataset_snippet

    __dummy_source_template = ("%(process)smaxEvents = cms.untracked.PSet( "
                               "input = cms.untracked.int32(%(nEvents)s) )\n"
                               "readFiles = cms.untracked.vstring()\n"
                               "secFiles = cms.untracked.vstring()\n"
                               "%(process)ssource = cms.Source(\"PoolSource\",\n"
                               "%(tab)s                    secondaryFileNames ="
                               "secFiles,\n"
                               "%(tab)s                    fileNames = readFiles\n"
                               ")\n"
                               "readFiles.extend(['dummy_File.root'])\n")
        
    def __find_lt( self, a, x ):
        'Find rightmost value less than x'
        i = bisect.bisect_left( a, x )
        if i:
            return i-1
        raise ValueError

    def __find_ge( self, a, x):
        'Find leftmost item greater than or equal to x'
        i = bisect.bisect_left( a, x )
        if i != len( a ):
            return i
        raise ValueError

    def __getData( self, dasQuery, dasLimit = 0 ):
        dasData = das_client.get_data( 'https://cmsweb.cern.ch',
                                       dasQuery, 0, dasLimit, False )
        jsondict = json.loads( dasData )
        # Check, if the DAS query fails
        if jsondict["status"] != 'ok':
            msg = "Status not 'ok', but:", jsondict["status"]
            raise AllInOneError(msg)
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
        print "Requesting file information for '%s' from DAS..."%( self.__name ),
        data = self.__getData( dasQuery_files, dasLimit )
        print "Done."
        data = [ entry["file"] for entry in data ]
        if len( data ) == 0:
            msg = ("No files are available for the dataset '%s'. This can be "
                   "due to a typo or due to a DAS problem. Please check the "
                   "spelling of the dataset and/or retry to run "
                   "'validateAlignments.py'."%( self.name() ))
            raise AllInOneError( msg )
        fileInformationList = []
        for file in data:
            fileName = file[0]["name"]
            fileCreationTime = file[0]["creation_time"]
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
        print "Requesting run information for '%s' from DAS..."%( self.__name ),
        data = self.__getData( dasQuery_runs )
        print "Done."
        data = [ entry["run"][0] for entry in data ]
        data.sort( key = lambda run: run["creation_time"] )
        self.__runList = data
        return data

    __source_template= ("%(importCms)s"
                        "import FWCore.PythonUtilities.LumiList as LumiList\n\n"
                        "%(goodLumiSecStr)s"
                        "%(process)smaxEvents = cms.untracked.PSet( "
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

    def convertTimeToRun( self, begin = None, end = None,
                          firstRun = None, lastRun = None,
                          shortTuple = True ):
        if ( begin and firstRun ) or ( end and lastRun ):
            msg = ( "The Usage of "
                    + "'begin' & 'firstRun' " * int( bool( begin and
                                                           firstRun ) )
                    + "and " * int( bool( ( begin and firstRun ) and
                                         ( end and lastRun ) ) )
                    + "'end' & 'lastRun' " * int( bool( end and lastRun ) )
                    + "is ambigous." )
            raise AllInOneError( msg )

        runList = [ run["run_number"] for run in self.__getRunList() ]
        runTimeList = [ run["creation_time"] for run in self.__getRunList() ]
        if begin:
            try:
                runIndex = self.__find_ge( runTimeList, begin )
            except ValueError:
                msg = ( "Your 'begin' is after the creation time of the last "
                        "run in the dataset\n'%s'"%( self.__name ) )
                raise AllInOneError( msg )
            firstRun = runList[runIndex]
            begin = None
        if end:
            try:
                runIndex = self.__find_lt( runTimeList, end )
            except ValueError:
                msg = ( "Your 'end' is before the creation time of the first "
                        "run in the dataset\n'%s'"%( self.__name ) )
                raise AllInOneError( msg )
            lastRun = runList[runIndex]
            end = None
        if shortTuple:
            return firstRun, lastRun
        else:
            return begin, end, firstRun, lastRun

    def dataType( self ):
        return self.__dataType
    
    def datasetSnippet( self, jsonPath = None, begin = None, end = None,
                        firstRun = None, lastRun = None, nEvents = None,
                        crab = False ):
        if self.__predefined:
            return ("process.load(\"Alignment.OfflineValidation.%s_cff\")\n"
                    "process.maxEvents = cms.untracked.PSet(\n"
                    "    input = cms.untracked.int32(%s)\n"
                    ")"
                    %( self.__name, nEvents ))
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
                                               repMap = theMap,
                                               crab = crab )
        return datasetSnippet

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

    def fileList( self ):
        if self.__fileList:
            return self.__fileList
        fileList = [ fileInfo["name"] \
                     for fileInfo in self.fileInfoList() ]
        self.__fileList = fileList
        return fileList
    
    def fileInfoList( self ):
        return self.__getFileInfoList( self.__dasLimit )

    def name( self ):
        return self.__name

    def predefined( self ):
        return self.__predefined

    def runList( self ):
        if self.__runList:
            return self.__runList
        return self.__getRunList()


if __name__ == '__main__':
    print "Start testing..."
    datasetName = '/MinimumBias/Run2012D-TkAlMinBias-v1/ALCARECO'
    jsonFile = ( '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/'
                 'Collisions12/8TeV/Prompt/'
                 'Cert_190456-207898_8TeV_PromptReco_Collisions12_JSON.txt' )
    dataset = Dataset( datasetName )
    print dataset.datasetSnippet( nEvents = 100,jsonPath = jsonFile,
                                  firstRun = "207983",
                                  end = "2012-11-28 00:00:00" )
    dataset.dump_cff( outName = "Dataset_Test_TkAlMinBias_Run2012D",
                      jsonPath = jsonFile,
                      firstRun = "207983",
                      end = "2012-11-28 00:00:00" )
