# idea stolen from:
# http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/
#        PhysicsTools/PatAlgos/python/tools/cmsswVersionTools.py
import das_client
import json
import os
import bisect
import re
import datetime
from FWCore.PythonUtilities.LumiList import LumiList
from TkAlExceptions import AllInOneError


class Dataset:
    def __init__( self, datasetName, dasLimit = 0, tryPredefinedFirst = True ):
        self.__name = datasetName

        self.__dasLimit = dasLimit
        self.__fileList = None
        self.__fileInfoList = None
        self.__runList = None
        self.__alreadyStored = False

        # check, if dataset name matches CMS dataset naming scheme
        if re.match( r'/.+/.+/.+', self.__name ):
            self.__official = True
            fileName = "Dataset" + self.__name.replace("/","_") + "_cff.py"
        else:
            self.__official = False
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
        if self.__official and not tryPredefinedFirst:
            self.__predefined = False
        elif os.path.exists( searchPath1 ):
            self.__predefined = True
            self.__filename = searchPath1
        elif os.path.exists( searchPath2 ):
            msg = ("The predefined dataset '%s' does exist in '%s', but "
                   "you need to run 'scram b' first."
                   %( self.__name, searchPath2 ))
            if self.__official:
                print msg
                print "Getting the data from DAS again.  To go faster next time, run scram b."
            else:
                raise AllInOneError( msg )
        elif os.path.exists( searchPath3 ):
            self.__predefined = True
            self.__filename = searchPath3
        elif self.__official:
            self.__predefined = False
        else:
            msg = ("The predefined dataset '%s' does not exist. Please "
                   "create it first or check for typos."%( self.__name ))
            raise AllInOneError( msg )

        if self.__predefined and self.__official:
            self.__name = "Dataset" + self.__name.replace("/","_")

        self.__dataType = self.__getDataType()
        self.__magneticField = self.__getMagneticField()

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
                                        if self.__findInJson(run, "run_number") >= firstRun ]
                if lastRun:
                    selectedRunList = [ run for run in selectedRunList \
                                        if self.__findInJson(run, "run_number") <= lastRun ]
                lumiList = [ str( self.__findInJson(run, "run_number") ) + ":1-" \
                             + str( self.__findInJson(run, "run_number") ) + ":max" \
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

    def __findInJson(self, jsondict, strings):
        if isinstance(strings, str):
            strings = [ strings ]

        if len(strings) == 0:
            return jsondict
        if isinstance(jsondict,dict):
            if strings[0] in jsondict:
                try:
                    return self.__findInJson(jsondict[strings[0]], strings[1:])
                except KeyError:
                    pass
        else:
            for a in jsondict:
                if strings[0] in a:
                    try:
                        return self.__findInJson(a[strings[0]], strings[1:])
                    except (TypeError, KeyError):  #TypeError because a could be a string and contain strings[0]
                        pass
        #if it's not found
        raise KeyError("Can't find " + strings[0])

    def __getData( self, dasQuery, dasLimit = 0 ):
        dasData = das_client.get_data( 'https://cmsweb.cern.ch',
                                       dasQuery, 0, dasLimit, False )
        if isinstance(dasData, str):
            jsondict = json.loads( dasData )
        else:
            jsondict = dasData
        # Check, if the DAS query fails
        try:
            error = self.__findInJson(jsondict,["data","error"])
        except KeyError:
            error = None
        if error or self.__findInJson(jsondict,"status") != 'ok' or "data" not in jsondict:
            msg = ("The DAS query returned a error.  Here is the output\n" + str(jsondict) +
                   "\nIt's possible that this was a server error.  If so, it may work if you try again later")
            raise AllInOneError(msg)
        return self.__findInJson(jsondict,"data")

    def __getDataType( self ):
        if self.__predefined:
            with open(self.__filename) as f:
                f.readline()
                f.readline()
                datatype = f.readline().replace("\n",'')
                if "#data type: " in datatype:
                    return datatype.replace("#data type: ","")
                else:
                    return "unknown"

        dasQuery_type = ( 'dataset dataset=%s | grep dataset.datatype,'
                          'dataset.name'%( self.__name ) )
        data = self.__getData( dasQuery_type )

        try:
            return self.__findInJson(data, ["dataset", "datatype"])
        except KeyError:
            print ("Cannot find the datatype of the dataset '%s'\n"
                   "It may not be possible to automatically find the magnetic field,\n"
                   "and you will not be able run in CRAB mode"
                   %( self.name() ))
            return "unknown"

    def __getMagneticField( self ):
        Bfieldlocation = os.path.join( os.environ["CMSSW_RELEASE_BASE"], "python", "Configuration", "StandardSequences" )
        Bfieldlist = [ f.replace("MagneticField_",'').replace("_cff.py",'') \
                           for f in os.listdir(Bfieldlocation) \
                               if f.startswith("MagneticField_") and f.endswith("_cff.py") and f != "MagneticField_cff.py" ]
        Bfieldlist.sort( key = lambda Bfield: -len(Bfield) ) #Put it in order of decreasing length, so that searching in the name gives the longer match

        if self.__predefined:
            with open(self.__filename) as f:
                f.readline()
                f.readline()
                datatype = f.readline().replace("\n", '')
                Bfield = f.readline().replace("\n", '')
                if "#magnetic field: " in Bfield:
                    Bfield = Bfield.replace("#magnetic field: ", "")
                    if Bfield in Bfieldlist or Bfield == "unknown":
                        return Bfield
                    else:
                        print "Your dataset has magnetic field '%s', which does not exist in your CMSSW version!" % Bfield
                        print "Using Bfield='unknown' - this will revert to the default"
                        return "unknown"
                elif datatype == "#data type: data":
                    return "AutoFromDBCurrent"           #this should be in the "#magnetic field" line, but for safety in case it got messed up
                else:
                    return "unknown"

        if self.__dataType == "data":
            return "AutoFromDBCurrent"

        dasQuery_type = ( 'dataset dataset=%s'%( self.__name ) )             #try to find the magnetic field from DAS
        data = self.__getData( dasQuery_type )                               #it seems to be there for the newer (7X) MC samples, except cosmics

        try:
            Bfield = self.__findInJson(data, ["dataset", "mcm", "sequences", "magField"])
            if Bfield in Bfieldlist:
                return Bfield
            elif Bfield == "":
                pass
            else:
                print "Your dataset has magnetic field '%s', which does not exist in your CMSSW version!" % Bfield
                print "Using Bfield='unknown' - this will revert to the default magnetic field"
                return "unknown"
        except KeyError:
            pass

        for possibleB in Bfieldlist:
            if possibleB in self.__name.replace("TkAlCosmics0T", ""):         #for some reason all cosmics dataset names contain this string
                return possibleB

        return "unknown"

    def __getFileInfoList( self, dasLimit ):
        if self.__fileInfoList:
            return self.__fileInfoList
        dasQuery_files = ( 'file dataset=%s | grep file.name, file.nevents, '
                           'file.creation_time, '
                           'file.modification_time'%( self.__name ) )
        print "Requesting file information for '%s' from DAS..."%( self.__name ),
        data = self.__getData( dasQuery_files, dasLimit )
        print "Done."
        data = [ self.__findInJson(entry,"file") for entry in data ]
        if len( data ) == 0:
            msg = ("No files are available for the dataset '%s'. This can be "
                   "due to a typo or due to a DAS problem. Please check the "
                   "spelling of the dataset and/or retry to run "
                   "'validateAlignments.py'."%( self.name() ))
            raise AllInOneError( msg )
        fileInformationList = []
        for file in data:
            fileName = self.__findInJson(file, "name")
            fileCreationTime = self.__findInJson(file, "creation_time")
            fileNEvents = self.__findInJson(file, "nevents")
            # select only non-empty files
            if fileNEvents == 0:
                continue
            fileDict = { "name": fileName,
                         "creation_time": fileCreationTime,
                         "nevents": fileNEvents
                         }
            fileInformationList.append( fileDict )
        fileInformationList.sort( key=lambda info: self.__findInJson(info,"name") )
        return fileInformationList

    def __getRunList( self ):
        if self.__runList:
            return self.__runList
        dasQuery_runs = ( 'run dataset=%s | grep run.run_number,'
                          'run.creation_time'%( self.__name ) )
        print "Requesting run information for '%s' from DAS..."%( self.__name ),
        data = self.__getData( dasQuery_runs )
        print "Done."
        data = [ self.__findInJson(entry,"run") for entry in data ]
        data.sort( key = lambda run: self.__findInJson(run, "run_number") )
        self.__runList = data
        return data

    __source_template= ("%(header)s"
                        "%(importCms)s"
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

    def __datetime(self, stringForDas):
        if len(stringForDas) != 8:
            raise AllInOneError(stringForDas + " is not a valid date string.\n"
                              + "DAS accepts dates in the form 'yyyymmdd'")
        year = stringForDas[:4]
        month = stringForDas[4:6]
        day = stringForDas[6:8]
        return datetime.date(int(year), int(month), int(day))

    def __dateString(self, date):
        return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)

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

        if begin or end:
            runList = [ self.__findInJson(run, "run_number") for run in self.__getRunList() ]

        if begin:
            lastdate = begin
            for delta in [ 1, 5, 10, 20, 30 ]:                       #try searching for about 2 months after begin
                firstdate = lastdate
                lastdate = self.__dateString(self.__datetime(firstdate) + datetime.timedelta(delta))
                dasQuery_begin = "run date between[%s,%s]" % (firstdate, lastdate)
                begindata = self.__getData(dasQuery_begin)
                if len(begindata) > 0:
                    begindata.sort(key = lambda run: self.__findInJson(run, ["run", "run_number"]))
                    try:
                        runIndex = self.__find_ge( runList, self.__findInJson(begindata[0], ["run", "run_number"]))
                    except ValueError:
                        msg = ( "Your 'begin' is after the creation time of the last "
                                "run in the dataset\n'%s'"%( self.__name ) )
                        raise AllInOneError( msg )
                    firstRun = runList[runIndex]
                    begin = None
                    break

        if begin:
            raise AllInOneError("No runs within a reasonable time interval after your 'begin'."
                                "Try using a 'begin' that has runs soon after it (within 2 months at most)")

        if end:
            firstdate = end
            for delta in [ 1, 5, 10, 20, 30 ]:                       #try searching for about 2 months before end
                lastdate = firstdate
                firstdate = self.__dateString(self.__datetime(lastdate) - datetime.timedelta(delta))
                dasQuery_end = "run date between[%s,%s]" % (firstdate, lastdate)
                enddata = self.__getData(dasQuery_end)
                if len(enddata) > 0:
                    enddata.sort(key = lambda run: self.__findInJson(run, ["run", "run_number"]))
                    try:
                        runIndex = self.__find_lt( runList, self.__findInJson(enddata[-1], ["run", "run_number"]))
                    except ValueError:
                        msg = ( "Your 'end' is before the creation time of the first "
                                "run in the dataset\n'%s'"%( self.__name ) )
                        raise AllInOneError( msg )
                    lastRun = runList[runIndex]
                    end = None
                    break

        if end:
            raise AllInOneError("No runs within a reasonable time interval before your 'end'."
                                "Try using an 'end' that has runs soon before it (within 2 months at most)")

        if shortTuple:
            return firstRun, lastRun
        else:
            return begin, end, firstRun, lastRun

    def dataType( self ):
        return self.__dataType

    def magneticField( self ):
        return self.__magneticField
    
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
                   "importCms": "",
                   "header": ""
                   }
        datasetSnippet = self.__createSnippet( jsonPath = jsonPath,
                                               begin = begin,
                                               end = end,
                                               firstRun = firstRun,
                                               lastRun = lastRun,
                                               repMap = theMap,
                                               crab = crab )
        if jsonPath == "" and begin == "" and end == "" and firstRun == "" and lastRun == "":
            try:
                self.dump_cff()
            except AllInOneError, e:
                print "Can't store the dataset as a cff:"
                print e
                print "This may be inconvenient in the future, but will not cause a problem for this validation."
        return datasetSnippet

    def dump_cff( self, outName = None, jsonPath = None, begin = None,
                  end = None, firstRun = None, lastRun = None ):
        if self.__alreadyStored:
            return     
        self.__alreadyStored = True
        if outName == None:
            outName = "Dataset" + self.__name.replace("/", "_")
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
                   "importCms": "import FWCore.ParameterSet.Config as cms\n",
                   "header": "#Do not delete, put anything before, or (unless you know what you're doing) change these comments\n"
                             "#%s\n"
                             "#data type: %s\n"
                             "#magnetic field: %s\n"
                             %(self.__name, self.__dataType, self.__magneticField)
                   }
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
        fileList = [ self.__findInJson(fileInfo,"name") \
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
                                  firstRun = "207800",
                                  end = "20121128")
    dataset.dump_cff( outName = "Dataset_Test_TkAlMinBias_Run2012D",
                      jsonPath = jsonFile,
                      firstRun = "207800",
                      end = "20121128" )
