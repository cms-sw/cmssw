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
    def __init__( self, datasetName, dasLimit = 0, tryPredefinedFirst = True,
                  cmssw = os.environ["CMSSW_BASE"], cmsswrelease = os.environ["CMSSW_RELEASE_BASE"]):
        self.__name = datasetName
        self.__origName = datasetName
        self.__dasLimit = dasLimit
        self.__fileList = None
        self.__fileInfoList = None
        self.__runList = None
        self.__alreadyStored = False
        self.__cmssw = cmssw
        self.__cmsswrelease = cmsswrelease
        self.__firstusedrun = None
        self.__lastusedrun = None
        self.__parentDataset = None
        self.__parentFileList = None
        self.__parentFileInfoList = None

        # check, if dataset name matches CMS dataset naming scheme
        if re.match( r'/.+/.+/.+', self.__name ):
            self.__official = True
            fileName = "Dataset" + self.__name.replace("/","_") + "_cff.py"
        else:
            self.__official = False
            fileName = self.__name + "_cff.py"

        searchPath1 = os.path.join( self.__cmssw, "python",
                                    "Alignment", "OfflineValidation",
                                    fileName )
        searchPath2 = os.path.join( self.__cmssw, "src",
                                    "Alignment", "OfflineValidation",
                                    "python", fileName )
        searchPath3 = os.path.join( self.__cmsswrelease,
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

    __source_template= ("%(header)s"
                        "%(importCms)s"
                        "import FWCore.PythonUtilities.LumiList as LumiList\n\n"
                        "%(goodLumiSecStr)s"
                        "readFiles = cms.untracked.vstring()\n"
                        "secFiles = cms.untracked.vstring()\n"
                        "%(process)ssource = cms.Source(\"PoolSource\",\n"
                        "%(lumiStr)s"
                        "%(tab)s                    secondaryFileNames ="
                        "secFiles,\n"
                        "%(tab)s                    fileNames = readFiles\n"
                        ")\n"
                        "%(files)s\n"
                        "%(lumiSecExtend)s\n"
                        "%(process)smaxEvents = cms.untracked.PSet( "
                        "input = cms.untracked.int32(%(nEvents)s) )\n"
                        "%(skipEventsString)s\n")

    __dummy_source_template = ("readFiles = cms.untracked.vstring()\n"
                               "secFiles = cms.untracked.vstring()\n"
                               "%(process)ssource = cms.Source(\"PoolSource\",\n"
                               "%(tab)s                    secondaryFileNames ="
                               "secFiles,\n"
                               "%(tab)s                    fileNames = readFiles\n"
                               ")\n"
                               "readFiles.extend(['dummy_File.root'])\n"
                               "%(process)smaxEvents = cms.untracked.PSet( "
                               "input = cms.untracked.int32(%(nEvents)s) )\n"
                               "%(skipEventsString)s\n")

    def __createSnippet( self, jsonPath = None, begin = None, end = None,
                         firstRun = None, lastRun = None, repMap = None,
                         crab = False, parent = False ):
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
        if self.predefined() and (jsonPath or begin or end or firstRun or lastRun):
            msg = ( "The parameters 'JSON', 'begin', 'end', 'firstRun', and 'lastRun'"
                    "only work for official datasets, not predefined _cff.py files" )
            raise AllInOneError( msg )
        goodLumiSecStr = ""
        lumiStr = ""
        lumiSecExtend = ""
        if firstRun or lastRun or jsonPath:
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
                theLumiList = None
                try:
                    theLumiList = LumiList ( filename = jsonPath )
                except ValueError:
                    pass

                if theLumiList is not None:
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
                else:
                    with open(jsonPath) as f:
                        jsoncontents = f.read()
                        if "process.source.lumisToProcess" in jsoncontents:
                            msg = "%s is not a json file, but it seems to be a CMSSW lumi selection cff snippet.  Trying to use it" % jsonPath
                            if firstRun or lastRun:
                                msg += ("\n  (after applying firstRun and/or lastRun)")
                            msg += ".\nPlease note that, depending on the format of this file, it may not work as expected."
                            msg += "\nCheck your config file to make sure that it worked properly."
                            print msg

                            self.__firstUsedRun = -1
                            self.__lastUsedRun = -1
                            if firstRun or lastRun:
                                jsoncontents = re.sub("\d+:(\d+|max)-\d+:(\d+|max)", self.getForceRunRangeFunction(firstRun, lastRun), jsoncontents)
                            lumiSecExtend = jsoncontents
                            splitLumiList = [[""]]

            if not len(splitLumiList[0][0]) == 0:
                lumiSecStr = [ "',\n'".join( lumis ) \
                               for lumis in splitLumiList ]
                lumiSecStr = [ "lumiSecs.extend( [\n'" + lumis + "'\n] )" \
                               for lumis in lumiSecStr ]
                lumiSecExtend = "\n".join( lumiSecStr )
                self.__firstusedrun = splitLumiList[0][0].split(":")[0]
                self.__lastusedrun = splitLumiList[-1][-1].split(":")[0]
        else:
            self.__firstusedrun = self.__findInJson(self.__getRunList()[0],"run_number")
            self.__lastusedrun = self.__findInJson(self.__getRunList()[-1],"run_number")

        if crab:
            files = ""
        else:
            splitFileList = list( self.__chunks( self.fileList(), 255 ) )
            fileStr = [ "',\n'".join( files ) for files in splitFileList ]
            fileStr = [ "readFiles.extend( [\n'" + files + "'\n] )" \
                        for files in fileStr ]
            files = "\n".join( fileStr )

            if parent:
                splitParentFileList = list( self.__chunks( self.fileList(parent = True), 255 ) )
                parentFileStr = [ "',\n'".join( parentFiles ) for parentFiles in splitParentFileList ]
                parentFileStr = [ "secFiles.extend( [\n'" + parentFiles + "'\n] )" \
                            for parentFiles in parentFileStr ]
                parentFiles = "\n".join( parentFileStr )
                files += "\n\n" + parentFiles


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

    def forcerunrange(self, firstRun, lastRun, s):
        """s must be in the format run1:lum1-run2:lum2"""
        s = s.group()
        print s
        run1 = s.split("-")[0].split(":")[0]
        lum1 = s.split("-")[0].split(":")[1]
        run2 = s.split("-")[1].split(":")[0]
        lum2 = s.split("-")[1].split(":")[1]
        if int(run2) < firstRun or int(run1) > lastRun:
            return ""
        if int(run1) < firstRun or firstRun < 0:
            run1 = firstRun
            lum1 = 1
        if int(run2) > lastRun:
            run2 = lastRun
            lum2 = "max"
        if int(run1) < self.__firstUsedRun:
            self.__firstUsedRun = int(run1)
        if int(run2) > self.__lastUsedRun:
            self.__lastUsedRun = int(run2)
        return "%s:%s-%s:%s" % (run1, lum1, run2, lum2)

    def getForceRunRangeFunction(self, firstRun, lastRun):
        def forcerunrangefunction(s):
            return self.forcerunrange(firstRun, lastRun, s)
        return forcerunrangefunction

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
                datatype = None
                for line in f.readlines():
                    if line.startswith("#data type: "):
                        if datatype is not None:
                            raise AllInOneError(self.__filename + " has multiple 'data type' lines.")
                        datatype = line.replace("#data type: ", "").replace("\n","")
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

    def __getParentDataset( self ):
        dasQuery = "parent dataset=" + self.__name
        data = self.__getData( dasQuery )
        try:
            return self.__findInJson(data, ["parent", "name"])
        except KeyError:
            raise AllInOneError("Cannot find the parent of the dataset '" + self.__name + "'\n"
                                "Here is the DAS output:\n" + str(jsondict) +
                                "\nIt's possible that this was a server error.  If so, it may work if you try again later")

    def __getMagneticField( self ):
        Bfieldlocation = os.path.join( self.__cmsswrelease, "python", "Configuration", "StandardSequences" )
        Bfieldlist = [ f.replace("MagneticField_",'').replace("_cff.py",'') \
                           for f in os.listdir(Bfieldlocation) \
                               if f.startswith("MagneticField_") and f.endswith("_cff.py") and f != "MagneticField_cff.py" ]
        Bfieldlist.sort( key = lambda Bfield: -len(Bfield) ) #Put it in order of decreasing length, so that searching in the name gives the longer match

        if self.__predefined:
            with open(self.__filename) as f:
                datatype = None
                Bfield = None
                for line in f.readlines():
                    if line.startswith("#data type: "):
                        if datatype is not None:
                            raise AllInOneError(self.__filename + " has multiple 'data type' lines.")
                        datatype = line.replace("#data type: ", "").replace("\n","")
                    if line.startswith("#magnetic field: "):
                        if Bfield is not None:
                            raise AllInOneError(self.__filename + " has multiple 'magnetic field' lines.")
                        Bfield = line.replace("#magnetic field: ", "").replace("\n","")
                if Bfield is not None:
                    Bfield = Bfield.split(",")[0]
                    if Bfield in Bfieldlist or Bfield == "unknown":
                        return Bfield
                    else:
                        print "Your dataset has magnetic field '%s', which does not exist in your CMSSW version!" % Bfield
                        print "Using Bfield='unknown' - this will revert to the default"
                        return "unknown"
                elif datatype == "data":
                    return "AutoFromDBCurrent"           #this should be in the "#magnetic field" line, but for safety in case it got messed up
                else:
                    return "unknown"

        if self.__dataType == "data":
            return "AutoFromDBCurrent"

        dasQuery_B = ( 'dataset dataset=%s'%( self.__name ) )             #try to find the magnetic field from DAS
        data = self.__getData( dasQuery_B )                               #it seems to be there for the newer (7X) MC samples, except cosmics

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

    def __getMagneticFieldForRun( self, run = -1, tolerance = 0.5 ):
        """For MC, this returns the same as the previous function.
           For data, it gets the magnetic field from the runs.  This is important for
           deciding which template to use for offlinevalidation
        """
        if "T" in self.__magneticField:                       #for MC
            Bfield = self.__magneticField.split("T")[0]
            return float(Bfield) / 10.0                       #e.g. 38T and 38T_PostLS1 both return 3.8
        if self.__predefined:
            with open(self.__filename) as f:
                Bfield = None
                for line in f.readlines():
                    if line.startswith("#magnetic field: ") and "," in line:
                        if Bfield is not None:
                            raise AllInOneError(self.__filename + " has multiple 'magnetic field' lines.")
                        return float(line.replace("#magnetic field: ", "").split(",")[1])

        if run > 0:
            dasQuery = ('run = %s'%run)                         #for data
            data = self.__getData(dasQuery)
            try:
                return self.__findInJson(data, ["run","bfield"])
            except KeyError:
                return "unknown Can't get the magnetic field for run %s from DAS" % run

        #run < 0 - find B field for the first and last runs, and make sure they're compatible
        #  (to within tolerance)
        #NOT FOOLPROOF!  The magnetic field might go up and then down, or vice versa
        if self.__firstusedrun is None or self.__lastusedrun is None:
            return "unknown Can't get the exact magnetic field for the dataset until data has been retrieved from DAS."
        firstrunB = self.__getMagneticFieldForRun(self.__firstusedrun)
        lastrunB = self.__getMagneticFieldForRun(self.__lastusedrun)
        try:
            if abs(firstrunB - lastrunB) <= tolerance:
                return .5*(firstrunB + lastrunB)
            print firstrunB, lastrunB, tolerance
            return ("unknown The beginning and end of your run range for %s\n"
                    "have different magnetic fields (%s, %s)!\n"
                    "Try limiting the run range using firstRun, lastRun, begin, end, or JSON,\n"
                    "or increasing the tolerance (in dataset.py) from %s.") % (self.__name, firstrunB, lastrunB, tolerance)
        except TypeError:
            if "unknown" in firstrunB:
                return firstrunB
            else:
                return lastrunB

    def __getFileInfoList( self, dasLimit, parent = False ):
        if self.__predefined:
            if parent:
                extendstring = "secFiles.extend"
            else:
                extendstring = "readFiles.extend"
            with open(self.__fileName) as f:
                files = []
                copy = False
                for line in f.readlines():
                    if "]" in line:
                        copy = False
                    if copy:
                        files.append({name: line.translate(None, "', " + '"')})
                    if extendstring in line and "[" in line and "]" not in line:
                        copy = True
            return files

        if self.__fileInfoList and not parent:
            return self.__fileInfoList
        if self.__parentFileInfoList and parent:
            return self.__parentFileInfoList

        if parent:
            searchdataset = self.parentDataset()
        else:
            searchdataset = self.__name
        dasQuery_files = ( 'file dataset=%s | grep file.name, file.nevents, '
                           'file.creation_time, '
                           'file.modification_time'%( searchdataset ) )
        print "Requesting file information for '%s' from DAS..."%( searchdataset ),
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
            fileName = 'unknown'
            try:
                fileName = self.__findInJson(file, "name")
                fileCreationTime = self.__findInJson(file, "creation_time")
                fileNEvents = self.__findInJson(file, "nevents")
            except KeyError:
                print ("DAS query gives bad output for file '%s'.  Skipping it.\n"
                       "It may work if you try again later.") % fileName
                fileNEvents = 0
            # select only non-empty files
            if fileNEvents == 0:
                continue
            fileDict = { "name": fileName,
                         "creation_time": fileCreationTime,
                         "nevents": fileNEvents
                         }
            fileInformationList.append( fileDict )
        fileInformationList.sort( key=lambda info: self.__findInJson(info,"name") )
        if parent:
            self.__parentFileInfoList = fileInformationList
        else:
            self.__fileInfoList = fileInformationList
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
        if not self.__dataType:
            self.__dataType = self.__getDataType()
        return self.__dataType

    def magneticField( self ):
        if not self.__magneticField:
            self.__magneticField = self.__getMagneticField()
        return self.__magneticField

    def magneticFieldForRun( self, run = -1 ):
        return self.__getMagneticFieldForRun(run)

    def parentDataset( self ):
        if not self.__parentDataset:
            self.__parentDataset = self.__getParentDataset()
        return self.__parentDataset

    def datasetSnippet( self, jsonPath = None, begin = None, end = None,
                        firstRun = None, lastRun = None, crab = False, parent = False ):
        if self.__predefined and parent:
                with open(self.__filename) as f:
                    if "secFiles.extend" not in f.read():
                        msg = ("The predefined dataset '%s' does not contain secondary files, "
                               "which your validation requires!") % self.__name
                        if self.__official:
                            self.__name = self.__origName
                            self.__predefined = False
                            print msg
                            print ("Retreiving the files from DAS.  You will be asked if you want "
                                   "to overwrite the old dataset.\n"
                                   "It will still be compatible with validations that don't need secondary files.")
                        else:
                            raise AllInOneError(msg)

        if self.__predefined:
            snippet = ("process.load(\"Alignment.OfflineValidation.%s_cff\")\n"
                       "process.maxEvents = cms.untracked.PSet(\n"
                       "    input = cms.untracked.int32(.oO[nEvents]Oo. / .oO[parallelJobs]Oo.)\n"
                       ")\n"
                       "process.source.skipEvents=cms.untracked.uint32(.oO[nIndex]Oo.*.oO[nEvents]Oo./.oO[parallelJobs]Oo.)"
                       %(self.__name))
            if not parent:
                with open(self.__filename) as f:
                    if "secFiles.extend" in f.read():
                        snippet += "\nprocess.source.secondaryFileNames = cms.untracked.vstring()"
            return snippet
        theMap = { "process": "process.",
                   "tab": " " * len( "process." ),
                   "nEvents": ".oO[nEvents]Oo. / .oO[parallelJobs]Oo.",
                   "skipEventsString": "process.source.skipEvents=cms.untracked.uint32(.oO[nIndex]Oo.*.oO[nEvents]Oo./.oO[parallelJobs]Oo.)\n",
                   "importCms": "",
                   "header": ""
                   }
        datasetSnippet = self.__createSnippet( jsonPath = jsonPath,
                                               begin = begin,
                                               end = end,
                                               firstRun = firstRun,
                                               lastRun = lastRun,
                                               repMap = theMap,
                                               crab = crab,
                                               parent = parent )
        if jsonPath == "" and begin == "" and end == "" and firstRun == "" and lastRun == "":
            try:
                self.dump_cff(parent = parent)
            except AllInOneError, e:
                print "Can't store the dataset as a cff:"
                print e
                print "This may be inconvenient in the future, but will not cause a problem for this validation."
        return datasetSnippet

    def dump_cff( self, outName = None, jsonPath = None, begin = None,
                  end = None, firstRun = None, lastRun = None, parent = False ):
        if self.__alreadyStored:
            return
        self.__alreadyStored = True
        if outName == None:
            outName = "Dataset" + self.__name.replace("/", "_")
        packageName = os.path.join( "Alignment", "OfflineValidation" )
        if not os.path.exists( os.path.join(
            self.__cmssw, "src", packageName ) ):
            msg = ("You try to store the predefined dataset'%s'.\n"
                   "For that you need to check out the package '%s' to your "
                   "private relase area in\n"%( outName, packageName )
                   + self.__cmssw )
            raise AllInOneError( msg )
        theMap = { "process": "",
                   "tab": "",
                   "nEvents": str( -1 ),
                   "skipEventsString": "",
                   "importCms": "import FWCore.ParameterSet.Config as cms\n",
                   "header": "#Do not delete or (unless you know what you're doing) change these comments\n"
                             "#%(name)s\n"
                             "#data type: %(dataType)s\n"
                             "#magnetic field: .oO[magneticField]Oo.\n"    #put in magnetic field later
                             %{"name": self.__name,                        #need to create the snippet before getting the magnetic field
                               "dataType": self.__dataType}                #so that we know the first and last runs
                   }
        dataset_cff = self.__createSnippet( jsonPath = jsonPath,
                                            begin = begin,
                                            end = end,
                                            firstRun = firstRun,
                                            lastRun = lastRun,
                                            repMap = theMap,
                                            parent = parent)
        magneticField = self.__magneticField
        if magneticField == "AutoFromDBCurrent":
            magneticField = "%s, %s" % (magneticField, str(self.__getMagneticFieldForRun()).replace("\n"," ")[0])
        dataset_cff = dataset_cff.replace(".oO[magneticField]Oo.",magneticField)
        filePath = os.path.join( self.__cmssw, "src", packageName,
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

    def fileList( self, parent = False ):
        if self.__fileList and not parent:
            return self.__fileList
        if self.__parentFileList and parent:
            return self.__parentFileList

        fileList = [ self.__findInJson(fileInfo,"name") \
                     for fileInfo in self.fileInfoList(parent) ]

        if not parent:
            self.__fileList = fileList
        else:
            self.__parentFileList = fileList
        return fileList

    def fileInfoList( self, parent = False ):
        return self.__getFileInfoList( self.__dasLimit, parent )

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
    print dataset.datasetSnippet( jsonPath = jsonFile,
                                  firstRun = "207800",
                                  end = "20121128")
    dataset.dump_cff( outName = "Dataset_Test_TkAlMinBias_Run2012D",
                      jsonPath = jsonFile,
                      firstRun = "207800",
                      end = "20121128" )
