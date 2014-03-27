import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *
from PhysicsTools.PatAlgos.tools.helpers import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from Configuration.AlCa.autoCond import autoCond

import os
import socket
from subprocess import *
import json
import das_client


## ------------------------------------------------------
## Automatic pick-up of RelVal input files
## ------------------------------------------------------

class PickRelValInputFiles( ConfigToolBase ):
    """  Picks up RelVal input files automatically and
  returns a vector of strings with the paths to be used in [PoolSource].fileNames
    PickRelValInputFiles( cmsswVersion, relVal, dataTier, condition, globalTag, maxVersions, skipFiles, numberOfFiles, debug )
    - useDAS       : switch to perform query in DAS rather than in DBS
                     optional; default: False
    - cmsswVersion : CMSSW release to pick up the RelVal files from
                     optional; default: the current release (determined automatically from environment)
    - formerVersion: use the last before the last valid CMSSW release to pick up the RelVal files from
                     applies also, if 'cmsswVersion' is set explicitly
                     optional; default: False
    - relVal       : RelVal sample to be used
                     optional; default: 'RelValTTbar'
    - dataTier     : data tier to be used
                     optional; default: 'GEN-SIM-RECO'
    - condition    : identifier of GlobalTag as defined in Configurations/PyReleaseValidation/python/autoCond.py
                     possibly overwritten, if 'globalTag' is set explicitly
                     optional; default: 'startup'
    - globalTag    : name of GlobalTag as it is used in the data path of the RelVals
                     optional; default: determined automatically as defined by 'condition' in Configurations/PyReleaseValidation/python/autoCond.py
      !!!            Determination is done for the release one runs in, not for the release the RelVals have been produced in.
      !!!            Example of deviation: data RelVals (CMSSW_4_1_X) might not only have the pure name of the GlobalTag 'GR_R_311_V2' in the full path,
                     but also an extension identifying the data: 'GR_R_311_V2_RelVal_wzMu2010B'
    - maxVersions  : max. versioning number of RelVal to check
                     optional; default: 9
    - skipFiles    : number of files to skip for a found RelVal sample
                     optional; default: 0
    - numberOfFiles: number of files to pick up
                     setting it to negative values, returns all found ('skipFiles' remains active though)
                     optional; default: -1
    - debug        : switch to enable enhanced messages in 'stdout'
                     optional; default: False
    """

    _label             = 'pickRelValInputFiles'
    _defaultParameters = dicttypes.SortedKeysDict()

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'useDAS'       , False                                                               , '' )
        self.addParameter( self._defaultParameters, 'cmsswVersion' , os.getenv( "CMSSW_VERSION" )                                        , 'auto from environment' )
        self.addParameter( self._defaultParameters, 'formerVersion', False                                                               , '' )
        self.addParameter( self._defaultParameters, 'relVal'       , 'RelValTTbar'                                                       , '' )
        self.addParameter( self._defaultParameters, 'dataTier'     , 'GEN-SIM-RECO'                                                      , '' )
        self.addParameter( self._defaultParameters, 'condition'    , 'startup'                                                           , '' )
        self.addParameter( self._defaultParameters, 'globalTag'    , autoCond[ self.getDefaultParameters()[ 'condition' ].value ][ : -5 ], 'auto from \'condition\'' )
        self.addParameter( self._defaultParameters, 'maxVersions'  , 3                                                                   , '' )
        self.addParameter( self._defaultParameters, 'skipFiles'    , 0                                                                   , '' )
        self.addParameter( self._defaultParameters, 'numberOfFiles', -1                                                                  , 'all' )
        self.addParameter( self._defaultParameters, 'debug'        , False                                                               , '' )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def __call__( self
                , useDAS        = None
                , cmsswVersion  = None
                , formerVersion = None
                , relVal        = None
                , dataTier      = None
                , condition     = None
                , globalTag     = None
                , maxVersions   = None
                , skipFiles     = None
                , numberOfFiles = None
                , debug         = None
                ):
        if useDAS is None:
            useDAS = self.getDefaultParameters()[ 'useDAS' ].value
        if cmsswVersion is None:
            cmsswVersion = self.getDefaultParameters()[ 'cmsswVersion' ].value
        if formerVersion is None:
            formerVersion = self.getDefaultParameters()[ 'formerVersion' ].value
        if relVal is None:
            relVal = self.getDefaultParameters()[ 'relVal' ].value
        if dataTier is None:
            dataTier = self.getDefaultParameters()[ 'dataTier' ].value
        if condition is None:
            condition = self.getDefaultParameters()[ 'condition' ].value
        if globalTag is None:
            globalTag = autoCond[ condition ][ : -5 ] # auto from 'condition'
        if maxVersions is None:
            maxVersions = self.getDefaultParameters()[ 'maxVersions' ].value
        if skipFiles is None:
            skipFiles = self.getDefaultParameters()[ 'skipFiles' ].value
        if numberOfFiles is None:
            numberOfFiles = self.getDefaultParameters()[ 'numberOfFiles' ].value
        if debug is None:
            debug = self.getDefaultParameters()[ 'debug' ].value
        self.setParameter( 'useDAS'       , useDAS )
        self.setParameter( 'cmsswVersion' , cmsswVersion )
        self.setParameter( 'formerVersion', formerVersion )
        self.setParameter( 'relVal'       , relVal )
        self.setParameter( 'dataTier'     , dataTier )
        self.setParameter( 'condition'    , condition )
        self.setParameter( 'globalTag'    , globalTag )
        self.setParameter( 'maxVersions'  , maxVersions )
        self.setParameter( 'skipFiles'    , skipFiles )
        self.setParameter( 'numberOfFiles', numberOfFiles )
        self.setParameter( 'debug'        , debug )
        return self.apply()

    def messageEmptyList( self ):
        print '%s DEBUG: Empty file list returned'%( self._label )
        print '    This might be overwritten by providing input files explicitly to the source module in the main configuration file.'

    def apply( self ):
        useDAS        = self._parameters[ 'useDAS'        ].value
        cmsswVersion  = self._parameters[ 'cmsswVersion'  ].value
        formerVersion = self._parameters[ 'formerVersion' ].value
        relVal        = self._parameters[ 'relVal'        ].value
        dataTier      = self._parameters[ 'dataTier'      ].value
        condition     = self._parameters[ 'condition'     ].value # only used for GT determination in initialization, if GT not explicitly given
        globalTag     = self._parameters[ 'globalTag'     ].value
        maxVersions   = self._parameters[ 'maxVersions'   ].value
        skipFiles     = self._parameters[ 'skipFiles'     ].value
        numberOfFiles = self._parameters[ 'numberOfFiles' ].value
        debug         = self._parameters[ 'debug'         ].value

        filePaths = []

        # Determine corresponding CMSSW version for RelVals
        preId      = '_pre'
        patchId    = '_patch'    # patch releases
        hltPatchId = '_hltpatch' # HLT patch releases
        dqmPatchId = '_dqmpatch' # DQM patch releases
        slhcId     = '_SLHC'     # SLHC releases
        rootId     = '_root'     # ROOT test releases
        ibId       = '_X_'       # IBs
        if patchId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( patchId )[ 0 ]
        elif hltPatchId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( hltPatchId )[ 0 ]
        elif dqmPatchId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( dqmPatchId )[ 0 ]
        elif rootId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( rootId )[ 0 ]
        elif slhcId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( slhcId )[ 0 ]
        elif ibId in cmsswVersion or formerVersion:
            outputTuple = Popen( [ 'scram', 'l -c CMSSW' ], stdout = PIPE, stderr = PIPE ).communicate()
            if len( outputTuple[ 1 ] ) != 0:
                print '%s INFO : SCRAM error'%( self._label )
                if debug:
                    print '    from trying to determine last valid releases before \'%s\''%( cmsswVersion )
                    print
                    print outputTuple[ 1 ]
                    print
                    self.messageEmptyList()
                return filePaths
            versions = { 'last'      :''
                       , 'lastToLast':''
                       }
            for line in outputTuple[ 0 ].splitlines():
                version = line.split()[ 1 ]
                if cmsswVersion.split( ibId )[ 0 ] in version or cmsswVersion.rpartition( '_' )[ 0 ] in version:
                    if not ( patchId in version or hltPatchId in version or dqmPatchId in version or slhcId in version or ibId in version or rootId in version ):
                        versions[ 'lastToLast' ] = versions[ 'last' ]
                        versions[ 'last' ]       = version
                        if version == cmsswVersion:
                            break
            # FIXME: ordering of output problematic ('XYZ_pre10' before 'XYZ_pre2', no "formerVersion" for 'XYZ_pre1')
            if formerVersion:
                # Don't use pre-releases as "former version" for other releases than CMSSW_X_Y_0
                if preId in versions[ 'lastToLast' ] and not preId in versions[ 'last' ] and not versions[ 'last' ].endswith( '_0' ):
                    versions[ 'lastToLast' ] = versions[ 'lastToLast' ].split( preId )[ 0 ] # works only, if 'CMSSW_X_Y_0' esists ;-)
                # Use pre-release as "former version" for CMSSW_X_Y_0
                elif versions[ 'last' ].endswith( '_0' ) and not ( preId in versions[ 'lastToLast' ] and versions[ 'lastToLast' ].startswith( versions[ 'last' ] ) ):
                    versions[ 'lastToLast' ] = ''
                    for line in outputTuple[ 0 ].splitlines():
                        version      = line.split()[ 1 ]
                        versionParts = version.partition( preId )
                        if versionParts[ 0 ] == versions[ 'last' ] and versionParts[ 1 ] == preId:
                            versions[ 'lastToLast' ] = version
                        elif versions[ 'lastToLast' ] != '':
                            break
                # Don't use CMSSW_X_Y_0 as "former version" for pre-releases
                elif preId in versions[ 'last' ] and not preId in versions[ 'lastToLast' ] and versions[ 'lastToLast' ].endswith( '_0' ):
                    versions[ 'lastToLast' ] = '' # no alternative :-(
                cmsswVersion = versions[ 'lastToLast' ]
            else:
                cmsswVersion = versions[ 'last' ]

        # Debugging output
        if debug:
            print '%s DEBUG: Called with...'%( self._label )
            for key in self._parameters.keys():
               print '    %s:\t'%( key ),
               print self._parameters[ key ].value,
               if self._parameters[ key ].value is self.getDefaultParameters()[ key ].value:
                   print ' (default)'
               else:
                   print
               if key == 'cmsswVersion' and cmsswVersion != self._parameters[ key ].value:
                   if formerVersion:
                       print '    ==> modified to last to last valid release %s (s. \'formerVersion\' parameter)'%( cmsswVersion )
                   else:
                       print '    ==> modified to last valid release %s'%( cmsswVersion )

        # Check domain
        domain = socket.getfqdn().split( '.' )
        domainSE = ''
        if len( domain ) == 0:
            print '%s INFO : Cannot determine domain of this computer'%( self._label )
            if debug:
                self.messageEmptyList()
            return filePaths
        elif os.uname()[0] == "Darwin":
            print '%s INFO : Running on MacOSX without direct access to RelVal files.'%( self._label )
            if debug:
                self.messageEmptyList()
            return filePaths
        elif len( domain ) == 1:
            print '%s INFO : Running on local host \'%s\' without direct access to RelVal files'%( self._label, domain[ 0 ] )
            if debug:
                self.messageEmptyList()
            return filePaths
        if not ( ( domain[ -2 ] == 'cern' and domain[ -1 ] == 'ch' ) or ( domain[ -2 ] == 'fnal' and domain[ -1 ] == 'gov' ) ):
            print '%s INFO : Running on site \'%s.%s\' without direct access to RelVal files'%( self._label, domain[ -2 ], domain[ -1 ] )
            if debug:
                self.messageEmptyList()
            return filePaths
        if domain[ -2 ] == 'cern':
            domainSE = 'T2_CH_CERN'
        elif domain[ -2 ] == 'fnal':
            domainSE = 'T1_US_FNAL_MSS'
        if debug:
            print '%s DEBUG: Running at site \'%s.%s\''%( self._label, domain[ -2 ], domain[ -1 ] )
            print '%s DEBUG: Looking for SE \'%s\''%( self._label, domainSE )

        # Find files
        validVersion = 0
        dataset    = ''
        datasetAll = '/%s/%s-%s-v*/%s'%( relVal, cmsswVersion, globalTag, dataTier )
        if useDAS:
            if debug:
                print '%s DEBUG: Using DAS query'%( self._label )
            dasLimit = numberOfFiles
            if dasLimit <= 0:
                dasLimit += 1
            for version in range( maxVersions, 0, -1 ):
                filePaths    = []
                filePathsTmp = []
                fileCount    = 0
                dataset = '/%s/%s-%s-v%i/%s'%( relVal, cmsswVersion, globalTag, version, dataTier )
                dasQuery = 'file dataset=%s | grep file.name'%( dataset )
                if debug:
                    print '%s DEBUG: Querying dataset \'%s\' with'%( self._label, dataset )
                    print '    \'%s\''%( dasQuery )
                # partially stolen from das_client.py for option '--format=plain', needs filter ("grep") in the query
                jsondict    = das_client.get_data( 'https://cmsweb.cern.ch', dasQuery, 0, dasLimit, False )
                if debug:
                    print '%s DEBUG: Received DAS JSON dictionary:'%( self._label )
                    print '    \'%s\''%( jsondict )
                if jsondict[ 'status' ] != 'ok':
                    print 'There was a problem while querying DAS with query \'%s\'. Server reply was:\n %s' % (dasQuery, jsondict)
                    exit( 1 )
                mongo_query = jsondict[ 'mongo_query' ]
                filters     = mongo_query[ 'filters' ]
                data        = jsondict[ 'data' ]
                if debug:
                    print '%s DEBUG: Query in JSON dictionary:'%( self._label )
                    print '    \'%s\''%( mongo_query )
                    print '%s DEBUG: Filters in query:'%( self._label )
                    print '    \'%s\''%( filters )
                    print '%s DEBUG: Data in JSON dictionary:'%( self._label )
                    print '    \'%s\''%( data )
                for row in data:
                    filePath = [ r for r in das_client.get_value( row, filters[ 'grep' ] ) ][ 0 ]
                    if debug:
                        print '%s DEBUG: Testing file entry \'%s\''%( self._label, filePath )
                    if len( filePath ) > 0:
                        if validVersion != version:
                            jsontestdict    = das_client.get_data( 'https://cmsweb.cern.ch', 'site dataset=%s | grep site.name'%( dataset ), 0, 999, False )
                            mongo_testquery = jsontestdict[ 'mongo_query' ]
                            testfilters = mongo_testquery[ 'filters' ]
                            testdata    = jsontestdict[ 'data' ]
                            if debug:
                                print '%s DEBUG: Received DAS JSON dictionary (site test):'%( self._label )
                                print '    \'%s\''%( jsontestdict )
                                print '%s DEBUG: Query in JSON dictionary (site test):'%( self._label )
                                print '    \'%s\''%( mongo_testquery )
                                print '%s DEBUG: Filters in query (site test):'%( self._label )
                                print '    \'%s\''%( testfilters )
                                print '%s DEBUG: Data in JSON dictionary (site test):'%( self._label )
                                print '    \'%s\''%( testdata )
                            foundSE = False
                            for testrow in testdata:
                                siteName = [ tr for tr in das_client.get_value( testrow, testfilters[ 'grep' ] ) ][ 0 ]
                                if siteName == domainSE:
                                    foundSE = True
                                    break
                            if not foundSE:
                                if debug:
                                    print '%s DEBUG: Possible version \'v%s\' not available on SE \'%s\''%( self._label, version, domainSE )
                                break
                            validVersion = version
                            if debug:
                                print '%s DEBUG: Valid version set to \'v%i\''%( self._label, validVersion )
                        if numberOfFiles == 0:
                            break
                        # protect from double entries ( 'unique' flag in query does not work here)
                        if not filePath in filePathsTmp:
                            filePathsTmp.append( filePath )
                            if debug:
                                print '%s DEBUG: File \'%s\' found'%( self._label, filePath )
                            fileCount += 1
                            # needed, since and "limit" overrides "idx" in 'get_data' (==> "idx" set to '0' rather than "skipFiles")
                            if fileCount > skipFiles:
                                filePaths.append( filePath )
                        elif debug:
                            print '%s DEBUG: File \'%s\' found again'%( self._label, filePath )
                if validVersion > 0:
                    if numberOfFiles == 0 and debug:
                        print '%s DEBUG: No files requested'%( self._label )
                    break
        else:
            if debug:
                print '%s DEBUG: Using DBS query'%( self._label )
            print '%s WARNING: DBS query disabled for DBS3 transition to new API'%( self._label )
            #for version in range( maxVersions, 0, -1 ):
                #filePaths = []
                #fileCount = 0
                #dataset = '/%s/%s-%s-v%i/%s'%( relVal, cmsswVersion, globalTag, version, dataTier )
                #dbsQuery = 'find file where dataset = %s'%( dataset )
                #if debug:
                    #print '%s DEBUG: Querying dataset \'%s\' with'%( self._label, dataset )
                    #print '    \'%s\''%( dbsQuery )
                #foundSE = False
                #for line in os.popen( 'dbs search --query="%s"'%( dbsQuery ) ).readlines():
                    #if line.find( '.root' ) != -1:
                        #if validVersion != version:
                            #if not foundSE:
                                #dbsSiteQuery = 'find dataset where dataset = %s and site = %s'%( dataset, domainSE )
                                #if debug:
                                    #print '%s DEBUG: Querying site \'%s\' with'%( self._label, domainSE )
                                    #print '    \'%s\''%( dbsSiteQuery )
                                #for lineSite in os.popen( 'dbs search --query="%s"'%( dbsSiteQuery ) ).readlines():
                                    #if lineSite.find( dataset ) != -1:
                                        #foundSE = True
                                        #break
                            #if not foundSE:
                                #if debug:
                                    #print '%s DEBUG: Possible version \'v%s\' not available on SE \'%s\''%( self._label, version, domainSE )
                                #break
                            #validVersion = version
                            #if debug:
                                #print '%s DEBUG: Valid version set to \'v%i\''%( self._label, validVersion )
                        #if numberOfFiles == 0:
                            #break
                        #filePath = line.replace( '\n', '' )
                        #if debug:
                            #print '%s DEBUG: File \'%s\' found'%( self._label, filePath )
                        #fileCount += 1
                        #if fileCount > skipFiles:
                            #filePaths.append( filePath )
                        #if not numberOfFiles < 0:
                            #if numberOfFiles <= len( filePaths ):
                                #break
                #if validVersion > 0:
                    #if numberOfFiles == 0 and debug:
                        #print '%s DEBUG: No files requested'%( self._label )
                    #break

        # Check output and return
        if validVersion == 0:
            print '%s WARNING : No RelVal file(s) found at all in datasets \'%s*\' on SE \'%s\''%( self._label, datasetAll, domainSE )
            if debug:
                self.messageEmptyList()
        elif len( filePaths ) == 0:
            print '%s WARNING : No RelVal file(s) picked up in dataset \'%s\''%( self._label, dataset )
            if debug:
                self.messageEmptyList()
        elif len( filePaths ) < numberOfFiles:
            print '%s INFO : Only %i RelVal file(s) instead of %i picked up in dataset \'%s\''%( self._label, len( filePaths ), numberOfFiles, dataset )

        if debug:
            print '%s DEBUG: returning %i file(s):\n%s'%( self._label, len( filePaths ), filePaths )
        return filePaths

pickRelValInputFiles = PickRelValInputFiles()
