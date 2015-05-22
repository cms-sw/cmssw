#!/usr/bin/env python
## Author: Anastasios Antoniadis

import sys, os, getpass, pprint, ast, re, tarfile, shutil, shelve, anydbm, optparse
import CMGTools.Production.eostools as eostools
from CMGTools.Production.cmgdbToolsApi import CmgdbToolsApi
from CMGTools.Production.findDSOnSav import validLogin
from optparse import *

class ValidationRunInfo( object ):

    def __init__( self, valRunDir, components_dict, analyzers_dict, 
                  release_info_dict, root_files_dict ):
        self.valRunDir = valRunDir
        self.components = components_dict
        self.analyzers = analyzers_dict
        self.release_info = release_info_dict
        self.root_files_info = root_files_dict
    
    def printComponents( self ):
        pp = pprint.PrettyPrinter( indent = 4 )
        pp.pprint( self.components )

    def printAnalyzers( self ):
        pp = pprint.PrettyPrinter( indent = 4 )
        pp.pprint( self.analyzers )

    def printReleaseInfo( self ):
        pp = pprint.PrettyPrinter( indent = 4 )
        pp.pprint( self.release_info )

    def printRootFilesInfo( self ):
        pp = pprint.PrettyPrinter( indent = 4 )
        pp.pprint( self.root_files_info )
        
    def getComponentsInfo( self ):
        return self.components

    def getAnalyzersInfo( self ):
        return self.analyzers

    def getReleaseInfo( self ):
        return self.release_info

    def getRootFilesInfo( self ):
        return self.root_files_info

    def getValRunDir( self ):
        return self.valRunDir

def getValidationRunInfoFromDisk( validationRunPath ):
    
    os.chdir( validationRunPath )     
    component_dirs = [ name for name in os.listdir(".") if os.path.isdir(name) and name != "Logger"]
    analyzers = []       
    check_analyzers = "yes"
    
    components = dict()
    root_files = dict()
    analyzers = dict()
    release_info = dict()

    for component_dir in component_dirs:
        log_file = os.path.join( component_dir, "log.txt" )

        with open(log_file, 'r') as f: # here I am reading each component's log file to get all the component and analyzer info
            target_type = "";
            target_name = "";
            
            for line in f:
                if ':' in line:
                    line = [ x.strip() for x in line.split( ':', 1 ) ]
                    if line[0] == "MCComponent":
                        if target_type == "":
                            continue
                        target_type = "component"
                        target_name = component_dir
                        components[component_dir] = dict()
                    
                    elif line[0] == "Analyzer":
                        if check_analyzers == "no":
                            break
                        target_type = "analyzer"
                        target_name = line[1]
                        analyzers[line[1]] = dict()
                    
                    else:
                        if target_type == "component":
                            if line[0] == "files":
                                 components[target_name][line[0]] = ast.literal_eval( line[1] )
                                 continue
                            components[target_name][line[0]] = line[1]
                        elif target_type == "analyzer":
                            analyzers[target_name][line[0]] = line[1]
            check_analyzers == "no"
            
        root_files[component_dir] = dict()
        for key in analyzers.keys():
            analyzer_dir_path = os.path.join( component_dir, key )
            root_files[component_dir][key] = [name for name in os.listdir( analyzer_dir_path ) \
                                                if os.path.isfile( os.path.join( analyzer_dir_path, name ) ) and re.compile( ".*\.root" ).match( name ) ]
        components[target_name]["dataset_on_eos"] = os.path.split("/"+components[target_name]["files"][0].split("//")[2] )[0]
        
    #Here I am handling the information of the logger file
    try:
    	tar = tarfile.open( "Logger.tgz" )
    	tar.extract( "Logger/logger_showtags.txt" )
    	tar.close()
    except:
	print "ERROR - Logger.tgz file not found, please add it to the directory you want to publish"
	exit( -1 )    
    try:
        showtagsFile = open( "Logger/logger_showtags.txt", 'r' )
        lines = showtagsFile.readlines()
        showtagsFile.close()
    except:
        print "ERROR: No showtags file found in logger"
        exit( -1 )

    release_info['Release'] = lines[0].split(":")[1].lstrip().rstrip()      # Get the release from the first line of showtags
    tagPattern = re.compile('^\s*(\S+)\s+(\S+)\s*$')   # Creates regexp to test incoming lines from 'showtags'
    tags = []

    for line in lines:
        m = tagPattern.match(line)  # Check if it is matches the tag pattern
        if m != None:
            package = m.group(2)    # get package name
            tag = m.group(1)        # get tag name
            if tag is not "NoCVS" and tag is not "NoTag":
                tags.append( {"package":package, "tag":tag } )

    release_info['Tags'] = tags
    os.system( "rm Logger/logger_showtags.txt" )
    os.system( "rm -rf Logger" )

    return ValidationRunInfo( "", components, analyzers, release_info, root_files )
    
def addInformationToCMGDB( dir_name, valRunInfo, development=False ):
    validationRunsLibraryPath = "/afs/cern.ch/user/a/anantoni/www/cmg-compare-validation-runs/ValidationRuns"
    cmgdbAPI = CmgdbToolsApi(development)
    cmgdbAPI.connect()

    #get all the information from the validation run object
    release_info = valRunInfo.getReleaseInfo()
    components_info = valRunInfo.getComponentsInfo()
    analyzers_info = valRunInfo.getAnalyzersInfo()
    root_files_info = valRunInfo.getRootFilesInfo()
    
    taghash = []
    for i in release_info['Tags']:
        a = hash( ( i['package'], i['tag'] ) )       # Create hash code for the tag set
        taghash.append(a)
    taghash.sort()
    endhash = hash( tuple( taghash ) )
		
    tagSetID = cmgdbAPI.getTagSetID( endhash )       # check if tag set is already on CMGDB
   
    if tagSetID is None:         # If it isn't found, add the tags, and the tag set
        if release_info['Tags']:
            tagIDs = []
            for row in release_info['Tags']:
                tagID = cmgdbAPI.addTag( row["package"],
                                         row["tag"] )
                if tagID is not None:
                    tagIDs.append( tagID )
			
            tagSetID = cmgdbAPI.addTagSet( release_info['Release'],
                                           endhash )
            for tagID in tagIDs:
                cmgdbAPI.addTagToSet( tagID, tagSetID )
			    	
    if tagSetID is not None: 
        validationRunID = cmgdbAPI.addValidationRun( tagSetID,
                                                     components_info[components_info.keys()[0]]['number of events processed'] )
    for component in components_info.keys():
        for analyzer in analyzers_info.keys():
            analyzerID = cmgdbAPI.addAnalyzer( analyzer )
            for root_file in root_files_info[component][analyzer]:
                
                datasetInfo = cmgdbAPI.addRootFilesToValidationRunWithAnalyzerOnDataset( validationRunID, 
                                                                           components_info[component]['dataset_on_eos'],
                                                                           component,
                                                                           analyzer, 
                                                                           root_file )
                destinationPath = os.path.join( validationRunsLibraryPath, 
                                                datasetInfo[1], 
                                                datasetInfo[2][1:], 
                                                repr(validationRunID) )
                os.system( "mkdir -p " + destinationPath  )
                os.system( "cp -r " + component + " " + destinationPath )
    
    
if __name__ == '__main__':
    parser = optparse.OptionParser()
   
    parser.usage = """
    %prg [options] <dir_name>	

    Use this script to publish validation run directories to the database.
    Example:
    publishValidationRunInfo.py [--dev] ./Val_run_dir
    """

    group = OptionGroup( parser, "Publish validation run info options", """These options affect the way you publish to Savannah and CMGDB""" )
    group.add_option( "-d", "--dev",
			action="store_true",
			dest="development",
			help="Choose between publishing to the official or development database",
			default=False )
    group.add_option("-u", "--username",
                        action = "store",
                        dest="username",
                        help="""Specify the username to access both the DBS and savannah servers.Default is $USER.""",
                        default=os.environ['USER'] )
    group.add_option("-p", "--password",
                        action = "store",
                        dest="password",
                        help="""Specify the password to access both the DBS and savannah servers.""",
		        default=None)

    parser.add_option_group( group )
    (options, args) = parser.parse_args()
    if len(args) != 1:
	parser.print_help()
	sys.exit(-1) 	

    if options.password == None:
        try:
            password = getpass.getpass("Enter NICE Password: ")
        except KeyboardInterrupt:
            print "Authentication Failed, exiting\n\n"
            sys.exit(1)
        options.password = password
    else:
	password = options.password

    if not validLogin(options.username, password):
        print "Authentication Failed, exiting\n\n"
        sys.exit(-1)

    dir_name = args[0]
    valRunInfo = getValidationRunInfoFromDisk( dir_name )
    valRunInfo.printComponents()
    valRunInfo.printAnalyzers()
    valRunInfo.printReleaseInfo()
    valRunInfo.printRootFilesInfo()
    addInformationToCMGDB( dir_name, valRunInfo, options.development )
    persistentObject = anydbm.open( "self.db", 'n')
    persistentObject.close()
    persistentObject = shelve.open( "self.db" )
    persistentObject["valRunInfo"] = valRunInfo
    persistentObject.close()


