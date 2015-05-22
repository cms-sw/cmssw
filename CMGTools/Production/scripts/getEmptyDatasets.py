#!/usr/bin/env python
## Author: Anastasios Antoniadis

import optparse
import CMGTools.Production.cx_Oracle as cx_Oracle
from CMGTools.Production.cmgdbApi import CmgdbApi
from CMGTools.Production.castorBaseDir import castorBaseDir
import CMGTools.Production.eostools as castortools
from optparse import *

if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.usage = """
    %prg [options] 

    Use this script to create a file with all the empty datasets on eos which are marked as open in the database.
    Example:
    publishValidationRunInfo.py [--dev --file emptyDatasets.txt] 
    """

    group = OptionGroup( parser, "options" )
    group.add_option( "-d", "--dev",
                        action="store_true",
                        dest="development",
                        help="Choose between publishing to the official or development database",
                        default=False )
    group.add_option("-f", "--file",
                        action = "store",
                        dest="filename",
                        help="""Specify the name of the file to be created""",
                        default="empty_datasets.txt")
    
    parser.add_option_group( group )
    (options, args) = parser.parse_args()

    cmgdbApi = CmgdbApi(options.development)
    cmgdbApi.connect()
    
    columns, rows = cmgdbApi.sql( "select file_owner, path_name from dataset_details where dataset_is_open='Y' and (file_owner='cmgtools' or file_owner='cmgtools_group')" )

    f = open( options.filename,'w')
    
    for row in rows:
        fileown = row[0]
        dsName = row[1]
        
	if fileown == "--cmgtools":
	    fileown = "cmgtools"
	
        lfnDir = castorBaseDir(fileown) + dsName
        castorDir = castortools.lfnToCastor( lfnDir )

        if castortools.datasetNotEmpty( castorDir, ".*root" ):
            continue
        else:
            f.write( fileown + "%" + dsName  + '\n' )
            
    f.close()
    
