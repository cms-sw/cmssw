#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## November 2nd 2011
# This script serves to query the CMGDB database
import os, sys, re, optparse
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/oracle/11.2.0.1.0p2/etc/profile.d/init.sh")
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/python/2.6.4-cms16/etc/profile.d/init.sh")
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/py2-cx-oracle/5.1/etc/profile.d/init.sh") 
import CMGTools.Production.cx_Oracle as cx_Oracle
from CMGTools.Production.cmgdbApi import CmgdbApi

# Make sure is being used as a script
if __name__ == '__main__':
    parser = optparse.OptionParser()
    cmgdbApi = CmgdbApi()
    cmgdbApi.connect()
    description = cmgdbApi.describe()
    parser.usage = """
Table Structure:
"""+description+"""

The database server being used is Oracle 11g, so a good idea would be to become farmiliar with the Oracle 11g query semantics.

Here are some example queries explained:
    If you want to get a list of tags that were used by cmgtools on the 7th March 2012, 
    - You will have to first make sure that you use the distinct() method to prevent repetitions
    - Once you have specified the information you want, you will then need to perform a join with the tags_in_sets table on tag_id
    - From the product of this join you then want to join with the dataset_details table, this time on tagset_id
    - You then specify the WHERE clause as where file_owner = 'cmgtools and the date...
    - To give the results of the entire day, you must do two things,
    -- First, use the trunc() method to truncate the timestamp to the desired length
    -- Then, use the to_timestamp() method in order to change the date you want to a timestamp

        "SELECT distinct(tags.tag_id), tags.tag, tags.package_name FROM tags INNER JOIN tags_in_sets ON tags_in_sets.tag_id = tags.tag_id JOIN dataset_details ON dataset_details.tagset_id = tags_in_sets.tagset_id WHERE dataset_details.file_owner = 'cmgtools' AND trunc(dataset_details.date_recorded) = to_timestamp('07-03-2012','DD-MM-YYYY') ORDER BY tags.tag_id"
    
    If you want to get a list of dataset names from the same time period, again from cmgtools,
    - You would first select the details you want to see, e.g dataset_fraction
    - Then you would use the same WHERE clause as in the previous example
        
        "SELECT dataset_id, path_name, date_recorded, dataset_fraction FROM dataset_details WHERE trunc(date_recorded) = to_timestamp('07-03-2012','DD-MM-YYYY') AND file_owner = 'cmgtools' ORDER BY dataset_id"
        
    If you want a list of datasets that used a certain tag
    - First specify the fields you want to select, in this case dataset id and name
    - Perform an INNER JOIN to on tagset_id to get
    - Then perform another join with the tags table, and specify the WHERE clause
    - Finally specify what to order the list by
    
        "SELECT dataset_details.dataset_id, dataset_details.path_name FROM dataset_details INNER JOIN tags_in_sets on tags_in_sets.tagset_id = dataset_details.tagset_id JOIN tags on tags.tag_id = tags_in_sets.tag_id WHERE tags.tag = 'B4_2_X_V00-03-00' AND tags.package_name = 'CommonTools/ParticleFlow' ORDER BY dataset_details.dataset_id"
    
    If you want to find the missing files on a particular dataset
    - First specify what you want to select
    - Then join on dataset id
    - Then specify which dataset
        
        "SELECT missing_files.missing_file from missing_files INNER JOIN dataset_details on dataset_details.dataset_id = missing_files.dataset_id WHERE path_name = '/QCD_Pt-20to30_EMEnriched_TuneZ2_7TeV-pythia6/Fall11-PU_S6_START44_V9B-v1--V3---cmgtools_group/AODSIM'"

Usage: 
-----
%prog -s <query>
%prog -a <args>
-----

Suggestions for more useful alias' are always welcome
Please experiment and email Peter Meckiffe with your suggestions for alias'
Currently the list is as follows:
getTags <path_name>
    SELECT distinct(tags.tag_id), tags.tag, tags.package_name from tags INNER JOIN tags_in_sets ON tags.tag_id = tags_in_sets.tag_id JOIN dataset_details ON dataset_details.tagset_id = tags_in_sets.tagset_id WHERE dataset_details.path_name = 'ARG1' ORDER BY tags.tag_id
getDatasetsAtDate <DD-MM-YYYY>
    SELECT distinct(dataset_id), path_name FROM dataset_details WHERE trunc(date_recorded) = TO_TIMESTAMP('ARG1','DD-MM-YYYY') ORDER BY dataset_id
getDatasetsAtDateWithUser <DD-MM-YYYY> <fileowner>
    SELECT distinct(dataset_id), path_name FROM dataset_details WHERE trunc(date_recorded) = TO_TIMESTAMP('ARG1','DD-MM-YYYY') and file_owner = 'ARG2' ORDER BY dataset_id
getDatasetsWithOwner <fileowner>
    SELECT distinct(dataset_id), path_name FROM dataset_details WHERE file_owner = 'ARG1' ORDER BY dataset_id
getMissingFiles <path_name>
    SELECT distinct(missing_files.missing_file) FROM missing_files INNER JOIN dataset_details ON dataset_details.dataset_id = missing_files.dataset_id WHERE dataset_details.path_name = 'ARG1'
getDuplicateFiles <path_name>
    SELECT distinct(duplicate_files.duplicate_file) FROM duplicate_files INNER JOIN dataset_details ON dataset_details.dataset_id = duplicate_files.dataset_id WHERE dataset_details.path_name = 'ARG1'
getBadJobs <path_name>
    SELECT distinct(bad_jobs.bad_job) FROM bad_jobs INNER JOIN dataset_details ON dataset_details.dataset_id = bad_jobs.bad_job WHERE dataset_details.path_name = 'ARG1'
getBadFiles <path_name>
    SELECT distinct(bad_files.bad_file) FROM bad_files INNER JOIN dataset_details ON dataset_details.dataset_id = bad_files.dataset_id WHERE dataset_details.path_name = 'ARG1'
getDatasetInfo <path_name>
    SELECT path_name, lfn, file_owner, dataset_entries, dataset_fraction, date_recorded FROM dataset_details WHERE path_name = 'ARG1'
getDatasetsMadeWithSameTagset <path_name>
    SELECT distinct(dataset_id), tagset_id, path_name FROM dataset_details WHERE tagset_id in (SELECT tagset_id FROM dataset_details WHERE path_name = 'ARG1')
    
e.g.
getInfo.py -a getTags /QCD_Pt-20to30_EMEnriched_TuneZ2_7TeV-pythia6/Fall11-PU_S6_START44_V9B-v1/AODSIM/V3

"""
    parser.add_option("-s", "--sql",
                      action = "store_true",
                      dest="sql",
                      help="Enter a raw sql query for cmgdb"
                      )
    parser.add_option("-a", "--alias",
                      action = "store",
                      dest="alias",
                      help="Enter query alias"
                      )

    (options, args) = parser.parse_args()
    
   
        
    
    # Dict of query alias'
    aliasDict = {"getTags":"SELECT distinct(tags.tag_id), tags.tag, tags.package_name from tags INNER JOIN tags_in_sets ON tags.tag_id = tags_in_sets.tag_id JOIN dataset_details ON dataset_details.tagset_id = tags_in_sets.tagset_id WHERE dataset_details.path_name = 'ARG1' ORDER BY tags.tag_id",
                 "getDatasetsAtDate":"SELECT distinct(dataset_id), path_name FROM dataset_details WHERE trunc(date_recorded) = TO_TIMESTAMP('ARG1','DD-MM-YYYY') ORDER BY dataset_id",
                 "getDatasetsAtDateWithOwner":"SELECT distinct(dataset_id), path_name FROM dataset_details WHERE trunc(date_recorded) = TO_TIMESTAMP('ARG1','DD-MM-YYYY') and file_owner = 'ARG2' ORDER BY dataset_id",
                 "getDatasetsWithOwner":"SELECT distinct(dataset_id), path_name FROM dataset_details WHERE file_owner = 'ARG1' ORDER BY dataset_id",
                 "getMissingFiles":"SELECT distinct(missing_files.missing_file) FROM missing_files INNER JOIN dataset_details ON dataset_details.dataset_id = missing_files.dataset_id WHERE dataset_details.path_name = 'ARG1'",
                 "getDuplicateFiles":"SELECT distinct(duplicate_files.duplicate_file) FROM duplicate_files INNER JOIN dataset_details ON dataset_details.dataset_id = duplicate_files.dataset_id WHERE dataset_details.path_name = 'ARG1'",
                 "getBadJobs":"SELECT distinct(bad_jobs.bad_job) FROM bad_jobs INNER JOIN dataset_details ON dataset_details.dataset_id = bad_jobs.bad_job WHERE dataset_details.path_name = 'ARG1'",
                 "getBadFiles":"SELECT distinct(bad_files.bad_file) FROM bad_files INNER JOIN dataset_details ON dataset_details.dataset_id = bad_files.dataset_id WHERE dataset_details.path_name = 'ARG1'",
                 "getDatasetInfo":"SELECT path_name, lfn, file_owner, dataset_entries, dataset_fraction, date_recorded FROM dataset_details WHERE path_name = 'ARG1'",
                 "getDatasetsMadeWithSameTagset":"SELECT distinct(dataset_id), tagset_id, path_name FROM dataset_details WHERE tagset_id in (SELECT tagset_id FROM dataset_details WHERE path_name = 'ARG1')"}

    # If there are no options selected exit.
    if options.sql is None and options.alias is None:
        parser.print_help()
        sys.exit(1)
    # Allow no less than one argument
    if len(args)<1:
        parser.print_help()
        sys.exit(1)

    
    # If its an SQL query, take 1st arg as query from command line
    if options.sql:
        query = args[0]
        
        # Check that only SELECT statements are being used
        ## TOTO: Make sure this is specified in the Oracle account
        select = re.compile('select', re.IGNORECASE)
        if not select.search(query):
            print "getDataset.py is for search uses only (SELECT queries). To publish, please use the publish.py script"
            sys.exit(1)
    # If an alias is specified    
    elif options.alias:
        # Check the alias is valid
        if options.alias in aliasDict:
            # Check the user has entered the correct number of arguments
            if len(args) != len(aliasDict[options.alias].split("ARG"))-1:
                print "Please use the correct amount of arguments %d are required in this alias" % (len(aliasDict[options.alias].split("ARG"))-1)
                sys.exit(1)
            
            # Sub Argument 1 into the query string
            query = re.sub("ARG1",args[0],aliasDict[options.alias])
            # Sub Argument 2
            if re.search("ARG2",aliasDict[options.alias]) and len(args)>1:
                query = re.sub("ARG2",args[1],query)
        
        # Alias was invalid
        else:
            print "Alias %s was not found current alias' are:" % options.alias
            for i in aliasDict:
                print i
            sys.exit(1)
    
    # Execute the Query
      
    columns, rows = cmgdbApi.sql(query)
    
    # Print out the column names
    colnames = ""
    for column in columns:
        colnames += str(column) + "\t"
    print colnames
    
    # Print out the results
    for row in rows:
        string = ""
        for column in row:
            string += str(column) + " ||\t"
        string = string.rstrip(" ||\t")
        print string
    
    
 
