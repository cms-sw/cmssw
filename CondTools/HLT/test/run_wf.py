#!/usr/bin/env python

"""
Example script to test reading from local sqlite db.
"""
from __future__ import print_function
import os
import sys
import ast
import optparse
import hashlib
import tarfile
import netrc
import getpass
import errno
import sqlite3
import json
import tempfile
import CondCore.Utilities.conddblib as conddb 

##############################################
def getCommandOutput(command):
##############################################
    """This function executes `command` and returns it output.
    Arguments:
    - `command`: Shell command to be invoked by this function.
    """
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        print('%s failed w/ exit code %d' % (command, err))
    return data

##############################################
def get_iovs(db, tag):
##############################################
    """Retrieve the list of IOVs from `db` for `tag`.

    Arguments:
    - `db`: database connection string
    - `tag`: tag of database record
    """

    ### unfortunately some gymnastics is needed here
    ### to make use of the conddb library

    officialdbs = { 
        # frontier connections
        'frontier://PromptProd/CMS_CONDITIONS'   :'pro',  
        'frontier://FrontierProd/CMS_CONDITIONS' :'pro', 
        'frontier://FrontierArc/CMS_CONDITIONS'  :'arc',            
        'frontier://FrontierInt/CMS_CONDITIONS'  :'int',            
        'frontier://FrontierPrep/CMS_CONDITIONS' :'dev', 
        }
    
    if db in officialdbs.keys():
        db = officialdbs[db]

    ## allow to use input sqlite files as well
    db = db.replace("sqlite_file:", "").replace("sqlite:", "")

    con = conddb.connect(url = conddb.make_url(db))
    session = con.session()
    IOV = session.get_dbtype(conddb.IOV)

    iovs = set(session.query(IOV.since).filter(IOV.tag_name == tag).all())
    if len(iovs) == 0:
        print("No IOVs found for tag '"+tag+"' in database '"+db+"'.")
        sys.exit(1)

    session.close()

    return sorted([int(item[0]) for item in iovs])


##############################################
def main():
##############################################
     
     defaultDB     = 'sqlite_file:mySqlite.db'
     defaultInTag  = 'myInTag'
     defaultOutTag = 'myOutTag'

     parser = optparse.OptionParser(usage = 'Usage: %prog [options] <file> [<file> ...]\n')
     
     parser.add_option('-f', '--inputDB',
                       dest = 'inputDB',
                       default = defaultDB,
                       help = 'file to inspect',
                       )
     
     parser.add_option('-i', '--inputTag',
                       dest = 'InputTag',
                       default = defaultInTag,
                       help = 'tag to be inspected',
                       )

     parser.add_option('-d', '--destTag',
                       dest = 'destTag',
                       default = defaultOutTag,
                       help = 'tag to be written',
                       )

     parser.add_option("-C", '--clean', 
                       dest="doTheCleaning",
                       action="store_true",
                       default = True,
                       help = 'if true remove the transient files',
                       )

     (options, arguments) = parser.parse_args()

     sqlite_db_url = options.inputDB
     if(".db" in sqlite_db_url):
         sqlite_db_url = "sqlite_file:{0}".format(sqlite_db_url)

     ##########
     # code to get it working on sqlite files without CMSSW

     # db = sqlite3.connect(sqlite_db_url)
     # cursor = db.cursor()
     # cursor.execute("SELECT * FROM IOV;")
     # IOVs = cursor.fetchall()
     # sinces = []
     # for element in IOVs:
     #    sinces.append(element[1])

     sinces = get_iovs(options.inputDB,options.InputTag)

     print(sinces)

     for i,since in enumerate(sinces):
          #print i,since
         
         print("============================================================")
         if(i<len(sinces)-1):
             command = 'conddb_import -c sqlite_file:'+options.InputTag+"_IOV_"+str(sinces[i])+'.db -f '+sqlite_db_url+" -i "+options.InputTag+" -t "+options.InputTag+" -b "+str(sinces[i])+" -e "+str(sinces[i+1]-1)
             print(command)
             getCommandOutput(command)
         else:
             command = 'conddb_import -c sqlite_file:'+options.InputTag+"_IOV_"+str(sinces[i])+'.db -f '+sqlite_db_url+" -i "+options.InputTag+" -t "+options.InputTag+" -b "+str(sinces[i])
             print(command)
             getCommandOutput(command)
             
         # update the trigger bits
             
         cmsRunCommand="cmsRun AlCaRecoTriggerBitsRcdUpdate_TEMPL_cfg.py inputDB=sqlite_file:"+options.InputTag+"_IOV_"+str(sinces[i])+".db inputTag="+options.InputTag+" outputDB=sqlite_file:"+options.InputTag+"_IOV_"+str(sinces[i])+"_updated.db outputTag="+options.destTag+" firstRun="+str(sinces[i])
         print(cmsRunCommand)
         getCommandOutput(cmsRunCommand)
     
         # merge the output
          
         mergeCommand = 'conddb_import -f sqlite_file:'+options.InputTag+"_IOV_"+str(sinces[i])+'_updated.db -c sqlite_file:'+options.destTag+".db -i "+options.destTag+" -t "+options.destTag+" -b "+str(sinces[i])
         print(mergeCommand)
         getCommandOutput(mergeCommand)

         # clean the house
         
         if(options.doTheCleaning):
             cleanCommand = 'rm -fr *updated*.db *IOV_*.db'
             getCommandOutput(cleanCommand)
         else:
             print("======> keeping the transient files")
          

if __name__ == "__main__":        
     main()
