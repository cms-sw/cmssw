#!/usr/bin/env python

"""
Example script to test reading from local sqlite db.
"""
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
from CondCore.Utilities.CondDBFW import querying

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
        print '%s failed w/ exit code %d' % (command, err)
    return data

##############################################
def main():
##############################################
     
     defaultFile  = 'mySqlite.db'
     defaultInTag = 'myInTag'
     defaultOutTag= 'myOutTag'

     parser = optparse.OptionParser(usage = 'Usage: %prog [options] <file> [<file> ...]\n')
     
     parser.add_option('-f', '--file',
                       dest = 'file',
                       default = defaultFile,
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

     parser.add_option('-C', '--clean',
                       dest = 'doTheCleaning',
                       default = True,
                       help = 'if true remove the transient files',
                       )

     (options, arguments) = parser.parse_args()

     sqlite_db_url = options.file
     db = sqlite3.connect(sqlite_db_url)
     cursor = db.cursor()
     cursor.execute("SELECT * FROM IOV;")
     IOVs = cursor.fetchall()
     sinces = []

     for element in IOVs:
          sinces.append(element[1])

     print sinces

     for i,since in enumerate(sinces):
          #print i,since

          print "============================================================"
          if(i<len(sinces)-1):
               command = 'conddb_import -c sqlite_file:'+sqlite_db_url.rstrip(".db")+"_IOV_"+str(sinces[i])+'.db -f sqlite_file:'+sqlite_db_url+" -i "+options.InputTag+" -t "+options.InputTag+" -b "+str(sinces[i])+" -e "+str(sinces[i+1]-1)
               print command
               getCommandOutput(command)
          else:
               command = 'conddb_import -c sqlite_file:'+sqlite_db_url.rstrip(".db")+"_IOV_"+str(sinces[i])+'.db -f sqlite_file:'+sqlite_db_url+" -i "+options.InputTag+" -t "+options.InputTag+" -b "+str(sinces[i])
               print command
               getCommandOutput(command)
               
          # update the trigger bits

          cmsRunCommand="cmsRun AlCaRecoTriggerBitsRcdUpdate_TEMPL_cfg.py inputDB=sqlite_file:"+sqlite_db_url.rstrip(".db")+"_IOV_"+str(sinces[i])+".db inputTag="+options.InputTag+" outputDB=sqlite_file:"+sqlite_db_url.rstrip(".db")+"_IOV_"+str(sinces[i])+"_updated.db outputTag="+options.destTag+" firstRun="+str(sinces[i])
          print cmsRunCommand
          getCommandOutput(cmsRunCommand)
     
          # merge the output
          
          mergeCommand = 'conddb_import -f sqlite_file:'+sqlite_db_url.rstrip(".db")+"_IOV_"+str(sinces[i])+'_updated.db -c sqlite_file:'+options.destTag+".db -i "+options.destTag+" -t "+options.destTag+" -b "+str(sinces[i])
          print mergeCommand
          getCommandOutput(mergeCommand)

          # clean the house
     
          if(ast.literal_eval(options.doTheCleaning)):
              cleanCommand = 'rm -fr *updated*.db *IOV_*.db'
              getCommandOutput(cleanCommand)
          else:
              print "======> keeping the transient files"
          

if __name__ == "__main__":        
     main()
