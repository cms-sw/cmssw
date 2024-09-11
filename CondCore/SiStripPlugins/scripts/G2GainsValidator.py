from __future__ import print_function 

from datetime import datetime
import ROOT
import configparser as ConfigParser
import datetime
import glob
import json
import numpy
import optparse
import os
import re
import sqlalchemy
import string
import subprocess
import sys
import time
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
        print ('%s failed w/ exit code %d' % (command, err))
    return data

##############################################
def getCerts() -> str:
##############################################
    cert_path = os.getenv('X509_USER_CERT', '')
    key_path = os.getenv('X509_USER_KEY', '')

    certs = ""
    if cert_path:
        certs += f' --cert {cert_path}'
    else:
        print("No certificate, nor proxy provided for Tier0 access")
    if key_path:
        certs += f' --key {key_path}'
    return certs

##############################################
def build_curl_command(url, proxy="", certs="", timeout=30, retries=3, user_agent="MyUserAgent"):
##############################################
    """Builds the curl command with the appropriate proxy, certs, and options."""
    cmd = f'/usr/bin/curl -k -L --user-agent "{user_agent}" '

    if proxy:
        cmd += f'--proxy {proxy} '
    else:
        cmd += f'{certs} '

    cmd += f'--connect-timeout {timeout} --retry {retries} {url}'
    return cmd

##############################################
def get_hlt_fcsr(session):
##############################################
    RunInfo = session.get_dbtype(conddb.RunInfo)
    lastRun = session.query(sqlalchemy.func.max(RunInfo.run_number)).scalar()
    fcsr = lastRun+1
    return int(fcsr)

##############################################
def getFCSR(proxy="", certs=""):
##############################################
    url = "https://cmsweb.cern.ch/t0wmadatasvc/prod/firstconditionsaferun"
    cmd = build_curl_command(url, proxy=proxy, certs=certs)
    out = subprocess.check_output(cmd, shell=True)
    response = json.loads(out)["result"][0]
    return int(response)

##############################################
def getPromptGT(proxy="", certs=""):
##############################################
    url = "https://cmsweb.cern.ch/t0wmadatasvc/prod/reco_config"
    cmd = build_curl_command(url, proxy=proxy, certs=certs)
    out = subprocess.check_output(cmd, shell=True)
    response = json.loads(out)["result"][0]['global_tag']
    return response

##############################################
def getExpressGT(proxy="", certs=""):
##############################################
    url = "https://cmsweb.cern.ch/t0wmadatasvc/prod/express_config"
    cmd = build_curl_command(url, proxy=proxy, certs=certs)
    out = subprocess.check_output(cmd, shell=True)
    response = json.loads(out)["result"][0]['global_tag']
    return response

##############################################
def resetSynchonization(db_name):
##############################################
    import sqlite3

    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Execute the SQL command to update the database
    cursor.execute("UPDATE TAG SET SYNCHRONIZATION='any' WHERE SYNCHRONIZATION='express';")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

##############################################
if __name__ == "__main__":
##############################################

    parser = optparse.OptionParser(usage = 'Usage: %prog [options] <file> [<file> ...]\n')
     
    parser.add_option('-t', '--validationTag',
                      dest = 'validationTag',
                      default = "SiStripApvGainAfterAbortGap_PCL_multirun_v0_prompt",
                      help = 'validation tag',
                      )
     
    parser.add_option('-s', '--since',
                      dest = 'since',
                      default = -1,
                      help = 'sinces to copy from validation tag',
                      )

    parser.add_option('-p', '--proxy',
                      dest = 'proxy',
                      default = "",
                      help = 'proxy to use for curl requests',
                      )

    parser.add_option('-u', '--user-mode',
                      dest='user_mode',
                      action='store_true',
                      default=False,
                      help='Enable user mode with specific X509 user certificate and key')

    (options, arguments) = parser.parse_args()

    if options.user_mode:
        os.environ['X509_USER_KEY'] = os.path.expanduser('~/.globus/userkey.pem')
        os.environ['X509_USER_CERT'] = os.path.expanduser('~/.globus/usercert.pem')
        print("User mode enabled. Using X509_USER_KEY and X509_USER_CERT from ~/.globus/")

    certs = ""
    if not options.proxy:
        certs = getCerts()

    FCSR = getFCSR(proxy=options.proxy, certs=certs)
    promptGT  = getPromptGT(proxy=options.proxy, certs=certs)
    expressGT = getExpressGT(proxy=options.proxy, certs=certs)

    con = conddb.connect(url = conddb.make_url("pro"))
    session = con.session()

    HLTFCSR = get_hlt_fcsr(session)
    print ("Current next HLT run", HLTFCSR, "| curret FCSR:", FCSR ,"| Express Global Tag",expressGT,"| Prompt Global Tag",promptGT)
    IOV     = session.get_dbtype(conddb.IOV)
    TAG     = session.get_dbtype(conddb.Tag)
    GT      = session.get_dbtype(conddb.GlobalTag)
    GTMAP   = session.get_dbtype(conddb.GlobalTagMap)
    RUNINFO = session.get_dbtype(conddb.RunInfo)

    myGTMap = session.query(GTMAP.record, GTMAP.label, GTMAP.tag_name).\
        filter(GTMAP.global_tag_name == str(expressGT)).\
        order_by(GTMAP.record, GTMAP.label).\
        all()

    ## connect to prep DB and get the list of IOVs to look at
    con2 = conddb.connect(url = conddb.make_url("dev"))
    session2 = con2.session()
    validationTagIOVs = session2.query(IOV.since,IOV.payload_hash,IOV.insertion_time).filter(IOV.tag_name == options.validationTag).all()

    ### fill the list of IOVs to be validated
    IOVsToValidate=[]
    if(options.since==-1):
        IOVsToValidate.append(validationTagIOVs[-1][0])
        print("Changing the default validation tag since to:",IOVsToValidate[0])
        
    else:
        for entry in validationTagIOVs:
            if(options.since!=1 and int(entry[0])>=int(options.since)):
                print("Appending to the validation list:",entry[0],entry[1],entry[2])
                IOVsToValidate.append(entry[0])
            
    for element in myGTMap:
    #print element
        Record = element[0]
        Label  = element[1]
        Tag = element[2]
        if(Record=="SiStripApvGain2Rcd"):
            TagIOVs = session.query(IOV.since,IOV.payload_hash,IOV.insertion_time).filter(IOV.tag_name == Tag).all()
            sorted_TagIOVs = sorted(TagIOVs, key=lambda x: x[0])
            lastG2Payload = sorted_TagIOVs[-1]
            print("Last G2 Prompt payload has IOV since:",lastG2Payload[0],"payload hash:",lastG2Payload[1],"insertion time:",lastG2Payload[2])

            # Get and print the current working directory
            current_directory = os.getcwd()
            print("Current Working Directory:", current_directory)

            command = 'conddb_import -c sqlite_file:toCompare.db -f frontier://FrontierProd/CMS_CONDITIONS -i '+str(Tag) +' -t '+str(Tag)+' -b '+str(lastG2Payload[0])
            print(command)
            getCommandOutput(command)

            # set syncrhonization to any
            resetSynchonization("toCompare.db")

            for i,theValidationTagSince in enumerate(IOVsToValidate):

                command = 'conddb_import -c sqlite_file:toCompare.db -f frontier://FrontierPrep/CMS_CONDITIONS -i '+str(options.validationTag) +' -t '+str(Tag)+' -b '+str(theValidationTagSince)
                if(theValidationTagSince < lastG2Payload[0]):
                    print("The last available IOV in the validation tag is older than the current last express IOV, taking FCSR as a since!")
                    command = 'conddb_import -c sqlite_file:toCompare.db -f frontier://FrontierPrep/CMS_CONDITIONS -i '+str(options.validationTag) +' -t '+str(Tag)+' -b '+str(FCSR+i)

                print(command)
                getCommandOutput(command)

                command = '${CMSSW_BASE}/src/CondCore/SiStripPlugins/scripts/testCompare.sh SiStripApvGain_FromParticles_GR10_v1_express '+str(lastG2Payload[0])+' '+str(theValidationTagSince)+' '+current_directory+'/toCompare.db'
                if(theValidationTagSince < lastG2Payload[0]):
                    command = '${CMSSW_BASE}/src/CondCore/SiStripPlugins/scripts/testCompare.sh SiStripApvGain_FromParticles_GR10_v1_express '+str(lastG2Payload[0])+' '+str(FCSR+i)+' '+current_directory+'/toCompare.db'
                print(command)
                getCommandOutput(command)
