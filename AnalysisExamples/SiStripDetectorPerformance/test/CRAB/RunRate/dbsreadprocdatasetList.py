#!/usr/bin/env python
#

import sys,os,string,getopt,time,re
import xml.sax.handler
import xml.sax
## DBS API
from dbsCgiApi import DbsCgiApi, DbsCgiDatabaseError
from dbsException import DbsException
from dbsApi import DbsApi, DbsApiException, InvalidDataTier

usage="Usage: python dbsreaddataset.py --DBSAddress=MCLocal_4/Writer --datasetPath=/TEST-TAC-120-DAQ/RAW/* --logfile=logs_TOB/proddatasetlist.txt \n"

valid = ['DBSAddress=','datasetPath=','logfile=']

try:
    opts, args = getopt.getopt(sys.argv[1:], "", valid)
except getopt.GetoptError, ex:
    print usage
    print str(ex)
    sys.exit(1)
                                                                                    

DEFAULT_URL ="http://cmsdbs.cern.ch/cms/prod/comp/DBS/CGIServer/prodquery"
#dbinstance = "MCLocal_4/Writer"
# datasetPath = "/TEST-TAC-120-DAQ/RAW/CMSSW_1_2_0-RAW-Run-00000540"
datasetPath = None


for opt, arg in opts:
    if opt == "--DBSAddress":
        dbsinstance = arg    
    if opt == "--datasetPath":
        datasetPath = arg
    if opt == "--logfile":
        logfile = arg    
    if opt == "--help":
        print usage
        sys.exit(1)

if dbsinstance == None:
    print "--DBSAddress option not provided. For example : --DBSAddress=MCLocal_4/Writer \n"
    print usage
    sys.exit(1)
    
if datasetPath == None:
    print "--datasetPath option not provided. For example : --datasetPath=/primarydataset/datatier/processeddataset \n"
    print usage
    sys.exit(1)

if logfile == None:
    print "--logfile option not provided. For example : --logfile=logs_TOB/proddatasetlist.txt \n"
    print usage
    sys.exit(1)

f = open(logfile, "w");

inargs = {'instance' : dbsinstance}
api = DbsCgiApi(DEFAULT_URL, inargs)

datasets = api.listProcessedDatasets(datasetPath)

for dataset in datasets:

    f.write("DatasetName: %s \n"%(dataset))
#
f.close()
