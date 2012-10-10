#!/usr/bin/env python

#-----------------------------------------------------
# original author: Andrea Lucaroni
# Revision:        $Revision: 1.1 $
# Last update:     $Date: 2011/06/28 19:34:22 $
# by:              $Author: mussgill $
#-----------------------------------------------------

from xml.dom import minidom
import re
import json
import os 
import stat
import sys

#include DBS
from DBSAPI.dbsApiException import DbsException
import DBSAPI.dbsApi
from DBSAPI.dbsApiException import *

# include XML-RPC client library
# RR API uses XML-RPC webservices interface for data access
import xmlrpclib

import array
import pickle as pk

from optparse import OptionParser
#####DEBUG
DEBUG = 0

#size file
def filesize1(n):    
    info = os.stat(n)
    sz = info[stat.ST_SIZE]
    return sz
            
### lumiCalc
def printLumi(file,namefile):
    if(filesize1(file) != 0):
        string= "lumiCalc.py -c frontier://LumiCalc/CMS_LUMI_PROD -i "
        string1= " --nowarning overview >"
        string2= string + file + string1 + namefile
        data = os.system(string2)
    else:
        data = ""
        print "0 lumi are not avaible"
    return data

###file  dbs
def DBSquery(dataset,site,run):

    url = "http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet"
    args = {}
    args['url']     = url
    args['level']   = 'CRITICAL'
    api = DBSAPI.dbsApi.DbsApi(args)
    files = api.listFiles(path=dataset,tier_list =site,runNumber=run)
    return files

###file cff data
def makecff(file_list,namefile):
    file1 = open(namefile ,'w')
    stringS  =           "process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )\n"
    stringS  = stringS + "readFiles = cms.untracked.vstring()\n"
    stringS  = stringS + "secFiles = cms.untracked.vstring()\n"
    stringS  = stringS + "\n"
    file1.write(stringS)
    
    filecount = 0
    extendOpen = 0
    for filename in file_list:
        
        if extendOpen == 0:
            stringS  = "readFiles.extend([\n"
            file1.write(stringS)
            extendOpen = 1
            
        stringS  =           "     '"
        stringS  = stringS + str(filename)
        stringS  = stringS + "',\n"
        file1.write(stringS)
        filecount = filecount + 1
        if filecount == 50:
            stringS  = "])\n\n"
            file1.write(stringS)
            filecount = 0
            extendOpen = 0

    if extendOpen == 1:
        stringS  =           "])\n\n"
        file1.write(stringS)
    
    stringS  =           "process.source = cms.Source(\"PoolSource\",\n"
    stringS  = stringS + "         fileNames = readFiles,\n"
    stringS  = stringS + "         secondaryFileNames = secFiles\n"
    stringS  = stringS + ")\n"
    file1.write(stringS)
    file1.close()


def defineOptions():
    parser = OptionParser()
    parser.add_option("-w", "--workspace",
                      dest="workspaceName",
                      default="GLOBAL",
                      help="define workspace: GLOBAL TRACKER ")

    parser.add_option("-r", "--regexp",
                      dest="regexp",
                      type="string",
                      default='groupName : LIKE %Collisions10% , runNumber : = 136088',
                      help=" \"{runNumber} >= 148127 and {runNumber} < 148128 \" ")

    parser.add_option("-d", "--datasetPath",
                      dest="dataset", \
                      default="/MinimumBias/Run2010A-TkAlMinBias-Dec22ReReco_v1/ALCARECO",
                      help="For example : --datasetPath /MinimumBias/Run2010A-TkAlMinBias-Dec22ReReco_v1/ALCARECO")
    
    parser.add_option("-s", "--site",
                      dest="site",
                      default="T2_CH_CAF",
                      help="For example : site T2_CH_CAF")                 

    parser.add_option("-i", "--info",
                      action="store_true",
                      dest="info",
                      default=False,
                      help="printout the column names on which it's possible to cut")
    
    (options, args) = parser.parse_args()
    if len(sys.argv) == 1:
	print("\nUsage: %s --help"%sys.argv[0])
        sys.exit(0)
    
    return options

    
def serverQuery(workspaceName,regexp):

    # get handler to RR XML-RPC server
    server = xmlrpclib.ServerProxy('http://cms-service-runregistry-api.web.cern.ch/cms-service-runregistry-api/xmlrpc')
    if DEBUG:
        print regexp
    data = server.RunDatasetTable.export(workspaceName,'xml_all' ,regexp)
    return data

#----------------------------------------------------

def getText(nodelist):
    rc = ""
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
    return rc

def getElement(obj,name):
    return obj.getElementsByTagName(name)

def printObj(obj,name):
    return getText(getElement(obj,name)[0].childNodes)


def getData(doc,options,dataset,site):
    server = xmlrpclib.ServerProxy('http://cms-service-runregistry-api.web.cern.ch/cms-service-runregistry-api/xmlrpc')
    runs = getElement(doc,'RUN')
    txtLongData=""
    txtkey=""
    lista=[]
    
    sep="\t"

    for run in runs:
        txtrun=printObj(run,'NUMBER') + sep + printObj(run,'HLTKEY')
        txtLongData+= txtrun + sep + "\n" 

    for run in runs:
        test=printObj(run,'HLTKEY')
        if not (test in lista):
            lista.append(test)

        file2=open("lista_key.txt",'w')
        for pkey in range(len(lista)):
            pwkey = lista[pkey] +"\n"
            file2.write(pwkey)

        file2.close()

    for i in range(len(lista)):
        if DEBUG:
            print lista[i]
        nameDBS=""
        nameDBS=str(i)+".data"
        name=""
        name=str(i)+".json"
        nameLumi=""
        nameLumi=str(i)+".lumi"
        file1 = open( name ,'w')
        listaDBS = []
        if DEBUG:
            print nameDBS
        for run in runs:
            key=printObj(run,'HLTKEY')
            if (key == lista[i]):
                print "running......"
                if DEBUG:
                    print printObj(run,'NUMBER')
                txtruns = "{runNumber} >= " + printObj(run,'NUMBER') +  " and {runNumber} < " + str(int(printObj(run,'NUMBER'))+1)
                txtriv = txtruns + " and {cmpPix} in ('GOOD') and {cmpStrip} in ('GOOD') and {cmpTrack} in ('GOOD')"
                riv = server.RunDatasetTable.export('GLOBAL', 'csv_run_numbers',txtriv)
                if riv:
                    lumirun = server.RunLumiSectionRangeTable.export('GLOBAL', 'json',txtruns)
                    ###dbs file
                    file = DBSquery(dataset,site,str(printObj(run,'NUMBER')))                    
                    for uno in file:
                        stringDBS = {}
                        stringDBS = uno['LogicalFileName']
                        listaDBS    += [stringDBS]
                    ###
                    if DEBUG:
                        print lumirun
                    comp="{}"
                    if (lumirun == comp):
                        print "LUMI ZERO"
                    else:
                        file1.write(lumirun)
               
        file1.close()
        string=""
        string="sed -i 's/}{/, /g'"
        string2=""
        string2= string + " " + name
        os.system(string2)
        printLumi(name,nameLumi)
        os.system("sed -i 's/\//_/g' lista_key.txt")
        listaDBS2 =[]
        for rootLSF in listaDBS:
            if not (rootLSF in listaDBS2):
                listaDBS2.append(rootLSF)
        makecff(listaDBS2,nameDBS)
          
    return txtLongData

#---------------------------------------------

def extractData(mode,reg,dataset,site,options):
    doc = minidom.parseString(serverQuery(mode,reg))
    return getData(doc,options,dataset,site)

def getRegExp(regexp):
    items = regexp.split(',') 
    dd = {}
    for item in items:
        key,value = item.split(':')
        dd[key.replace(' ','')] = value
    return dd


#---------------------------------------------MAIN

options = defineOptions()
data=extractData(options.workspaceName,options.regexp,options.dataset,options.site,options)

