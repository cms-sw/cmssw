#!/usr/bin/env python 

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/DqmRrApi

from sys import stderr, exit
import xml.dom.minidom
from rrapi import RRApi, RRApiError

def queryRR(firstRun,lastRun,groupName):
    rrurl = "http://runregistry.web.cern.ch/runregistry/"
    stderr.write("Querying run registry for range [%d, %d], group name like %s ...\n" % (firstRun, lastRun, groupName))
    server = RRApi(rrurl)
    mycolumns = ['number', 'hltKeyDescription', 'runClassName']
    run_data = server.data(workspace = 'GLOBAL', table = 'runsummary', template = 'xml', columns = mycolumns, filter = {'datasetExists': '= true', 'number':'>= %d and <= %d'%(firstRun,lastRun), 'runClassName':"like '%%%s%%'"%groupName})
    ret = {}
    xml_data = xml.dom.minidom.parseString(run_data)
    xml_runs = xml_data.documentElement.getElementsByTagName("RunSummaryRowGlobal")
    for xml_run in xml_runs:
        ret[xml_run.getElementsByTagName("number")[0].firstChild.nodeValue] = xml_run.getElementsByTagName("hltKeyDescription")[0].firstChild.nodeValue
    return ret
