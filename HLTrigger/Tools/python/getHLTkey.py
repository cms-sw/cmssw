#!/usr/bin/env python 
from sys import stderr, exit
import commands


#
# E.P., 27 July 2010
# query to the Run Reistry taken from a script by Giovanni Petrucianni
#


from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] ")
parser.add_option("--firstRun",  dest="firstRun",  help="first run", type="int", metavar="RUN", default="1")
parser.add_option("--lastRun",   dest="lastRun",   help="last run",  type="int", metavar="RUN", default="9999999")
parser.add_option("--groupName", dest="groupName", help="select runs of name like NAME", metavar="NAME", default="Collisions%")
parser.add_option("--rrurl",     dest="rrurl",     help="run registry xmlrpc url", metavar="URL", default="http://cms-service-runregistry-api.web.cern.ch/cms-service-runregistry-api/xmlrpc")
parser.add_option("--HLTkey",    dest="HLTkey",    help="name of the HLTkey e.g. /cdaq/physics/Run2010/v3.1/HLT_1.6E30/V1",metavar="HLT")
parser.add_option("--perKey",    action="store_true",default=False,dest="perKey",help="list the runs per HLT key",metavar="perKey")
(options, args) = parser.parse_args()

def queryRR():
    stderr.write("Querying run registry for range [%d, %d], group name like %s ...\n" % (options.firstRun, options.lastRun, options.groupName))
    import xmlrpclib
    import xml.dom.minidom
    server = xmlrpclib.ServerProxy(options.rrurl)
    run_data = server.DataExporter.export('RUN', 'GLOBAL', 'xml_datasets', "{runNumber} >= %d AND {runNumber} <= %d AND {groupName} like '%s' AND {datasetName} = '/Global/Online/ALL'"  % (options.firstRun, options.lastRun, options.groupName))
    ret = {}
    xml_data = xml.dom.minidom.parseString(run_data)
    xml_runs = xml_data.documentElement.getElementsByTagName("RUN_DATASET")
    for xml_run in xml_runs:
        ret[xml_run.getElementsByTagName("RUN_NUMBER")[0].firstChild.nodeValue] = xml_run.getElementsByTagName("RUN_HLTKEY")[0].firstChild.nodeValue
    return ret

runKeys = queryRR()
runs = runKeys.keys(); runs.sort()

if options.perKey:
 	runsPerKey={}
	for run in runs:
		key = runKeys[run]
		if not key in runsPerKey.keys():
			tmpruns=[]
			tmpruns.append(run)
			runsPerKey[key] = tmpruns
		else:
			runsPerKey[key].append(run)
	theKeys = runsPerKey.keys()
	for key in theKeys:
		theruns = runsPerKey[key]
		topr=""
		for r in theruns:
			topr=topr+"\t"+r
		print key,topr
	exit(1)
			
if options.HLTkey:
	HLTkey = options.HLTkey
	print "List of runs taken with HLT key = ",HLTkey 
for run in runs:
    key = runKeys[run]

    if not options.HLTkey:
       print run,key	
    else:
	if key == options.HLTkey:
	   print run
