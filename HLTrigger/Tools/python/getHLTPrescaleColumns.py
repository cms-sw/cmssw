#!/usr/bin/env python 
from sys import stderr, exit
import commands

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] Trigger_Path")
parser.add_option("--firstRun",  dest="firstRun",  help="first run", type="int", metavar="RUN", default="1")
parser.add_option("--lastRun",   dest="lastRun",   help="last run",  type="int", metavar="RUN", default="9999999")
parser.add_option("--groupName", dest="groupName", help="select runs of name like NAME", metavar="NAME", default="Collisions%")
parser.add_option("--rrurl",     dest="rrurl",     help="run registry xmlrpc url", metavar="URL", default="http://cms-service-runregistry-api.web.cern.ch/cms-service-runregistry-api/xmlrpc")
parser.add_option("--jsonOut",   dest="jsonOut",   help="dump prescales in JSON format on FILE", metavar="FILE")
(options, args) = parser.parse_args()
if len(args) != 1:
    parser.print_usage()
    exit(2)
path = args[0]


edmCfgFromDB = "edmConfigFromDB --orcoff --format summary.ascii --paths " + path;
## my $pyPrintTable = "echo 'for X in process.PrescaleService.prescaleTable: print \"\%s \%s\" % (X.pathName.value(), X.prescales[0])'";
def getPrescalesFromKey(key):
    #stderr.write("\t%s ...\n" % key);
    cmd = ( edmCfgFromDB +" --configName "+key + " | grep -i "+ path + " | tail -1 | awk ' $2 ==\"%s\" {print $NL}' " ) % path
    res = commands.getoutput(cmd)
    res_split = res.split()
    psMap = {}
    aa=""
    if len(res)>0:
	for uu in range(3,len(res_split)-1):
		if uu % 2 == 1:
		   aa = aa + res_split[uu] + "\t"
	psMap[path] = aa
    else:
	psMap[path] = 0
    return psMap


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
prescaleTable = {}
runs = runKeys.keys(); runs.sort()
stderr.write("Querying ConfDB for prescales for path %s...\n" % (path));
jsout = {}
for run in runs:
    key = runKeys[run]
    if not prescaleTable.has_key(key):
        prescaleTable[key] = getPrescalesFromKey(key)
    psfactor = 1
    if prescaleTable[key].has_key(path): psfactor = prescaleTable[key][path]
    print "%s\t%s" % (run, psfactor)
    jsout[run] = psfactor

if options.jsonOut:
    stderr.write("Exporting to JSON file %s...\n" % (options.jsonOut))
    import json
    jsonFile = open(options.jsonOut, "w")
    jsonFile.write(json.dumps(jsout))
    jsonFile.close()
