#!/usr/bin/env python 
from sys import stderr, exit
import commands

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] Trigger_Path")
parser.add_option("--firstRun",  dest="firstRun",  help="first run", type="int", metavar="RUN", default="1")
parser.add_option("--lastRun",   dest="lastRun",   help="last run",  type="int", metavar="RUN", default="9999999")
parser.add_option("--groupName", dest="groupName", help="select runs of name like NAME", metavar="NAME", default="Collisions%")
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

from queryRR import queryRR

runKeys = queryRR(options.firstRun,options.lastRun,options.groupName)
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
