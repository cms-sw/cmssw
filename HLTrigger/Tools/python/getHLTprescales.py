#!/usr/bin/env python 
from __future__ import print_function
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


edmCfgFromDB = "edmConfigFromDB  --orcoff --format summary.ascii --paths " + path;
## my $pyPrintTable = "echo 'for X in process.PrescaleService.prescaleTable: print \"\%s \%s\" % (X.pathName.value(), X.prescales[0])'";
def getPrescalesFromKey(key):
    #stderr.write("\t%s ...\n" % key);
    cmd = ( edmCfgFromDB +" --configName "+key + " | grep -i "+ path + " | tail -1 | awk ' $2 ==\"%s\" {print $NL}' " ) % path
    res = commands.getoutput(cmd)
    res_split = res.split()
    psCols = []
    if len(res)>0:
        for uu in range(3,len(res_split)-1):
            if uu % 2 == 1:
                psCols.append(res_split[uu])
    return psCols

from queryRR import queryRR

def readIndex():
    asciiFile=open("columns.txt","read")
    mapIndex={}
    fl="go"
    while fl:
        fl=asciiFile.readline()
        if len(fl)>0:
            ll=fl.split()
            runnumber=ll[0]
            pindex=ll[1]
            mapIndex[runnumber]=pindex
    asciiFile.close()
    return mapIndex


MapIndex=readIndex()
runKeys = queryRR(options.firstRun,options.lastRun,options.groupName)
prescaleTable = {}
Absent = []
runs = runKeys.keys(); runs.sort()
stderr.write("Querying ConfDB for prescales for path %s...\n" % (path));
jsout = {}
for run in runs:
    key = runKeys[run]
    if key not in prescaleTable:
        prescaleTable[key] = getPrescalesFromKey(key)
    psfactor = 1
    absent=0
    if len(prescaleTable[key]) == 0:
        psfactor = 0
    else:
        if run in MapIndex:
            index = int(MapIndex[run])
            psfactor = prescaleTable[key][index]
        else:
            if int(run) < 138564:
                index = 0
                psfactor = prescaleTable[key][index]
            else:
                #print "... the run ",run," is not found in columns.txt ... Index is set to zero, need to check independently..."
                index=0
                psfactor = prescaleTable[key][index]
                Absent.append(run)
                absent=1
    if absent==0:
        print("%s\t%s" % (run, psfactor))
    else:
        print("%s\t%s\t (*)" % (run, psfactor))
    jsout[run] = psfactor

if len(Absent)>0:
    print("")
    print("(*) The following runs were not found in columns.txt (the run may be too recent, or the prescale index is not in OMDS).")
    print("For these runs, the prescale_index was assumed to be zero. You need to check independently.")
    for r in Absent:
        print("\t",r)
    print("")

if options.jsonOut:
    stderr.write("Exporting to JSON file %s...\n" % (options.jsonOut))
    import json
    jsonFile = open(options.jsonOut, "w")
    jsonFile.write(json.dumps(jsout))
    jsonFile.close()
