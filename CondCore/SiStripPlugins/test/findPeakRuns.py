'''
Script used to find all PEAK runs

Usage:

create the list of SiStripLatency::singleReadOutMode() IOVs / values with:
getPayloadData.py --plugin pluginSiStripLatency_PayloadInspector --plot plot_SiStripIsPeakModeHistory --tag SiStripLatency_GR10_v2_hlt --time_type Run --iovs '{"start_iov": "1", "end_iov": "400000"}' --db Prod --test > & log.json 

followed by:

findPeakRuns.py -f log.json > allPeakRuns.log
'''


from __future__ import print_function
import json
import ROOT
from pprint import pprint
from optparse import OptionParser
import CondCore.Utilities.conddblib as conddb

#__________________________________________
def findPeakIOVs(values):
    listOfIOVs=[]
    listOfModes=[]
    lastMode=1
    lastRun=1
    
    latestRun=-1
    latestMode=-9999

    for value in values:
        if (value['y']!=lastMode):
            listOfIOVs.append((lastRun,value['x']))
            if(lastMode==1):
                listOfModes.append('PEAK')
            else:
                listOfModes.append('DECO')
            lastMode=value['y']
            lastRun=value['x']
        else:
            latestRun=value['x']
            if(value['y']==1):
                latestMode='PEAK'
            else:
                latestMode='DECO'

    ## special case for the last open IOV
    listOfIOVs.append((lastRun,999999))
    listOfModes.append(latestMode)

    return dict(zip(listOfIOVs,listOfModes))

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="open FILE and extract info", metavar="FILE")
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print PEAK/DECO dictionary")

(options, args) = parser.parse_args()

with open(options.filename) as data_file:    
    data = json.load(data_file)
    values = data["data"]
    
IOVs = findPeakIOVs(values)
if(options.verbose):
    pprint(IOVs)

con = conddb.connect(url = conddb.make_url("pro"))
session = con.session()
RUNINFO = session.get_dbtype(conddb.RunInfo)

####################################
# Get the whole list of runs
####################################
AllRuns = session.query(RUNINFO.run_number, RUNINFO.start_time, RUNINFO.end_time).all()
for run in AllRuns:
    for key, value in IOVs.items():
        if(key[0]<run[0] and key[1]>run[0]):
            if(value=='PEAK'):
                print(run[0],"(",run[1],") IOV [",key[0],"-",key[1],") which is ",value," mode")
