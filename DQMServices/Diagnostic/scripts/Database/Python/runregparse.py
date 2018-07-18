#!/usr/bin/env python

# LM: version date: 01/02/2010 --> fixed dataset search and added json output file (optional)
# LM: updated 03/04/2010 --> adapted to new runreg api (and dcs status info)
# LM: updated 15/04/2010 --> added bfield threshold

# include XML-RPC client library
# RR API uses XML-RPC webservices interface for data access
import xmlrpclib,sys,ConfigParser,os,string,commands,time,re
# for json support
try: # FUTURE: Python 2.6, prior to 2.6 requires simplejson
    import json
except:
    try:
        import simplejson as json
    except:
        print "Please use lxplus or set an environment (for example crab) with json lib available"
        sys.exit(1)

global QF_Req,ls_temp_data,QF_ALL_SYS,EXCEPTION,EXRUN
EXCEPTION=False
EXRUN=-1

def invert_intervals(intervals,min_val=1,max_val=9999):
    # first order and merge in case 
    if not intervals:
        return []
    intervals=merge_intervals(intervals)
    intervals = sorted(intervals, key = lambda x: x[0])
    result = []
    if min_val==-1:
        # defin min and max
        (a,b)=intervals[0]
        min_val=a
    if max_val==-1:
        (a,b)=intervals[len(intervals)-1]
        max_val=b

    curr_min=min_val
    for (x,y) in intervals:
        if x>curr_min:
            result.append((curr_min,x-1))
        curr_min=y+1
    if curr_min<max_val:
        result.append((curr_min,max_val))

#    print min_val,max_val
    return result

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key = lambda x: x[0])
    result = []
    (a, b) = intervals[0]
    for (x, y) in intervals[1:]:
        if x <= b:
            b = max(b, y)
        else:
            result.append((a, b))
            (a, b) = (x, y)
    result.append((a, b))
    return result

def remove_html_tags(data):
    p = re.compile(r'<.*?>')
    newdata=p.sub('', data)
    newdata=newdata.replace("&nbsp;","")
    return newdata

def remove_extra_spaces(data):
    result= re.sub(r'\s', '', data)
    return result

def searchrun(runno):
    global QF_Req,ls_temp_data,QF_ALL_SYS,EXCEPTION,EXRUN
    intervallist=[]
    selectls=""

    for line in ls_temp_data.split("\n"):
        if runno in line:
#            print line
            try:
                if "%%%BAD LS INFO BEGIN%%%" in line:
                    selectls=line.split("%%%BAD LS INFO BEGIN%%%")[1]
                    selectls=selectls.split("%%%BAD LS INFO END%%%")[0]
                    selectls=remove_html_tags(selectls)
                    selectls=remove_extra_spaces(selectls)
                    # print selectls
                    for tag in QF_ALL_SYS:
                        selectls=selectls.replace(tag+":","\n"+tag+":")
                    # print selectls
                    
                    for line in selectls.split("\n"):
                        try:
                            tag=line.split(":")[0]
                            intervals=line.split(":")[1]
                        except:
                            continue
                        if tag in QF_Req.keys():
                            if QF_Req[tag]=="GOOD":
                                for interval in intervals.split(","):
                                    if "ALL" in interval:
                                        lmin=1
                                        lmax=9999
                                    else:
                                        strmin=interval.split('-')[0]
                                        strmax=interval.split('-')[1]
                                        lmin=int(strmin)
                                        if "END" in strmax: 
                                            lmax=9999
                                        else:
                                            lmax=int(strmax)
                                    intervallist.append((lmin,lmax))
            except:
                EXCEPTION=True
                EXRUN=int(runno)
    intervallist=merge_intervals(intervallist)
    # print runno, intervallist
    return intervallist



#main starts here#

QF_Req={}
GOODRUN={}
compactList = {} 

QF_ALL_SYS=["Hcal","Track","Strip","Egam","Es","Dt","Csc","Pix","Muon","Rpc","Castor","Jmet","Ecal","L1t","Hlt","NONE"]
QF_ALL_STAT=["GOOD","BAD","EXCL","NONE"]
DCS_ALL=['Bpix','Fpix','Tibtid','TecM','TecP','Tob','Ebminus','Ebplus','EeMinus','EePlus','EsMinus','EsPlus','HbheA','HbheB','HbheC','H0','Hf','Dtminus','Dtplus','Dt0','CscMinus','CscPlus','Rpc','Castor',"NONE"]

# reading config file
CONFIGFILE='runreg.cfg'
CONFIG = ConfigParser.ConfigParser()
print 'Reading configuration file from ',CONFIGFILE
CONFIG.read(CONFIGFILE)

DATASET=CONFIG.get('Common','Dataset')
GROUP=CONFIG.get('Common','Group')
HLTNAMEFILTER=CONFIG.get('Common','HLTnameFilter')
ADDRESS=CONFIG.get('Common','RunReg')
RUNMIN=CONFIG.get('Common','Runmin')
RUNMAX=CONFIG.get('Common','Runmax')
QFLAGS=CONFIG.get('Common','QFLAGS')
BFIELD=CONFIG.get('Common','BField_thr')
LSPARSE=CONFIG.get('Common','LSCOMMENT')
DCSSTAT=CONFIG.get('Common','DCS')
DCSLIST=string.split(DCSSTAT,',')

OUTPUTFILENAME=CONFIG.get('Common',"OutputFileName")

LSCOMMENT=True
if "TRUE" in LSPARSE.upper() or "1" in LSPARSE.upper() or "YES" in LSPARSE.upper():
    LSCOMMENT=True
elif "FALSE" in LSPARSE.upper() or "0" in LSPARSE.upper() or "NO" in LSPARSE.upper():
    LSCOMMENT=False
else:
    print "Error in parsing LSCOMMENT cfg parameter: LSPARSE"
    sys.exit(1)

QFlist=string.split(QFLAGS,',')
for QF in QFlist:
    syst=string.split(QF,":")[0]
    value=string.split(QF,":")[1]
    if syst not in QF_ALL_SYS or value not in QF_ALL_STAT:
        print "QFLAG not valid:",syst,value 
        sys.exit(1)
    QF_Req[syst]=value

for dcs in DCSLIST:
    if dcs not in DCS_ALL:
        print "DCS not valid:",dcs
        sys.exit(1)


CFGLIST=CONFIG.items('Common')
JSONFILE=CONFIG.get('Common','JSONFILE')

try:
    BFIELD_float=float(BFIELD)
except:
    print "BFIELD threshold value not understood:",BFIELD
    sys.exit(1)

# report the request

print "You asked for the runreg info in the run range:"+RUNMIN+"-"+RUNMAX
print "for dataset: "+DATASET
print "with the following quality flags:"
for SS in QF_Req.keys():
    print SS, QF_Req[SS]
print "and with the following DCS status:"
for dcs in DCSLIST:
    print dcs
print "Manual bad LS in comment column:",LSCOMMENT
#sys.exit(1)
 
# get handler to RR XML-RPC server
FULLADDRESS=ADDRESS+"/xmlrpc"
print "RunRegistry from: ",FULLADDRESS
server = xmlrpclib.ServerProxy(FULLADDRESS)

# build up selection in RUN table
sel_runtable="{groupName} ='"+GROUP+"' and {runNumber} >= "+RUNMIN+" and {runNumber} <= "+RUNMAX+" and {bfield}>"+BFIELD+" and {datasetName} LIKE '"+DATASET+"'"

# the lumisection selection is on the Express dataset:
sel_dstable="{groupName} ='"+GROUP+"' and {runNumber} >= "+RUNMIN+" and {runNumber} <= "+RUNMAX+" and {bfield}>"+BFIELD+" and {datasetName} LIKE '%Express%'"

for key in QF_Req.keys():
    if key != "NONE" and QF_Req[key]!="NONE":
        sel_runtable+=" and {cmp"+key+"} = '"+QF_Req[key]+"'"
        sel_dstable+=" and {cmp"+key+"} = '"+QF_Req[key]+"'"
#print sel_runtable

# build up selection in RUNLUMISECTION table, not requestuing bfield here because only runs in the run table selection will be considered
sel_dcstable="{groupName} ='"+GROUP+"' and {runNumber} >= "+RUNMIN+" and {runNumber} <= "+RUNMAX
for dcs in DCSLIST:
    if dcs !="NONE":
        sel_dcstable+=" and {parDcs"+dcs+"} = 1"
# = 'True'"
# print sel_dcstable

Tries=0
print " " 
while Tries<10:
    try:
        print "Accessing run registry...."
        dcs_data = server.DataExporter.export('RUNLUMISECTION', 'GLOBAL', 'json', sel_dcstable)
        run_data = server.DataExporter.export('RUN', 'GLOBAL', 'csv_runs', sel_runtable)
        ls_temp_data = server.DataExporter.export('RUN', 'GLOBAL', 'csv_datasets', sel_dstable)
        break
    except:
        print "Something wrong in accessing runregistry, retrying in 3s...."
        Tries=Tries+1
        time.sleep(3)
if Tries==10:
    print "Run registry unaccessible.....exiting now"
    sys.exit(1)
    
#print dcs_data
# print "run data: ", run_data
#print ls_temp_data
# find LS info in comment



LISTOFRUN=[]
selectedRuns = open(OUTPUTFILENAME, 'w')
print "Saving selected runs to file OUTPUTFILENAME"
for line in run_data.split("\n"):
    run=line.split(',')[0]
    if run.isdigit():
        hlt=line.split(',')[9]
        print "for run", run, "hlt is", hlt
        if HLTNAMEFILTER == "" or hlt.find(HLTNAMEFILTER):
            LISTOFRUN.append(run)
            selectedRuns.write(run+"\n")
selectedRuns.close()

selected_dcs={}
jsonlist=json.loads(dcs_data)


for element in jsonlist:
    if element in LISTOFRUN:
# first search manual ls certification
        if LSCOMMENT:
            # using LS intervals in comment
            manualbad_int=searchrun(element)
        # make a badlumi list
            dcsbad_int=invert_intervals(jsonlist[element])
            combined=[]
            for interval in  manualbad_int:
                combined.append(interval)
            for interval in  dcsbad_int:
                combined.append(interval)
            combined=merge_intervals(combined)
            combined=invert_intervals(combined)
            selected_dcs[element]=combined
        else:
            # using only DCS info
            selected_dcs[element]=jsonlist[element]
        # combined include bith manual LS and DCS LS

#JSONOUT=json.dumps(selected_dcs)
# WARNING: Don't use selected_dcs before dumping into file, it gets screwed up (don't know why!!)
if JSONFILE != "NONE":
    lumiSummary = open(JSONFILE, 'w')
    json.dump(selected_dcs, lumiSummary)
    lumiSummary.close() 
    print " "
    print "-------------------------------------------"
    print "Json file: ",JSONFILE," written."


# buildup cms snippet
selectlumi="process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange(\n"
ranges = []
runs_to_print = sorted(selected_dcs.keys())
for run in runs_to_print:
   blocks = sorted(selected_dcs[run])
   prevblock = [-2,-2]
   for lsrange in blocks:
       if lsrange[0] == prevblock[1]+1:
           print "Run: ",run,"- This lumi starts at ", lsrange[0], " previous ended at ", prevblock[1]+1, " so I should merge"
           prevblock[1] = lsrange[1]
           ranges[-1] = "\t'%s:%d-%s:%d',\n" % (run, prevblock[0],
run, prevblock[1])
       else:
           ranges.append("\t'%s:%d-%s:%d',\n" % (run, lsrange[0],
run, lsrange[1]))
           prevblock = lsrange
selectlumi += "".join(ranges)
selectlumi += ")"


print "-------------------------------------------"
print " "
print "CFG snippet to select:"
print selectlumi

if EXCEPTION:
    print "WARNING: Something wrong in manual lumisection selection tag for run: "+str(EXRUN)
