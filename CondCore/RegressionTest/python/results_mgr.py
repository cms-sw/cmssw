#!/usr/bin/env python
import sys
import getopt
import common_db
import results_db
import reference_db

def CmdUsage():
	print "Command line arguments :"
	print "--read_sel (-S) --full (-f) -i [runid] -l [test_label] -c [ cand_release:arch]: read the content of the results db"
	print "--read (-R): read the content of the results db) db"
	print "--del (-D) -i [runid] : delete the result entry for the run"

try:
    opts, args = getopt.getopt(sys.argv[1:], "SRFDi:l:c:h", ['read_sel', 'read','full','del','help'])
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    CmdUsage()
    sys.exit(2)
rflag = False
sflag = False
fflag = False
dflag = False
T_LABEL = None
RUNID = None
CRELEASE = None
RRELEASE = None
if( len(opts)==0 ):
    CmdUsage()
    sys.exit(2)    
for o, a in opts:
    if o in ("--read","-R" ):
        rflag = True
    elif o in ("--read_sel","-S" ):
        sflag = True
    elif o in ("--full","-F" ):
        fflag = True
    elif o in ("--del","-D" ):
        dflag = True
    elif o == "-l":
        T_LABEL = a
    elif o == "-i":
        RUNID = a
    elif o == "-c":
        CRELEASE = a
    elif o in ("--help","-h"):
        CmdUsage()
        sys.exit(2)
    else:
        assert False, "unhandled option"

conn = common_db.createDBConnection()
resDb = results_db.ResultsDB( conn ) 
        
mainOption = False
if(rflag == True):
    resDb.read()
elif(sflag == True):
    crel = None
    carch = None
    if( CRELEASE != None ):
	    tok = CRELEASE.split(":")
	    crel = tok[0]
	    carch = tok[1]
    rrel = None
    rarch = None
    resDb.readSelection( RUNID, T_LABEL, crel, carch, fflag )
elif( dflag == True ):
    if( RUNID == None ):
	    print 'ERROR: runid has not been provided.'
    else:
	    resDb.deleteRun( RUNID )

conn.close()
