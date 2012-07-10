#!/usr/bin/env python
import sys
import getopt
import common_db
import results_db
import reference_db

def CmdUsage():
	print "Command line arguments :"
	print "--create (-C): creates full regression test db schema"
	print "--drop (-D): drops full regression test db schema."
	print "--add (-A): -r [release] -a [arch] -p [path]: adds a reference release into db."
	print "--erase (-E): -r [release] -a [arch]: remove a reference release from db."
	print "--list (-L): lists the available reference releases"
	print "--read (-R): read the content of the results db) db"
	print "--create_res: creates results db schema"
	print "--drop_res: drops results db schema."
	print "--create_ref: creates reference db schema"
	print "--drop_ref: drops reference db schema."

try:
    opts, args = getopt.getopt(sys.argv[1:], "CDAELRr:a:p:h", ['create', 'drop', 'add', 'erase', 'list','read','create_res','drop_res','create_ref','drop_ref','help'])
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    CmdUsage()
    sys.exit(2)
c0flag = False
d0flag = False
c1flag = False
d1flag = False
aflag = False
eflag = False
lflag = False
rflag = False
RELEASE = None
ARCH = None
PATH = None
if( len(opts)==0 ):
    CmdUsage()
    sys.exit(2)    
for o, a in opts:
    if o in ("--create","-C"):
        c0flag = True
        c1flag = True
    elif o in ( "--drop", "-D"):
        d0flag = True
        d1flag = True
    elif o in ("--add","-A" ):
        aflag = True
    elif o in ("--erase","-E" ):
        eflag = True
    elif o in ("--list","-L" ):
        lflag = True
    elif o in ("--read","-R" ):
        rflag = True
    elif o == "-r":
        RELEASE = a
    elif o == "-a":
        ARCH = a
    elif o == "-p":
        PATH = a
    elif o == "--create_ref":
        c0flag = True
    elif o == "--create_res":
        c1flag = True
    elif o == "--drop_ref":
        d0flag = True
    elif o == "--drop_res":
        d1flag = True
    elif o in ("--help","-h"):
        CmdUsage()
        sys.exit(2)
    else:
        assert False, "unhandled option"

conn = common_db.createDBConnection()
refDb = reference_db.ReferenceDB( conn )
resDb = results_db.ResultsDB( conn ) 
        
mainOption = False
if(d0flag == True):
    refDb.drop()
    mainOption = True
if(d1flag == True):
    resDb.drop()
    mainOption = True
if(c0flag == True):
    refDb.create()
    mainOption = True
if(c1flag == True):
    resDb.create()
    mainOption = True

if(aflag == True and mainOption == False):
    mainOption = True
    if( RELEASE == None ):
        print "-r parameter has not been specified."
    if( ARCH == None ):
        print "-a parameter has not been specified."
    if( PATH == None ):
        print "-p parameter has not been specified."
    refDb.addRelease(RELEASE, ARCH, PATH)
if(eflag == True and mainOption == False):
    mainOption = True
    if( RELEASE == None ):
        print "-r parameter has not been specified."
    if( ARCH == None ):
        print "-a parameter has not been specified."
    refDb.deleteRelease(RELEASE, ARCH)
if(lflag == True and mainOption == False ):
    mainOption = True
    refDb.read()
if(rflag == True and mainOption == False):
    mainOption = True
    resDb.read()
conn.close()
