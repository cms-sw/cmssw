#! /usr/bin/env python2.4


#
# Main script to list, plot, draw, dump Ecal conditions
#
#
import os,sys,getopt

def usage():
    print "NAME "
    print "   EcalCondDB - inpect EcalCondDB"
    print
    print "SYNOPSIS"
    print "   EcalCondDB [options] [command] [command_options]"
    print
    print "OPTIONS"
    print
    print "Specify short options as '-o foo', long options as '--option=foo'"
    print
    print "   -c, --connect= [connectstring]"
    print "      specify database, default oracle://cms_orcoff_prod/CMS_COND_31X_ECAL"
    print
    print "   -P, --authpath=  [authenticationpath], default /afs/cern.ch/cms/DB/conddb "
    print
    print "COMMAND OPTIONS"
    print
    print "   -t, --tag= [tag] specify tag"
    print
    print "COMMANDS"
    
    print "   -l, --listtags=  list all tags and exit"
    print
    print "   -d, --dump= [file] dump record to xml file"
    print
    print "   -p, --plot= [file] plot record to file, extension specifies ",\
                    "format (.root,.png,.jpg,.gif,.svg)"
    print 
    print "   -q, --compare= [file]"
    print "         compare two tags and write histograms to file, extension",\
                    " specifies format."               
    print "         Example:"
    print "                 EcalCondDB -q -t tag1 -t tag2"
    print
    print "   -g, --histo= [file] make histograms and write to file"
    print
    
try:
    opts, args = getopt.getopt(sys.argv[1:],
                               "c:P:t:ld:p:q:g:",
                               ["connect=","authpath=","tag=","listtags",\
                                "dump=","plot=","compare=","histo="])
    
    if not len(opts):
        usage()
        sys.exit(0)

except getopt.GetoptError:
    usage()
    sys.exit(0)

dbName =  "oracle://cms_orcoff_prod/CMS_COND_31X_ECAL"
authpath= "/afs/cern.ch/cms/DB/conddb"

tags=[]
ntags=0

do_list=0
do_dump=0
do_plot=0
do_comp=0
do_hist=0

outfilename=None


for opt,arg in opts:

    if opt in ("-c","--connect"):
       dbName=arg
    
    if opt in ("-P","--authpath"):
       authpath=arg 

    if opt in ("-h","--help"):
       usage()
       sys.exit(0)

    if opt in ("-t","--tag"):
       tags.append(arg)
       ntags=ntags+1

    if opt in ("-l","--listtags"):
       do_list=1  
       
    if opt in ("-q","--compare"):
       do_comp=1
       outfilename=arg

    if opt in ("-d","--dump"):
       do_dump=1
       outfilename=arg

    if opt in ("-p","--plot"):
       do_plot=1 
       outfilename=arg
   
    if opt in ("-g","--histo"):
       do_hist=1
       outfilename=arg 


#done parsing options, now steer

import EcalCondTools
import DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)
from pluginCondDBPyInterface import *
          
a = FWIncantation()

rdbms = RDBMS(authpath)
db = rdbms.getDB(dbName)

if do_list :
   EcalCondTools.listTags(db)

if do_dump :
   if not len(tags):
      print "Must specify tag with -t"
      sys.exit(0)
   EcalCondTools.dumpXML(db,tags[0],outfilename)

if do_plot:
   if not len(tags):
       print "Must specify tag with -t"
       sys.exit(0)       
   EcalCondTools.plot(db,tags[0],outfilename) 
 
if do_comp:
   if len(tags) != 2 :
       print "Must give exactly two tags to compare: -t tag1 -t tag2"
       sys.exit(0)
   EcalCondTools.compare(tags[0],db,tags[1],db,outfilename)       

if do_hist:
   if not len(tags):
       print "Must specify tag with -t"
       sys.exit(0)       
   EcalCondTools.histo(db,tags[0],outfilename) 


    
