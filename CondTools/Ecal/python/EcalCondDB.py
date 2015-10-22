#! /usr/bin/env python


#
# Main script to list, plot, draw, dump Ecal conditions
#
#
import os,sys,getopt


def usage():
    print
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
    print "      specify database, default frontier://FrontierProd/CMS_COND_31X_ECAL"
    print
    print "   -P, --authpath=  [authenticationpath], default /afs/cern.ch/cms/DB/conddb "
    print
    print "COMMAND OPTIONS"
    print
    print "   -t, --tag= [tag,file] specify tag or xml file (histo,compare)"
    print
    print "   -s, --since= [runnumber] specify since"
    
    print "COMMANDS"
    
    print "   -l, --listtags  list all tags and exit"
    print
    print "   -m, --listiovs  list iovs for a given tag"
    print "                   Example EcalCondDB.py -t tag" 
    print
    print "   -d, --dump= [file] dump record to xml file"
    print "         Example EcalCondDB.py -d file.xml -t tag -s since"
    print
    print "   -p, --plot= [file] plot record to file, extension specifies ",\
                    "format (.root,.png,.jpg,.gif,.svg)"
    print "         Example: EcalCondDB.py -p=plot.png -t tag -s since"
    print
    
    print "   -q, --compare= [file]"
    print "         compare two tags and write histograms to file, extension",\
                    " specifies format."               
    print "         Example:"
    print "           EcalCondDB -q=h.root -t tag1 -s since1 -t tag2 -s since2"
    print
    print "   -g, --histo= [file] make histograms and write to file"
    print "         Example : "
    print "           EcalCondDB -g=h.png -t tag1 -s since1 "
    print
    
try:
    opts, args = getopt.getopt(sys.argv[1:],
                             "c:P:t:ld:p:q:g:ms:h",
                             ["connect=","authpath=","tag=","listtags",\
                              "dump=","plot=","compare=","histo=","listiovs",\
                              "since=","help"])
    
    if not len(opts):
        usage()
        sys.exit(0)

except getopt.GetoptError as ex:
    print
    print ex," , use -h or --help for help"
    sys.exit(0)

dbName =  "frontier://FrontierProd/CMS_COND_31X_ECAL"
authpath= "/afs/cern.ch/cms/DB/conddb"

tags=[]
sinces=[]



do_list=0
do_liio=0
do_dump=0
do_plot=0
do_comp=0
do_hist=0

outfilename=None

#shortcut for infinity
inf="4294967295"

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
       if arg.find(".xml")>0 :
           print "WARNING : plot from XML is not protected against changes"
           print "          in DetId or CondFormats"
           
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

    if opt in ("-m","--listiovs"):
       do_liio=1
       
    if opt in ("-s","--since"):
       sinces.append(arg)


   
#done parsing options, now steer


import DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)
from pluginCondDBPyInterface import *
          
a = FWIncantation()

rdbms = RDBMS(authpath)
db = rdbms.getReadOnlyDB(dbName)

import EcalCondTools

if do_list :
   EcalCondTools.listTags(db)

if do_dump :
   if not len(tags):
      print "Must specify tag with -t"
      sys.exit(0)
   EcalCondTools.dumpXML(db,tags[0],sinces[0],outfilename)

if do_plot:
   if not len(tags) or not len (sinces):
       print "Must specify tag with -t [tag] -s [since]"
       sys.exit(0)       
   EcalCondTools.plot(db,tags[0],sinces[0],outfilename) 
 
if do_comp:
   if len(tags) != 2 :
       print "Must give exactly two tags to compare: -t tag1 -t tag2"
       sys.exit(0)
   if  tags[0].find('.xml')<0 and  len(sinces)!=2 :
       print "Must specify tag, since  to compare  with -t [tag1] \
              -s [since1]  -t [tag2] -s [since2]     "
       sys.exit(0)
       
   EcalCondTools.compare(tags[0],db,sinces[0],
                         tags[1],db,sinces[1],outfilename)       

if do_hist:
   if not len(tags):
       print "Must specify tag, since,  with -t [tag] \
              -s [runsince]   (since  not needed for xml)"
       sys.exit(0)
       
   if  tags[0].find('.xml')<0 and  not len(sinces) :
       print "Must specify tag, since, with -t [tag] \
              -s [runsince]   "
       sys.exit(0)
       
   EcalCondTools.histo(db,tags[0],sinces[0],outfilename) 

if do_liio:
    if not len(tags):
       print "Must specify tag  with -t [tag]"
       sys.exit(0)
    EcalCondTools.listIovs(db,tags[0])
     
    
