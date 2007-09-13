#!/usr/bin/env python2.4

import string, sys, os, getopt, subprocess

def usage():

   print "Usage: "+sys.argv[0]+"  -r runlistfile -s filesstageing -q stageinfos -h"
   print " -h: help"
   sys.exit(2)

def dostaging(filelist):
   print "Asking stageing for files"
   for file in filelist:
      print file
      values1=['stager_get ','-M','/castor/cern.ch/cms'+str(file)]
      do_stager=subprocess.Popen(values1,0,"stager_get")

def staginginfo(filelist):
   
   print "Asking stager status informations"
   for file in filelist:
      print file
      file="/castor/cern.ch/cms"+file
      values1=['stager_qry','-M',file]
      stager_info=subprocess.Popen(values1,0, "stager_qry")




def main():

 try:
     opts, args = getopt.getopt(sys.argv[1:], "r:sqh", ["runlist=","filesstageing=","stagerinfos=","help"])

 except getopt.GetoptError:
     # print help information and exit:
     usage()
     sys.exit(2)

 
 runlist=None
 logdir="/tmp/"
 # stage of files
 do_staging= False
 # give stageing informations
 staging_infos= False

 outputfile_basedir ="/tmp/"

 for opt, arg in opts:

     if opt in ("-r","--runlist"):
         runlist=arg
         if (not os.path.exists(runlist)) :
            print sys.argv[0]+" File not found: "+runlist
            sys.exit(2)

     if opt in ("-h","--help"):
         usage()

     if opt in ("-s","--filesstageing"):
         do_staging = True
         
     if opt in ("-q","--filesstageinginfos"):
         staging_infos = True    

 if ( runlist==None and not (do_staging or staging_info)):
    usage()
    exit(2)


# work directory

 workdir= os.getcwd()

 runs_list=[]

 listofruns=open(runlist)

 runs_list= listofruns.read().split()
 
 if do_staging:
    dostaging(runs_list)
    

 if staging_infos:
    staginginfo(runs_list)
    


if __name__ == "__main__":
    main()
