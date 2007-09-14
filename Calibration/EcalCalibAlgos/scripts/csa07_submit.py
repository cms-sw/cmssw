#!/usr/bin/env python2.4

import string, sys, os, getopt, subprocess

def usage():

   print "Usage: "+sys.argv[0]+"  -c confile -r runlistfile -e destdir -h -d -n numberjobs"
   print " -h: help, -d:dryrun(do not submit jobs)"
   sys.exit(2)


def main():

 try:
     opts, args = getopt.getopt(sys.argv[1:], "c:r:e:n:sqhd", ["conffile=","runlist=","destdir=","numberjobs=","help","dryrun"])

 except getopt.GetoptError:
     # print help information and exit:
     usage()
     sys.exit(2)

 conffile_template = None
 runlist = None
 dryrun = 0
 
 logdir = "/tmp/"
 destdir = None
 have_destdir = 0
 outputfile_basedir = "/tmp/"
 
 # default number jobs
 n = 999999999

 # base names
 configfile_basename = "config-alcarecophisym-"
 outputfile_basename =  outputfile_basedir+"alcarecophisym-"
 logfile_basename = logdir
 errfile_basename = logdir

 
 for opt, arg in opts:
    
     if opt in ("-c", "--conffile"):
         conffile_template = arg
         if (not os.path.exists(conffile_template)) :
            print sys.argv[0]+" File not found: "+conffile_template
            sys.exit(2)

     if opt in ("-r","--runlist"):
         runlist=arg
         if (not os.path.exists(runlist)) :
            print sys.argv[0]+" File not found: "+runlist
            sys.exit(2)
              
     if opt in ("-e","--destdir"):
         destdir=arg
         have_destdir=1

     if opt in ("-h","--help"):
         usage()

     if opt in ("-d","--dryrun"): 
         dryrun=1

     if opt in ("-n","--njobs"):
         n=int(arg)
  
 # exit condition
 if (conffile_template==None or runlist==None):
    usage()
    exit(2)


 # work directory

 workdir = os.getcwd()
 runs_list=[]
 listofruns=open(runlist)
 runs_list= listofruns.read().split()

 # jobs organization and  prepare configuration file
 # alcarecophisym or calibration process

 nfilesroot = len(runs_list)/n
 
 for jobcount in range(n):
      # solving logic problem
      if jobcount != (n-1):
         inputfilefrag = runs_list[(jobcount*nfilesroot):((jobcount+1)*nfilesroot)]
      else:
         inputfilefrag = runs_list[(jobcount*nfilesroot):]
      
      outputfile = outputfile_basename+str(jobcount)+".root"
      configuration_file = open(conffile_template)
      data = configuration_file.read()
      filenamelist = str(inputfilefrag)
      filenamelist = filenamelist.replace("["," ")
      filenamelist = filenamelist.replace("]"," ")
      filenamelist = filenamelist.replace(",",",\n")   
      data = data.replace("INPUTFILES", filenamelist)
      data = data.replace("OUTPUTFILE", outputfile)
      data = data.replace("NUMBER",str(jobcount+1))
      conffile = configfile_basename+str(jobcount)+".cfg"
      outfile = open(conffile,"w")
      outfile.write(data)
      outfile.close
      configuration_file.close()
   
   
   #now launch job
   
      logfile = logfile_basename+str(jobcount)+".log"
      errfile = errfile_basename+str(jobcount)+".err"
       
      args=['bsub','-q8nh',"-o dummy","-e dummy",workdir+'/cmssw-job.csh',conffile,logfile,errfile,workdir,outputfile,destdir]   

#   to get email notification (debug)
#   args=['bsub','-q8nh',workdir+'/cmssw-job.csh',conffile,logfile,errfile,workdir,outputfile,destdir]   
      
      if (not dryrun):
         cmsRun = subprocess.Popen(args,0,"bsub")
        

if __name__ == "__main__":
    main()
