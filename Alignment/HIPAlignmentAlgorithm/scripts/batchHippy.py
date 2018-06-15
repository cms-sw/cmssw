#!/bin/env python

import sys
import imp
import copy
import os
import shutil
import pickle
import math
import pprint
import subprocess
from datetime import date
from optparse import OptionParser


class MyBatchManager:
   '''Batch manager specific to cmsRun processes.'''

   def __init__(self):
      # define options and arguments ====================================
      self.parser = OptionParser()
      self.parser.add_option("-o", "--outdir", dest="outputdir", type="string",
                                help="Name of the local output directory for your jobs. This directory will be created automatically.",
                                default="./")
      self.parser.add_option("--commoncfg", dest="commoncfg", type="string",
                                help="Name of the common config file.",
                                default="python/common_cff_py.txt")
      self.parser.add_option("--aligncfg", dest="aligncfg", type="string",
                                help="Name of the align. config file.",
                                default="python/align_tpl_py.txt")
      self.parser.add_option("--niter", dest="niter", type="int",
                                help="Number of iterations",
                                default="15")
      self.parser.add_option("--lst", "--listfile", "--lstfile", dest="lstfile", type="string",
                                help="lst file to read",
                                default=None)
      self.parser.add_option("--iovs", "--iovfile", dest="iovfile", type="string",
                                help="IOV list to read",
                                default=None)
      self.parser.add_option("--trkselcfg", "--trackselectionconfig", dest="trkselcfg", type="string",
                                help="Track selection config location",
                                default="python")
      self.parser.add_option("--notify", "--sendto", dest="sendto", type="string",
                                help="Email addresses (comma-separated) to notify when job is complete.",
                                default=None)
      self.parser.add_option("--deform", action="store_true",
                                dest="useSD", default=False,
                                help="Include surface deformations in alignment")
      self.parser.add_option("-f", "--force", action="store_true",
                                dest="force", default=False,
                                help="Don't ask any questions, just over-write")
      self.parser.add_option("--resubmit", action="store_true",
                                dest="resubmit", default=False,
                                help="Resubmit a job from the last iteration")
      self.parser.add_option("--redirectproxy", action="store_true",
                                dest="redirectproxy", default=False,
                                help="Redirect the proxy to a path visible in batch")
      self.parser.add_option("--dry", dest="dryRun", type="int",
                                default=0,
                                help="Do not submit jobs, just set up the cfg files")
      (self.opt,self.args) = self.parser.parse_args()

      self.checkProxy() # Check if Grid proxy initialized

      self.mkdir(self.opt.outputdir)

      if self.opt.lstfile is None:
         print "Unspecified lst file."
         sys.exit(1)
      if self.opt.iovfile is None:
         print "Unspecified IOV list."
         sys.exit(1)

      self.jobname = self.opt.outputdir.split('/')[-1]

      if self.opt.redirectproxy:
         print "Job {} is configured to redirect its Grid proxy.".format(self.jobname)
         self.redirectProxy()

      if self.opt.sendto is not None:
         self.opt.sendto.strip()
         self.opt.sendto.replace(","," ")
         print "Job {} is configured to notify {}.".format(self.jobname, self.opt.sendto)

      # Set numerical flags for iterator_py
      self.SDflag = 1 if self.opt.useSD else 0
      self.redirectproxyflag = 1 if self.opt.redirectproxy else 0


   def mkdir(self, dirname):
      mkdir = 'mkdir -p %s' % dirname
      ret = os.system( mkdir )
      if( ret != 0 ):
         print 'Please remove or rename directory: ', dirname
         sys.exit(4)

   def notify(self, desc):
      print desc
      if self.opt.sendto is not None:
         strcmd = "mail -s {1} {0} <<< \"{2}\"".format(self.opt.sendto, self.jobname, desc)
         os.system(strcmd)

   def checkLastIteration(self):
      lastIter=self.opt.niter
      doesExist=os.system("test -s {}/alignments_iter{}.db".format(self.opt.outputdir, lastIter))
      while (doesExist != 0):
         lastIter -= 1
         if lastIter < 0:
            break
         doesExist=os.system("test -s {}/alignments_iter{}.db".format(self.opt.outputdir, lastIter))
      return lastIter

   def finalize(self, ret):
      strresult=""
      exitCode=0
      if( ret != 0 ):
         strresult = "Jobs cannot be submitted for {}. Exiting...".format(self.jobname)
         exitCode=1
      elif self.opt.dryRun > 0:
         strresult = "Dry run setup is complete for {}.".format(self.jobname)
      else:
         lastIter=self.checkLastIteration()
         if lastIter == self.opt.niter:
            strresult = "The final iteration {}/alignments_iter{}.db is recorded successfully.".format(self.jobname, lastIter)
         elif lastIter>0:
            strresult = "The last successful iteration was {}/alignments_iter{}.db out of the {} requested iterations.".format(self.jobname, lastIter, self.opt.niter)
            exitCode=1
         else:
            strresult = "None of the {} iterations were successful in job {}.".format(self.opt.niter, self.jobname)
            exitCode=1
      self.notify(strresult)
      if exitCode!=0:
         sys.exit(strresult)

   def submitJobs(self):
      jobcmd=""
      if self.opt.resubmit:
         jobcmd = 'scripts/reiterator_py {} {} {} {} {} {}'.format(
         self.opt.niter,
         self.opt.outputdir,
         self.opt.iovfile
         )
      else:
         if self.opt.dryRun > 0:
            print 'Dry run option is enabled. Will not submit jobs to the queue'
         jobcmd = 'scripts/iterator_py {} {} {} {} {} {} {} {} {} {}'.format(
         self.opt.niter,
         self.opt.outputdir,
         self.opt.lstfile,
         self.opt.iovfile,
         self.opt.commoncfg,
         self.opt.aligncfg,
         self.opt.trkselcfg,
         self.SDflag,
         self.redirectproxyflag,
         self.opt.dryRun
         )
      ret = os.system( jobcmd )
      self.finalize(ret)

   def checkProxy(self):
      try:
         subprocess.check_call(["voms-proxy-info", "--exists"])
      except subprocess.CalledProcessError:
         print "Please initialize your proxy before submitting."
         sys.exit(1)

   def redirectProxy(self):
      local_proxy = subprocess.check_output(["voms-proxy-info", "--path"]).strip()
      new_proxy_path = os.path.join(self.opt.outputdir,".user_proxy")
      print "Copying local proxy {} to the job directory as {}.".format(local_proxy,new_proxy_path)
      shutil.copyfile(local_proxy, new_proxy_path)



if __name__ == '__main__':
   batchManager = MyBatchManager()
   batchManager.submitJobs()
