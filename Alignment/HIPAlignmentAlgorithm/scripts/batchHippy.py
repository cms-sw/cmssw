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
      self.parser.add_option("-f", "--force", action="store_true",
                                dest="force", default=False,
                                help="Don't ask any questions, just over-write")
      self.parser.add_option("--resubmit", action="store_true",
                                dest="resubmit", default=False,
                                help="Resubmit a job from the last iteration")
      (self.opt,self.args) = self.parser.parse_args()

      self.mkdir(self.opt.outputdir)

      if(self.opt.lstfile is None):
         print "Unspecified lst file."
         sys.exit(1)
      if(self.opt.iovfile is None):
         print "Unspecified IOV list."
         sys.exit(1)

   def mkdir( self, dirname ):
      mkdir = 'mkdir -p %s' % dirname
      ret = os.system( mkdir )
      if( ret != 0 ):
         print 'Please remove or rename directory: ', dirname
         sys.exit(4)

   def submitJobs(self):
      jobcmd=""
      if self.opt.resubmit:
         jobcmd = 'scripts/reiterator_py {} {} {} {} {} {}'.format(
         self.opt.niter,
         self.opt.outputdir,
         self.opt.iovfile
         )
      else:
         jobcmd = 'scripts/iterator_py {} {} {} {} {} {}'.format(
         self.opt.niter,
         self.opt.outputdir,
         self.opt.lstfile,
         self.opt.iovfile,
         self.opt.commoncfg,
         self.opt.aligncfg
         )
      ret = os.system( jobcmd )
      if( ret != 0 ):
         sys.exit('Jobs cannot be submitted')




if __name__ == '__main__':
   batchManager = MyBatchManager()
   batchManager.submitJobs()

