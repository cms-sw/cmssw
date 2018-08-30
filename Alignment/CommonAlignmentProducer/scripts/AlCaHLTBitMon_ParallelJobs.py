#!/usr/bin/env python

#-----------------------------------------------------
# original author: Andrea Lucaroni
# Revision:        $Revision: 1.53 $
# Last update:     $Date: 2011/06/28 19:01:36 $
# by:              $Author: mussgill $
#-----------------------------------------------------

from __future__ import print_function
import re
import json
import os 
import stat
import sys

import array
import pickle as pk

from optparse import OptionParser
#####DEBUG
DEBUG = 0

def getConfigTemplateFilename():
    template = os.path.expandvars('$CMSSW_BASE/src/Alignment/CommonAlignmentProducer/data/AlCaHLTBitMon_cfg_template_py')
    if os.path.exists(template):
        return template
    template = os.path.expandvars('$CMSSW_RELEASE_BASE/src/Alignment/CommonAlignmentProducer/data/AlCaHLTBitMon_cfg_template_py')
    if os.path.exists(template):
        return template
    return 'None'

def mkHLTKeyListList(hltkeylistfile):
    keylistlist = []
    f = open(hltkeylistfile, 'r')
    for line in f:
        keylistlist.append(line.replace('\n',''))
    f.close()
    return keylistlist

def parallelJobs(hltkeylistfile,jsonDir,globalTag,templateName,queue,cafsetup):
    PWD = os.path.abspath('.')

    if templateName == 'default':
        templateFile = getConfigTemplateFilename()
    else:
        templateFile = os.path.abspath(os.path.expandvars(templateName))

    tfile = open(templateFile, 'r')
    template = tfile.read()
    tfile.close()
    template = template.replace('%%%GLOBALTAG%%%', globalTag)

    keylistlist = mkHLTKeyListList(hltkeylistfile)
    index = 0
    for keylist in keylistlist:

        configName = 'AlCaHLTBitMon_%d_cfg.py'%index
        jobName = 'AlCaHLTBitMon_%d_job.csh'%index
        jsonFile = os.path.abspath('%(dir)s/%(index)d.json' % {'dir' : jsonDir, 'index' : index})
        dataFile = os.path.abspath('%(dir)s/%(index)d.data' % {'dir' : jsonDir, 'index' : index})
        logFile = 'AlCaHLTBitMon_%d'%index

        dfile = open(dataFile, 'r')
        data = dfile.read()
        dfile.close()

        config = template.replace('%%%JSON%%%', jsonFile);
        config = config.replace('%%%DATA%%%', data);
        config = config.replace('%%%KEYNAME%%%', keylist);
        config = config.replace('%%%LOGFILE%%%', logFile);

        cfile = open(configName, 'w')
        for line in config:
            cfile.write(line)
        cfile.close()

        jfile = open(jobName, 'w')
        jfile.write('#!/bin/tcsh\n\n')
        if cafsetup == True:
            jfile.write('source /afs/cern.ch/cms/caf/setup.csh\n\n')

        jfile.write('cd $1\n\n')
        jfile.write('eval `scramv1 run -csh`\n\n')
        jfile.write('cmsRun %s\n\n'%configName)
        jfile.write('cp %s.log $2'%logFile)
        jfile.close()

        if os.path.exists('./%s'%keylist) == False:
            os.system('mkdir ./%s'%keylist)

        os.system('chmod u+x %s'%jobName)
        #print('bsub -q %(queue)s %(jobname)s %(pwd)s %(pwd)s/%(outdir)s' % {'queue' : queue, 'jobname' : jobName, 'pwd' : PWD, 'outdir' : keylist})
        os.system('bsub -q %(queue)s %(jobname)s %(pwd)s %(pwd)s/%(outdir)s' % {'queue' : queue, 'jobname' : jobName, 'pwd' : PWD, 'outdir' : keylist})

        index = index + 1

def defineOptions():
    parser = OptionParser()

    parser.add_option("-k", "--keylist",
                      dest="hltKeyListFile",
                      default="lista_key.txt",
                      help="text file with HLT keys")

    parser.add_option("-j", "--json",
                      dest="jsonDir",
                      help="directory with the corresponding json files")

    parser.add_option("-g", "--globalTag",
                      dest="globalTag",
                      help="the global tag to use in the config files")

    parser.add_option("-t", "--template",
                      dest="template",
                      default="default",
                      help="the template to use for the config files")

    parser.add_option("-q", "--queue",
                      dest="queue",
                      default="cmscaf1nd",
                      help="the queue to use (default=cmscaf1nd)")

    parser.add_option("-c", "--cafsetup", action="store_true",
                      dest="cafsetup",
                      default=False,
                      help="wether the caf setup is sourced in the scripts")

    (options, args) = parser.parse_args()

    if len(sys.argv) == 1:
        print("\nUsage: %s --help"%sys.argv[0])
        sys.exit(0)

    if str(options.hltKeyListFile) == 'None':
        print("Please provide a file with HLT keys")
        sys.exit(0)

    if str(options.jsonDir) == 'None':
        print("Please provide a directory containing the json files")
        sys.exit(0)

    if str(options.globalTag) == 'None':
        print("Please provide a global tag")
        sys.exit(0)

    return options


#---------------------------------------------MAIN

options = defineOptions()
p = parallelJobs(options.hltKeyListFile,
                 options.jsonDir,
                 options.globalTag,
                 options.template,
                 options.queue,
                 options.cafsetup)
