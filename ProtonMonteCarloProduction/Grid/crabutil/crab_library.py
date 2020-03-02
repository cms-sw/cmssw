#!/usr/bin/env python

from crabutil import colors
import numpy as numpy

import time
import os
import sys

from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from CRABClient.UserUtilities import config, getUsernameFromSiteDB
from httplib import HTTPException

color = colors.Paint()

class CrabLibrary():

  def doSubmit(self, dataset, mode, era, year, xangle, mass, configfile, filesPerJob, tagname, enable, with_dataset, lfndir, config):

    timestr = time.strftime("%Y-%m-%d_UTC%H-%M-%S")
    pathfull = '/store/user/%s/%s_%s_%s_%s_%s_%s/' % (getUsernameFromSiteDB(), tagname, mode, era, mass, xangle, timestr) 

    print "\t" + color.BOLD + "Dataset: " + color.ENDC,
    print "\t" + color.OKGREEN + dataset + color.ENDC

    print "\t" + color.BOLD + "Config file: " + color.ENDC,
    print "\t" + color.OKGREEN + configfile + color.ENDC

    print "\t" + color.BOLD + "TagName: " + color.ENDC,
    print "\t" + color.OKGREEN + tagname + color.ENDC

    print "\t" + color.BOLD + "Era: " + color.ENDC,
    print "\t" + color.OKGREEN + era + color.ENDC

    print "\t" + color.BOLD + "Mode: " + color.ENDC,
    print "\t" + color.OKGREEN + mode + color.ENDC

    print "\t" + color.BOLD + "X-Angle: " + color.ENDC,
    print "\t" + color.OKGREEN + xangle + color.ENDC

    print "\t" + color.BOLD + "Mass: " + color.ENDC,
    print "\t" + color.OKGREEN + mass + color.ENDC

    print "\t" + color.BOLD + "Year: " + color.ENDC,
    print "\t" + color.OKGREEN + str(year) + color.ENDC

    print "\t" + color.BOLD + "Files per Job: " + color.ENDC,
    print "\t" + color.OKGREEN + str(filesPerJob) + color.ENDC

    print "\t" + color.BOLD + "Enable: " + color.ENDC,
    print "\t" + color.OKGREEN + str(enable) + color.ENDC

    print "\t" + color.BOLD + "With Dataset: " + color.ENDC,
    print "\t" + color.OKGREEN + str(with_dataset) + color.ENDC

    print "\t" + color.BOLD + "LFN output dir: " + color.ENDC,
    print "\t" + color.OKGREEN + str(pathfull) + color.ENDC

    timestr = time.strftime("%Y-%m-%d_UTC%H-%M-%S")

    if int(with_dataset):
        config.JobType.pluginName = 'Analysis'
        config.Data.inputDataset = dataset
    else:
        config.JobType.pluginName = 'PrivateMC'
        print "\t" + color.BOLD + color.HEADER + "-- Submittion without dataset --" + color.ENDC

    config.General.transferLogs = False
    config.General.transferOutputs = True
    config.JobType.maxMemoryMB = 2500
    config.Data.inputDBS = 'phys03'
    config.JobType.allowUndistributedCMSSW = True
    config.Data.splitting = 'EventBased' # or 'LumiBased' or 'Automatic' or 'FileBased'
    config.Data.unitsPerJob = int(filesPerJob)
    NJOBS = 5 #250
    config.Data.totalUnits = NJOBS * config.Data.unitsPerJob
    config.Data.outputPrimaryDataset = tagname + "_" + mode + "_" + era + "_" + str(mass) + "_" + str(xangle)
    config.Data.publication = True
    config.JobType.psetName = configfile
    config.JobType.outputFiles = ['output.root']
    config.General.workArea = 'crab_' + config.Data.outputPrimaryDataset + '_' + timestr
    config.General.requestName = config.Data.outputPrimaryDataset + "_" + timestr
    config.Data.outputDatasetTag = config.Data.outputPrimaryDataset + "_" + timestr
    config.Site.storageSite = 'T2_IT_Pisa' #T2_IT_Pisa, T2_CH_CERNBOX
    config.Data.outLFNDirBase = pathfull
    config.JobType.pyCfgParams = ["Mode="+mode,"Era="+era,"Mass="+str(mass),"XAngle="+str(xangle)]

    if int(enable):
    	res = crabCommand('submit', config = config)
    else:
    	print "\t" + color.BOLD + color.HEADER + "-- Submittion not enabled --" + color.ENDC

    print "\n"
