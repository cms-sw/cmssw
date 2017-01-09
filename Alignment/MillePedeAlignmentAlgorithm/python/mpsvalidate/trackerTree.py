#!/usr/bin/env python

##########################################################################
##
# Check if there is the trackerTree.root file.
##

import logging
import os


def check(config):
    logger = logging.getLogger("mpsvalidate")
    logger.info("Check if TrackerTree.root file exists")
    
    outputpath = os.path.join(os.environ['CMSSW_BASE'], "src", "Alignment", "MillePedeAlignmentAlgorithm", "python", "mpsvalidate", "TrackerTree.root")
    print(outputpath)

    # check if file exists
    if (not os.path.isfile(outputpath)):
        logger.info("TrackerTree.root file does not exist. It will be created now.")
        
        configpath = os.path.join(os.environ["CMSSW_BASE"], "src", "Alignment", "MillePedeAlignmentAlgorithm", "test", "trackerTree_cfg.py")
        logger.info("Path to the config file: {0}".format(configpath))
        
        
        os.system("cmsRun {0} outputpath={1}".format(configpath, outputpath))