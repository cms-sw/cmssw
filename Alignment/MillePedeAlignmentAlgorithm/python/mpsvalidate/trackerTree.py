#!/usr/bin/env python

##########################################################################
##
# Check if there is the trackerTree.root file.
##

import logging
import os

from Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper \
    import checked_out_MPS


def check(config):
    logger = logging.getLogger("mpsvalidate")
    logger.info("Check if TrackerTree.root file exists")
    
    outputpath = os.path.join(config.jobDataPath, ".TrackerTree.root")

    # check if file exists
    if not os.path.isfile(outputpath):
        logger.info("TrackerTree.root file does not exist. It will be created now.")
        
        configpath = os.path.join(config.mpspath, "test", "trackerTree_cfg.py")
        logger.info("Path to the config file: {0}".format(configpath))
        
        cmd = "cmsRun {0} outputFile={1}".format(configpath, outputpath)
        if config.globalTag != None: cmd += " globalTag="+config.globalTag
        if config.firstRun != None: cmd += " firstRun="+config.firstRun
        os.system(cmd+" > /dev/null 2>&1")

    return os.path.abspath(outputpath)
