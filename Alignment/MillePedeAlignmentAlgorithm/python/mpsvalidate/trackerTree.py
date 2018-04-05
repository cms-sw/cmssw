##########################################################################
##
# Check if there is the trackerTree.root file.
##

import logging
import os

from Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper \
    import checked_out_MPS
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools


def check(config):
    logger = logging.getLogger("mpsvalidate")
    logger.info("Check if TrackerTree.root file exists")
    
    outputpath = os.path.join(config.jobDataPath, ".TrackerTree.root")

    # check if file exists
    if not os.path.isfile(outputpath):
        logger.info("TrackerTree.root file does not exist. It will be created now.")
        
        configpath = os.path.join(config.mpspath, "test", "trackerTree_cfg.py")
        logger.info("Path to the config file: {0}".format(configpath))
        
        cmd = ["cmsRun", configpath, "outputFile="+outputpath]
        if config.globalTag != None: cmd.append("globalTag="+config.globalTag)
        if config.firstRun != None: cmd.append("firstRun="+config.firstRun)
        mps_tools.run_checked(cmd, suppress_stderr = True)

    return os.path.abspath(outputpath)
