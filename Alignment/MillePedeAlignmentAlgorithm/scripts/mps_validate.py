#!/usr/bin/env python3

##########################################################################
# Create histograms out of treeFile_merge.root . The pede.dump.gz file is
# parsed. The histograms are plotted as PNG files. The output data is
# created as PDF, HTML, ...
##

import argparse
import glob
import logging
import os
import shutil
import sys

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()

import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.style as mpsv_style
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.geometry as mpsv_geometry
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.bigModule as mpsv_bigModule
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.iniparser as mpsv_iniparser
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.dumpparser as mpsv_dumpparser
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.pdfCreator as mpsv_pdfCreator
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.htmlCreator as mpsv_htmlCreator
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.monitorPlot as mpsv_monitorPlot
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.trackerTree as mpsv_trackerTree
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.bigStructure as mpsv_bigStructure
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.beamerCreator as mpsv_beamerCreator
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.timeStructure as mpsv_timeStructure
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.additionalparser as mpsv_additionalparser



def main():
    # config logging module
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s (%(pathname)s line %(lineno)d): %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("mpsvalidate")
    
    # ArgumentParser
    parser = argparse.ArgumentParser(description="Validate your Alignment.")
    parser.add_argument(
        "-j", "--job", help="chose jobmX directory (default: ini-file)", default=-1, type=int)
    parser.add_argument(
        "-t", "--time", help="chose MillePedeUser_X Tree (default: ini-file)", default=-1, type=int)
    parser.add_argument("-i", "--ini", help="specify a ini file")
    parser.add_argument("-m", "--message",
                        help="identification on every plot", default="")
    parser.add_argument("-p", "--jobdatapath",
                        help="path to the jobm directory", default="")
    parser.add_argument("-o", "--outputpath",
                        help="outputpath", default="")
    parser.add_argument("-l", "--logging",
                        help="if this argument is given a logging file (validation.log) is saved in the current directory", action="store_true")
    parser.add_argument("-c", "--copy",
                        help="creates a copy of the validation_user.ini file in the current directory", action="store_true")
    args = parser.parse_args()

    # create config object
    config = mpsv_iniparser.ConfigData()
    
    # create logging handler
    if args.logging:
        handler = logging.FileHandler("validation.log", mode="w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s %(asctime)s (%(pathname)s line %(lineno)d): %(message)s",
                                      datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # parse default ini file
    logger.info("start to parse the default.ini")
    config.parseConfig(os.path.join(config.mpspath, "templates",
                                    "mpsvalidate_default.ini"))
    
    # copy of ini file in current directory
    if args.copy:
        logger.info("create copy of validation_user.ini in current directory")
        shutil.copy2(os.path.join(config.mpspath, "templates", "mpsvalidate_default.ini"),
                     "validation_user.ini")
        sys.exit()
        

    # parse user ini file
    if args.ini != None:
        logger.info("start to parse the user ini: {0}".format(args.ini))
        config.parseConfig(args.ini)

    # override ini configs with consol parameter
    config.parseParameter(args)

    # create output directories
    logger.info("create the output directories")
    if not os.path.exists(os.path.join(config.outputPath, "plots/pdf")):
        os.makedirs(os.path.join(config.outputPath, "plots/pdf"))
    if not os.path.exists(os.path.join(config.outputPath, "plots/png")):
        os.makedirs(os.path.join(config.outputPath, "plots/png"))

    # open root file and get TTree MillePedeUser_X
    logger.info("try to open the root file: {0}".format(os.path.join(config.jobDataPath, "treeFile_merge.root")))
    treeFile = ROOT.TFile(os.path.join(config.jobDataPath, "treeFile_merge.root"))
    MillePedeUser = treeFile.Get("MillePedeUser_{0}".format(config.jobTime))
    if not MillePedeUser:
        logger.error("Could not open TTree File MillePedeUser_{0} in {1}".format(
            config.jobTime, os.path.join(config.jobDataPath, "treeFile_merge.root")))
        return

    # set gStyle
    mpsv_style.setgstyle()
    
    # create alignables object
    alignables = mpsv_geometry.Alignables(config)
    
    # check if there is the TrackerTree.root file and if not create it
    mpsv_trackerTree.check(config)

    ##########################################################################
    # draw the plots of the millePedeMonitor_merge.root file
    #

    if config.showmonitor:
        try:
            logger.info("start to collect the plots of the millePedeMonitor_merge.root file")
            mpsv_monitorPlot.plot(config)
        except Exception as e:
            logging.error("millePedeMonitor_merge.root failure - {0} {1}".format(type(e), e))
            raise

    ##########################################################################
    # parse the alignment_merge.py file
    #

    if config.showadditional:
        logger.info("start to parse the alignment_merge.py file")
        try:
            additionalData = mpsv_additionalparser.AdditionalData()
            additionalData.parse(
                config, os.path.join(config.jobDataPath, "alignment_merge.py"))
        except Exception as e:
            logging.error("alignment_merge.py parser failure - {0} {1}".format(type(e), e))
            raise

    ##########################################################################
    # parse the file pede.dump.gz and return a PedeDumpData Object
    #

    if config.showdump:
        try:
            logger.info("start to parse the pede.dump.gz file")
            pedeDump = mpsv_dumpparser.parse(
                os.path.join(config.jobDataPath, "pede.dump.gz"), config)
        except Exception as e:
            logging.error("pede.dump.gz parser failure - {0} {1}".format(type(e), e))
            raise

    ##########################################################################
    # time dependend big structures
    #

    if config.showtime:
        try:
            logger.info("create the time dependent plots")
            mpsv_timeStructure.plot(treeFile, alignables, config)
        except Exception as e:
            logging.error("time dependent plots failure - {0} {1}".format(type(e), e))
            raise

    ##########################################################################
    # big structures
    #

    if config.showhighlevel:
        try:
            logger.info("create the high level plots")
            mpsv_bigStructure.plot(MillePedeUser, alignables, config)
        except Exception as e:
            logging.error("high level plots failure - {0} {1}".format(type(e), e))
            raise

    ##########################################################################
    # modules of a hole structure
    # and part of structure
    #

    if config.showmodule:
        try:
            logger.info("create the module plots")
            mpsv_bigModule.plot(MillePedeUser, alignables, config)
        except Exception as e:
            logging.error("module plots failure - {0} {1}".format(type(e), e))
            raise

    ##########################################################################
    # create TEX, beamer
    #

    if config.showtex:
        try:
            logger.info("create the latex file")
            mpsv_pdfCreator.create(alignables, pedeDump, additionalData,
                                   config.latexfile, config)
        except Exception as e:
            logging.error("latex creation failure - {0} {1}".format(type(e), e))
            raise
        
    if config.showbeamer:
        try:
            logger.info("create the latex beamer file")
            mpsv_beamerCreator.create(alignables, pedeDump, additionalData,
                                      "beamer.tex", config)
        except Exception as e:
            logging.error("beamer latex failure - {0} {1}".format(type(e), e))
            raise
        
    # delete latex temporary files
    for extension in ["aux", "log", "nav", "out", "snm", "toc"]:
        extension = "*." + extension
        pattern = os.path.join(config.outputPath, extension)
        logger.info("Remove temporary latex files: "+pattern)
        map(os.remove, glob.glob(pattern))
        
    if config.showhtml:
        try:
            logger.info("create the HTML file")
            mpsv_htmlCreator.create(alignables, pedeDump, additionalData,
                                    "html_file.html", config)
        except Exception as e:
            logging.error("HTML creation failure - {0} {1}".format(type(e), e))
            raise

if __name__ == "__main__":
    main()
