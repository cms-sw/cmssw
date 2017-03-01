#!/usr/bin/env python

##########################################################################
# Creates html out of the histograms, parsed data and a given template.
##

import logging
import os
import string

from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes import MonitorData, PedeDumpData
from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.geometry import Alignables, Structure


# create class to have delimiter %% which is not used in latex


class TexTemplate(string.Template):
    delimiter = "%%"


def create(alignables, pedeDump, additionalData, outputFile, config):
    logger = logging.getLogger("mpsvalidate")

    # load template
    with open(os.path.join(config.mpspath, "html_template.html"), "r") as template:
        data = template.read()
        template.close()

    # create object where data could be substituted
    data = TexTemplate(data)

    # output string
    out = ""

    # general information

    out += "<h1>General information</h1>\n"

    if (config.message):
        out += "Project: {0}\n<br>".format(config.message)
    out += "Input-Path: {0}\n<br>".format(config.jobDataPath)
    
    # alignment_merge.py
    try:
        out += "<h2>Alignment Configuration</h2>\n"
        out += "<b>PedeSteerer method:</b> {0}<br>\n".format(
            additionalData.pedeSteererMethod)
        out += "<b>PedeSteerer options:</b>\n"
        for line in additionalData.pedeSteererOptions:
            out += "{0}<br>\n".format(line)
        out += "<b>PedeSteerer command:</b> {0}<br>\n".format(
            additionalData.pedeSteererCommand)

        for selector in additionalData.pattern:
            out += "<b>{0}:</b><br>\n".format(additionalData.pattern[selector][3])
            for line in additionalData.pattern[selector][0]:
                for i in line:
                    out += "{0} ".format(i)
                out += "<br>\n"
            for line in additionalData.pattern[selector][2]:
                out += "{0} \n".format(line)
                out += "<br>\n"
    except Exception as e:
        logger.error("data not found - {0} {1}".format(type(e), e))
            
    # table of input files with number of tracks
    if (config.showmonitor):
        out += "<h2>Datasets with tracks</h2>\n"
        out += """<table border="1">
            <tr>
               <th>Dataset</th>
               <th>Number of used tracks</th>
            <tr>"""
        for monitor in MonitorData.monitors:
            out += """<tr>
                <th>{0}</th>
                <th>{1}</th>
                </tr>""".format(monitor.name, monitor.ntracks)
        try:
            if (pedeDump.nrec):
                out += """<tr>
                    <th>Number of records</th>
                    <th>{0}</th>
                    </tr>""".format(pedeDump.nrec)
        except Exception as e:
            logger.error("data not found - {0} {1}".format(type(e), e))
        out += """</table>"""
        out += "The information in this table is based on the monitor root files. Note that the number of tracks which where used in the pede step can differ from this table."

    # pede.dump.gz
    try:
        out += "<h2>Pede monitoring information</h2>\n"
        if (pedeDump.sumValue != 0):
            out += r"<b>Sum(Chi^2)/Sum(Ndf)</b> &= {0}<br> &= {1}".format(
                pedeDump.sumSteps, pedeDump.sumValue)
        else:
            out += r"<b>Sum(W*Chi^2)/Sum(Ndf)/<W></b> &= {0}<br> &= {1}".format(
                pedeDump.sumSteps, pedeDump.sumWValue)
        out += r"<b>with correction for down-weighting:</b> {0}<br>".format(
            pedeDump.correction)
        out += r"<b>Peak dynamic memory allocation:</b> {0} GB<br>".format(
            pedeDump.memory)
        out += r"<b>Total time:</b> {0} h {1} m {2} s<br>".format(
            pedeDump.time[0], pedeDump.time[1], pedeDump.time[2])
        out += r"<b>Number of records:</b> {0}<br>".format(pedeDump.nrec)
        out += r"<b>Total number of parameters:</b> {0}<br>".format(pedeDump.ntgb)
        out += r"<b>Number of variable parameters:</b> {0}<br>".format(pedeDump.nvgb)
        out += r"<b>Warning:</b><br>"
        for line in pedeDump.warning:

            # check if line empty
            if line.replace(r" ", r""):
                out += "{0}<br>\n".format(line)
    except Exception as e:
        logger.error("data not found - {0} {1}".format(type(e), e))

    # high level structures

    big = [x for x in config.outputList if (x.plottype == "big")]

    if big:
        out += "<h1>High-level parameters</h1>\n"
        for i in big:
            out += "<a href='plots/pdf/{0}.pdf'><img src='plots/png/{0}.png'></a>\n".format(
                i.filename)

    # time (IOV) dependent plots

    time = [x for x in config.outputList if (x.plottype == "time")]

    if time:
        out += "<h1>High-level parameters versus time (IOV)</h1>\n"
        # get list with names of the structures
        for structure in [x.name for x in time if x.parameter == "xyz"]:
            out += "<h2>{0}<h2>\n".format(structure)
            for mode in ["xyz", "rot"]:
                if any([x.filename for x in time if (x.parameter == mode and x.name == structure)]):
                    filename = [x.filename for x in time if (x.parameter == mode and x.name == structure)][0]
                    out += "<a href='plots/pdf/{0}.pdf'><img src='plots/png/{0}.png'></a>\n".format(
                        filename)

    # hole modules

    # check if there are module plots
    if any(x for x in config.outputList if (x.plottype == "mod" and x.number == "")):
        out += "<h1>Module-level parameters</h1>\n"

        # loop over all structures
        for moduleName in [x.name for x in alignables.structures]:

            # check if there is a plot for this module
            if any(x for x in config.outputList if (x.plottype == "mod" and x.number == "" and x.name == moduleName)):
                out += "<h2>{0}</h2>\n".format(moduleName)

                # loop over modes
                for mode in ["xyz", "rot", "dist"]:

                    # get module plot
                    module = [x for x in config.outputList if (
                        x.plottype == "mod" and x.number == "" and x.name == moduleName and x.parameter == mode)]
                    # get list of sub module plots
                    moduleSub = [x for x in config.outputList if (
                        x.plottype == "subMod" and x.number != "" and x.name == moduleName and x.parameter == mode)]

                    # check if plot there is a plot in this mode
                    if module:
                        out += "<a href='plots/pdf/{0}.pdf'><img src='plots/png/{0}.png'></a>\n".format(module[
                                                                                                        0].filename)

                        # loop over submodules
                        for plot in moduleSub:
                            out += "<a href='plots/pdf/{0}.pdf'><img src='plots/png/{0}.png'></a>\n".format(
                                plot.filename)

    # plot taken from the millePedeMonitor_merge.root file

    if any(x for x in config.outputList if x.plottype == "monitor"):
        out += "<h1>Monitor</h1>\n"
        for plot in [x for x in config.outputList if x.plottype == "monitor"]:
            out += "<h3>{0}</h3>\n".format(plot.name)
            out += "<a href='plots/pdf/{0}.pdf'><img src='plots/png/{0}.png'></a>\n".format(
                plot.filename)

    data = data.substitute(message=config.message, out=out)

    with open(os.path.join(config.outputPath, outputFile), "w") as output:
        output.write(data)
