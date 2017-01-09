#!/usr/bin/env python

##########################################################################
# Creates beamer out of the histograms, parsed data and a given template.
##

import logging
import os
import string

from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes import MonitorData, PedeDumpData
from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.geometry import Alignables, Structure


# create class to have delimiter %% which is not used in latex


class TexTemplate(string.Template):
    delimiter = "%%"


class Out:

    def __init__(self):
        self.text = ""

    def addSlide(self, head, text):
        self.text += "\\begin{{frame}}[t]{{{0}}}\n".format(head)
        self.text += text
        self.text += """\\vfill
                        \\rule{0.9\paperwidth}{1pt}
                        \insertnavigation{0.89\paperwidth}
                        \\end{frame}\n"""
                        
    def addSlide_fragile(self, head, text):
        self.text += "\\begin{{frame}}[fragile=singleslide]{{{0}}}\n".format(head)
        self.text += text
        self.text += """\\vfill
                        \\rule{0.9\paperwidth}{1pt}
                        \insertnavigation{0.89\paperwidth}
                        \\end{frame}\n"""
                        
    def add(self, text):
        self.text += text + "\n"


def create(alignables, pedeDump, additionalData, outputFile, config):
    logger = logging.getLogger("mpsvalidate")
    
    # load template
    with open(os.path.join(config.mpspath, "beamer_template.tex"), "r") as template:
        data = template.read()
        template.close()

    # create object where data could be substituted
    data = TexTemplate(data)

    # output string
    out = Out()
    text = ""
    
    # title page
    if (config.message):
        text += """\centering
                    \\vspace*{{4cm}}
                    \Huge\\bfseries Alignment Validation\par
                    \\vspace{{2cm}}	
                    \scshape\huge Alignment Campaign\\\\ {{{0}}}\par
                    \\vfill
                    \large \\today\par""".format(config.message)
    else:
        text += """\centering
                    \\vspace*{4cm}
                    \Huge\\bfseries Alignment Validation\par
                    \\vfill
                    \large \\today\par"""
    out.addSlide("", text)
    
    # table of contents
    text = "\\tableofcontents"
    out.addSlide("Overview", text)

    # general information
    out.add("\section{General information}")
    text = ""
    if (config.message):
        text = "Project: {{{0}}}\\\\\n".format(config.message)
    text += "Input-Path:\n"
    text += "\\begin{verbatim}\n"
    text += config.jobDataPath+"\n"
    text += "\\end{verbatim}\n"
    out.addSlide_fragile("General information", text)
    
    # alignment_merge.py
    try:
        out.add("\subsection{Alignment Configuration}")
        text = "\\textbf{{PedeSteerer method:}} {{{0}}}\\\\\n".format(
            additionalData.pedeSteererMethod)
        text += "\\textbf{{PedeSteerer options:}}\\\\\n"
        for line in additionalData.pedeSteererOptions:
            text += "{{{0}}}\\\\\n".format(line)
        text += "\\textbf{{PedeSteerer command:}} {0}\\\\\n".format(
            additionalData.pedeSteererCommand)
        out.addSlide("Alignment Configuration", text)
    except Exception as e:
        logger.error("data not found - {0} {1}".format(type(e), e))
    
    # table of input files with number of tracks
    if (config.showmonitor):
        out.add("\subsection{Datasets with tracks}")
        text = """\\begin{table}[h]
            \centering
            \caption{Datasets with tracks}
            \\begin{tabular}{cc}
            \hline
            Dataset & Number of used tracks \\\\
            \hline \n"""
        try:
            for monitor in MonitorData.monitors:
                text += "{0} & {1}\\\\\n".format(monitor.name, monitor.ntracks)
        except Exception as e:
            logger.error("data not found - {0} {1}".format(type(e), e))
        if (pedeDump.nrec):
            text += "Number of records & {0}\\\\\n".format(pedeDump.nrec)
        text += """\hline
                  \end{tabular}\n
                  \end{table}\n"""
        text += "The information in this table is based on the monitor root files. Note that the number of tracks which where used in the pede step can differ from this table.\n"
        out.addSlide("Datasets with tracks", text)

    # pede.dump.gz
    out.add("\subsection{Pede monitoring information}")
    try:
        if (pedeDump.sumValue != 0):
            text = r"\begin{{align*}}Sum(Chi^2)/Sum(Ndf) &= {0}\\ &= {1}\end{{align*}}".format(
                pedeDump.sumSteps, pedeDump.sumValue)
        else:
            text = r"\begin{{align*}}Sum(W*Chi^2)/Sum(Ndf)/<W> &= {0}\\ &= {1}\end{{align*}}".format(
                pedeDump.sumSteps, pedeDump.sumWValue)
        text += r"with correction for down-weighting: {0}\\".format(
            pedeDump.correction)
        text += r"Peak dynamic memory allocation: {0} GB\\".format(pedeDump.memory)
        text += r"Total time: {0} h {1} m {2} s\\".format(
            pedeDump.time[0], pedeDump.time[1], pedeDump.time[2])
        text += r"Number of records: {0}\\".format(pedeDump.nrec)
        text += r"Total number of parameters: {0}\\".format(pedeDump.ntgb)
        text += r"Number of variable parameters: {0}\\".format(pedeDump.nvgb)
        out.addSlide("Pede monitoring information", text)
    except Exception as e:
        logger.error("data not found - {0} {1}".format(type(e), e))
        
    # Parameter plots
    out.add("\section{Parameter plots}")
    
    # high level Structures
    out.add("\subsection{High-level parameters}")
    big = [x for x in config.outputList if (x.plottype == "big")]

    for i in big:
        text = "\includegraphics[height=0.85\\textheight]{{{0}/plots/pdf/{1}.pdf}}\n".format(
            config.outputPath, i.filename)

        out.addSlide("High-level parameters", text)

    # time (IOV) dependent plots
    out.add("\subsection{High-level parameters versus time (IOV)}")
    time = [x for x in config.outputList if (x.plottype == "time")]

    if time:
        # get list with names of the structures
        for structure in [x.name for x in time if x.parameter == "xyz"]:
            for mode in ["xyz", "rot"]:
                text = "\\framesubtitle{{{0}}}\n".format(structure)
                if any([x.filename for x in time if (x.parameter == mode and x.name == structure)]):
                    filename = [x.filename for x in time if (x.parameter == mode and x.name == structure)][0]
                    text += "\includegraphics[height=0.85\\textheight]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                        config.outputPath, filename)

                out.addSlide("High-level parameters versus time (IOV)", text)

    # hole modules
    out.add("\subsection{Module-level parameters}")
    # check if there are module plots
    if any(x for x in config.outputList if (x.plottype == "mod" and x.number == "")):

        # loop over all structures
        for moduleName in [x.name for x in alignables.structures]:

            # check if there is a plot for this module
            if any(x for x in config.outputList if (x.plottype == "mod" and x.number == "" and x.name == moduleName)):

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
                        text = "\\framesubtitle{{{0}}}\n".format(moduleName)
                        text += "\includegraphics[height=0.85\\textheight]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                            config.outputPath, module[0].filename)

                        out.addSlide("Module-level parameters", text)

                        # loop over submodules
                        for plot in moduleSub:
                            text = "\\framesubtitle{{{0}}}\n".format(
                                moduleName)
                            text += "\includegraphics[height=0.85\\textheight]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                                config.outputPath, plot.filename)

                            out.addSlide("Module-level parameters", text)

    # plot taken from the millePedeMonitor_merge.root file
    out.add("\section{Monitor plots}")
    for plot in [x for x in config.outputList if x.plottype == "monitor"]:
        text = "\\framesubtitle{{{0}}}\n".format(plot.name)
        text += "\includegraphics[height=0.85\\textheight]{{{0}/plots/pdf/{1}.pdf}}\n".format(
            config.outputPath, plot.filename)
        out.addSlide("Monitor", text)

    data = data.substitute(out=out.text)

    with open(os.path.join(config.outputPath, outputFile), "w") as output:
        output.write(data)
        output.close()

    # TODO run pdflatex
    for i in range(2):
        os.system("pdflatex -output-directory={0}  {1}/{2}".format(
            config.outputPath, config.outputPath, outputFile))
