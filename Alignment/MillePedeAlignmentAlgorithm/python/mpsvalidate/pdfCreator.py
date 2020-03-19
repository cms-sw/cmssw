##########################################################################
# Creates pdf out of the histograms, parsed data and a given template.
##

from builtins import range
import logging
import os
import string
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes as mpsv_classes


# create class to have delimiter %% which is not used in latex


class TexTemplate(string.Template):
    delimiter = "%%"


def create(alignables, pedeDump, additionalData, outputFile, config):
    logger = logging.getLogger("mpsvalidate")

    # load template
    with open(os.path.join(config.mpspath, "templates",
                           "mpsvalidate_tex_template.tex")) as template:
        data = template.read()
        template.close()

    # create object where data could be substituted
    data = TexTemplate(data)

    # output string
    out = ""
    
    # title page
    if (config.message):
        out += """\\begin{{titlepage}}
                    \centering
                    \\vspace*{{4cm}}
                    \Huge\\bfseries Alignment Validation\par
                    \\vspace{{2cm}}	
                    \scshape\huge Alignment Campaign\\\\ {{{0}}}\par
                    \\vfill
                    \large \\today\par
                    \\end{{titlepage}}
                    \\tableofcontents
                    \\newpage""".format(config.message)
    else:
        out += """\\begin{titlepage}
                    \centering
                    \\vspace*{4cm}
                    \Huge\\bfseries Alignment Validation\par
                    \\vfill
                    \large \\today\par
                    \\end{titlepage}
                    \\tableofcontents
                    \\newpage"""

    # general information

    out += "\section{{General information}}\n"

    if (config.message):
        out += "Project: {{{0}}}\\\\\n".format(config.message)
    out += "Input-Path:\n"
    out += "\\begin{verbatim}\n"
    out += config.jobDataPath+"\n"
    out += "\\end{verbatim}\n"

    # alignment_merge.py
    try:
        out += "\subsection{Alignment Configuration}\n"
        out += "\\textbf{{PedeSteerer method:}} {{{0}}}\\\\\n".format(
            additionalData.pede_steerer_method)
        out += "\\textbf{{PedeSteerer options:}}\\\\\n"
        for line in additionalData.pede_steerer_options:
            out += "{{{0}}}\\\\\n".format(line)
        out += "\\textbf{{PedeSteerer command:}} {0}\\\\\n".format(
            additionalData.pede_steerer_command)

        for i in sorted(additionalData.selectors):
            out += "\\textbf{{{0}:}}\n".format(additionalData.selectors[i]["name"])
            out += "\\begin{verbatim}\n"
            for line in additionalData.selectors[i]["selector"].dumpPython().split("\n"):
                out += line + "\n"
            out += "\\end{verbatim}\n"

        if len(additionalData.iov_definition) > 0:
            out += "\\textbf{{IOV defintion:}}\n"
            out += "\\begin{verbatim}\n"
            for line in additionalData.iov_definition.dumpPython().split("\n"):
                out += line + "\n"
            out += "\\end{verbatim}\n\n"
        out += "\n"

    except Exception as e:
        logger.error("data not found - {0} {1}".format(type(e), e))

    # table of input files with number of tracks
    if config.showmonitor:
        out += "\subsection{Datasets with tracks}\n"
        out += """\\begin{table}[h]
            \centering
            \caption{Datasets with tracks}
            \\begin{tabular}{ccc}
            \hline
            Dataset & Number of used tracks & Weight \\\\
            \hline \n"""
        for monitor in mpsv_classes.MonitorData.monitors:
            out += "{0} & {1} & {2}\\\\\n".format(monitor.name, monitor.ntracks,
                                                  monitor.weight if monitor.weight != None else "--")
        try:
            if (pedeDump.nrec):
                out += "\hline\nNumber of records & {0}\\\\\n".format(pedeDump.nrec)
        except Exception as e:
            logger.error("data not found - {0} {1}".format(type(e), e))
        out += """\hline
                  \end{tabular}\n
                  \end{table}\n"""
        out += "The information in this table is based on the monitor root files. Note that the number of tracks which where used in the pede step can differ from this table.\n"
    try:
        # pede.dump.gz
        if config.showdump:
            out += "\subsection{{Pede monitoring information}}\n"
            if (pedeDump.sumValue != 0):
                out += r"\begin{{align*}}Sum(Chi^2)/Sum(Ndf) &= {0}\\ &= {1}\end{{align*}}".format(
                    pedeDump.sumSteps, pedeDump.sumValue)
            else:
                out += r"\begin{{align*}}Sum(W*Chi^2)/Sum(Ndf)/<W> &= {0}\\ &= {1}\end{{align*}}".format(
                    pedeDump.sumSteps, pedeDump.sumWValue)
            out += r"with correction for down-weighting: {0}\\".format(
                pedeDump.correction)
            out += r"Peak dynamic memory allocation: {0} GB\\".format(
                pedeDump.memory)
            out += r"Total time: {0} h {1} m {2} s\\".format(
                pedeDump.time[0], pedeDump.time[1], pedeDump.time[2])
            out += r"Number of records: {0}\\".format(pedeDump.nrec)
            out += r"Total number of parameters: {0}\\".format(pedeDump.ntgb)
            out += r"Number of variable parameters: {0}\\".format(pedeDump.nvgb)
            out += r"Warning:\\"
            for line in pedeDump.warning:

                # check if line empty
                if line.replace(r" ", r""):
                    out += "\\begin{verbatim}\n"
                    out += line + "\n"
                    out += "\\end{verbatim}\n"

            out += "\section{{Parameter plots}}\n"
    except Exception as e:
        logger.error("data not found - {0} {1}".format(type(e), e))

    # high level structures
    if config.showhighlevel:
        big = [x for x in config.outputList if (x.plottype == "big")]

        if big:
            out += "\subsection{{High-level parameters}}\n"
            for i in big:
                out += "\includegraphics[width=\linewidth]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                    config.outputPath, i.filename)

    # time (IOV) dependent plots
    if config.showtime:
        time = [x for x in config.outputList if (x.plottype == "time")]

        if time:
            out += "\subsection{{High-level parameters versus time (IOV)}}\n"
            # get list with names of the structures
            for structure in [x.name for x in time if x.parameter == "xyz"]:
                out += "\subsubsection{{{0}}}\n".format(structure)
                for mode in ["xyz", "rot"]:
                    if any([x.filename for x in time if (x.parameter == mode and x.name == structure)]):
                        filename = [x.filename for x in time if (x.parameter == mode and x.name == structure)][0]
                        out += "\includegraphics[width=\linewidth]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                            config.outputPath, filename)

    # hole modules
    if config.showmodule:
        # check if there are module plots
        if any(x for x in config.outputList if (x.plottype == "mod" and x.number == "")):
            out += "\subsection{{Module-level parameters}}\n"

            # loop over all structures
            for moduleName in [x.name for x in alignables.structures]:

                # check if there is a plot for this module
                if any(x for x in config.outputList if (x.plottype == "mod" and x.number == "" and x.name == moduleName)):
                    out += "\subsubsection{{{0}}}\n".format(moduleName)
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
                            out += "\includegraphics[width=\linewidth]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                                config.outputPath, module[0].filename)
                            if config.showsubmodule:
                                # loop over submodules
                                for plot in moduleSub:
                                    out += "\includegraphics[width=\linewidth]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                                        config.outputPath, plot.filename)

    # plot taken from the millePedeMonitor_merge.root file
    if config.showmonitor:
        if any(x for x in config.outputList if x.plottype == "monitor"):
            out += "\section{{Monitor plots}}\n"

            lastdataset = ""
            for plot in [x for x in config.outputList if x.plottype == "monitor"]:
                # all plots of a dataset together in one section
                if (lastdataset != plot.name):
                    out += "\subsection{{{0}}}\n".format(plot.name)
                lastdataset = plot.name
                out += "\includegraphics[width=\linewidth]{{{0}/plots/pdf/{1}.pdf}}\n".format(
                    config.outputPath, plot.filename)

    data = data.substitute(out=out)

    with open(os.path.join(config.outputPath, outputFile), "w") as output:
        output.write(data)
        output.close()

    # TODO run pdflatex
    for i in range(2):
        os.system("pdflatex -output-directory={0}  {1}/{2}".format(
            config.outputPath, config.outputPath, outputFile))
