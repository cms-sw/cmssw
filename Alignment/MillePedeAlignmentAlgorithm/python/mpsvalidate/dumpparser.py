#!/usr/bin/env python

##########################################################################
# Parse the pede.dump.gz file and returns a pedeDump object with the
# parsed information of the file.
##

import gzip
import logging
import re

from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes import PedeDumpData


def parse(path, config):
    logger = logging.getLogger("mpsvalidate")
    
    # parse pede.dump.gz

    pedeDump = PedeDumpData()

    # only recognize warning the first time
    warningBool = False

    # save lines in list
    try:
        with gzip.open(path) as gzipFile:
            dumpFile = gzipFile.readlines()
    except IOError:
        logger.error("PedeDump: {0} does not exist".format(path))
        return

    for i, line in enumerate(dumpFile):
        # Sum(Chi^2)/Sum(Ndf)
        if ("Sum(Chi^2)/Sum(Ndf) =" in line):
            number = []
            number.append(map(float, re.findall(
                r"[-+]?\d*\.\d+", dumpFile[i])))
            number.append(map(int, re.findall(r"[-+]?\d+", dumpFile[i + 1])))
            number.append(map(float, re.findall(
                r"[-+]?\d*\.\d+", dumpFile[i + 2])))
            pedeDump.sumSteps = "{0} / ( {1} - {2} )".format(
                number[0][0], number[1][0], number[1][1])
            pedeDump.sumValue = number[2][0]

        # Sum(W*Chi^2)/Sum(Ndf)/<W>
        if ("Sum(W*Chi^2)/Sum(Ndf)/<W> =" in line):
            number = []
            number.append(map(float, re.findall(
                r"[-+]?\d*\.\d+", dumpFile[i])))
            number.append(map(int, re.findall(r"[-+]?\d+", dumpFile[i + 1])))
            number.append(map(float, re.findall(
                r"[-+]?\d*\.\d+", dumpFile[i + 2])))
            number.append(map(float, re.findall(
                r"[-+]?\d*\.\d+", dumpFile[i + 3])))
            pedeDump.sumSteps = "{0} / ( {1} - {2} ) / {3}".format(
                number[0][0], number[1][0], number[1][1], number[2][0])
            pedeDump.sumWValue = number[3][0]

        if ("with correction for down-weighting" in line):
            number = map(float, re.findall(r"[-+]?\d*\.\d+", dumpFile[i]))
            pedeDump.correction = number[0]

        # Peak dynamic memory allocation
        if ("Peak dynamic memory allocation:" in line):
            number = map(float, re.findall(r"[-+]?\d*\.\d+", dumpFile[i]))
            pedeDump.memory = number[0]

        # total time
        if ("Iteration-end" in line):
            number = map(int, re.findall(r"\d+", dumpFile[i + 1]))
            pedeDump.time = number[:3]

        # warings
        if ("WarningWarningWarningWarning" in line and warningBool == False):
            warningBool = True
            j = i + 8
            while ("Warning" not in dumpFile[j]):
                pedeDump.warning.append(dumpFile[j])
                j += 1

        # nrec number of records
        if (" = number of records" in line):
            number = map(int, re.findall("\d+", dumpFile[i]))
            pedeDump.nrec = number[0]

        # ntgb total number of parameters
        if (" = total number of parameters" in line):
            number = map(int, re.findall("\d+", dumpFile[i]))
            pedeDump.ntgb = number[0]

        # nvgb number of variable parameters
        if (" = number of variable parameters" in line):
            number = map(int, re.findall("\d+", dumpFile[i]))
            pedeDump.nvgb = number[0]

    return pedeDump
