#!/usr/bin/env python

##########################################################################
# Parse the alignment_merge.py file for additional information
#

import logging


class AdditionalData:
    """ stores the additional information of the alignment_merge.py file
    """

    def __init__(self):
        self.pedeSteererMethod = ""
        self.pedeSteererOptions = []
        self.pedeSteererCommand = ""

        # safe the selector information Rigid, Bowed, TwoBowed
        self.selector = [[] for x in range(3)]
        self.selectorTag = [[] for x in range(3)]
        self.selectorThird = [[] for x in range(3)]

        # string to find the information and variables where to safe the
        # information (searchstring: [selector list, seletor tag, third element,
        # name])
        self.pattern = {
            "process.AlignmentProducer.ParameterBuilder.SelectorRigid = cms.PSet(": [self.selector[0], self.selectorTag[0], self.selectorThird[0], "SelectorRigid"],
            "process.AlignmentProducer.ParameterBuilder.SelectorBowed = cms.PSet(": [self.selector[1], self.selectorTag[1], self.selectorThird[1], "SelectorBowed"],
            "process.AlignmentProducer.ParameterBuilder.SelectorTwoBowed = cms.PSet(": [self.selector[2], self.selectorTag[2], self.selectorThird[2], "SelectorTwoBowed"]
        }

    def parse(self, config, path):
        logger = logging.getLogger("mpsvalidate")
        
        # open aligment_merge.py file
        try:
            with open(path) as inputFile:
                mergeFile = inputFile.readlines()
        except IOError:
            logger.error("AdditionalData: {0} does not exist".format(path))
            return

        # search pattern

        # loop over lines
        for index, line in enumerate(mergeFile):
            try:
                # search for SelectorRigid, SelectorBowed and SelectorTwoBowed
                for string in self.pattern:
                    if (string in line):
                        # extract data
                        for lineNumber in range(index + 2, index + 8):
                            mergeFile[lineNumber] = mergeFile[lineNumber].split("#", 1)[0]
                            # break at the end of the SelectorRigid
                            if (")" in mergeFile[lineNumber]):
                                break
                            self.pattern[string][0].append(
                                mergeFile[lineNumber].replace("\"", "'").strip("', \n").split(","))
                            # check if third argument
                            if (len(self.pattern[string][0][-1]) > 2):
                                self.pattern[string][1].append(
                                    self.pattern[string][0][-1][2])
                    # check for third arguments
                    if ("'" not in line.replace("\"", "'")):
                        for tag in self.pattern[string][1]:
                            if tag in line:
                                self.pattern[string][2].append(line.strip("\n").replace("#", ""))
                                # add following lines
                                for lineNumber in range(index + 1, index + 5):
                                    # new process or blank line
                                    if ("process" in mergeFile[lineNumber] or "\n" == mergeFile[lineNumber]):
                                        break
                                    # different tag
                                    if (any(x in mergeFile[lineNumber] for x in self.pattern[string][1])):
                                        break
                                    self.pattern[string][2].append(mergeFile[lineNumber].strip("\n").replace("#", ""))
            except Exception as e:
                logging.error("Selector Parsing error")

            # search for pedeSteererMethod
            if ("process.AlignmentProducer.algoConfig.pedeSteerer.method" in line and "#" not in line):
                try:
                    self.pedeSteererMethod = line.replace("\"", "'").split("'")[1]
                except Exception as e:
                    logger.error("AdditionalParser: pedeSteererMethod not found - {0}".format(e))

            # search for pedeSteererOptions
            if ("process.AlignmentProducer.algoConfig.pedeSteerer.options" in line and "#" not in line):
                for lineNumber in range(index + 1, index + 15):
                    if (lineNumber<len(mergeFile)):
                        if ("]" in mergeFile[lineNumber]):
                            break
                        self.pedeSteererOptions.append(
                            mergeFile[lineNumber].replace("\"", "'").strip("', \n"))

            # search for pedeSteererCommand
            if ("process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand" in line and "#" not in line):
                try:
                    self.pedeSteererCommand = line.replace("\"", "'").split("'")[1]
                except Exception as e:
                    logger.error("AdditionalParser: pedeSteererCommand not found - {0}".format(e))
