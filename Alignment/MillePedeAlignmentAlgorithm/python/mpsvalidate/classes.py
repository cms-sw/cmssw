#!/usr/bin/env python

##########################################################################
# Classes which are needed by the mps_validate.py file.
##


class PlotData:
    """ Hold information about XYZ
    """

    def __init__(self, mode):
        self.numberOfBins = [0, 0, 0]
        self.maxShift = [0, 0, 0]
        self.minShift = [0, 0, 0]
        self.maxBinShift = [0, 0, 0]
        # used binShift
        self.binShift = [0, 0, 0]
        self.hiddenEntries = [0, 0, 0]
        self.binPosition = [1, 1, 1]
        self.usedRange = [0, 0, 0]
        self.histo = []
        self.histoAxis = []
        # plot title and text
        self.title = 0
        self.text = 0
        self.label = ""
        self.objid = 0
        # switch mode for position, rotation, distortion
        if (mode == "xyz"):
            self.xyz = {0: "X", 1: "Y", 2: "Z"}
            self.data = [0, 1, 2]
            self.unit = "#mum"
        if (mode == "rot"):
            self.xyz = {0: "#alpha", 1: "#beta", 2: "#gamma"}
            self.data = [3, 4, 5]
            self.unit = ""
        if (mode == "dist"):
            self.xyz = {0: "A", 1: "B", 2: "C"}
            self.data = [6, 7, 8]
            self.unit = ""


class PedeDumpData:
    """ information out of the pede.dump.gz file
    """

    def __init__(self):
        self.sumValue = 0
        self.sumWValue = 0
        self.sumSteps = ""
        self.correction = 0
        self.memory = 0
        self.time = []
        self.warning = []
        # number of records
        self.nrec = 0
        # total numer of parameters
        self.ntgb = 0
        # number of variable parameters
        self.nvgb = 0

    def printLog(self):
        if (self.sumValue != 0):
            print "Sum(Chi^2)/Sum(Ndf) = {0} = {1}".format(self.sumSteps, self.sumValue)
        else:
            print "Sum(W*Chi^2)/Sum(Ndf)/<W> = {0} = {1}".format(self.sumSteps, self.sumWValue)
        print "with correction for down-weighting: {0}".format(self.correction)
        print "Peak dynamic memory allocation: {0} GB".format(self.memory)
        print "Total time: {0} h {1} m {2} s".format(self.time[0], self.time[1], self.time[2])
        print "Number of records: {0}".format(self.nrec)
        print "Total number of parameters: {0}".format(self.ntgb)
        print "Number of variable parameters: {0}".format(self.nvgb)
        print "Warning:"
        for line in self.warning:
            print line


class MonitorData:
    """ information out of the monitor root files
    """
    monitors = []

    def __init__(self, name, ntracks):
        self.name = name
        self.ntracks = ntracks
        self.monitors.append(self)


class OutputData:
    """ stores the information about the data which should be part of the Output
    """

    def __init__(self, plottype="", name="", number="", parameter="", filename=""):
        self.plottype = plottype
        self.name = name
        self.number = number
        self.parameter = parameter
        self.filename = filename
