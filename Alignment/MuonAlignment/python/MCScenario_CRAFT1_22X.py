# To use this script:
#     it's an ordinary Python script, not a CMSSW configuration file (though it writes and runs CMSSW configuration files)
#         * you MUST have an inertGlobalPositionRcd.db in your working directory
#         * you MUST NOT have an MCScenario_CRAFT1_22X.db
#
#     to make the inertGlobalPositionRcd:
#         cmsRun Alignment/MuonAlignment/python/makeGlobalPositionRcd_cfg.py
#
#     to get rid of the MCScenario_CRAFT1_22X.db:
#         rm MCScenario_CRAFT1_22X.db    (naturally)
#
#     to run this script:
#         python MCScenario_CRAFT1_22X.py
#
#     it will create
#         * MCScenario_CRAFT1_22X.xml            the XML file with randomly-distributed values, created directly by define_scenario()
#         * convert_cfg.py                       the conversion configuration file
#         * MCScenario_CRAFT1_22X.db             the SQLite database created from the XML
#         * check_cfg.py                         configuration file that converts the SQLite file back into XML
#         * MCScenario_CRAFT1_22X_CHECKME.xml    converted back, so that we can check the values that were saved to the database
#
#     to check the output in Excel, do this
#         ./Alignment/MuonAlignment/python/geometryXMLtoCSV.py < MCScenario_CRAFT1_22X_CHECKME.xml > MCScenario_CRAFT1_22X_CHECKME.csv
#         and then open MCScenario_CRAFT1_22X_CHECKME.csv in Excel

import random, os
from math import *

# # useful, but not currently used
# def mean(xlist):
#   s, n = 0., 0.
#   for x in xlist:
#     s += x
#     n += 1.
#   return s/n
# def stdev(xlist):
#   s, s2, n = 0., 0., 0.
#   for x in xlist:
#     s += x
#     s2 += x**2
#     n += 1.
#   return sqrt(s2/n - (s/n)**2)

#### called once at the end of this script
def make_scenario_sqlite():
    scenario = define_scenario()
    write_xml(scenario, "MCScenario_CRAFT1_22X.xml")
    write_conversion_cfg("convert_cfg.py", "MCScenario_CRAFT1_22X.xml", "MCScenario_CRAFT1_22X.db")
    cmsRun("convert_cfg.py")
    write_check_cfg("check_cfg.py", "MCScenario_CRAFT1_22X.db", "MCScenario_CRAFT1_22X_CHECKME.xml")
    cmsRun("check_cfg.py")
#### that's it!  everything this uses is defined below

def write_conversion_cfg(fileName, xmlFileName, dbFileName):
    outfile = file(fileName, "w")
    outfile.write("""
from Alignment.MuonAlignment.convertXMLtoSQLite_cfg import *
process.MuonGeometryDBConverter.fileName = "%(xmlFileName)s"
process.PoolDBOutputService.connect = "sqlite_file:%(dbFileName)s"
""" % vars())

def write_check_cfg(fileName, dbFileName, xmlFileName):
    outfile = file(fileName, "w")
    outfile.write("""
from Alignment.MuonAlignment.convertSQLitetoXML_cfg import *
process.PoolDBESSource.connect = "sqlite_file:%(dbFileName)s"
process.MuonGeometryDBConverter.outputXML.fileName = "%(xmlFileName)s"
process.MuonGeometryDBConverter.outputXML.relativeto = "ideal"
process.MuonGeometryDBConverter.outputXML.suppressDTChambers = False
process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = True
process.MuonGeometryDBConverter.outputXML.suppressDTLayers = True
process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = False
process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = False
""" % vars())

def cmsRun(fileName):
    os.system("cmsRun %(fileName)s" % vars())

########### writing a scenario in XML ##############################################################

# only needed to make the output XML readable
DTpreferred_order = {"wheel":1, "station":2, "sector":3, "superlayer":4, "layer":5}
CSCpreferred_order = {"endcap":1, "station":2, "ring":3, "chamber":4, "layer":5}
def DTsorter(a, b): return cmp(DTpreferred_order[a], DTpreferred_order[b])
def CSCsorter(a, b): return cmp(CSCpreferred_order[a], CSCpreferred_order[b])

# an instance of this class corresponds to one <DTChamber ... /> or <CSCStation ... />, etc.
class Alignable:
    def __init__(self, alignabletype, **location):
        self.alignabletype = alignabletype
        self.location = location

    def writeXML(self):
        parameters = self.location.keys()
        if self.alignabletype[0:2] == "DT":
            parameters.sort(DTsorter)
        else:
            parameters.sort(CSCsorter)

        output = ["<", self.alignabletype, " "]
        for parameter in parameters:
            output.extend([parameter, "=\"", str(self.location[parameter]), "\" "])
        output.append("/>")

        return "".join(output)

preferred_order = {"x":1, "y":2, "z":3, "phix":4, "phiy":5, "phiz":6}
def sorter(a, b): return cmp(preferred_order[a], preferred_order[b])

# an instance of this class corresponds to one <setposition ... />
class Position:
    def __init__(self, **location):
        self.location = location

    def writeXML(self):
        parameters = self.location.keys()
        parameters.sort(sorter)

        output = ["<setposition relativeto=\"ideal\" "]
        for parameter in parameters:
            output.extend([parameter, "=\"", str(self.location[parameter]), "\" "])
        output.append("/>")

        return "".join(output)

# an instance of this class corresponds to one <operation> ... </operation> in the XML file
class Operation:
    def __init__(self, alignable, position):
        self.alignable = alignable
        self.position = position

    def writeXML(self):
        output = ["<operation> ", self.alignable.writeXML(), " ", self.position.writeXML(), " </operation>\n"]
        return "".join(output)

def write_xml(scenario, fileName):
    # a scenario is an ordered list of Operations
    XMLlist = ["<MuonAlignment>\n"]
    for operation in scenario:
        XMLlist.append(operation.writeXML())
    XMLlist.append("</MuonAlignment>\n")
    XMLstring = "".join(XMLlist)
        
    outfile = file(fileName, "w")
    outfile.write(XMLstring)

########### defining the actual scenario ##############################################################

# this is the interesting part: where we define a scenario for CRAFT1 MC
def define_scenario():
    # get some data we'll need later
    DTnominal = get_DTnominal()
    DTaligned = get_DTaligned()
    CSCphotogrammetry = get_CSCphotogrammetry()

    # this will be a list of operations to write to an XML file
    scenario = []

    # 1. DT chambers were aligned relative to the tracker, so the uncertainty in their positions
    #    is global: chambers in the same wheel or station are uncorrelated.
    #    One exception: chambers in the same sector are correlated due to the fact that they lie
    #    on the same track.  Therefore:
    #        0.1 cm Gaussian errors for each chamber
    #        0.1 cm Gaussian errors for each sector
    #        on top of nominal values read off of the plots, so that we correctly account for
    #        which chambers were not aligned due to insufficient statistics
    #    If there's so little statistics that we can't even read the value off the plot, we do
    #        0.45 cm Gaussian errors in local x
    #        0.23 cm Gaussian errors in local y
    #        (the scale of the misalignments before aligning)
    #    DTnominal contains post-alignment x and y nominal values, read off the plots by hand

    sector_xshifts = {}
    sector_yshifts = {}
    for sector in 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14:
        sector_xshifts[sector] = random.gauss(0, 0.1)
        sector_yshifts[sector] = random.gauss(0, 0.1)

    for dtchamber in DTnominal:
        if dtchamber.nominalx is None:
            errx = random.gauss(0, 0.45)
        else:
            errx = random.gauss(dtchamber.nominalx, 0.1) + sector_xshifts[dtchamber.sector]

        if dtchamber.nominaly is None:
            erry = random.gauss(0, 0.23)
        else:
            erry = random.gauss(dtchamber.nominaly, 0.1) + sector_yshifts[dtchamber.sector]

        # 0.25 cm in Z would correspond to the scale of the z residuals vs z slopes in the plots
        errz = random.gauss(0, 0.25)

        # phix assumed to be the same order as phiy: see below
        errphix = random.gauss(0, 0.003)

        # magnetic field measurements had to correct for phiy misalignment:
        # I read off a 3 mrad value from their plots
        errphiy = random.gauss(0, 0.003)

        # phiz can be very accurately measured from the rphi residuals vs z, and it was aligned
        # it is not succeptable to any track-bias problems, because it is a local measurement
        # (unlike x and y)
        # the value here comes from repeatability of the measurement on a second iteration: 0.058 mrad
        # of course, if the chamber was not aligned, it should get 1.0 mrad, the scale of misalignments
        was_aligned = False
        for aligneddt in DTaligned:
            if aligneddt.wheel == dtchamber.wheel and aligneddt.station == dtchamber.station and aligneddt.sector == dtchamber.sector:
                was_aligned = True
                break

        if was_aligned:
            errphiz = random.gauss(0, 5.8e-05)
        else:
            errphiz = random.gauss(0, 0.001)

        # now turn this into an operation
        alignable = Alignable("DTChamber", wheel = dtchamber.wheel, station = dtchamber.station, sector = dtchamber.sector)
        position = Position(x = errx, y = erry, z = errz, phix = errphix, phiy = errphiy, phiz = errphiz)
        scenario.append(Operation(alignable, position))

    # 2. CSC chambers have fully-hierarchical misalignments because
    #    they haven't been aligned directly to the tracker yet (no *direct*
    #    measurement until first beams)

    # Before starting, let's build a list of chambers
    CSCchambers = []
    for endcap in 1, 2:
        for station, ring in (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1):
            if station > 1 and ring == 1:
                nchambers = 18
            else:
                nchambers = 36

            for chamber in range(1, nchambers+1):
                CSCchambers.append(CSCChamber(endcap = endcap, station = station, ring = ring, chamber = chamber))
            
    # First, the layer uncertainties: x only for simplicity, observed 0.0092 cm in overlaps alignment test
    for chamber in CSCchambers:
        for layer in 1, 2, 3, 4, 5, 6:
            alignable = Alignable("CSCLayer", endcap = chamber.endcap, station = chamber.station, ring = chamber.ring, chamber = chamber.chamber, layer = layer)
            position = Position(x = random.gauss(0, 0.0092), y = 0, z = 0, phix = 0, phiy = 0, phiz = 0)
            scenario.append(Operation(alignable, position))

    # Next, the chamber errors: we actually know these from
    # photogrammetry (next time, photogrammetry will be a starting
    # point for the alignment we upload)
    for chamber in CSCchambers:
        # the z and phix values come from DCOPS measurements, however
        errz = random.gauss(0, 0.17)
        errphix = random.gauss(0, 0.00068)

        # the phiy values can't be measured by DCOPS or
        # photogrammetry, but some values were measured by the
        # overlaps alignment test
        errphiy = random.gauss(0, 0.0023)

        photogrammetry = None
        for chamberPG in CSCphotogrammetry:
            if chamberPG.endcap == chamber.endcap and chamberPG.station == chamber.station and chamberPG.ring == chamber.ring and chamberPG.chamber == chamber.chamber:
                photogrammetry = chamberPG
                break
        
        alignable = Alignable("CSCChamber", endcap = chamber.endcap, station = chamber.station, ring = chamber.ring, chamber = chamber.chamber)

        if photogrammetry is not None:
            position = Position(x = photogrammetry.x, y = photogrammetry.y, z = errz, phix = errphix, phiy = errphiy, phiz = photogrammetry.phiz)
        else:
            # when we don't have explicit PG for a chamber, the uncertainty comes from the RMS of the ones we do have
            position = Position(x = random.gauss(0, 0.12), y = random.gauss(0, 0.10), z = errz, phix = errphix, phiy = errphiy, phiz = random.gauss(0, 0.00056))

        scenario.append(Operation(alignable, position))

    # Finally, the disk errors: some reasonable values
    for endcap in 1, 2:
        for station in 1, 2, 3, 4:
            alignable = Alignable("CSCStation", endcap = endcap, station = station)
            position = Position(x = random.gauss(0, 0.05), y = random.gauss(0, 0.05), z = random.gauss(0, 0.3), phix = random.gauss(0, 0.0005), phiy = random.gauss(0, 0.0005), phiz = random.gauss(0, 0.0005))
            scenario.append(Operation(alignable, position))

    return scenario

# the data we used in define_scenario(), put at the end because it's long

# approximate DT chamber positions after alignment, determined by eye, looking at the plots
# the actual values are derived from a Gaussian with these means
# "None" means that there wasn't even enough data to determine a nominal position
# this is a complete list of chambers
class DTChamber:
    def __init__(self, **location):
        self.__dict__.update(location)
def get_DTnominal():
    DTnominal = []
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 2, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 3, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 4, nominalx = 3, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 5, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 6, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 8, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 9, nominalx = 7.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 10, nominalx = 10, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 11, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 1, sector = 12, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 1, nominalx = 5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 2, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 3, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 4, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 5, nominalx = 0, nominaly = 0.5))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 6, nominalx = 0, nominaly = 1))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 8, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 9, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 1, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 1, nominalx = 6, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 2, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 3, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 4, nominalx = 1, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 5, nominalx = -1, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 6, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 7, nominalx = -1, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 8, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 9, nominalx = 3, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 1, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 1, nominalx = 6, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 2, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 3, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 4, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 5, nominalx = 0, nominaly = -1.5))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 6, nominalx = -4, nominaly = -0.5))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 8, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 9, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 11, nominalx = 0, nominaly = -0.5))
    DTnominal.append(DTChamber(wheel = 1, station = 1, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 2, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 3, nominalx = 3, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 4, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 5, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 6, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 8, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 9, nominalx = 3, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 10, nominalx = 3, nominaly = 2))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 11, nominalx = 4, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 1, sector = 12, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 2, nominalx = 5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 3, nominalx = 6, nominaly = 2))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 4, nominalx = 1, nominaly = -4))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 5, nominalx = -2, nominaly = -6))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 6, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 8, nominalx = 9, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 9, nominalx = 5, nominaly = -4))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 10, nominalx = -1, nominaly = -4))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 11, nominalx = 8, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -2, station = 2, sector = 12, nominalx = 8, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 2, nominalx = 0, nominaly = -0.5))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 3, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 4, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 5, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 6, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 8, nominalx = None, nominaly = -2))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 9, nominalx = 2, nominaly = -2))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 10, nominalx = 0, nominaly = -0.5))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 2, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 1, nominalx = 6, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 2, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 3, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 4, nominalx = 0, nominaly = 5))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 5, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 6, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 8, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 9, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 2, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 2, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 3, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 4, nominalx = -0.5, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 5, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 6, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 8, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 9, nominalx = 0, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 2, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 2, nominalx = 1.5, nominaly = -2))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 3, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 4, nominalx = -3, nominaly = 2.5))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 5, nominalx = -5, nominaly = 2.5))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 6, nominalx = -6, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 8, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 9, nominalx = 0, nominaly = 4))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 11, nominalx = 0, nominaly = 3))
    DTnominal.append(DTChamber(wheel = 2, station = 2, sector = 12, nominalx = 3.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 2, nominalx = 9, nominaly = -4))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 3, nominalx = 0, nominaly = -2))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 4, nominalx = 0, nominaly = -1.5))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 5, nominalx = 0, nominaly = -2.5))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 6, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 8, nominalx = 5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 9, nominalx = 0, nominaly = -3))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 10, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -2, station = 3, sector = 12, nominalx = 6, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 1, nominalx = 8, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 2, nominalx = 0, nominaly = -0.5))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 3, nominalx = 8, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 4, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 5, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 6, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 8, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 9, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = -1, station = 3, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 2, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 3, nominalx = 0, nominaly = -1))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 4, nominalx = 0, nominaly = 0.5))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 5, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 6, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 7, nominalx = -1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 8, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 9, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 0, station = 3, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 2, nominalx = 0, nominaly = 0.5))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 3, nominalx = 0, nominaly = 0.5))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 4, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 5, nominalx = 0, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 6, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 7, nominalx = -3, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 8, nominalx = None, nominaly = -0.75))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 9, nominalx = 0, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 10, nominalx = 0, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 11, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 1, station = 3, sector = 12, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 2, nominalx = 3, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 3, nominalx = 0, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 4, nominalx = 0, nominaly = 2))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 5, nominalx = 0, nominaly = 1))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 6, nominalx = -5, nominaly = 3))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 7, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 8, nominalx = 3, nominaly = 8))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 9, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 10, nominalx = 0, nominaly = 0))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 11, nominalx = 0, nominaly = -2))
    DTnominal.append(DTChamber(wheel = 2, station = 3, sector = 12, nominalx = 0, nominaly = -5))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 2, nominalx = 9, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 3, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 4, nominalx = 5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 5, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 6, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 7, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 8, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 9, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 10, nominalx = -1, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 11, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 12, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 13, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -2, station = 4, sector = 14, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 2, nominalx = -1.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 3, nominalx = -0.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 4, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 5, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 6, nominalx = 0.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 7, nominalx = -5, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 8, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 9, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 10, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 11, nominalx = -1, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 12, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 13, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = -1, station = 4, sector = 14, nominalx = -1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 1, nominalx = 3, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 2, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 3, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 4, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 5, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 6, nominalx = 0.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 7, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 8, nominalx = 0.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 9, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 10, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 11, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 12, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 13, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 0, station = 4, sector = 14, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 2, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 3, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 4, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 5, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 6, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 7, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 8, nominalx = -3, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 9, nominalx = 0.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 10, nominalx = -1.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 11, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 12, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 13, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 1, station = 4, sector = 14, nominalx = -2, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 1, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 2, nominalx = 2, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 3, nominalx = -0.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 4, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 5, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 6, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 7, nominalx = -6, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 8, nominalx = None, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 9, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 10, nominalx = 3, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 11, nominalx = -0.5, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 12, nominalx = 1, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 13, nominalx = 0, nominaly = None))
    DTnominal.append(DTChamber(wheel = 2, station = 4, sector = 14, nominalx = -2, nominaly = None))
    return DTnominal

# this is just a list of which chambers were aligned in phiz
def get_DTaligned():
    DTaligned = []
    DTaligned.append(DTChamber(wheel = -2, station = 2, sector = 10))
    DTaligned.append(DTChamber(wheel = -2, station = 3, sector = 3))
    DTaligned.append(DTChamber(wheel = -2, station = 3, sector = 4))
    DTaligned.append(DTChamber(wheel = -2, station = 3, sector = 5))
    DTaligned.append(DTChamber(wheel = -2, station = 3, sector = 9))
    DTaligned.append(DTChamber(wheel = -2, station = 3, sector = 10))
    DTaligned.append(DTChamber(wheel = -2, station = 3, sector = 11))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 3))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 13))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 5))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 8))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 9))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 10))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 14))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 11))
    DTaligned.append(DTChamber(wheel = -2, station = 4, sector = 12))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 3))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 4))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 5))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 8))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 9))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 10))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 11))
    DTaligned.append(DTChamber(wheel = -1, station = 1, sector = 12))
    DTaligned.append(DTChamber(wheel = -1, station = 2, sector = 2))
    DTaligned.append(DTChamber(wheel = -1, station = 2, sector = 4))
    DTaligned.append(DTChamber(wheel = -1, station = 2, sector = 6))
    DTaligned.append(DTChamber(wheel = -1, station = 2, sector = 10))
    DTaligned.append(DTChamber(wheel = -1, station = 2, sector = 11))
    DTaligned.append(DTChamber(wheel = -1, station = 2, sector = 12))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 2))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 4))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 5))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 6))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 9))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 10))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 11))
    DTaligned.append(DTChamber(wheel = -1, station = 3, sector = 12))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 2))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 3))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 4))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 13))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 5))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 8))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 9))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 10))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 14))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 11))
    DTaligned.append(DTChamber(wheel = -1, station = 4, sector = 12))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 5))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 6))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 10))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 3))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 11))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 4))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 8))
    DTaligned.append(DTChamber(wheel = 0, station = 1, sector = 12))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 5))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 9))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 2))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 6))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 10))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 3))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 11))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 4))
    DTaligned.append(DTChamber(wheel = 0, station = 2, sector = 12))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 5))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 9))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 6))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 10))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 3))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 11))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 4))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 8))
    DTaligned.append(DTChamber(wheel = 0, station = 3, sector = 12))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 2))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 3))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 4))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 5))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 6))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 9))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 10))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 14))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 11))
    DTaligned.append(DTChamber(wheel = 0, station = 4, sector = 12))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 2))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 3))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 4))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 5))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 8))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 9))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 10))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 11))
    DTaligned.append(DTChamber(wheel = 1, station = 1, sector = 12))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 2))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 3))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 4))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 5))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 6))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 8))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 9))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 10))
    DTaligned.append(DTChamber(wheel = 1, station = 2, sector = 11))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 2))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 3))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 4))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 5))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 6))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 9))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 10))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 11))
    DTaligned.append(DTChamber(wheel = 1, station = 3, sector = 12))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 2))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 3))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 4))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 13))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 5))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 6))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 8))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 9))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 10))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 11))
    DTaligned.append(DTChamber(wheel = 1, station = 4, sector = 12))
    DTaligned.append(DTChamber(wheel = 2, station = 2, sector = 10))
    DTaligned.append(DTChamber(wheel = 2, station = 2, sector = 11))
    DTaligned.append(DTChamber(wheel = 2, station = 3, sector = 3))
    DTaligned.append(DTChamber(wheel = 2, station = 3, sector = 4))
    DTaligned.append(DTChamber(wheel = 2, station = 3, sector = 5))
    DTaligned.append(DTChamber(wheel = 2, station = 3, sector = 10))
    DTaligned.append(DTChamber(wheel = 2, station = 3, sector = 11))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 3))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 4))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 13))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 5))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 8))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 14))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 11))
    DTaligned.append(DTChamber(wheel = 2, station = 4, sector = 12))
    return DTaligned

# exact photogrammetry data for the CSCs, which have not been uploaded to the database yet
# they are therefore, unfortunately, a known error, rather than an alignment correction (that will change with the next reprocessing)
# this is not a complete list!  missing chambers (most notably ME1/1) should get a random value
class CSCChamber:
    def __init__(self, **location):
        self.__dict__.update(location)
def get_CSCphotogrammetry():
    CSCphotogrammetry = []
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 1, x = -0.13329762, y = 0.12841226, z = 0.100811556521762, phix = 0.00197246358979321, phiz = 0.000189263205103485))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 3, x = -0.00777817, y = -0.07823767, z = 0.0791115565217524, phix = 0.000245313589793417, phiz = -0.000892836794896601))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 5, x = 0.02264694, y = -0.00846476, z = -0.0781884434782114, phix = -0.000143053589793174, phiz = 0.000179863205103548))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 7, x = 0.0348601, y = -0.08814248, z = -0.0240884434782629, phix = -0.00102182358979309, phiz = -0.000158426794896638))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 9, x = 0.14330193, y = -0.04584235, z = 0.106811556521734, phix = 0.00198281358979336, phiz = -0.000786736794896603))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 11, x = 0.003615, y = 0.05033697, z = 0.0616115565217115, phix = -0.000275993589793034, phiz = -0.000170346794896581))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 13, x = -0.00205643, y = 0.09153855, z = 0.120111556521692, phix = 0.00280307358979316, phiz = 1.36320510346621e-06))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 15, x = 0.0078258, y = 0.05469822, z = -0.020588443478232, phix = -0.000510953589793262, phiz = 0.000563063205103465))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 2, x = -0.14771888, y = 0.04597149, z = 0.180823756521704, phix = 0.000413693589793274, phiz = -0.0014656967948965))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 6, x = -0.01027533, y = -0.10647866, z = 0.0999237565217754, phix = 0.00109375358979337, phiz = -0.00038428679489666))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 8, x = -0.00381304, y = -0.10133644, z = -0.0743762434782411, phix = -0.000843343589793333, phiz = -0.00053258679489665))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 10, x = 0.03256889, y = 0.15095426, z = 0.142723756521718, phix = 0.00240738358979321, phiz = -0.000316186794896645))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 12, x = -0.04596194, y = 0.12965172, z = -0.0655762434782901, phix = -4.59835897933639e-05, phiz = 0.000415513205103357))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 14, x = 0.06358684, y = 0.31720637, z = -0.0518762434783184, phix = 0.00277375358979327, phiz = -0.00122278679489662))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 16, x = 0.02429694, y = -0.1236167, z = 0.0656237565217452, phix = 0.00123156358979338, phiz = 0.00031591320510338))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 1, chamber = 18, x = -0.02368683, y = -0.01145061, z = -0.28737624347832, phix = 0.000199313589793204, phiz = 0.000124613205103374))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 1, x = 0.133877, y = 0.172, z = 0.127911556521781, phix = 0.00160113358979309, phiz = -0.000391926794896635))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 3, x = -0.08149324, y = 0.24258861, z = 0.0608115565216849, phix = -0.000580473589793085, phiz = 3.86320510337157e-06))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 5, x = -0.14671489, y = 0.02911814, z = -0.0365884434783084, phix = -4.55735897931912e-05, phiz = 0.000391723205103389))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 7, x = -0.09442243, y = 0.03445556, z = 0.232711556521735, phix = 0.000355553589793123, phiz = -3.2456794896607e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 9, x = -0.03045391, y = -0.04467214, z = 0.0417115565217046, phix = -0.000224903589793113, phiz = 0.000363393205103479))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 11, x = 0.01840026, y = -0.03562783, z = -0.0161884434783133, phix = 0.000416273589793137, phiz = -0.000240746794896607))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 13, x = 0.01576475, y = 0.03869465, z = 0.0394115565217135, phix = -3.04358979320607e-06, phiz = -0.000174886794896523))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 15, x = 0.13722411, y = 0.13759821, z = 0.201211556521685, phix = 0.000798863589793209, phiz = -0.000329066794896526))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 17, x = 0.15899721, y = -0.00058444, z = 0.036811556521684, phix = 0.00041023358979343, phiz = -0.00062320679489658))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 19, x = 0.0548253, y = 0.141, z = 0.0135115565217347, phix = -0.000431603589793228, phiz = -0.000342656794896534))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 21, x = 0.13939665, y = 0.11227876, z = 0.0830115565216829, phix = -0.000452753589793032, phiz = -0.000556786794896569))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 23, x = 0.05072309, y = 0.10340977, z = -0.0557884434782636, phix = -0.000258373589793093, phiz = -0.00058092679489663))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 25, x = -0.08732435, y = 0.04074979, z = 0.104611556521718, phix = -0.000285613589793188, phiz = 0.000144893205103358))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 27, x = 0.02857068, y = 0.02920064, z = -0.0273884434782303, phix = -0.000303773589793399, phiz = 8.07432051035395e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 29, x = -0.04425003, y = 0.04268277, z = 0.00581155652173493, phix = -0.000115453589793047, phiz = 0.000276603205103365))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 31, x = -0.03427819, y = 0.05137156, z = -0.0776884434782232, phix = -0.00062594358979341, phiz = 0.000102453205103492))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 6, x = -0.1850657, y = -0.05669352, z = -0.00647624347823239, phix = 8.20435897931463e-05, phiz = 0.000556603205103423))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 8, x = -0.09363843, y = -0.00760112, z = 0.167723756521696, phix = 0.000838863589793027, phiz = 0.000220463205103494))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 10, x = 0.107269, y = -0.169, z = -0.150876243478251, phix = -0.000264363589793104, phiz = -0.000813676794896523))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 12, x = 0.24113212, y = -0.11460865, z = -0.124776243478323, phix = -0.000167093589793177, phiz = -0.000877806794896463))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 14, x = 0.19660012, y = -0.02221225, z = -0.150776243478276, phix = -0.000179283589793256, phiz = -0.000471946794896594))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 18, x = 0.18556472, y = 0.01370347, z = 0.0199237565217345, phix = 0.000525733589793143, phiz = -0.000690276794896638))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 20, x = 0.16529897, y = 0.07455478, z = -0.153276243478217, phix = -0.000756863589793111, phiz = -0.000409726794896592))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 22, x = 0.13178845, y = -0.0282643, z = -0.0733762434782648, phix = -0.000309983589793406, phiz = -0.00077386679489666))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 24, x = -0.01284236, y = 0.17344584, z = 0.194723756521739, phix = 0.00134647358979313, phiz = -0.000298046794896534))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 26, x = -0.00973978, y = -0.00514822, z = -0.0883762434782511, phix = -0.000297803589793267, phiz = -0.000472186794896468))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 28, x = 0.0172497, y = 0.072, z = -0.100976243478271, phix = -0.000452693589793396, phiz = 0.000263673205103343))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 30, x = 0.05762959, y = 0.1293418, z = -0.155376243478258, phix = -0.000334203589793203, phiz = -8.04667948965943e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 32, x = -0.06894813, y = 0.09045599, z = -0.000776243478298966, phix = -0.000161023589793208, phiz = -0.000594606794896535))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 2, ring = 2, chamber = 34, x = -0.00536987, y = 0.14469911, z = -0.466676243478219, phix = -0.00217893358979339, phiz = -0.000328786794896452))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 1, x = 0.12464372, y = 0.04490414, z = 0.123572234042626, phix = 0.00202107, phiz = -0.000307866794896583))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 3, x = 0.15132085, y = -0.08185831, z = -0.112627765957427, phix = 0.00020493, phiz = 0.000627833205103512))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 5, x = 0.01280388, y = -0.15387293, z = -0.0651277659574134, phix = 0.00021062, phiz = 0.000880133205103384))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 7, x = 0.02664976, y = -0.15505183, z = -0.107527765957457, phix = 0.00036428, phiz = 0.000888433205103345))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 9, x = -0.04118758, y = -0.01772642, z = -0.111227765957437, phix = 0.00015373, phiz = 0.000436733205103401))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 13, x = 0.02267438, y = -0.00042916, z = 0.219272234042592, phix = 0.00325534, phiz = 0.000288633205103528))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 15, x = 0.08233609, y = 0.13275095, z = -0.0844277659574573, phix = 0.00245855, phiz = -0.000113066794896532))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 17, x = 0.09382761, y = 0.07213797, z = 0.175972234042547, phix = 0.00239662, phiz = -0.000579756794896635))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 2, x = 0.15656207, y = 0.02191579, z = -0.034515565957463, phix = 0.00080234, phiz = -0.00123830679489645))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 4, x = -0.04226705, y = -0.19952867, z = -0.11061556595746, phix = -0.00029032, phiz = -0.000574016794896659))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 6, x = 0.00359069, y = -0.18779042, z = -0.130015565957478, phix = 0.00031314, phiz = 0.000114283205103538))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 8, x = 0.0369314, y = -0.21795999, z = -0.138215565957466, phix = 0.00010816, phiz = -0.000127406794896645))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 12, x = 0.0735391, y = 0.04400669, z = -0.1987155659574, phix = 0.00014802, phiz = 0.000804493205103451))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 14, x = 0.09799262, y = -0.000866, z = 0.0442844340425381, phix = 0.00217521, phiz = 0.000632793205103344))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 16, x = 0.02742219, y = 0.09710766, z = -0.16681556595745, phix = -0.00052941, phiz = -0.000235916794896651))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 1, chamber = 18, x = 0.0797748, y = 0.04470836, z = -0.126515565957448, phix = -0.00023344, phiz = -0.000785606794896587))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 3, x = 0.24270578, y = 0.22647915, z = 0.37667223404253, phix = 0.00101778, phiz = -0.000660866794896631))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 5, x = 0.20566583, y = 0.0449228, z = 0.400072234042568, phix = 0.00151614, phiz = -0.00041672679489646))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 7, x = 0.08320257, y = -0.09211108, z = 0.0119722340425596, phix = -0.00017015, phiz = -0.00024754679489658))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 9, x = 0.07127921, y = -0.02224214, z = -0.00772776595738378, phix = -0.00039811, phiz = -5.33967948965763e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 11, x = 0.01301877, y = 0.05732852, z = 0.144772234042534, phix = 7.596e-05, phiz = -0.000149256794896635))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 13, x = -0.1487237, y = 0.040403, z = 0.110772234042543, phix = 0.00011849, phiz = 0.000304893205103518))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 17, x = -0.17635042, y = 0.03247408, z = -0.0108277659574014, phix = 3.647e-05, phiz = 0.000573213205103507))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 19, x = -0.104746, y = 0.048, z = 0.018772234042558, phix = 0.00011243, phiz = -0.000262656794896454))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 21, x = -0.08678077, y = 0.06971623, z = 0.19947223404256, phix = 0.00138555, phiz = 0.000116793205103383))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 23, x = -0.16101753, y = 0.04975218, z = 0.0872722340425298, phix = 0.00010333, phiz = 0.000380933205103462))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 25, x = -0.05300766, y = -0.03218804, z = -0.0124277659574545, phix = -5.775e-05, phiz = -9.48867948966647e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 27, x = -0.13058955, y = 0.06299871, z = 0.151572234042533, phix = 0.000951, phiz = 0.00100925320510337))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 33, x = 0.07476999, y = 0.11053344, z = 0.00167223404253036, phix = 2.735e-05, phiz = -0.000499276794896586))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 35, x = 0.0886567, y = 0.08210497, z = 0.19157223404261, phix = 0.00196552, phiz = 0.00068086320510341))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 2, x = 0.34228066, y = 0.1647853, z = -0.0418155659574495, phix = -4.254e-05, phiz = -0.00060692679489649))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 4, x = 0.16835445, y = 0.10559846, z = -0.0600155659574284, phix = -0.00016404, phiz = -0.000428786794896663))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 6, x = 0.07656583, y = 0.14728633, z = -0.0418155659574495, phix = -1.823e-05, phiz = -0.000159606794896572))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 8, x = 0.07389798, y = 0.00862586, z = 0.0621844340425923, phix = 0.00057421, phiz = 0.000859533205103347))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 10, x = 0.00072212, y = -0.108, z = 0.0420844340425219, phix = -0.00011548, phiz = -0.000236326794896602))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 12, x = 0.02307, y = 0.00312937, z = 0.118184434042632, phix = 0.00055002, phiz = 0.000267813205103495))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 14, x = -0.09049769, y = -0.03422373, z = 0.0258844340426094, phix = 0.00019139, phiz = 1.95320510343322e-06))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 16, x = -0.1561102, y = -0.0536092, z = -0.0365155659574157, phix = 0.00030991, phiz = -1.38667948965665e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 18, x = -0.1354657, y = -0.05158886, z = -0.0281155659574779, phix = 0.00016107, phiz = 0.000290273205103553))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 20, x = -0.12498458, y = 0.01870579, z = -0.0312155659573818, phix = 6.383e-05, phiz = -0.00045027679489662))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 22, x = -0.09976921, y = 0.08719465, z = -0.078915565957459, phix = 0.00041633, phiz = -5.61267948966382e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 24, x = -0.16315435, y = 0.04404276, z = -0.284015565957475, phix = 0.00080837, phiz = 6.80532051033911e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 26, x = -0.06971606, y = -0.00320663, z = -0.0725155659574739, phix = -0.00024617, phiz = -0.000197806794896449))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 28, x = -0.0572999, y = 0.001, z = -0.120415565957387, phix = -0.00014282, phiz = 0.000316323205103552))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 30, x = 0.1250197, y = 0.08733898, z = -0.164115565957445, phix = -0.00027353, phiz = -3.95367948966552e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 32, x = 0.10196627, y = -0.02732263, z = -0.189215565957397, phix = 0.00011853, phiz = -0.000503396794896638))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 3, ring = 2, chamber = 34, x = 0.28748391, y = 0.13206325, z = 0.0599844340425761, phix = -0.00070807, phiz = -0.000373216794896658))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 1, x = 0.26028907, y = 0.04293448, z = 0.0904827888887212, phix = 0.00250464, phiz = -0.000670866794896474))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 3, x = 0.1463711, y = -0.25187623, z = -0.0912172111112568, phix = -0.00035959, phiz = 0.000408833205103543))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 5, x = 0.09046847, y = -0.15926442, z = 0.0073827888887763, phix = -0.00021199, phiz = -2.98667948965825e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 7, x = 0.0438613, y = -0.10796729, z = 0.0304827888887758, phix = 0.00200935, phiz = 0.000378433205103335))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 9, x = -0.15457732, y = -0.02322306, z = 0.159882788888694, phix = 0.00236903, phiz = 0.000976733205103386))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 11, x = -0.00616344, y = 0.03958037, z = -0.0521172111112946, phix = 0.00032738, phiz = 0.000530343205103367))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 13, x = 0.11624442, y = 0.07325792, z = -0.0691172111112337, phix = 0.00028886, phiz = -0.0015113667948965))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 15, x = 0.00733436, y = 0.12079347, z = 0.0731827888887437, phix = 0.00578472, phiz = 3.693320510334e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 17, x = 0.18204451, y = 0.16740803, z = 0.0963827888887181, phix = 0.00245209, phiz = -0.000534756794896563))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 2, x = 0.19053227, y = 0.06150758, z = -0.164805011111298, phix = 0.00119356, phiz = 0.000290693205103443))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 4, x = 0.14836591, y = -0.19202158, z = -0.034605011111239, phix = -0.00019904, phiz = 0.000455983205103427))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 6, x = 0.05767508, y = -0.1484085, z = 0.126394988888705, phix = 0.00297234, phiz = -8.5716794896662e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 8, x = -0.04550471, y = -0.07249477, z = 0.217194988888764, phix = 0.00323649, phiz = 0.000352593205103391))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 10, x = -0.15364184, y = 0.03999977, z = -0.0812050111112512, phix = -0.0002825, phiz = 0.000396193205103534))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 12, x = -0.02404163, y = 0.16460966, z = -0.0639050111112738, phix = -5.133e-05, phiz = 3.44932051035141e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 14, x = 0.01165971, y = 0.07375365, z = -0.18990501111125, phix = 0.00213179, phiz = 0.000392793205103548))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 16, x = 0.09344805, y = 0.05873951, z = 0.01039498888872, phix = 0.00026974, phiz = -0.00105891679489645))
    CSCphotogrammetry.append(CSCChamber(endcap = 1, station = 4, ring = 1, chamber = 18, x = 0.28073448, y = 0.09238311, z = -0.0649050111112501, phix = -5.136e-05, phiz = -0.0006706067948965))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 1, x = 0.03272088, y = -0.00231938, z = 0.223832788888899, phix = -4.087e-05, phiz = 0.000648533205103385))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 3, x = 0.03252691, y = -0.21329506, z = 0.0805327888888314, phix = -0.00034748, phiz = -0.000610166794896561))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 5, x = -0.04444004, y = -0.12882215, z = 0.0351327888888591, phix = -0.00015839, phiz = 0.000540133205103377))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 7, x = -0.07443857, y = -0.15687053, z = -0.149467211111187, phix = -0.0003066, phiz = 0.0003884332051034))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 9, x = -0.08803488, y = -0.017575, z = 0.18413278888886, phix = 0.00140058, phiz = 0.00018673320510354))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 11, x = -0.12245858, y = 0.09361748, z = 0.0172327888888049, phix = -8.688e-05, phiz = 0.000980343205103429))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 13, x = -0.01943353, y = -0.05640462, z = 0.178632788888876, phix = 0.00184027, phiz = 0.000318633205103502))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 15, x = -0.03791445, y = -0.05836686, z = 0.0351327888888591, phix = -0.00021972, phiz = 0.000366933205103503))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 17, x = -0.05723164, y = -0.01589905, z = -0.0915672111111689, phix = -0.00099663, phiz = 0.000175243205103426))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 2, x = 0.00627867, y = 0.09758223, z = 0.0805449888888461, phix = -0.00034738, phiz = 0.000149693205103496))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 4, x = -0.03569325, y = -0.29128251, z = -0.0599550111111284, phix = 0.00122623, phiz = -0.00171401679489658))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 6, x = -0.05611354, y = -0.19428799, z = 0.00704498888887883, phix = -0.00080735, phiz = -0.000395716794896472))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 8, x = -0.10874219, y = -0.15329142, z = 0.0696449888888537, phix = 0.00189114, phiz = 0.000712593205103529))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 10, x = 0.02597123, y = 0.03641629, z = 0.0687449888888523, phix = -6.132e-05, phiz = 0.000276193205103414))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 12, x = -0.13010765, y = 0.02570703, z = 0.0162449888888432, phix = -6.645e-05, phiz = 4.44932051033575e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 14, x = -0.10392442, y = -0.05472459, z = -0.102455011111147, phix = 0.00018908, phiz = -1.72067948964738e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 16, x = 0.05584571, y = 0.0326816, z = 0.145044988888799, phix = 0.0007616, phiz = -0.00128591679489665))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 1, chamber = 18, x = 0.03211637, y = 0.04981882, z = 0.179544988888892, phix = 0.00124163, phiz = -0.00114960679489662))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 1, x = 0.22039, y = 0.007, z = 0.158532788888806, phix = 0.00078386, phiz = -0.000932726794896643))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 3, x = 0.29361847, y = 0.20262756, z = -0.043667211111142, phix = 0.00055291, phiz = -0.00128586679489651))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 5, x = 0.1949147, y = 0.04611162, z = 0.12673278888883, phix = -0.0006532, phiz = -0.000748726794896459))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 7, x = 0.17218076, y = -0.07422582, z = -0.00956721111117531, phix = -0.00046174, phiz = -0.000537546794896482))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 9, x = 0.08478393, y = -0.01072205, z = 0.0308327888888016, phix = 3.645e-05, phiz = -0.000443396794896467))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 11, x = -0.01407191, y = -0.00186653, z = -0.104867211111127, phix = -0.00129126, phiz = -0.000349256794896613))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 13, x = -0.06913078, y = -0.05173802, z = -0.0454672111111449, phix = 0.00071091, phiz = 7.48932051033435e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 15, x = -0.09916713, y = 0.05335057, z = 0.00193278888889381, phix = 7.291e-05, phiz = -0.00058092679489663))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 17, x = -0.10756977, y = 0.01600526, z = -0.112567211111127, phix = -0.00038894, phiz = -5.67867948966239e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 19, x = -0.114618, y = -0.063, z = -0.0926672111111202, phix = -0.0004985, phiz = 0.000122653205103518))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 21, x = -0.05356331, y = 0.01186642, z = -0.0895672111111026, phix = -0.00041948, phiz = 3.67932051035247e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 23, x = -0.07187409, y = 0.02064121, z = -0.0984672111111422, phix = -0.00032824, phiz = 8.0933205103495e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 25, x = -0.0033596, y = 0.043819, z = -0.0273672111111409, phix = -0.00021277, phiz = -0.000134886794896483))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 27, x = 0.02775875, y = -0.05530164, z = 0.0241327888888918, phix = -0.00044538, phiz = -0.000560746794896483))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 29, x = 0.05594185, y = -0.09544598, z = 0.118232788888804, phix = 0.00038297, phiz = -0.000276606794896495))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 31, x = 0.0856224, y = -0.03369766, z = -0.0734672111111649, phix = -0.00034041, phiz = -0.000682456794896646))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 33, x = 0.19020673, y = 0.05074749, z = 0.146732788888812, phix = 0.00040419, phiz = -0.000893276794896591))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 35, x = 0.2355044, y = 0.13768152, z = 0.0107327888888449, phix = 0.00042238, phiz = -0.000800136794896655))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 2, x = 0.26555003, y = 0.07169519, z = -0.177355011111104, phix = 0.00027035, phiz = -0.00106092679489667))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 4, x = 0.29206468, y = 0.14387087, z = 0.119344988888884, phix = 2.734e-05, phiz = -0.000870786794896494))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 6, x = 0.26291198, y = -0.01878629, z = -0.0183550111111117, phix = -0.00022481, phiz = -0.000523606794896603))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 8, x = 0.079854, y = 0.09167118, z = 0.0288449888888636, phix = 6.08e-06, phiz = 1.95332051033947e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 10, x = 0.0117629, y = -0.145, z = -0.0100550111111488, phix = 0.00013371, phiz = -0.000276326794896642))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 12, x = -0.04487803, y = -0.08707076, z = 0.0118449888888108, phix = -3.038e-05, phiz = 9.78132051034919e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 14, x = -0.14775082, y = -0.1911316, z = -0.118055011111096, phix = -0.00042225, phiz = 6.19532051033822e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 16, x = -0.28789927, y = -0.06734383, z = -0.100755011111119, phix = -0.00015495, phiz = 0.000416133205103364))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 18, x = -0.15946815, y = 0.09547742, z = -0.280455011111144, phix = 0.00075979, phiz = -7.97267948966507e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 22, x = 0.007991, y = -0.05815918, z = -0.0720550111111606, phix = -0.00045282, phiz = 0.000233873205103485))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 24, x = 0.11316226, y = -0.03478701, z = -0.0126550111111783, phix = 2.736e-05, phiz = -0.000501946794896568))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 26, x = 0.08673581, y = -0.06757907, z = -0.0516550111111655, phix = 3.952e-05, phiz = -0.000727806794896591))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 28, x = 0.0985838, y = -0.105, z = 0.0114449888887975, phix = 0.00013071, phiz = -0.000613676794896545))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 30, x = 0.07790039, y = -0.0625869, z = 0.151544988888872, phix = 0.00025833, phiz = -0.000149536794896488))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 32, x = 0.1015898, y = 0.16358274, z = -0.0937550111111705, phix = -0.00014283, phiz = -0.000484396794896647))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 34, x = 0.12916856, y = 0.14227349, z = 0.046144988888841, phix = 0.00031603, phiz = -0.00101521679489647))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 2, ring = 2, chamber = 36, x = 0.1425567, y = 0.16091767, z = -0.183955011111152, phix = 0.00024306, phiz = -0.000971066794896558))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 1, x = -0.05592091, y = -0.14282063, z = 0.167858832652996, phix = 0.0026577435897931, phiz = 0.00103046320510347))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 3, x = 0.1039447, y = -0.1957025, z = -0.0721411673470129, phix = -0.000483853589793204, phiz = -0.000253836794896545))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 5, x = 0.22401635, y = -0.16053853, z = 0.0650588326529942, phix = 0.000711423589793048, phiz = -9.01367948966669e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 7, x = 0.17233366, y = -0.09751403, z = -0.0904411673469667, phix = -0.000318813589793033, phiz = -0.0010684267948966))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 11, x = 0.16700208, y = 0.0094963, z = -0.0979411673470167, phix = -0.000335773589793431, phiz = -0.00019034679489649))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 13, x = 0.12813915, y = -0.0839952, z = -0.057041167346938, phix = -0.000472173589793135, phiz = -0.000228636794896486))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 15, x = -0.0086396, y = -0.00589939, z = -0.252241167346938, phix = 0.000460983589793196, phiz = 0.00041306320510337))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 17, x = -0.13718235, y = -0.05254309, z = 0.0204588326530484, phix = 0.00424077358979303, phiz = 0.00105175320510353))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 2, x = -0.02190078, y = -0.17458443, z = 0.0972710326529977, phix = 0.00199795358979328, phiz = 0.000293303205103346))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 4, x = 0.10542526, y = -0.08774745, z = -0.0608289673469926, phix = 0.000341573589793265, phiz = 0.00127402320510339))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 6, x = 0.23065378, y = -0.211988, z = -0.102928967346998, phix = -0.000335823589793128, phiz = 0.000435713205103383))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 8, x = 0.20999028, y = -0.18088317, z = -0.0596289673469528, phix = 3.98435897933761e-05, phiz = -0.00186258679489648))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 10, x = 0.17675613, y = 0.17165519, z = -0.0909289673469402, phix = -0.000301633589793083, phiz = -0.00069618679489647))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 12, x = 0.08131728, y = 0.04188537, z = 0.0653710326530472, phix = 0.00205482358979315, phiz = -0.000734486794896627))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 14, x = 0.12765714, y = 0.05066342, z = -0.11122896734696, phix = -6.83135897931012e-05, phiz = -0.000292786794896527))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 16, x = -0.07081855, y = -0.06756401, z = -0.0979289673470021, phix = -0.000278893589793172, phiz = -0.000423086794896665))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 1, chamber = 18, x = -0.01241019, y = -0.074799, z = -0.11892896734696, phix = 0.00017073358979326, phiz = 0.00116961320510334))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 1, x = -0.0355435, y = -0.061, z = 0.328658832652991, phix = 0.000881333589793376, phiz = 4.55832051033944e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 3, x = -0.06492433, y = -0.02806702, z = 0.274658832653017, phix = 0.00192036358979321, phiz = 0.00025086320510348))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 5, x = -0.00642146, y = 0.0019381, z = 0.154158832653025, phix = 0.000297743589793187, phiz = 0.000737723205103347))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 7, x = 0.17362686, y = -0.07526945, z = -0.0766411673470202, phix = -1.51835897931444e-05, phiz = -0.000562456794896526))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 9, x = 0.11667637, y = -0.07061518, z = 0.0530588326530506, phix = 0.000689573589793058, phiz = -5.66067948966076e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 11, x = 0.20698017, y = -0.204446, z = 0.0430588326530597, phix = -0.000312903589793423, phiz = -0.00100074679489648))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 17, x = 0.1138125, y = 0.03608083, z = 0.131058832653025, phix = 0.000255263589793342, phiz = -0.000163206794896453))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 19, x = 0.28681, y = 0.088, z = 0.0249588326530557, phix = 0.000580393589793127, phiz = -0.000322656794896625))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 21, x = 0.18493719, y = 0.10118553, z = 0.017758832653044, phix = -0.000115483589793309, phiz = -0.000306786794896485))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 23, x = 0.20579137, y = 0.09907056, z = -0.0236411673470229, phix = 3.03358979326685e-06, phiz = -0.000360926794896521))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 27, x = 0.15471734, y = 0.05874433, z = 0.0692588326529631, phix = -0.00110916358979342, phiz = -0.000229256794896493))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 29, x = -0.0747244, y = 0.01395451, z = -0.0486411673470002, phix = -0.000516523589793161, phiz = 0.000906603205103496))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 31, x = -0.1540025, y = -0.08125983, z = -0.00184116734703821, phix = -0.000258383589793032, phiz = -0.000367546794896478))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 33, x = -0.12090094, y = 0.06177962, z = 0.111158832653018, phix = -0.000191443589793073, phiz = -0.00029872679489662))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 35, x = -0.23299965, y = -0.06010303, z = 0.0279588326529847, phix = 0.00010329358979323, phiz = 0.000884133205103499))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 2, x = -0.1024115, y = -0.1330872, z = -0.0330289673470361, phix = 5.46935897932758e-05, phiz = 0.000265933205103375))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 4, x = 0.03507118, y = -0.11874506, z = 0.0184710326529967, phix = 0.000136723589793453, phiz = -3.12167948965936e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 6, x = 0.00338251, y = -0.10124991, z = -0.0581289673469882, phix = 0.00100262358979343, phiz = 0.000352603205103552))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 8, x = 0.22526629, y = -0.09919686, z = -0.109528967347046, phix = 0.00013065358979304, phiz = -0.000189536794896528))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 10, x = 0.276132, y = -0.016, z = -0.0436289673469901, phix = -0.000498353589793455, phiz = -0.000143676794896574))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 12, x = 0.26641055, y = -0.07880361, z = -0.0211289673469537, phix = -3.64535897931065e-05, phiz = -0.000707806794896459))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 14, x = 0.14754805, y = -0.16910982, z = -0.00172896734704864, phix = 0.000401013589793065, phiz = -0.000161946794896561))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 16, x = 0.3647294, y = 0.05572985, z = 0.0501710326529974, phix = -0.000240013589793209, phiz = -0.000556126794896583))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 20, x = 0.15005933, y = 0.16731772, z = 0.180271032652968, phix = 0.000938843589793359, phiz = -0.000169726794896574))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 22, x = 0.17484547, y = 0.13515875, z = 0.141471032653044, phix = 0.00220951358979317, phiz = -0.000153866794896595))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 24, x = 0.03258097, y = -0.06178537, z = -0.162928967346943, phix = -0.000343433589793063, phiz = 0.00038195320510348))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 26, x = 0.1069721, y = 0.01728697, z = -0.0344289673470257, phix = 0.000246173589793075, phiz = 5.78132051034519e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 28, x = 0.00508872, y = 0.198, z = -0.13432896734696, phix = -0.000158003589793354, phiz = 0.000963673205103488))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 30, x = 0.01415843, y = 0.07627337, z = -0.155428967347007, phix = -0.000285693589793146, phiz = 0.000249533205103347))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 32, x = -0.16045866, y = -0.07118195, z = -0.0343289673470508, phix = -0.000200593589793419, phiz = 0.000208393205103352))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 34, x = -0.13893205, y = -0.07263736, z = 0.105571032652961, phix = 0.00129169358979343, phiz = -1.37867948966086e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 3, ring = 2, chamber = 36, x = -0.13171023, y = -0.02174079, z = 0.00387103265302358, phix = -1.82235897933206e-05, phiz = -0.000305926794896605))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 1, x = -0.00375728, y = -0.00589311, z = -0.0512505444444287, phix = -0.000147613589793439, phiz = -9.23679489650553e-06))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 3, x = -0.01626346, y = -0.14298179, z = 0.0462494555556532, phix = 0.000898963589793256, phiz = 0.000268163205103367))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 5, x = 0.02867642, y = -0.16968825, z = 0.00524945555559952, phix = 0.00253598358979325, phiz = 0.000549863205103529))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 7, x = 0.03438047, y = -0.14473991, z = -0.0171505444443483, phix = 0.00053938358979323, phiz = 0.000581573205103547))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 9, x = 0.24275029, y = 0.0473612, z = -0.116150544444395, phix = 0.00152774358979325, phiz = -0.00115673679489658))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 11, x = 0.20408056, y = 0.06235783, z = 0.130349455555574, phix = 0.00205402358979313, phiz = -0.000380346794896624))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 13, x = 0.14304759, y = 0.06378925, z = 0.119849455555595, phix = 0.00204731358979305, phiz = -0.0009086367948965))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 15, x = 0.06046132, y = -0.03910823, z = -0.0708505444443972, phix = 0.000340203589793154, phiz = -0.000226936794896604))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 17, x = 0.08508658, y = 0.02064638, z = -0.133450544444372, phix = 0.000179803589793204, phiz = -0.000315246794896584))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 2, x = -0.12675358, y = 0.00518459, z = 0.0118616555556628, phix = 0.00364648358979304, phiz = -0.000223696794896533))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 4, x = 0.05012777, y = -0.12128076, z = -0.0163383444444207, phix = -0.000160423589793302, phiz = 0.00103402320510337))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 6, x = 0.19262729, y = -0.16105645, z = -0.0619383444443429, phix = 0.000744833589793392, phiz = -0.000794286794896459))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 8, x = 0.29741182, y = -0.1207007, z = -0.0112383444443367, phix = 0.000423723589793157, phiz = -0.00092258679489654))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 10, x = 0.16217173, y = 0.03773458, z = -0.0773383444443425, phix = -0.000186163589793187, phiz = -7.61867948966266e-05))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 12, x = 0.19304015, y = 0.06632182, z = 0.216961655555565, phix = 0.00270921358979338, phiz = -0.000624486794896573))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 14, x = 0.06906927, y = 0.07974044, z = 0.160161655555612, phix = 0.00243311358979316, phiz = 0.000127213205103338))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 16, x = -0.04404145, y = 0.00910093, z = -0.192738344444365, phix = 0.000468713589793291, phiz = 0.000114913205103484))
    CSCphotogrammetry.append(CSCChamber(endcap = 2, station = 4, ring = 1, chamber = 18, x = 0.04367915, y = 0.05204417, z = 0.0577616555556233, phix = 0.00249092358979309, phiz = -0.00102338679489655))
    return CSCphotogrammetry

# run it all!
make_scenario_sqlite()
