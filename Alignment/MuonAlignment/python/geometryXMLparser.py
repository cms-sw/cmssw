#!/usr/bin/python

# XML must come from MuonGeometryDBConverter; it must not be hand-made
# Example configuration that will work
# 
# PSet outputXML = {
#     string fileName = "tmp.xml"
#     string relativeto = "container"   # keep in mind which relativeto you used when interpreting positions and angles!
#     bool survey = false               # important: survey must be false
#     bool rawIds = false               # important: rawIds must be false
#     bool eulerAngles = false
# }

# External libraries (standard in Python >= 2.4, at least)
import xml.sax

class Alignable:
    def pos(self):
        return self.x, self.y, self.z
    def covariance(self):
        return (self.xx, self.xy, self.xz), (self.xy, self.yy, self.yz), (self.xz, self.yz, self.zz)

class DTAlignable:
    def index(self):
        i = []
        try: i.append(self.wheel)
        except AttributeError: pass
        try: i.append(self.station)
        except AttributeError: pass
        try: i.append(self.sector)
        except AttributeError: pass
        try: i.append(self.superlayer)
        except AttributeError: pass
        try: i.append(self.layer)
        except AttributeError: pass
        return tuple(i)

class CSCAlignable:
    def index(self):
        i = []
        try: i.append(self.endcap)
        except AttributeError: pass
        try: i.append(self.station)
        except AttributeError: pass
        try: i.append(self.ring)
        except AttributeError: pass
        try: i.append(self.chamber)
        except AttributeError: pass
        try: i.append(self.layer)
        except AttributeError: pass
        return tuple(i)

class Operation:
    def __init__(self):
        self.chambers = []
        self.setposition = {}
        self.setape = {}

# This class is a subclass of something which knows how to parse XML
class MuonGeometry(xml.sax.handler.ContentHandler):
    def __init__(self, stream):
        self.dt = {}
        self.csc = {}
        self._operation = None

        parser = xml.sax.make_parser()
        parser.setContentHandler(self)
        parser.parse(stream)

        self.dtWheels, self.dtStations, self.dtChambers, self.dtSuperLayers, self.dtLayers = [], [], [], [], []
        for index, alignable in self.dt.iteritems():
            if len(index) == 1: self.dtWheels.append(alignable)
            elif len(index) == 2: self.dtStations.append(alignable)
            elif len(index) == 3: self.dtChambers.append(alignable)
            elif len(index) == 4: self.dtSuperLayers.append(alignable)
            elif len(index) == 5: self.dtLayers.append(alignable)

#         if len(self.dtWheels) == 0: del self.dtWheels
#         if len(self.dtStations) == 0: del self.dtStations
#         if len(self.dtChambers) == 0: del self.dtChambers
#         if len(self.dtSuperLayers) == 0: del self.dtSuperLayers
#         if len(self.dtLayers) == 0: del self.dtLayers
            
        self.cscEndcaps, self.cscStations, self.cscRings, self.cscChambers, self.cscLayers = [], [], [], [], []
        for index, alignable in self.csc.iteritems():
            if len(index) == 1: self.cscEndcaps.append(alignable)
            elif len(index) == 2: self.cscStations.append(alignable)
            elif len(index) == 3: self.cscRings.append(alignable)
            elif len(index) == 4: self.cscChambers.append(alignable)
            elif len(index) == 5: self.cscLayers.append(alignable)

#         if len(self.cscEndcaps) == 0: del self.cscEndcaps
#         if len(self.cscStations) == 0: del self.cscStations
#         if len(self.cscRings) == 0: del self.cscRings
#         if len(self.cscChambers) == 0: del self.cscChambers
#         if len(self.cscLayers) == 0: del self.cscLayers

    # what to do when you get to a <startelement>
    def startElement(self, tag, attrib):
        attrib = dict(attrib.items())
        if "rawId" in attrib: raise Exception, "Please use \"rawIds = false\""
        if "aa" in attrib: raise Exception, "Please use \"survey = false\""

        if tag == "MuonAlignment": pass

        elif tag == "collection": raise NotImplementedError, "<collection /> and <collection> blocks aren't implemented yet"

        elif tag == "operation":
            self._operation = Operation()

        elif self._operation is None: raise Exception, "All chambers and positions must be enclosed in <operation> blocks"

        elif tag == "setposition":
            self._operation.setposition["relativeto"] = str(attrib["relativeto"])

            for name in "x", "y", "z":
                self._operation.setposition[name] = float(attrib[name])
            try:
                for name in "phix", "phiy", "phiz":
                    self._operation.setposition[name] = float(attrib[name])
            except KeyError:
                for name in "alpha", "beta", "gamma":
                    self._operation.setposition[name] = float(attrib[name])

        elif tag == "setape":
            for name in "xx", "xy", "xz", "yy", "yz", "zz":
                self._operation.setposition[name] = float(attrib[name])

        elif tag[0:2] == "DT":
            alignable = DTAlignable()
            for name in "wheel", "station", "sector", "superlayer", "layer":
                if name in attrib:
                    alignable.__dict__[name] = int(attrib[name])
            self._operation.chambers.append(alignable)

        # <CSC...>: print endcap/station/ring/chamber/layer
        elif tag[0:3] == "CSC":
            alignable = CSCAlignable()
            for name in "endcap", "station", "ring", "chamber", "layer":
                if name in attrib:
                    alignable.__dict__[name] = int(attrib[name])
            self._operation.chambers.append(alignable)

    # what to do when you get to an </endelement>
    def endElement(self, tag):
        if tag == "operation":
            if self._operation is None: raise Exception, "Unbalanced <operation></operation>"
            for c in self._operation.chambers:
                c.__dict__.update(self._operation.setposition)
                c.__dict__.update(self._operation.setape)
                if isinstance(c, DTAlignable): self.dt[c.index()] = c
                elif isinstance(c, CSCAlignable): self.csc[c.index()] = c
