#!/usr/bin/env python

# XML must come from MuonGeometryDBConverter; it must not be hand-made
# Example configuration that will work
# 
# PSet outputXML = {
#     string fileName = "tmp.xml"
#     string relativeto = "container"   # keep in mind which relativeto you used when interpreting positions and angles!
#     bool survey = false               # important: survey must be false
#     bool rawIds = false               # important: rawIds must be false
#     bool eulerAngles = false
# 
#     untracked bool suppressDTBarrel = true
#     untracked bool suppressDTWheels = true
#     untracked bool suppressDTStations = true
#     untracked bool suppressDTChambers = true
#     untracked bool suppressDTSuperLayers = true
#     untracked bool suppressDTLayers = true
#     untracked bool suppressCSCEndcaps = true
#     untracked bool suppressCSCStations = true
#     untracked bool suppressCSCRings = true
#     untracked bool suppressCSCChambers = true
#     untracked bool suppressCSCLayers = false
# }

# External libraries (standard in Python >= 2.4, at least)
from xml.sax import handler, make_parser
from sys import stdin

# Headers for the CSV file
print "Alignable, wheel, station, sector, superlayer, layer, relativeto, x, y, z, angletype, phix, phiy, phiz, xx, xy, xz, yy, yz, zz"
print ", endcap, station, ring, chamber, layer, , , , , , alpha, beta, gamma, , , , , , "

# This class is a subclass of something which knows how to parse XML
class ContentHandler(handler.ContentHandler):
    # what to do when you get to a <startelement>
    def startElement(self, tag, attrib):
        attrib = dict(attrib.items())
        if "rawId" in attrib: raise Exception, "Please use \"rawIds = false\""
        if "aa" in attrib: raise Exception, "Please use \"survey = false\""

        # <DT...>: print wheel/station/sector/superlayer/layer
        if tag[0:2] == "DT":
            print tag,  # ending with a comma means "don't print end-of-line character"
            for a in "wheel", "station", "sector", "superlayer", "layer":
                if a in attrib:
                    print (", %s" % attrib[a]),
                else:
                    print ", ",

        # <CSC...>: print endcap/station/ring/chamber/layer
        elif tag[0:3] == "CSC":
            print tag,
            for a in "endcap", "station", "ring", "chamber", "layer":
                if a in attrib:
                    print (", %s" % attrib[a]),
                else:
                    print ", ",

        # <setposition>: print x, y, z and phix, phiy, phiz or alpha, beta, gamma
        elif tag == "setposition":
            print (", %(relativeto)s, %(x)s, %(y)s, %(z)s" % attrib),
            if "phix" in attrib:
                print (", phixyz, %(phix)s, %(phiy)s, %(phiz)s" % attrib),
            else:
                print (", Euler, %(alpha)s, %(beta)s, %(gamma)s" % attrib),

        # <setape>: print xx, xy, xz, yy, yz, zz
        elif tag == "setape":
            print (", %(xx)s, %(xy)s, %(xz)s, %(yy)s, %(yz)s, %(zz)s" % attrib),

    # what to do when you get to an </endelement>
    def endElement(self, tag):
        if tag == "operation":
            print ""  # end current line (note: no comma)

# Actually make it and use it on "stdin" (a file object)
parser = make_parser()
parser.setContentHandler(ContentHandler())
parser.parse(stdin)
