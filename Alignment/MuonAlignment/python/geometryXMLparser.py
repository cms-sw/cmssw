#!/usr/bin/python

# XML must come from MuonGeometryDBConverter; not hand-made
# Example configuration that will work
# 
# PSet outputXML = {
#     string fileName = "tmp.xml"
#     string relativeto = "container"   # keep in mind which relativeto you used when interpreting positions and angles!
#     bool survey = false               # important: survey must be false
#     bool rawIds = false               # important: rawIds must be false
#     bool eulerAngles = false
#     int32 precision = 8
# }

def dtorder(a, b):
  for ai, bi, name in zip(list(a) + [0]*(5 - len(a)), \
                          list(b) + [0]*(5 - len(b)), \
                          ("wheel", "station", "sector", "superlayer", "layer")):
    exec("a%s = %d" % (name, ai))
    exec("b%s = %d" % (name, bi))

  if awheel == bwheel and astation == bstation:

    if asector != bsector:
      if astation == 4: sectororder = [0, 1, 2, 3, 4, 13, 5, 6, 7, 8, 9, 10, 14, 11, 12]
      elif awheel == 0: sectororder = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
      else: sectororder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      return cmp(sectororder.index(asector), sectororder.index(bsector))

    elif asuperlayer != bsuperlayer:
      superlayerorder = [0, 1, 3, 2]
      return cmp(superlayerorder.index(asuperlayer), superlayerorder.index(bsuperlayer))

  return cmp(a, b)

def cscorder(a, b):
  for ai, bi, name in zip(list(a) + [0]*(5 - len(a)), \
                          list(b) + [0]*(5 - len(b)), \
                          ("endcap", "station", "ring", "chamber", "layer")):
    exec("a%s = %d" % (name, ai))
    exec("b%s = %d" % (name, bi))

  if astation == 1 and aring == 3: return cmp(a, b)

  elif aendcap == bendcap and astation == bstation and aring == bring and achamber != bchamber:
    if achamber == 0: return -1 # upper hierarchy comes first
    if bchamber == 0: return  1 # upper hierarchy comes first
    if achamber % 2 == 1 and bchamber % 2 == 0: return -1  # odds come first
    elif achamber % 2 == 0 and bchamber % 2 == 1: return 1 # evens come after

  return cmp(a, b)

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
    def __init__(self, stream=None):
        self.dt = {}
        self.csc = {}
        self._operation = None

        if stream is not None:
          parser = xml.sax.make_parser()
          parser.setContentHandler(self)
          parser.parse(stream)

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

    # writing back to xml
    def xml(self, stream=None, precision=8):
      if precision == None: format = "%g"
      else: format = "%." + str(precision) + "f"

      if stream == None:
        output = []
        writeline = lambda x: output.append(x)
      else:
        writeline = lambda x: stream.write(x)

      writeline("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
      writeline("<?xml-stylesheet type=\"text/xml\" href=\"MuonAlignment.xsl\"?>\n")
      writeline("<MuonAlignment>\n\n")

      dtkeys = self.dt.keys()
      dtkeys.sort(dtorder)
      csckeys = self.csc.keys()
      csckeys.sort(cscorder)

      def f(number): return format % number

      def position_ape(ali, attributes):
        writeline("  <%s%s />\n" % (level, attributes))
        writeline("  <setposition relativeto=\"%s\" x=\"%s\" y=\"%s\" z=\"%s\" phix=\"%s\" phiy=\"%s\" phiz=\"%s\" />\n" % \
                  (ali.relativeto, f(ali.x), f(ali.y), f(ali.z), f(ali.phix), f(ali.phiy), f(ali.phiz)))

        if "xx" in ali.__dict__:
          writeline("  <setape xx=\"%s\" xy=\"%s\" xz=\"%s\" yy=\"%s\" yz=\"%s\" zz=\"%s\" />\n" % \
                    (f(ali.xx), f(ali.xy), f(ali.xz), f(ali.yy), f(ali.yz), f(ali.zz)))

      for key in dtkeys:
        writeline("<operation>\n")

        if len(key) == 0: level = "DTBarrel"
        elif len(key) == 1: level = "DTWheel "
        elif len(key) == 2: level = "DTStation "
        elif len(key) == 3: level = "DTChamber "
        elif len(key) == 4: level = "DTSuperLayer "
        elif len(key) == 5: level = "DTLayer "

        ali = self.dt[key]
        attributes = " ".join(["%s=\"%d\"" % (name, value) for name, value in zip(("wheel", "station", "sector", "superlayer", "layer"), key)])
        position_ape(ali, attributes)

        writeline("</operation>\n\n")

      for key in csckeys:
        writeline("<operation>\n")

        if len(key) == 1: level = "CSCEndcap "
        elif len(key) == 2: level = "CSCStation "
        elif len(key) == 3: level = "CSCRing "
        elif len(key) == 4: level = "CSCChamber "
        elif len(key) == 5: level = "CSCLayer "

        ali = self.csc[key]
        attributes = " ".join(["%s=\"%d\"" % (name, value) for name, value in zip(("endcap", "station", "ring", "chamber", "layer"), key)])
        position_ape(ali, attributes)

        writeline("</operation>\n\n")

      writeline("</MuonAlignment>\n")
      if stream == None: return "".join(output)
