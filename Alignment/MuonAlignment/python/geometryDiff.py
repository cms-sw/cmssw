#!/usr/bin/python

from Alignment.MuonAlignment.geometryXMLparser import MuonGeometry, dtorder, cscorder
import sys

geom1 = MuonGeometry(file(sys.argv[1]))
geom2 = MuonGeometry(file(sys.argv[2]))

def loopover(which):
  if which == "DT":
    keys = geom1.dt.keys()
    keys.sort(dtorder)

  elif which == "CSC":
    keys = geom1.csc.keys()
    keys.sort(cscorder)

  else: raise Exception

  for key in keys:
    if which == "DT" and "phix" in geom1.dt[key].__dict__:
      thevars = "x", "y", "z", "phix", "phiy", "phiz"
    elif which == "CSC" and "phix" in geom1.csc[key].__dict__:
      thevars = "x", "y", "z", "phix", "phiy", "phiz"
    else:
      thevars = "x", "y", "z", "alpha", "beta", "gamma"

    if which == "DT":
      g1 = geom1.dt[key]
      g2 = geom2.dt[key]
    else:
      g1 = geom1.csc[key]
      g2 = geom2.csc[key]

    for var in thevars:
      g1v = eval("g1.%s" % var)
      g2v = eval("g2.%s" % var)

      if var in ("x", "y", "z"): epsilon = 0.0001
      else: epsilon = 0.000001

      if abs(g1v - g2v) > epsilon:
        print "%s %s difference in %s: %g - %g = %g" % (which, str(key), var, g1v, g2v, g1v - g2v)

loopover("DT")
loopover("CSC")
