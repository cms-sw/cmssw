#!/usr/bin/python

from Alignment.MuonAlignment.geometryXMLparser import MuonGeometry, dtorder, cscorder
import sys, getopt

usage = "Usage: geometryDiff.py [-h|--help] [-e|--epsilon epsilon] geometry1.xml geometry2.xml"

try:
  opts, args = getopt.getopt(sys.argv[1:], "he:", ["help", "epsilon="])
except getopt.GetoptError, msg:
  print >>sys.stderr, usage
  sys.exit(2)

if len(args) != 2:
  print >>sys.stderr, usage
  sys.exit(2)

opts = dict(opts)

if "-h" in opts or "--help" in opts:
  print usage
  sys.exit(0)

epsilon = 1e-6
if "-e" in opts: epsilon = float(opts["-e"])
if "--epsilon" in opts: epsilon = float(opts["--epsilon"])

geom1 = MuonGeometry(file(args[0]))
geom2 = MuonGeometry(file(args[1]))

from math import sin, cos, sqrt
sqrtepsilon = sqrt(epsilon)

def matrixmult(a, b):
  return [[sum([i*j for i, j in zip(row, col)]) for col in zip(*b)] for row in a]

def transpose(a):
  return [[a[j][i] for j in range(len(a[i]))] for i in range(len(a))]

def rotFromPhi(g):
  phix, phiy, phiz = g.phix, g.phiy, g.phiz
  rotX = [[1.,         0.,         0.,       ],
          [0.,         cos(phix),  sin(phix),],
          [0.,        -sin(phix),  cos(phix),]]
  rotY = [[cos(phiy),  0.,        -sin(phiy),],
          [0.,         1.,         0.,       ],
          [sin(phiy),  0.,         cos(phiy),]]
  rotZ = [[cos(phiz),  sin(phiz),  0.,       ],
          [-sin(phiz), cos(phiz),  0.,       ],
          [0.,         0.,         1.,       ]]
  return matrixmult(rotX, matrixmult(rotY, rotZ))

def rotFromEuler(g):
  s1, s2, s3 = sin(g.alpha), sin(g.beta), sin(g.gamma)
  c1, c2, c3 = cos(g.alpha), cos(g.beta), cos(g.gamma)
  return [[c2 * c3,    c1 * s3 + s1 * s2 * c3,   s1 * s3 - c1 * s2 * c3,],
          [-c2 * s3,   c1 * c3 - s1 * s2 * s3,   s1 * c3 + c1 * s2 * s3,],
          [s2,        -s1 * c2,                  c1 * c2,               ]]

def loopover(which):
  if which == "DT":
    keys = geom1.dt.keys()
    keys.sort(dtorder)

  elif which == "CSC":
    keys = geom1.csc.keys()
    keys.sort(cscorder)

  else: raise Exception

  for key in keys:
    if which == "DT":
      g1 = geom1.dt[key]
      g2 = geom2.dt[key]
    else:
      g1 = geom1.csc[key]
      g2 = geom2.csc[key]

    if g1.relativeto != g2.relativeto:
      print "%s %s relativeto=\"%s\" versus relativeto=\"%s\"" % (which, str(key), g1.relativeto, g2.relativeto)

    if abs(g1.x - g2.x) > epsilon or abs(g1.y - g2.y) > epsilon or abs(g1.z - g2.z) > epsilon:
      print "%s %s position difference: (%g, %g, %g) - (%g, %g, %g) = (%g, %g, %g)" % \
            (which, str(key), g1.x, g1.y, g1.z, g2.x, g2.y, g2.z, g1.x - g2.x, g1.y - g2.y, g1.z - g2.z)

    if "phix" in g1.__dict__:
      g1type = "phi"
      g1a, g1b, g1c = g1.phix, g1.phiy, g1.phiz
      g1rot = rotFromPhi(g1)
    else:
      g1type = "euler"
      g1a, g1b, g1c = g1.alpha, g1.beta, g1.gamma
      g1rot = rotFromEuler(g1)

    if "phix" in g2.__dict__:
      g2type = "phi"
      g2a, g2b, g2c = g2.phix, g2.phiy, g2.phiz
      g2rot = rotFromPhi(g2)
    else:
      g2type = "euler"
      g2a, g2b, g2c = g2.alpha, g2.beta, g2.gamma
      g2rot = rotFromEuler(g2)
    
    diff = matrixmult(g1rot, transpose(g2rot))
    if abs(diff[0][0] - 1.) > sqrtepsilon or abs(diff[1][1] - 1.) > sqrtepsilon or abs(diff[2][2] - 1.) > sqrtepsilon or \
       abs(diff[0][1]) > epsilon or abs(diff[0][2]) > epsilon or abs(diff[1][2]) > epsilon:
      print "%s %s rotation difference: %s(%g, %g, %g) - %s(%g, %g, %g) = %s" % \
            (which, str(key), g1type, g1a, g1b, g1c, g2type, g2a, g2b, g2c, str(diff))

loopover("DT")
loopover("CSC")


