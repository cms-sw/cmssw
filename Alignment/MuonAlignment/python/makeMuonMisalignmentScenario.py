from optparse import OptionParser
from random import gauss
from math import sqrt
import os
execfile("Alignment/MuonAlignment/data/idealTransformation.py")

### Get variances and covariances from the commandline

parser = OptionParser(usage="Usage: python %prog outputName [options] (default is unit matrix times 1e-15)")

parser.add_option("--xx", dest="xx", help="variance of x (cm*cm)", default="1e-15")
parser.add_option("--xy", dest="xy", help="covariance of x and y (cm*cm)", default="0")
parser.add_option("--xz", dest="xz", help="covariance of x and z (cm*cm)", default="0")
parser.add_option("--xphix", dest="xphix", help="covariance of x and phix (cm*rad)", default="0")
parser.add_option("--xphiy", dest="xphiy", help="covariance of x and phiy (cm*rad)", default="0")
parser.add_option("--xphiz", dest="xphiz", help="covariance of x and phiz (cm*rad)", default="0")

parser.add_option("--yy", dest="yy", help="variance of y (cm*cm)", default="1e-15")
parser.add_option("--yz", dest="yz", help="covariance of y and z (cm*cm)", default="0")
parser.add_option("--yphix", dest="yphix", help="covariance of y and phix (cm*rad)", default="0")
parser.add_option("--yphiy", dest="yphiy", help="covariance of y and phiy (cm*rad)", default="0")
parser.add_option("--yphiz", dest="yphiz", help="covariance of y and phiz (cm*rad)", default="0")

parser.add_option("--zz", dest="zz", help="variance of z (cm*cm)", default="1e-15")
parser.add_option("--zphix", dest="zphix", help="covariance of z and phix (cm*rad)", default="0")
parser.add_option("--zphiy", dest="zphiy", help="covariance of z and phiy (cm*rad)", default="0")
parser.add_option("--zphiz", dest="zphiz", help="covariance of z and phiz (cm*rad)", default="0")

parser.add_option("--phixphix", dest="phixphix", help="variance of phix (rad*rad)", default="1e-15")
parser.add_option("--phixphiy", dest="phixphiy", help="covariance of phix and phiy (rad*rad)", default="0")
parser.add_option("--phixphiz", dest="phixphiz", help="covariance of phix and phiz (rad*rad)", default="0")

parser.add_option("--phiyphiy", dest="phiyphiy", help="variance of phiy (rad*rad)", default="1e-15")
parser.add_option("--phiyphiz", dest="phiyphiz", help="covariance of phiy and phiz (rad*rad)", default="0")

parser.add_option("--phizphiz", dest="phizphiz", help="variance of phiz (rad*rad)", default="1e-15")

parser.add_option("-f", dest="force", help="force overwrite of output files", action="store_true")

options, args = parser.parse_args()
if args is None or len(args) != 1:
    parser.print_help()
    exit(-1)
outputName = args[0]

if not options.force:
    if os.path.exists(outputName + ".xml"):
        raise Exception, (outputName + ".xml exists!")
    if os.path.exists(outputName + "_convert_cfg.py"):
        raise Exception, (outputName + "_convert_cfg.py exists!")
    if os.path.exists(outputName + ".db"):
        raise Exception, (outputName + ".db exists!")
    if os.path.exists(outputName + "_correlations.txt"):
        raise Exception, (outputName + "_correlations.txt exists!")

components = "xx", "xy", "xz", "xphix", "xphiy", "xphiz", "yy", "yz", "yphix", "yphiy", "yphiz", "zz", "zphix", "zphiy", "zphiz", "phixphix", "phixphiy", "phixphiz", "phiyphiy", "phiyphiz", "phizphiz"
for component in components:
    exec("%s = float(options.%s)" % (component, component))

### Print out user's choices as diagnostics

print "Spread in each parameter: x %g mm" % (sqrt(xx)*10.)
print "                          y %g mm" % (sqrt(yy)*10.)
print "                          z %g mm" % (sqrt(zz)*10.)
print "                          phix %g mrad" % (sqrt(phixphix)*1000.)
print "                          phiy %g mrad" % (sqrt(phiyphiy)*1000.)
print "                          phiz %g mrad" % (sqrt(phizphiz)*1000.)
print

print "Covariance matrix (x, y, z, phix, phiy, phiz):"
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xx, xy, xz, xphix, xphiy, xphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xy, yy, yz, yphix, yphiy, yphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xz, yz, zz, zphix, zphiy, zphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xphix, yphix, zphix, phixphix, phixphiy, phixphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xphiy, yphiy, zphiy, phixphiy, phiyphiy, phiyphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xphiz, yphiz, zphiz, phixphiz, phiyphiz, phizphiz)
print

print "Correlation (x, y, z, phix, phiy, phiz):"
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xx/sqrt(xx)/sqrt(xx), xy/sqrt(xx)/sqrt(yy), xz/sqrt(xx)/sqrt(zz), xphix/sqrt(xx)/sqrt(phixphix), xphiy/sqrt(xx)/sqrt(phiyphiy), xphiz/sqrt(xx)/sqrt(phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xy/sqrt(yy)/sqrt(xx), yy/sqrt(yy)/sqrt(yy), yz/sqrt(yy)/sqrt(zz), yphix/sqrt(yy)/sqrt(phixphix), yphiy/sqrt(yy)/sqrt(phiyphiy), yphiz/sqrt(yy)/sqrt(phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xz/sqrt(zz)/sqrt(xx), yz/sqrt(zz)/sqrt(yy), zz/sqrt(zz)/sqrt(zz), zphix/sqrt(zz)/sqrt(phixphix), zphiy/sqrt(zz)/sqrt(phiyphiy), zphiz/sqrt(zz)/sqrt(phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xphix/sqrt(phixphix)/sqrt(xx), yphix/sqrt(phixphix)/sqrt(yy), zphix/sqrt(phixphix)/sqrt(zz), phixphix/sqrt(phixphix)/sqrt(phixphix), phixphiy/sqrt(phixphix)/sqrt(phiyphiy), phixphiz/sqrt(phixphix)/sqrt(phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xphiy/sqrt(phiyphiy)/sqrt(xx), yphiy/sqrt(phiyphiy)/sqrt(yy), zphiy/sqrt(phiyphiy)/sqrt(zz), phixphiy/sqrt(phiyphiy)/sqrt(phixphix), phiyphiy/sqrt(phiyphiy)/sqrt(phiyphiy), phiyphiz/sqrt(phiyphiy)/sqrt(phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (xphiz/sqrt(phizphiz)/sqrt(xx), yphiz/sqrt(phizphiz)/sqrt(yy), zphiz/sqrt(phizphiz)/sqrt(zz), phixphiz/sqrt(phizphiz)/sqrt(phixphix), phiyphiz/sqrt(phizphiz)/sqrt(phiyphiy), phizphiz/sqrt(phizphiz)/sqrt(phizphiz))
print

for correlation_coefficient in [abs(xy/sqrt(xx)/sqrt(yy)), abs(xz/sqrt(xx)/sqrt(zz)), abs(xphix/sqrt(xx)/sqrt(phixphix)), abs(xphiy/sqrt(xx)/sqrt(phiyphiy)), abs(xphiz/sqrt(xx)/sqrt(phizphiz)), \
                                abs(yz/sqrt(yy)/sqrt(zz)), abs(yphix/sqrt(yy)/sqrt(phixphix)), abs(yphiy/sqrt(yy)/sqrt(phiyphiy)), abs(yphiz/sqrt(yy)/sqrt(phizphiz)),
                                abs(zphix/sqrt(zz)/sqrt(phixphix)), abs(zphiy/sqrt(zz)/sqrt(phiyphiy)), abs(zphiz/sqrt(zz)/sqrt(phizphiz)),
                                abs(phixphiy/sqrt(phixphix)/sqrt(phiyphiy)), abs(phixphiz/sqrt(phixphix)/sqrt(phizphiz)),
                                abs(phiyphiz/sqrt(phiyphiy)/sqrt(phizphiz))]:
    if correlation_coefficient > 1.:
        raise Exception, "Correlations must not be larger than one!"

### Some useful mathematical transformations (why don't we have access to numpy?)

def mmult(a, b):
    """Matrix multiplication: mmult([[11, 12], [21, 22]], [[-1, 0], [0, 1]]) returns [[-11, 12], [-21, 22]]"""
    return [[sum([i*j for i, j in zip(row, col)]) for col in zip(*b)] for row in a]

def mvdot(m, v):
    """Applies matrix m to vector v: mvdot([[-1, 0], [0, 1]], [12, 55]) returns [-12, 55]"""
    return [i[0] for i in mmult(m, [[vi] for vi in v])]

def mtrans(a):
    """Matrix transposition: mtrans([[11, 12], [21, 22]]) returns [[11, 21], [12, 22]]"""
    return [[a[j][i] for j in range(len(a[i]))] for i in range(len(a))]

def cholesky(A):
    """Cholesky decomposition of the correlation matrix to properly normalize the transformed random deviates"""

    # A = L * D * L^T = (L * D^0.5) * (L * D^0.5)^T where we want (L * D^0.5), the "square root" of A
    # find L and D from A using recurrence relations
    L = {}
    D = {}
    for j in range(len(A[0])):
        D[j] = A[j][j] - sum([L[j,k]**2 * D[k] for k in range(j)])
        for i in range(len(A)):
            if i > j:
                L[i,j] = (A[i][j] - sum([L[i,k] * L[j,k] * D[k] for k in range(j)])) / D[j]

    L = [[    1.,     0.,     0.,     0.,     0., 0.],
         [L[1,0],     1.,     0.,     0.,     0., 0.],
         [L[2,0], L[2,1],     1.,     0.,     0., 0.],
         [L[3,0], L[3,1], L[3,2],     1.,     0., 0.],
         [L[4,0], L[4,1], L[4,2], L[4,1],     1., 0.],
         [L[5,0], L[5,1], L[5,2], L[5,1], L[5,0], 1.]]

    Dsqrt = [[sqrt(D[0]),         0.,          0.,         0.,         0.,         0.],
             [        0., sqrt(D[1]),          0.,         0.,         0.,         0.],
             [        0.,         0., sqrt(D[2]),          0.,         0.,         0.],
             [        0.,         0.,          0., sqrt(D[3]),         0.,         0.],
             [        0.,         0.,          0.,         0., sqrt(D[4]),         0.],
             [        0.,         0.,          0.,         0.,         0., sqrt(D[5])]]

    return mmult(L, Dsqrt)

matrix = [[   xx,    xy,    xz,    xphix,    xphiy,    xphiz],
          [   xy,    yy,    yz,    yphix,    yphiy,    yphiz],
          [   xz,    yz,    zz,    zphix,    zphiy,    zphiz],
          [xphix, yphix, zphix, phixphix, phixphiy, phixphiz],
          [xphiy, yphiy, zphiy, phixphiy, phiyphiy, phiyphiz],
          [xphiz, yphiz, zphiz, phixphiz, phiyphiz, phizphiz]]

chomat = cholesky(matrix)

### Generate correlated random misalignments for all chambers

def random6dof():
    randomunit = [gauss(0., 1.), gauss(0., 1.), gauss(0., 1.), gauss(0., 1.), gauss(0., 1.), gauss(0., 1.)]
    return mvdot(chomat, randomunit)

misal = {}

for wheel in -2, -1, 0, 1, 2:
    for station in 1, 2, 3, 4:
        for sector in range(1, 14+1):
            if station != 4 and sector > 12: continue

            misal["DT", wheel, station, 0, sector] = random6dof()

for endcap in 1, 2:
    for station in 1, 2, 3, 4:
        for ring in 1, 2, 3:
            if station > 1 and ring == 3: continue
            for sector in range(1, 36+1):
                if station > 1 and ring == 1 and sector > 18: continue

                misal["CSC", endcap, station, ring, sector] = random6dof()

### More diagnostics

sum_x = 0.
sum_y = 0.
sum_z = 0.
sum_phix = 0.
sum_phiy = 0.
sum_phiz = 0.

sum_xx = 0.
sum_xy = 0.
sum_xz = 0.
sum_xphix = 0.
sum_xphiy = 0.
sum_xphiz = 0.
sum_yy = 0.
sum_yz = 0.
sum_yphix = 0.
sum_yphiy = 0.
sum_yphiz = 0.
sum_zz = 0.
sum_zphix = 0.
sum_zphiy = 0.
sum_zphiz = 0.
sum_phixphix = 0.
sum_phixphiy = 0.
sum_phixphiz = 0.
sum_phiyphiy = 0.
sum_phiyphiz = 0.
sum_phizphiz = 0.

for xi, yi, zi, phixi, phiyi, phizi in misal.values():
    sum_x += xi
    sum_y += yi
    sum_z += zi
    sum_phix += phixi
    sum_phiy += phiyi
    sum_phiz += phizi
    
    sum_xx += xi*xi
    sum_xy += xi*yi
    sum_xz += xi*zi
    sum_xphix += xi*phixi
    sum_xphiy += xi*phiyi
    sum_xphiz += xi*phizi
    sum_yy += yi*yi
    sum_yz += yi*zi
    sum_yphix += yi*phixi
    sum_yphiy += yi*phiyi
    sum_yphiz += yi*phizi
    sum_zz += zi*zi
    sum_zphix += zi*phixi
    sum_zphiy += zi*phiyi
    sum_zphiz += zi*phizi
    sum_phixphix += phixi*phixi
    sum_phixphiy += phixi*phiyi
    sum_phixphiz += phixi*phizi
    sum_phiyphiy += phiyi*phiyi
    sum_phiyphiz += phiyi*phizi
    sum_phizphiz += phizi*phizi

ave_x = sum_x/float(len(misal))
ave_y = sum_y/float(len(misal))
ave_z = sum_z/float(len(misal))
ave_phix = sum_phix/float(len(misal))
ave_phiy = sum_phiy/float(len(misal))
ave_phiz = sum_phiz/float(len(misal))

ave_xx = sum_xx/float(len(misal))
ave_xy = sum_xy/float(len(misal))
ave_xz = sum_xz/float(len(misal))
ave_xphix = sum_xphix/float(len(misal))
ave_xphiy = sum_xphiy/float(len(misal))
ave_xphiz = sum_xphiz/float(len(misal))
ave_yy = sum_yy/float(len(misal))
ave_yz = sum_yz/float(len(misal))
ave_yphix = sum_yphix/float(len(misal))
ave_yphiy = sum_yphiy/float(len(misal))
ave_yphiz = sum_yphiz/float(len(misal))
ave_zz = sum_zz/float(len(misal))
ave_zphix = sum_zphix/float(len(misal))
ave_zphiy = sum_zphiy/float(len(misal))
ave_zphiz = sum_zphiz/float(len(misal))
ave_phixphix = sum_phixphix/float(len(misal))
ave_phixphiy = sum_phixphiy/float(len(misal))
ave_phixphiz = sum_phixphiz/float(len(misal))
ave_phiyphiy = sum_phiyphiy/float(len(misal))
ave_phiyphiz = sum_phiyphiz/float(len(misal))
ave_phizphiz = sum_phizphiz/float(len(misal))

print "Estimated covariance matrix from %d chambers (x, y, z, phix, phiy, phiz):" % len(misal)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xx, ave_xy, ave_xz, ave_xphix, ave_xphiy, ave_xphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xy, ave_yy, ave_yz, ave_yphix, ave_yphiy, ave_yphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xz, ave_yz, ave_zz, ave_zphix, ave_zphiy, ave_zphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xphix, ave_yphix, ave_zphix, ave_phixphix, ave_phixphiy, ave_phixphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xphiy, ave_yphiy, ave_zphiy, ave_phixphiy, ave_phiyphiy, ave_phiyphiz)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xphiz, ave_yphiz, ave_zphiz, ave_phixphiz, ave_phiyphiz, ave_phizphiz)
print

print "Estimated correlation from %d chambers (x, y, z, phix, phiy, phiz):" % len(misal)
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xx/sqrt(ave_xx)/sqrt(ave_xx), ave_xy/sqrt(ave_xx)/sqrt(ave_yy), ave_xz/sqrt(ave_xx)/sqrt(ave_zz), ave_xphix/sqrt(ave_xx)/sqrt(ave_phixphix), ave_xphiy/sqrt(ave_xx)/sqrt(ave_phiyphiy), ave_xphiz/sqrt(ave_xx)/sqrt(ave_phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xy/sqrt(ave_yy)/sqrt(ave_xx), ave_yy/sqrt(ave_yy)/sqrt(ave_yy), ave_yz/sqrt(ave_yy)/sqrt(ave_zz), ave_yphix/sqrt(ave_yy)/sqrt(ave_phixphix), ave_yphiy/sqrt(ave_yy)/sqrt(ave_phiyphiy), ave_yphiz/sqrt(ave_yy)/sqrt(ave_phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xz/sqrt(ave_zz)/sqrt(ave_xx), ave_yz/sqrt(ave_zz)/sqrt(ave_yy), ave_zz/sqrt(ave_zz)/sqrt(ave_zz), ave_zphix/sqrt(ave_zz)/sqrt(ave_phixphix), ave_zphiy/sqrt(ave_zz)/sqrt(ave_phiyphiy), ave_zphiz/sqrt(ave_zz)/sqrt(ave_phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xphix/sqrt(ave_phixphix)/sqrt(ave_xx), ave_yphix/sqrt(ave_phixphix)/sqrt(ave_yy), ave_zphix/sqrt(ave_phixphix)/sqrt(ave_zz), ave_phixphix/sqrt(ave_phixphix)/sqrt(ave_phixphix), ave_phixphiy/sqrt(ave_phixphix)/sqrt(ave_phiyphiy), ave_phixphiz/sqrt(ave_phixphix)/sqrt(ave_phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xphiy/sqrt(ave_phiyphiy)/sqrt(ave_xx), ave_yphiy/sqrt(ave_phiyphiy)/sqrt(ave_yy), ave_zphiy/sqrt(ave_phiyphiy)/sqrt(ave_zz), ave_phixphiy/sqrt(ave_phiyphiy)/sqrt(ave_phixphix), ave_phiyphiy/sqrt(ave_phiyphiy)/sqrt(ave_phiyphiy), ave_phiyphiz/sqrt(ave_phiyphiy)/sqrt(ave_phizphiz))
print "%11.8f %11.8f %11.8f %11.8f %11.8f %11.8f" % (ave_xphiz/sqrt(ave_phizphiz)/sqrt(ave_xx), ave_yphiz/sqrt(ave_phizphiz)/sqrt(ave_yy), ave_zphiz/sqrt(ave_phizphiz)/sqrt(ave_zz), ave_phixphiz/sqrt(ave_phizphiz)/sqrt(ave_phixphix), ave_phiyphiz/sqrt(ave_phizphiz)/sqrt(ave_phiyphiy), ave_phizphiz/sqrt(ave_phizphiz)/sqrt(ave_phizphiz))
print

### Delete all three files at once to make sure the user never sees
### stale data (e.g. from a stopped process due to failed conversion)

if os.path.exists(outputName + ".xml"):
    os.unlink(outputName + ".xml")
if os.path.exists(outputName + "_convert_cfg.py"):
    os.unlink(outputName + "_convert_cfg.py")
if os.path.exists(outputName + ".db"):
    os.unlink(outputName + ".db")
if os.path.exists(outputName + "_correlations.txt"):
    os.unlink(outputName + "_correlations.txt")

### Print out the list of correlations

txtfile = file(outputName + "_correlations.txt", "w")
for wheel in -2, -1, 0, 1, 2:
    for station in 1, 2, 3, 4:
        for sector in range(1, 14+1):
            if station != 4 and sector > 12: continue
            txtfile.write("DT %(wheel)d %(station)d %(sector)d %(xx)g %(xy)g %(xz)g %(xphix)g %(xphiy)g %(xphiz)g %(yy)g %(yz)g %(yphix)g %(yphiy)g %(yphiz)g %(zz)g %(zphix)g %(zphiy)g %(zphiz)g %(phixphix)g %(phixphiy)g %(phixphiz)g %(phiyphiy)g %(phiyphiz)g %(phizphiz)g\n" % vars())

for endcap in 1, 2:
    for station in 1, 2, 3, 4:
        for ring in 1, 2, 3:
            if station > 1 and ring == 3: continue
            for sector in range(1, 36+1):
                if station > 1 and ring == 1 and sector > 18: continue
                txtfile.write("CSC %(endcap)d %(station)d %(ring)d %(sector)d %(xx)g %(xy)g %(xz)g %(xphix)g %(xphiy)g %(xphiz)g %(yy)g %(yz)g %(yphix)g %(yphiy)g %(yphiz)g %(zz)g %(zphix)g %(zphiy)g %(zphiz)g %(phixphix)g %(phixphiy)g %(phixphiz)g %(phiyphiy)g %(phiyphiz)g %(phizphiz)g\n" % vars())

### Make an XML representation of the misalignment

xmlfile = file(outputName + ".xml", "w")
xmlfile.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
xmlfile.write("<?xml-stylesheet type=\"text/xml\" href=\"MuonAlignment.xsl\"?>\n")
xmlfile.write("<MuonAlignment>\n\n")

for (system, whendcap, station, ring, sector), (xi, yi, zi, phixi, phiyi, phizi) in misal.items():
    if system == "DT": wheel = whendcap
    if system == "CSC": endcap = whendcap

    rot = rotation[system, whendcap, station, ring, sector]
    localape = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    globalape = mmult(rot, mmult(localape, mtrans(rot)))
    globalxx = globalape[0][0]
    globalxy = globalape[0][1]
    globalxz = globalape[0][2]
    globalyy = globalape[1][1]
    globalyz = globalape[1][2]
    globalzz = globalape[2][2]

    xmlfile.write("<operation>\n")

    if system == "DT":
        xmlfile.write("    <DTChamber wheel=\"%(wheel)d\" station=\"%(station)d\" sector=\"%(sector)d\" />\n" % vars())
    if system == "CSC":
        xmlfile.write("    <CSCChamber endcap=\"%(endcap)d\" station=\"%(station)d\" ring=\"%(ring)d\" chamber=\"%(sector)d\" />\n" % vars())

        # ME1/1a is called "ring 4", but it should always get exactly the same alignment constants as the corresponding ME1/1b ("ring 1")
        if (station, ring) == (1, 1):
            xmlfile.write("    <CSCChamber endcap=\"%(endcap)d\" station=\"%(station)d\" ring=\"4\" chamber=\"%(sector)d\" />\n" % vars())

    xmlfile.write("    <setposition relativeto=\"ideal\" x=\"%(xi)g\" y=\"%(yi)g\" z=\"%(zi)g\" phix=\"%(phixi)g\" phiy=\"%(phiyi)g\" phiz=\"%(phizi)g\" />\n" % vars())
    xmlfile.write("    <setape xx=\"%(globalxx)g\" xy=\"%(globalxy)g\" xz=\"%(globalxz)g\" yy=\"%(globalyy)g\" yz=\"%(globalyz)g\" zz=\"%(globalzz)g\" />\n" % vars())
    xmlfile.write("</operation>\n\n")

xmlfile.write("</MuonAlignment>\n")
xmlfile.close()

### Convert it to an SQLite file with CMSSW

cfgfile = file(outputName + "_convert_cfg.py", "w")

cfgfile.write("""import FWCore.ParameterSet.Config as cms

process = cms.Process("CONVERT")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.MuonGeometryDBConverter = cms.EDAnalyzer("MuonGeometryDBConverter",
                                                 input = cms.string("xml"),
                                                 fileName = cms.string("%(outputName)s.xml"),
                                                 shiftErr = cms.double(1000.),
                                                 angleErr = cms.double(6.28),

                                                 output = cms.string("db")
                                                 )

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:%(outputName)s.db"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                            cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))

process.Path = cms.Path(process.MuonGeometryDBConverter)
""" % vars())

print "To create an SQLite file for this geometry (%(outputName)s.db), run the following:" % vars()
print
os.system("echo cmsRun %s_convert_cfg.py" % outputName)

