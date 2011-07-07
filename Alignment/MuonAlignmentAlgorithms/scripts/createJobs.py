#! /usr/bin/env python

import os, sys, optparse, math

copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""
    if copyargs[i].find(" ") != -1:
        copyargs[i] = "\"%s\"" % copyargs[i]
commandline = " ".join(copyargs)

usage = """./%prog DIRNAME ITERATIONS INITIALGEOM INPUTFILES [options]

Creates (overwrites) a directory for each of the iterations and creates (overwrites)
submitJobs.sh with the submission sequence and dependencies.

DIRNAME        directories will be named DIRNAME01, DIRNAME02, etc.
ITERATIONS     number of iterations
INITIALGEOM    SQLite file containing muon geometry with tag names
               DTAlignmentRcd, DTAlignmentErrorRcd, CSCAlignmentRcd, CSCAlignmentErrorRcd
INPUTFILES     Python file defining 'fileNames', a list of input files as
               strings (create with findQualityFiles.py)"""

parser = optparse.OptionParser(usage)
parser.add_option("-j", "--jobs",
                   help="approximate number of \"gather\" subjobs",
                   type="int",
                   default=50,
                   dest="subjobs")
parser.add_option("-s", "--submitJobs",
                   help="alternate name of submitJobs.sh script (please include .sh extension); a file with this name will be OVERWRITTEN",
                   type="string",
                   default="submitJobs.sh",
                   dest="submitJobs")
parser.add_option("-b", "--big",
                  help="if invoked, subjobs will also be run on cmscaf1nd",
                  action="store_true",
                  dest="big")
parser.add_option("--mapplots",
                  help="if invoked, draw \"map plots\"",
                  action="store_true",
                  dest="mapplots")
parser.add_option("--segdiffplots",
                  help="if invoked, draw \"segment-difference plots\"",
                  action="store_true",
                  dest="segdiffplots")
parser.add_option("--curvatureplots",
                  help="if invoked, draw \"curvature plots\"",
                  action="store_true",
                  dest="curvatureplots")
parser.add_option("--globalTag",
                  help="GlobalTag for alignment/calibration conditions (typically all conditions except muon and tracker alignment)",
                  type="string",
                  default="CRAFT0831X_V1::All",
                  dest="globaltag")
parser.add_option("--trackerconnect",
                  help="connect string for tracker alignment (frontier://... or sqlite_file:...)",
                  type="string",
                  default="",
                  dest="trackerconnect")
parser.add_option("--trackeralignment",
                  help="name of TrackerAlignmentRcd tag",
                  type="string",
                  default="Alignments",
                  dest="trackeralignment")
parser.add_option("--trackerAPEconnect",
                  help="connect string for tracker APEs (frontier://... or sqlite_file:...)",
                  type="string",
                  default="",
                  dest="trackerAPEconnect")
parser.add_option("--trackerAPE",
                  help="name of TrackerAlignmentErrorRcd tag (tracker APEs)",
                  type="string",
                  default="AlignmentErrors",
                  dest="trackerAPE")
parser.add_option("--gprcdconnect",
                  help="connect string for GlobalPositionRcd (frontier://... or sqlite_file:...)",
                  type="string",
                  default="",
                  dest="gprcdconnect")
parser.add_option("--gprcd",
                  help="name of GlobalPositionRcd tag",
                  type="string",
                  default="SurveyGeometry",
                  dest="gprcd")
parser.add_option("--iscosmics",
                  help="if invoked, use cosmic track refitter instead of the standard one",
                  action="store_true",
                  dest="iscosmics")
parser.add_option("--station123params",
                  help="alignable parameters for DT stations 1, 2, 3 (see SWGuideAlignmentAlgorithms#Selection_of_what_to_align)",
                  type="string",
                  default="111111",
                  dest="station123params")
parser.add_option("--station4params",
                  help="alignable parameters for DT station 4",
                  type="string",
                  default="100011",
                  dest="station4params")
parser.add_option("--cscparams",
                  help="alignable parameters for CSC chambers",
                  type="string",
                  default="100011",
                  dest="cscparams")
parser.add_option("--minTrackPt",
                  help="minimum allowed track transverse momentum (in GeV)",
                  type="string",
                  default="100",
                  dest="minTrackPt")
parser.add_option("--maxTrackPt",
                  help="maximum allowed track transverse momentum (in GeV)",
                  type="string",
                  default="200",
                  dest="maxTrackPt")
parser.add_option("--minTrackerHits",
                  help="minimum number of tracker hits",
                  type="int",
                  default=15,
                  dest="minTrackerHits")
parser.add_option("--maxTrackerRedChi2",
                  help="maximum tracker chi^2 per degrees of freedom",
                  type="string",
                  default="10",
                  dest="maxTrackerRedChi2")
parser.add_option("--allowTIDTEC",
                  help="if invoked, allow tracks that pass through the tracker's !TID/!TEC region (recommended)",
                  action="store_true",
                  dest="allowTIDTEC")
parser.add_option("--twoBin",
                  help="if invoked, apply the \"two-bin method\" to control charge-antisymmetric errors",
                  action="store_true",
                  dest="twoBin")
parser.add_option("--weightAlignment",
                  help="if invoked, segments will be weighted by ndf/chi^2 in the alignment",
                  action="store_true",
                  dest="weightAlignment")
parser.add_option("--minAlignmentSegments",
                  help="minimum number of segments required to align a chamber",
                  type="int",
                  default=5,
                  dest="minAlignmentHits")
parser.add_option("--combineME11",
                  help="treat ME1/1a and ME1/1b as the same objects",
                  action="store_true",
                  dest="combineME11")
parser.add_option("--maxEvents",
                  help="maximum number of events",
                  type="string",
                  default="-1",
                  dest="maxEvents")
parser.add_option("--skipEvents",
                  help="number of events to be skipped",
                  type="string",
                  default="0",
                  dest="skipEvents")
parser.add_option("--validationLabel",
                  help="if given nonempty string RUNLABEL, diagnostics and creation of plots will be run in the end of the last iteration; the RUNLABEL will be used to mark a run; the results will be put into a RUNLABEL_DATESTAMP.tgz tarball",
                  type="string",
                  default="",
                  dest="validationLabel")
parser.add_option("--maxResSlopeY",
                  help="maximum residual slope y component",
                  type="string",
                  default="10",
                  dest="maxResSlopeY")
parser.add_option("--motionPolicyNSigma",
                  help="minimum nsigma(deltax) position displacement in order to move a chamber for the final alignment result; default NSIGMA=3",
                  type="int",
                  default=3,
                  dest="motionPolicyNSigma")

if len(sys.argv) < 5:
    raise SystemError, "Too few arguments.\n\n"+parser.format_help()

DIRNAME = sys.argv[1]
ITERATIONS = int(sys.argv[2])
INITIALGEOM = sys.argv[3]
INPUTFILES = sys.argv[4]

options, args = parser.parse_args(sys.argv[5:])
mapplots_ingeneral = options.mapplots
segdiffplots_ingeneral = options.segdiffplots
curvatureplots_ingeneral = options.curvatureplots
globaltag = options.globaltag
trackerconnect = options.trackerconnect
trackeralignment = options.trackeralignment
trackerAPEconnect = options.trackerAPEconnect
trackerAPE = options.trackerAPE
gprcdconnect = options.gprcdconnect
gprcd = options.gprcd
iscosmics = str(options.iscosmics)
station123params = options.station123params
station4params = options.station4params
cscparams = options.cscparams
minTrackPt = options.minTrackPt
maxTrackPt = options.maxTrackPt
minTrackerHits = str(options.minTrackerHits)
maxTrackerRedChi2 = options.maxTrackerRedChi2
allowTIDTEC = str(options.allowTIDTEC)
twoBin = str(options.twoBin)
weightAlignment = str(options.weightAlignment)
minAlignmentHits = str(options.minAlignmentHits)
combineME11 = str(options.combineME11)
maxEvents = options.maxEvents
skipEvents = options.skipEvents
validationLabel = options.validationLabel
maxResSlopeY = options.maxResSlopeY
theNSigma = options.motionPolicyNSigma

execfile(INPUTFILES)
stepsize = int(math.ceil(1.*len(fileNames)/options.subjobs))
pwd = str(os.getcwdu())

bsubfile = ["#!/bin/sh", ""]
bsubnames = []
last_align = None

#####################################################################
# step 0: convert initial geometry to xml
INITIALXML = INITIALGEOM + '.xml'
if INITIALGEOM[-3:]=='.db':
  INITIALXML = INITIALGEOM[:-3] + '.xml'
print "Converting",INITIALGEOM,"to",INITIALXML," ...will be done in several seconds..."
exit_code = os.system("./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py  %s %s" % (INITIALGEOM,INITIALXML))
if exit_code>0:
  print "problem: conversion exited with code:", exit_code
  sys.exit()

#####################################################################

directory = ""
for iteration in range(1, ITERATIONS+1):
    if iteration == 1:
        inputdb = INITIALGEOM
        inputdbdir = directory[:]
    else:
        inputdb = director + ".db"
        inputdbdir = directory[:]

    directory = "%s%02d/" % (DIRNAME, iteration)
    director = directory[:-1]
    os.system("rm -rf %s; mkdir %s" % (directory, directory))
    os.system("cp gather_cfg.py %s" % directory)
    os.system("cp align_cfg.py %s" % directory)

    bsubfile.append("cd %s" % directory)

    mapplots = False
    if mapplots_ingeneral and (iteration == 1 or iteration == ITERATIONS): mapplots = True
    segdiffplots = False
    if segdiffplots_ingeneral and (iteration == 1 or iteration == ITERATIONS): segdiffplots = True
    curvatureplots = False
    if curvatureplots_ingeneral and (iteration == 1 or iteration == ITERATIONS): curvatureplots = True

    for jobnumber in range(options.subjobs):
        gather_fileName = "%sgather%03d.sh" % (directory, jobnumber)
        inputfiles = " ".join(fileNames[jobnumber*stepsize:(jobnumber+1)*stepsize])

        if mapplots or segdiffplots or curvatureplots: copyplots = "plotting*.root"
        else: copyplots = ""

        copytrackerdb = ""
        if trackerconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerconnect[12:]
        if trackerAPEconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerAPEconnect[12:]
        if gprcdconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % gprcdconnect[12:]

        if len(inputfiles) > 0:
            file(gather_fileName, "w").write("""#/bin/sh
# %(commandline)s

export ALIGNMENT_CAFDIR=`pwd`

cd %(pwd)s
eval `scramv1 run -sh`
export ALIGNMENT_AFSDIR=`pwd`

export ALIGNMENT_INPUTFILES='%(inputfiles)s'
export ALIGNMENT_ITERATION=%(iteration)d
export ALIGNMENT_JOBNUMBER=%(jobnumber)d
export ALIGNMENT_MAPPLOTS=%(mapplots)s
export ALIGNMENT_SEGDIFFPLOTS=%(segdiffplots)s
export ALIGNMENT_CURVATUREPLOTS=%(curvatureplots)s
export ALIGNMENT_GLOBALTAG=%(globaltag)s
export ALIGNMENT_INPUTDB=%(inputdb)s
export ALIGNMENT_TRACKERCONNECT=%(trackerconnect)s
export ALIGNMENT_TRACKERALIGNMENT=%(trackeralignment)s
export ALIGNMENT_TRACKERAPECONNECT=%(trackerAPEconnect)s
export ALIGNMENT_TRACKERAPE=%(trackerAPE)s
export ALIGNMENT_GPRCDCONNECT=%(gprcdconnect)s
export ALIGNMENT_GPRCD=%(gprcd)s
export ALIGNMENT_ISCOSMICS=%(iscosmics)s
export ALIGNMENT_STATION123PARAMS=%(station123params)s
export ALIGNMENT_STATION4PARAMS=%(station4params)s
export ALIGNMENT_CSCPARAMS=%(cscparams)s
export ALIGNMENT_MINTRACKPT=%(minTrackPt)s
export ALIGNMENT_MAXTRACKPT=%(maxTrackPt)s
export ALIGNMENT_MINTRACKERHITS=%(minTrackerHits)s
export ALIGNMENT_MAXTRACKERREDCHI2=%(maxTrackerRedChi2)s
export ALIGNMENT_ALLOWTIDTEC=%(allowTIDTEC)s
export ALIGNMENT_TWOBIN=%(twoBin)s
export ALIGNMENT_WEIGHTALIGNMENT=%(weightAlignment)s
export ALIGNMENT_MINALIGNMENTHITS=%(minAlignmentHits)s
export ALIGNMENT_COMBINEME11=%(combineME11)s
export ALIGNMENT_MAXEVENTS=%(maxEvents)s
export ALIGNMENT_SKIPEVENTS=%(skipEvents)s
export ALIGNMENT_MAXRESSLOPEY=%(maxResSlopeY)s

cp -f %(directory)sgather_cfg.py %(inputdbdir)s%(inputdb)s %(copytrackerdb)s $ALIGNMENT_CAFDIR/
cd $ALIGNMENT_CAFDIR/
ls -l
cmsRun gather_cfg.py
ls -l
cp -f *.tmp %(copyplots)s $ALIGNMENT_AFSDIR/%(directory)s
""" % vars())
            os.system("chmod +x %s" % gather_fileName)
            bsubfile.append("echo %sgather%03d.sh" % (directory, jobnumber))

            if last_align is None: waiter = ""
            else: waiter = "-w \"ended(%s)\"" % last_align            
            if options.big: queue = "cmscaf1nd"
            else: queue = "cmscaf1nh"

            bsubfile.append("bsub -R \"type==SLC5_64\" -q %s -J \"%s_gather%03d\" %s gather%03d.sh" % (queue, director, jobnumber, waiter, jobnumber))

            bsubnames.append("ended(%s_gather%03d)" % (director, jobnumber))

    file("%sconvert-db-to-xml_cfg.py" % directory, "w").write("""from Alignment.MuonAlignment.convertSQLitetoXML_cfg import *
process.PoolDBESSource.connect = \"sqlite_file:%(directory)s%(director)s.db\"
process.MuonGeometryDBConverter.outputXML.fileName = \"%(directory)s%(director)s.xml\"
process.MuonGeometryDBConverter.outputXML.relativeto = \"ideal\"
process.MuonGeometryDBConverter.outputXML.suppressDTChambers = False
process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = True
process.MuonGeometryDBConverter.outputXML.suppressDTLayers = True
process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = False
process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = True

process.MuonGeometryDBConverter.getAPEs = True
process.PoolDBESSource.toGet = cms.VPSet(
    cms.PSet(record = cms.string(\"DTAlignmentRcd\"), tag = cms.string(\"DTAlignmentRcd\")),
    cms.PSet(record = cms.string(\"DTAlignmentErrorRcd\"), tag = cms.string(\"DTAlignmentErrorRcd\")),
    cms.PSet(record = cms.string(\"CSCAlignmentRcd\"), tag = cms.string(\"CSCAlignmentRcd\")),
    cms.PSet(record = cms.string(\"CSCAlignmentErrorRcd\"), tag = cms.string(\"CSCAlignmentErrorRcd\")),
      )
""" % vars())

    copytrackerdb = ""
    if trackerconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerconnect[12:]
    if trackerAPEconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerAPEconnect[12:]
    if gprcdconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % gprcdconnect[12:]

    file("%salign.sh" % directory, "w").write("""#!/bin/sh
# %(commandline)s

export ALIGNMENT_CAFDIR=`pwd`

cd %(pwd)s
eval `scramv1 run -sh`
export ALIGNMENT_AFSDIR=`pwd`
export ALIGNMENT_INPUTDB=%(inputdb)s
export ALIGNMENT_ITERATION=%(iteration)d
export ALIGNMENT_GLOBALTAG=%(globaltag)s
export ALIGNMENT_TRACKERCONNECT=%(trackerconnect)s
export ALIGNMENT_TRACKERALIGNMENT=%(trackeralignment)s
export ALIGNMENT_TRACKERAPECONNECT=%(trackerAPEconnect)s
export ALIGNMENT_TRACKERAPE=%(trackerAPE)s
export ALIGNMENT_GPRCDCONNECT=%(gprcdconnect)s
export ALIGNMENT_GPRCD=%(gprcd)s
export ALIGNMENT_ISCOSMICS=%(iscosmics)s
export ALIGNMENT_STATION123PARAMS=%(station123params)s
export ALIGNMENT_STATION4PARAMS=%(station4params)s
export ALIGNMENT_CSCPARAMS=%(cscparams)s
export ALIGNMENT_MINTRACKPT=%(minTrackPt)s
export ALIGNMENT_MAXTRACKPT=%(maxTrackPt)s
export ALIGNMENT_MINTRACKERHITS=%(minTrackerHits)s
export ALIGNMENT_MAXTRACKERREDCHI2=%(maxTrackerRedChi2)s
export ALIGNMENT_ALLOWTIDTEC=%(allowTIDTEC)s
export ALIGNMENT_TWOBIN=%(twoBin)s
export ALIGNMENT_WEIGHTALIGNMENT=%(weightAlignment)s
export ALIGNMENT_MINALIGNMENTHITS=%(minAlignmentHits)s
export ALIGNMENT_COMBINEME11=%(combineME11)s
export ALIGNMENT_MAXRESSLOPEY=%(maxResSlopeY)s

cp -f %(directory)salign_cfg.py %(directory)sconvert-db-to-xml_cfg.py %(inputdbdir)s%(inputdb)s %(directory)s*.tmp  %(copytrackerdb)s $ALIGNMENT_CAFDIR/
cd $ALIGNMENT_CAFDIR/
export ALIGNMENT_ALIGNMENTTMP=`ls alignment*.tmp`

ls -l
cmsRun align_cfg.py
cp -f MuonAlignmentFromReference_report.py $ALIGNMENT_AFSDIR/%(directory)s%(director)s_report.py
cp -f MuonAlignmentFromReference_outputdb.db $ALIGNMENT_AFSDIR/%(directory)s%(director)s.db
cp -f MuonAlignmentFromReference_plotting.root $ALIGNMENT_AFSDIR/%(directory)s%(director)s.root

cd $ALIGNMENT_AFSDIR
cmsRun %(directory)sconvert-db-to-xml_cfg.py

# if it's 1st or last iteration, combine _plotting.root files into one:
if [ \"$ALIGNMENT_ITERATION\" == \"1\" ] || [ \"$ALIGNMENT_ITERATION\" == \"%(ITERATIONS)s\" ]; then
  nfiles=$(ls %(directory)splotting0*.root 2> /dev/null | wc -l)
  if [ \"$nfiles\" != \"0\" ]; then
    hadd -f1 %(directory)s%(director)s_plotting.root %(directory)splotting0*.root
    #if [ $? == 0 ]; then rm %(directory)splotting0*.root; fi
  fi
fi

# if it's last iteration, apply chamber motion policy
if [ \"$ALIGNMENT_ITERATION\" == \"%(ITERATIONS)s\" ]; then
  # convert this iteration's geometry into detailed xml
  ./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py %(directory)s%(director)s.db %(directory)s%(director)s_extra.xml
  # perform motion policy 
  ./Alignment/MuonAlignmentAlgorithms/scripts/motionPolicyChamber.py \
      %(INITIALXML)s  %(directory)s%(director)s_extra.xml \
      %(directory)s%(director)s_report.py \
      %(directory)s%(director)s_final.xml \
      --nsigma %(theNSigma)s
  # convert the resulting xml into the final sqlite geometry
  ./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py %(directory)s%(director)s_final.xml %(directory)s%(director)s_final.db
fi

""" % vars())
    os.system("chmod +x %salign.sh" % directory)

    bsubfile.append("echo %salign.sh" % directory)
    bsubfile.append("bsub -R \"type==SLC5_64\" -q cmscaf1nd -J \"%s_align\" -w \"%s\" align.sh" % (director, " && ".join(bsubnames)))
    bsubfile.append("cd ..")
    bsubnames = []
    last_align = "%s_align" % director
    
    # after the last iteration (optionally) do diagnostics run
    if len(validationLabel) and iteration == ITERATIONS:
        # do we have plotting files created?
        directory1 = "%s01/" % DIRNAME
        director1 = directory1[:-1]

        file("%svalidation.sh" % directory, "w").write("""#!/bin/sh
# %(commandline)s

export ALIGNMENT_CAFDIR=`pwd`
mkdir files
mkdir out

cd %(pwd)s
eval `scramv1 run -sh`
ALIGNMENT_AFSDIR=`pwd`
ALIGNMENT_ITERATION=%(iteration)d
ALIGNMENT_MAPPLOTS=None
ALIGNMENT_SEGDIFFPLOTS=None
ALIGNMENT_CURVATUREPLOTS=None

# copy the scripts to CAFDIR
cd Alignment/MuonAlignmentAlgorithms/scripts/
cp -f plotscripts.py $ALIGNMENT_CAFDIR/
cp -f mutypes.py $ALIGNMENT_CAFDIR/
cp -f alignmentValidation.py $ALIGNMENT_CAFDIR/
cp -f phiedges_fitfunctions.C $ALIGNMENT_CAFDIR/
cp -f createTree.py $ALIGNMENT_CAFDIR/
cd -
cp Alignment/MuonAlignmentAlgorithms/test/browser/tree* $ALIGNMENT_CAFDIR/out/

# copy the results to CAFDIR
cp -f %(directory1)s%(director1)s_report.py $ALIGNMENT_CAFDIR/files/
cp -f %(directory)s%(director)s_report.py $ALIGNMENT_CAFDIR/files/
cp -f %(directory1)s%(director1)s.root $ALIGNMENT_CAFDIR/files/
cp -f %(directory)s%(director)s.root $ALIGNMENT_CAFDIR/files/
if [ -e %(directory1)s%(director1)s_plotting.root ] && [ -e %(directory)s%(director)s_plotting.root ]; then
  cp -f %(directory1)s%(director1)s_plotting.root $ALIGNMENT_CAFDIR/files/
  cp -f %(directory)s%(director)s_plotting.root $ALIGNMENT_CAFDIR/files/
  ALIGNMENT_MAPPLOTS=%(mapplots)s
  ALIGNMENT_SEGDIFFPLOTS=%(segdiffplots)s
  ALIGNMENT_CURVATUREPLOTS=%(curvatureplots)s
fi

cd $ALIGNMENT_CAFDIR/
echo \" ### Start running ###\"
date

# do fits and median plots first 
./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  --createDirSructure --dt --csc --fit --median

if [ $ALIGNMENT_MAPPLOTS == \"True\" ]; then
  ./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  --dt --csc --map
fi

if [ $ALIGNMENT_SEGDIFFPLOTS == \"True\" ]; then
  ./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  --dt --csc --segdiff
fi                   

if [ $ALIGNMENT_CURVATUREPLOTS == \"True\" ]; then
  ./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  --dt --csc --curvature
fi

# run simple diagnostic
./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out --dt --csc --diagnostic

# fill the tree browser structure: 
./createTree.py -i $ALIGNMENT_CAFDIR/out

timestamp=`date \"+%%y-%%m-%%d %%H:%%M:%%S\"`
echo \"%(validationLabel)s.plots (${timestamp})\" > out/label.txt

ls -l out/
timestamp=`date +%%Y%%m%%d%%H%%M%%S`
tar czf %(validationLabel)s_${timestamp}.tgz out
cp -f %(validationLabel)s_${timestamp}.tgz $ALIGNMENT_AFSDIR/

""" % vars())
        os.system("chmod +x %svalidation.sh" % directory)
        
        bsubfile.append("echo %svalidation.sh" % directory)
        bsubfile.append("bsub -R \"type==SLC5_64\" -q cmscaf1nd -J \"%s_validation\" -w \"ended(%s)\" validation.sh" % (director, last_align))
        bsubfile.append("cd ..")
        
    bsubfile.append("")


file(options.submitJobs, "w").write("\n".join(bsubfile))
os.system("chmod +x %s" % options.submitJobs)

