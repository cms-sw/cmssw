#! /usr/bin/env python

from __future__ import print_function
import os, sys, optparse, math

copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""
    if copyargs[i].find(" ") != -1:
        copyargs[i] = "\"%s\"" % copyargs[i]
commandline = " ".join(copyargs)

prog = sys.argv[0]

usage = """./%(prog)s DIRNAME ITERATIONS INITIALGEOM INPUTFILES [options]

Creates (overwrites) a directory for each of the iterations and creates (overwrites)
submitJobs.sh with the submission sequence and dependencies.

DIRNAME        directories will be named DIRNAME01, DIRNAME02, etc.
ITERATIONS     number of iterations
INITIALGEOM    SQLite file containing muon geometry with tag names
               DTAlignmentRcd, DTAlignmentErrorExtendedRcd, CSCAlignmentRcd, CSCAlignmentErrorExtendedRcd
INPUTFILES     Python file defining 'fileNames', a list of input files as
               strings (create with findQualityFiles.py)""" % vars()

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
parser.add_option("-u", "--user_mail",
                  help="if invoked, send mail to a specified email destination. If \"-u\" is not present, the default destination LSB_MAILTO in lsf.conf will be used",
                  type="string",
                  dest="user_mail")
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
                  help="connect string for tracker alignment (frontier://FrontierProd/CMS_COND_310X_ALIGN or sqlite_file:...)",
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
                  help="name of TrackerAlignmentErrorExtendedRcd tag (tracker APEs)",
                  type="string",
                  default="AlignmentErrorsExtended",
                  dest="trackerAPE")
parser.add_option("--trackerBowsconnect",
                  help="connect string for tracker Surface Deformations (frontier://... or sqlite_file:...)",
                  type="string",
                  default="",
                  dest="trackerBowsconnect")
parser.add_option("--trackerBows",
                  help="name of TrackerSurfaceDeformationRcd tag",
                  type="string",
                  default="TrackerSurfaceDeformations",
                  dest="trackerBows")
parser.add_option("--gprcdconnect",
                  help="connect string for GlobalPositionRcd (frontier://... or sqlite_file:...)",
                  type="string",
                  default="",
                  dest="gprcdconnect")
parser.add_option("--gprcd",
                  help="name of GlobalPositionRcd tag",
                  type="string",
                  default="GlobalPosition",
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
                  default="0",
                  dest="minTrackPt")
parser.add_option("--maxTrackPt",
                  help="maximum allowed track transverse momentum (in GeV)",
                  type="string",
                  default="1000",
                  dest="maxTrackPt")
parser.add_option("--minTrackP",
                  help="minimum allowed track momentum (in GeV)",
                  type="string",
                  default="0",
                  dest="minTrackP")
parser.add_option("--maxTrackP",
                  help="maximum allowed track momentum (in GeV)",
                  type="string",
                  default="10000",
                  dest="maxTrackP")
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
parser.add_option("--notAllowTIDTEC",
                  help="if invoked, do not allow tracks that pass through the tracker's TID||TEC region (not recommended)",
                  action="store_true",
                  dest="notAllowTIDTEC")
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
parser.add_option("--notCombineME11",
                  help="if invoced, treat ME1/1a and ME1/1b as separate objects",
                  action="store_true",
                  dest="notCombineME11")
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
parser.add_option("--noCleanUp",
                  help="if invoked, temporary plotting???.root and *.tmp files would not be removed at the end of each align job",
                  action="store_true",
                  dest="noCleanUp")
parser.add_option("--noCSC",
                  help="if invoked, CSC endcap chambers would not be processed",
                  action="store_true",
                  dest="noCSC")
parser.add_option("--noDT",
                  help="if invoked, DT barrel chambers would not be processed",
                  action="store_true",
                  dest="noDT")
parser.add_option("--createMapNtuple",
                  help="if invoked while mapplots are switched on, a special ntuple would be created",
                  action="store_true",
                  dest="createMapNtuple")
parser.add_option("--inputInBlocks",
                  help="if invoked, assume that INPUTFILES provides a list of files already groupped into job blocks, -j has no effect in that case",
                  action="store_true",
                  dest="inputInBlocks")
parser.add_option("--json",
                  help="If present with JSON file as argument, use JSON file for good lumi mask. "+\
                  "The latest JSON file is available at /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/Prompt/",
                  type="string",
                  default="",
                  dest="json")
parser.add_option("--createAlignNtuple",
                  help="if invoked, debug ntuples with residuals would be created during gather jobs",
                  action="store_true",
                  dest="createAlignNtuple")
parser.add_option("--residualsModel",
                  help="functional residuals model. Possible vaslues: pureGaussian2D (default), pureGaussian, GaussPowerTails, ROOTVoigt, powerLawTails",
                  type="string",
                  default="pureGaussian2D",
                  dest="residualsModel")
parser.add_option("--useResiduals",
                  help="select residuals to use, possible values: 1111, 1110, 1100, 1010, 0010 that correspond to x y dxdz dydz residuals",
                  type="string",
                  default="1110",
                  dest="useResiduals")
parser.add_option("--peakNSigma",
                  help="if >0, only residuals peaks within n-sigma multidimentional ellipsoid would be considered in the alignment fit",
                  type="string",
                  default="-1.",
                  dest="peakNSigma")
parser.add_option("--preFilter",
                  help="if invoked, MuonAlignmentPreFilter module would be invoked in the Path's beginning. Can significantly speed up gather jobs.",
                  action="store_true",
                  dest="preFilter")
parser.add_option("--muonCollectionTag",
                  help="If empty, use trajectories. If not empty, it's InputTag of muons collection to use in tracker muons based approach, e.g., 'newmuons' or 'muons'",
                  type="string",
                  default="",
                  dest="muonCollectionTag")
parser.add_option("--maxDxy",
                  help="maximum track impact parameter with relation to beamline",
                  type="string",
                  default="1000.",
                  dest="maxDxy")
parser.add_option("--minNCrossedChambers",
                  help="minimum number of muon chambers that a track is required to cross",
                  type="string",
                  default="3",
                  dest="minNCrossedChambers")
parser.add_option("--extraPlots",
                  help="produce additional plots with geometry, reports differences, and corrections visulizations",
                  action="store_true",
                  dest="extraPlots")

if len(sys.argv) < 5:
    raise SystemError("Too few arguments.\n\n"+parser.format_help())

DIRNAME = sys.argv[1]
ITERATIONS = int(sys.argv[2])
INITIALGEOM = sys.argv[3]
INPUTFILES = sys.argv[4]

options, args = parser.parse_args(sys.argv[5:])
user_mail = options.user_mail
mapplots_ingeneral = options.mapplots
segdiffplots_ingeneral = options.segdiffplots
curvatureplots_ingeneral = options.curvatureplots
globaltag = options.globaltag
trackerconnect = options.trackerconnect
trackeralignment = options.trackeralignment
trackerAPEconnect = options.trackerAPEconnect
trackerAPE = options.trackerAPE
trackerBowsconnect = options.trackerBowsconnect
trackerBows = options.trackerBows
gprcdconnect = options.gprcdconnect
gprcd = options.gprcd
iscosmics = str(options.iscosmics)
station123params = options.station123params
station4params = options.station4params
cscparams = options.cscparams
muonCollectionTag = options.muonCollectionTag
minTrackPt = options.minTrackPt
maxTrackPt = options.maxTrackPt
minTrackP = options.minTrackP
maxTrackP = options.maxTrackP
maxDxy = options.maxDxy
minTrackerHits = str(options.minTrackerHits)
maxTrackerRedChi2 = options.maxTrackerRedChi2
minNCrossedChambers = options.minNCrossedChambers
allowTIDTEC = str(not options.notAllowTIDTEC)
twoBin = str(options.twoBin)
weightAlignment = str(options.weightAlignment)
minAlignmentHits = str(options.minAlignmentHits)
combineME11 = str(not options.notCombineME11)
maxEvents = options.maxEvents
skipEvents = options.skipEvents
validationLabel = options.validationLabel
maxResSlopeY = options.maxResSlopeY
theNSigma = options.motionPolicyNSigma
residualsModel = options.residualsModel
peakNSigma = options.peakNSigma
preFilter = not not options.preFilter
extraPlots = options.extraPlots
useResiduals = options.useResiduals


#print "check: ", allowTIDTEC, combineME11, preFilter

doCleanUp = not options.noCleanUp
createMapNtuple = not not options.createMapNtuple
createAlignNtuple = not not options.createAlignNtuple

doCSC = True
if options.noCSC: doCSC = False
doDT = True
if options.noDT: doDT = False
if options.noCSC and options.noDT:
    print("cannot do --noCSC and --noDT at the same time!")
    sys.exit()

json_file = options.json

fileNames=[]
fileNamesBlocks=[]
execfile(INPUTFILES)
njobs = options.subjobs
if (options.inputInBlocks):
    njobs = len(fileNamesBlocks)
    if njobs==0:
        print("while --inputInBlocks is specified, the INPUTFILES has no blocks!")
        sys.exit()

stepsize = int(math.ceil(1.*len(fileNames)/options.subjobs))

pwd = str(os.getcwd())

copytrackerdb = ""
if trackerconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerconnect[12:]
if trackerAPEconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerAPEconnect[12:]
if trackerBowsconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerBowsconnect[12:]
if gprcdconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % gprcdconnect[12:]


#####################################################################
# step 0: convert initial geometry to xml
INITIALXML = INITIALGEOM + '.xml'
if INITIALGEOM[-3:]=='.db':
    INITIALXML = INITIALGEOM[:-3] + '.xml'
print("Converting",INITIALGEOM,"to",INITIALXML," ...will be done in several seconds...")
print("./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py  %s %s --gprcdconnect %s --gprcd %s" % (INITIALGEOM,INITIALXML,gprcdconnect,gprcd))
exit_code = os.system("./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py  %s %s --gprcdconnect %s --gprcd %s" % (INITIALGEOM,INITIALXML,gprcdconnect,gprcd))
if exit_code>0:
    print("problem: conversion exited with code:", exit_code)
    sys.exit()

#####################################################################

def writeGatherCfg(fname, my_vars):
    file(fname, "w").write("""#/bin/sh
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
export ALIGNMENT_TRACKERBOWSCONNECT=%(trackerBowsconnect)s
export ALIGNMENT_TRACKERBOWS=%(trackerBows)s
export ALIGNMENT_GPRCDCONNECT=%(gprcdconnect)s
export ALIGNMENT_GPRCD=%(gprcd)s
export ALIGNMENT_ISCOSMICS=%(iscosmics)s
export ALIGNMENT_STATION123PARAMS=%(station123params)s
export ALIGNMENT_STATION4PARAMS=%(station4params)s
export ALIGNMENT_CSCPARAMS=%(cscparams)s
export ALIGNMENT_MUONCOLLECTIONTAG=%(muonCollectionTag)s
export ALIGNMENT_MINTRACKPT=%(minTrackPt)s
export ALIGNMENT_MAXTRACKPT=%(maxTrackPt)s
export ALIGNMENT_MINTRACKP=%(minTrackP)s
export ALIGNMENT_MAXTRACKP=%(maxTrackP)s
export ALIGNMENT_MAXDXY=%(maxDxy)s
export ALIGNMENT_MINTRACKERHITS=%(minTrackerHits)s
export ALIGNMENT_MAXTRACKERREDCHI2=%(maxTrackerRedChi2)s
export ALIGNMENT_MINNCROSSEDCHAMBERS=%(minNCrossedChambers)s
export ALIGNMENT_ALLOWTIDTEC=%(allowTIDTEC)s
export ALIGNMENT_TWOBIN=%(twoBin)s
export ALIGNMENT_WEIGHTALIGNMENT=%(weightAlignment)s
export ALIGNMENT_MINALIGNMENTHITS=%(minAlignmentHits)s
export ALIGNMENT_COMBINEME11=%(combineME11)s
export ALIGNMENT_MAXEVENTS=%(maxEvents)s
export ALIGNMENT_SKIPEVENTS=%(skipEvents)s
export ALIGNMENT_MAXRESSLOPEY=%(maxResSlopeY)s
export ALIGNMENT_DO_DT=%(doDT)s
export ALIGNMENT_DO_CSC=%(doCSC)s
export ALIGNMENT_JSON=%(json_file)s
export ALIGNMENT_CREATEMAPNTUPLE=%(createMapNtuple)s
#export ALIGNMENT_CREATEALIGNNTUPLE=%(createAlignNtuple)s
export ALIGNMENT_PREFILTER=%(preFilter)s


if [ \"zzz$ALIGNMENT_JSON\" != \"zzz\" ]; then
  cp -f $ALIGNMENT_JSON $ALIGNMENT_CAFDIR/
fi

cp -f %(directory)sgather_cfg.py %(inputdbdir)s%(inputdb)s %(copytrackerdb)s $ALIGNMENT_CAFDIR/
cd $ALIGNMENT_CAFDIR/
ls -l
cmsRun gather_cfg.py
ls -l
cp -f *.tmp %(copyplots)s $ALIGNMENT_AFSDIR/%(directory)s
""" % my_vars)

#####################################################################

def writeAlignCfg(fname, my_vars):
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
export ALIGNMENT_TRACKERBOWSCONNECT=%(trackerBowsconnect)s
export ALIGNMENT_TRACKERBOWS=%(trackerBows)s
export ALIGNMENT_GPRCDCONNECT=%(gprcdconnect)s
export ALIGNMENT_GPRCD=%(gprcd)s
export ALIGNMENT_ISCOSMICS=%(iscosmics)s
export ALIGNMENT_STATION123PARAMS=%(station123params)s
export ALIGNMENT_STATION4PARAMS=%(station4params)s
export ALIGNMENT_CSCPARAMS=%(cscparams)s
export ALIGNMENT_MINTRACKPT=%(minTrackPt)s
export ALIGNMENT_MAXTRACKPT=%(maxTrackPt)s
export ALIGNMENT_MINTRACKP=%(minTrackP)s
export ALIGNMENT_MAXTRACKP=%(maxTrackP)s
export ALIGNMENT_MINTRACKERHITS=%(minTrackerHits)s
export ALIGNMENT_MAXTRACKERREDCHI2=%(maxTrackerRedChi2)s
export ALIGNMENT_ALLOWTIDTEC=%(allowTIDTEC)s
export ALIGNMENT_TWOBIN=%(twoBin)s
export ALIGNMENT_WEIGHTALIGNMENT=%(weightAlignment)s
export ALIGNMENT_MINALIGNMENTHITS=%(minAlignmentHits)s
export ALIGNMENT_COMBINEME11=%(combineME11)s
export ALIGNMENT_MAXRESSLOPEY=%(maxResSlopeY)s
export ALIGNMENT_CLEANUP=%(doCleanUp)s
export ALIGNMENT_CREATEALIGNNTUPLE=%(createAlignNtuple)s
export ALIGNMENT_RESIDUALSMODEL=%(residualsModel)s
export ALIGNMENT_PEAKNSIGMA=%(peakNSigma)s
export ALIGNMENT_USERESIDUALS=%(useResiduals)s

cp -f %(directory)salign_cfg.py %(inputdbdir)s%(inputdb)s %(directory)s*.tmp  %(copytrackerdb)s $ALIGNMENT_CAFDIR/

export ALIGNMENT_PLOTTINGTMP=`find %(directory)splotting0*.root -maxdepth 1 -size +0 -print 2> /dev/null`

# if it's 1st or last iteration, combine _plotting.root files into one:
if [ \"$ALIGNMENT_ITERATION\" != \"111\" ] || [ \"$ALIGNMENT_ITERATION\" == \"%(ITERATIONS)s\" ]; then
  #nfiles=$(ls %(directory)splotting0*.root 2> /dev/null | wc -l)
  if [ \"zzz$ALIGNMENT_PLOTTINGTMP\" != \"zzz\" ]; then
    hadd -f1 %(directory)s%(director)s_plotting.root %(directory)splotting0*.root
    #if [ $? == 0 ] && [ \"$ALIGNMENT_CLEANUP\" == \"True\" ]; then rm %(directory)splotting0*.root; fi
  fi
fi

if [ \"$ALIGNMENT_CLEANUP\" == \"True\" ] && [ \"zzz$ALIGNMENT_PLOTTINGTMP\" != \"zzz\" ]; then
  rm $ALIGNMENT_PLOTTINGTMP
fi

cd $ALIGNMENT_CAFDIR/
export ALIGNMENT_ALIGNMENTTMP=`find alignment*.tmp -maxdepth 1 -size +1k -print 2> /dev/null`
ls -l

cmsRun align_cfg.py
cp -f MuonAlignmentFromReference_report.py $ALIGNMENT_AFSDIR/%(directory)s%(director)s_report.py
cp -f MuonAlignmentFromReference_outputdb.db $ALIGNMENT_AFSDIR/%(directory)s%(director)s.db
cp -f MuonAlignmentFromReference_plotting.root $ALIGNMENT_AFSDIR/%(directory)s%(director)s.root

cd $ALIGNMENT_AFSDIR
./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py %(directory)s%(director)s.db %(directory)s%(director)s.xml --noLayers --gprcdconnect $ALIGNMENT_GPRCDCONNECT --gprcd $ALIGNMENT_GPRCD

export ALIGNMENT_ALIGNMENTTMP=`find %(directory)salignment*.tmp -maxdepth 1 -size +1k -print 2> /dev/null`
if [ \"$ALIGNMENT_CLEANUP\" == \"True\" ] && [ \"zzz$ALIGNMENT_ALIGNMENTTMP\" != \"zzz\" ]; then
  rm $ALIGNMENT_ALIGNMENTTMP
  echo " "
fi

# if it's not 1st or last iteration, do some clean up:
if [ \"$ALIGNMENT_ITERATION\" != \"1\" ] && [ \"$ALIGNMENT_ITERATION\" != \"%(ITERATIONS)s\" ]; then
  if [ \"$ALIGNMENT_CLEANUP\" == \"True\" ] && [ -e %(directory)s%(director)s.root ]; then
    rm %(directory)s%(director)s.root
  fi
fi

# if it's last iteration, apply chamber motion policy
if [ \"$ALIGNMENT_ITERATION\" == \"%(ITERATIONS)s\" ]; then
  # convert this iteration's geometry into detailed xml
  ./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py %(directory)s%(director)s.db %(directory)s%(director)s_extra.xml --gprcdconnect $ALIGNMENT_GPRCDCONNECT --gprcd $ALIGNMENT_GPRCD
  # perform motion policy 
  ./Alignment/MuonAlignmentAlgorithms/scripts/motionPolicyChamber.py \
      %(INITIALXML)s  %(directory)s%(director)s_extra.xml \
      %(directory)s%(director)s_report.py \
      %(directory)s%(director)s_final.xml \
      --nsigma %(theNSigma)s
  # convert the resulting xml into the final sqlite geometry
  ./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py %(directory)s%(director)s_final.xml %(directory)s%(director)s_final.db --gprcdconnect $ALIGNMENT_GPRCDCONNECT --gprcd $ALIGNMENT_GPRCD
fi

""" % my_vars)

#####################################################################

def writeValidationCfg(fname, my_vars):
    file(fname, "w").write("""#!/bin/sh
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
ALIGNMENT_EXTRAPLOTS=%(extraPlots)s
export ALIGNMENT_GPRCDCONNECT=%(gprcdconnect)s
export ALIGNMENT_GPRCD=%(gprcd)s
export ALIGNMENT_DO_DT=%(doDT)s
export ALIGNMENT_DO_CSC=%(doCSC)s


# copy the scripts to CAFDIR
cd Alignment/MuonAlignmentAlgorithms/scripts/
cp -f plotscripts.py $ALIGNMENT_CAFDIR/
cp -f mutypes.py $ALIGNMENT_CAFDIR/
cp -f alignmentValidation.py $ALIGNMENT_CAFDIR/
cp -f phiedges_fitfunctions.C $ALIGNMENT_CAFDIR/
cp -f createTree.py $ALIGNMENT_CAFDIR/
cp -f signConventions.py $ALIGNMENT_CAFDIR/
cp -f convertSQLiteXML.py $ALIGNMENT_CAFDIR/
cp -f wrapperExtraPlots.sh $ALIGNMENT_CAFDIR/
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

dtcsc=""
if [ $ALIGNMENT_DO_DT == \"True\" ]; then
  dtcsc="--dt"
fi
if [ $ALIGNMENT_DO_CSC == \"True\" ]; then
  dtcsc="${dtcsc} --csc"
fi


cd $ALIGNMENT_CAFDIR/
echo \" ### Start running ###\"
date

# do fits and median plots first 
./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  --createDirSructure --dt --csc --fit --median

if [ $ALIGNMENT_MAPPLOTS == \"True\" ]; then
  ./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  $dtcsc --map
fi

if [ $ALIGNMENT_SEGDIFFPLOTS == \"True\" ]; then
  ./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  $dtcsc --segdiff
fi                   

if [ $ALIGNMENT_CURVATUREPLOTS == \"True\" ]; then
  ./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  $dtcsc --curvature
fi

if [ $ALIGNMENT_EXTRAPLOTS == \"True\" ]; then
  if [ \"zzz%(copytrackerdb)s\" != \"zzz\" ]; then
    cp -f $ALIGNMENT_AFSDIR/%(copytrackerdb)s $ALIGNMENT_CAFDIR/
  fi
  cp $ALIGNMENT_AFSDIR/inertGlobalPositionRcd.db .
  ./convertSQLiteXML.py $ALIGNMENT_AFSDIR/%(INITIALGEOM)s g0.xml --noLayers  --gprcdconnect $ALIGNMENT_GPRCDCONNECT --gprcd $ALIGNMENT_GPRCD
  ./wrapperExtraPlots.sh -n $ALIGNMENT_ITERATION -i $ALIGNMENT_AFSDIR -0 g0.xml -z -w %(station123params)s %(dir_no_)s
  mkdir out/extra
  cd %(dir_no_)s
  mv MB ../out/extra/
  mv ME ../out/extra/
  cd -
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

""" % my_vars)


#####################################################################

#SUPER_SPECIAL_XY_AND_DXDZ_ITERATIONS = True
SUPER_SPECIAL_XY_AND_DXDZ_ITERATIONS = False

bsubfile = ["#!/bin/sh", ""]
bsubnames = []
last_align = None
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

    dir_no_ = DIRNAME
    if DIRNAME[-1]=='_': dir_no_ = DIRNAME[:-1]

    os.system("rm -rf %s; mkdir %s" % (directory, directory))
    os.system("cp Alignment/MuonAlignmentAlgorithms/python/gather_cfg.py %s" % directory)
    os.system("cp Alignment/MuonAlignmentAlgorithms/python/align_cfg.py %s" % directory)

    bsubfile.append("cd %s" % directory)

    mapplots = False
    if mapplots_ingeneral and (iteration == 1 or iteration == 3 or iteration == 5 or iteration == 7 or iteration == 9 or iteration == ITERATIONS): mapplots = True
    segdiffplots = False
    if segdiffplots_ingeneral and (iteration == 1 or iteration == ITERATIONS): segdiffplots = True
    curvatureplots = False
    if curvatureplots_ingeneral and (iteration == 1 or iteration == ITERATIONS): curvatureplots = True

    ### gather.sh runners for njobs
    for jobnumber in range(njobs):
        if not options.inputInBlocks:
            inputfiles = " ".join(fileNames[jobnumber*stepsize:(jobnumber+1)*stepsize])
        else:
            inputfiles = " ".join(fileNamesBlocks[jobnumber])

        if mapplots or segdiffplots or curvatureplots: copyplots = "plotting*.root"
        else: copyplots = ""

        if len(inputfiles) > 0:
            gather_fileName = "%sgather%03d.sh" % (directory, jobnumber)
            writeGatherCfg(gather_fileName, vars())
            os.system("chmod +x %s" % gather_fileName)
            bsubfile.append("echo %sgather%03d.sh" % (directory, jobnumber))

            if last_align is None: waiter = ""
            else: waiter = "-w \"ended(%s)\"" % last_align            
            if options.big: queue = "cmscaf1nd"
            else: queue = "cmscaf1nh"

            bsubfile.append("bsub -R \"type==SLC6_64\" -q %s -J \"%s_gather%03d\" -u youremail.tamu.edu %s gather%03d.sh" % (queue, director, jobnumber, waiter, jobnumber))

            bsubnames.append("ended(%s_gather%03d)" % (director, jobnumber))


    ### align.sh
    if SUPER_SPECIAL_XY_AND_DXDZ_ITERATIONS:
        if ( iteration == 1 or iteration == 3 or iteration == 5 or iteration == 7 or iteration == 9):
            tmp = station123params, station123params, useResiduals 
            station123params, station123params, useResiduals = "000010", "000010", "0010"
            writeAlignCfg("%salign.sh" % directory, vars())
            station123params, station123params, useResiduals = tmp
        elif ( iteration == 2 or iteration == 4 or iteration == 6 or iteration == 8 or iteration == 10):
            tmp = station123params, station123params, useResiduals 
            station123params, station123params, useResiduals = "110001", "100001", "1100"
            writeAlignCfg("%salign.sh" % directory, vars())
            station123params, station123params, useResiduals = tmp
    else:
        writeAlignCfg("%salign.sh" % directory, vars())

    os.system("chmod +x %salign.sh" % directory)

    bsubfile.append("echo %salign.sh" % directory)
    if user_mail: bsubfile.append("bsub -R \"type==SLC6_64\" -q cmscaf1nd -J \"%s_align\" -u %s -w \"%s\" align.sh" % (director, user_mail, " && ".join(bsubnames)))
    else: bsubfile.append("bsub -R \"type==SLC6_64\" -q cmscaf1nd -J \"%s_align\" -w \"%s\" align.sh" % (director, " && ".join(bsubnames)))

    #bsubfile.append("cd ..")
    bsubnames = []
    last_align = "%s_align" % director


    ### after the last iteration (optionally) do diagnostics run
    if len(validationLabel) and iteration == ITERATIONS:
        # do we have plotting files created?
        directory1 = "%s01/" % DIRNAME
        director1 = directory1[:-1]

        writeValidationCfg("%svalidation.sh" % directory, vars())
        os.system("chmod +x %svalidation.sh" % directory)

        bsubfile.append("echo %svalidation.sh" % directory)
        if user_mail: bsubfile.append("bsub -R \"type==SLC6_64\" -q cmscaf1nd -J \"%s_validation\" -u %s -w \"ended(%s)\" validation.sh" % (director, user_mail, last_align))
        else: bsubfile.append("bsub -R \"type==SLC6_64\" -q cmscaf1nd -J \"%s_validation\" -w \"ended(%s)\" validation.sh" % (director, last_align))

    bsubfile.append("cd ..")
    bsubfile.append("")


file(options.submitJobs, "w").write("\n".join(bsubfile))
os.system("chmod +x %s" % options.submitJobs)
