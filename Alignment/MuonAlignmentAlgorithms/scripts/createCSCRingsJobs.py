#! /usr/bin/env python

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

DIRNAME        directories will be named DIRNAME01, DIRNAME02...
ITERATIONS     number of iterations
INITIALGEOM    SQLite file containing muon geometry with tag names
               DTAlignmentRcd, DTAlignmentErrorExtendedRcd, CSCAlignmentRcd, CSCAlignmentErrorExtendedRcd
INPUTFILES     Python file defining 'fileNames', a list of input files as
               strings (create with findQualityFiles.py)""" % vars()

parser = optparse.OptionParser(usage)
parser.add_option("--validationLabel",
                  help="[REQUIRED] the label to be used to mark a run; the results will be put into a LABEL_DATESTAMP.tgz tarball",
                  type="string",
                  default="",
                  dest="validationLabel")
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
                  help="name of TrackerAlignmentErrorExtendedRcd tag (tracker APEs)",
                  type="string",
                  default="AlignmentErrorsExtended",
                  dest="trackerAPE")
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
                  help="if invoked, treat ME1/1a and ME1/1b as separate objects",
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
parser.add_option("--maxResSlopeY",
                  help="maximum residual slope y component",
                  type="string",
                  default="10",
                  dest="maxResSlopeY")
parser.add_option("--ring2only",
                  help="if invoked, use only ring 2 results to align all rings in corresponding disks",
                  action="store_true",
                  dest="ring2only")
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
parser.add_option("--preFilter",
                  help="if invoked, MuonAlignmentPreFilter module would be invoked in the Path's beginning. Can significantly speed up gather jobs.",
                  action="store_true",
                  dest="preFilter")
parser.add_option("--useTrackerMuons",
                  help="use tracker muons approach instead of the default global muons tracks based one",
                  action="store_true",
                  dest="useTrackerMuons")
parser.add_option("--muonCollectionTag",
                  help="InputTag of muons collection to use in tracker muons based approach",
                  type="string",
                  default="newmuons",
                  dest="muonCollectionTag")
parser.add_option("--maxDxy",
                  help="maximum track impact parameter with relation to beamline",
                  type="string",
                  default="1000.",
                  dest="maxDxy")
parser.add_option("--minNCrossedChambers",
                  help="minimum number of muon chambers that a track is required to cross",
                  type="string",
                  default="2",
                  dest="minNCrossedChambers")

if len(sys.argv) < 5:
    raise SystemError, "Too few arguments.\n\n"+parser.format_help()

DIRNAME = sys.argv[1]
ITERATIONS = int(sys.argv[2])
INITIALGEOM = sys.argv[3]
INPUTFILES = sys.argv[4]

options, args = parser.parse_args(sys.argv[5:])
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
preFilter = not not options.preFilter
useTrackerMuons = options.useTrackerMuons

createMapNtuple=False
if options.createMapNtuple: createMapNtuple=True

ring2only = ""
if options.ring2only: ring2only = "--ring2only"
inputInBlocks = ""
if options.inputInBlocks: inputInBlocks = "--inputInBlocks"

json_file = options.json

if validationLabel == '':
  print "\nOne or more of REQUIRED options is missing!\n"
  parser.print_help()
  sys.exit()

fileNames=[]
fileNamesBlocks=[]
execfile(INPUTFILES)
njobs = options.subjobs
if (options.inputInBlocks):
  njobs = len(fileNamesBlocks)
  if njobs==0:
    print "while --inputInBlocks is specified, the INPUTFILES has no blocks!"
    sys.exit()

stepsize = int(math.ceil(1.*len(fileNames)/njobs))

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
# do two iterations of gather + align jobs
directory = ""
for iteration in range(1, ITERATIONS+1):
    if iteration == 1:
        inputdb = INITIALGEOM
        inputdbdir = directory[:]
        inputxml = INITIALXML
    else:
        inputdb = director + ".db"
        inputdbdir = directory[:]
        inputxml = director + ".xml"

    directory = "%s%02d/" % (DIRNAME, iteration)
    director = directory[:-1]
    directory1 = "%s01/" % DIRNAME
    director1 = directory1[:-1]
    os.system("rm -rf %s; mkdir %s" % (directory, directory))
    os.system("cp gather_cfg.py %s" % directory)
    #os.system("cp align_cfg.py %s" % directory)

    bsubfile.append("cd %s" % directory)

    for jobnumber in range(njobs):
        gather_fileName = "%sgather%03d.sh" % (directory, jobnumber)
        if not options.inputInBlocks:
          inputfiles = " ".join(fileNames[jobnumber*stepsize:(jobnumber+1)*stepsize])
        else:
          inputfiles = " ".join(fileNamesBlocks[jobnumber])

        copyplots = "plotting*.root"

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
export ALIGNMENT_MAPPLOTS=True
export ALIGNMENT_SEGDIFFPLOTS=True
export ALIGNMENT_CURVATUREPLOTS=False
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
export ALIGNMENT_DO_DT='False'
export ALIGNMENT_DO_CSC='True'
export ALIGNMENT_JSON=%(json_file)s
export ALIGNMENT_CREATEMAPNTUPLE=%(createMapNtuple)s
export ALIGNMENT_PREFILTER=%(preFilter)s
export ALIGNMENT_USETRACKERMUONS=%(useTrackerMuons)s

if [ \"zzz$ALIGNMENT_JSON\" != \"zzz\" ]; then
  cp -f $ALIGNMENT_JSON $ALIGNMENT_CAFDIR/
fi

cp -f %(directory)sgather_cfg.py %(inputdbdir)s%(inputdb)s %(copytrackerdb)s $ALIGNMENT_CAFDIR/
cd $ALIGNMENT_CAFDIR/
ls -l
cmsRun gather_cfg.py
ls -l
cp -f %(copyplots)s $ALIGNMENT_AFSDIR/%(directory)s
""" % vars())
            os.system("chmod +x %s" % gather_fileName)
            bsubfile.append("echo %sgather%03d.sh" % (directory, jobnumber))

            if last_align is None: waiter = ""
            else: waiter = "-w \"ended(%s)\"" % last_align
            if options.big: queue = "cmscaf1nd"
            else: queue = "cmscaf1nh"

            bsubfile.append("bsub -R \"type==SLC5_64\" -q %s -J \"%s_gather%03d\" %s gather%03d.sh" % (queue, director, jobnumber, waiter, jobnumber))

            bsubnames.append("ended(%s_gather%03d)" % (director, jobnumber))

    copytrackerdb = ""
    if trackerconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerconnect[12:]
    if trackerAPEconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % trackerAPEconnect[12:]
    if gprcdconnect[0:12] == "sqlite_file:": copytrackerdb += "%s " % gprcdconnect[12:]

    file("%salign.sh" % directory, "w").write("""#!/bin/sh
# %(commandline)s

export ALIGNMENT_CAFDIR=`pwd`
mkdir files
mkdir out

cd %(pwd)s
eval `scramv1 run -sh`
export ALIGNMENT_AFSDIR=`pwd`

# combine _plotting.root files into one:
nfiles=$(ls %(directory)splotting0*.root 2> /dev/null | wc -l)
if [ \"$nfiles\" != \"0\" ]; then
  flist=""
  for fn in %(directory)splotting0*.root
  do
    FILESIZE=$(stat -c%%s "$fn")
    if [ $FILESIZE -gt 1000 ]; then
      echo $fn, $FILESIZE
      flist="$flist $fn"
    fi
  done
  echo $flist
  #hadd -f1 %(directory)s%(director)s_plotting.root %(directory)splotting0*.root
  hadd -f1 %(directory)s%(director)s_plotting.root $flist
  #if [ $? == 0 ]; then rm %(directory)splotting0*.root; fi
fi

# copy plotting and db files to CAFDIR
cp -f %(directory)s%(director)s_plotting.root  $ALIGNMENT_CAFDIR/files
cp -f inertGlobalPositionRcd.db %(inputdbdir)s%(inputdb)s  %(inputdbdir)s%(inputxml)s  %(copytrackerdb)s  $ALIGNMENT_CAFDIR/

# copy the scripts to CAFDIR
cd Alignment/MuonAlignmentAlgorithms/scripts/
cp -f plotscripts.py $ALIGNMENT_CAFDIR/
cp -f mutypes.py $ALIGNMENT_CAFDIR/
cp -f alignmentValidation.py $ALIGNMENT_CAFDIR/
cp -f phiedges_fitfunctions.C $ALIGNMENT_CAFDIR/
cp -f convertSQLiteXML.py $ALIGNMENT_CAFDIR/
cp -f alignCSCRings.py $ALIGNMENT_CAFDIR/
cp -f signConventions.py $ALIGNMENT_CAFDIR/
cd -

cd $ALIGNMENT_CAFDIR/
ls -l

# run alignment validation to produce map plots and sin fit results
./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out  --csc --map --segdiff --createDirSructure

# align CSC rings using the fit results from the previous step
./alignCSCRings.py -d $ALIGNMENT_CAFDIR/out -l %(validationLabel)s -x %(inputxml)s %(ring2only)s

# convert ring-aligned xml geometry into sqlite
./convertSQLiteXML.py %(inputxml)s.ring.xml %(director)s.db

# convert the new sqlite into proper chambers and layers xml
./convertSQLiteXML.py %(director)s.db %(director)s.xml

#copy all good stuff to $ALIGNMENT_AFSDIR/%(directory)s
tar czf %(director)s_%(validationLabel)s.tgz out
cp -f %(director)s_%(validationLabel)s.tgz $ALIGNMENT_AFSDIR/%(directory)s
cp -f out/tmp_test_results_map__%(validationLabel)s.pkl  $ALIGNMENT_AFSDIR/%(directory)s%(director)s.pkl
cp -f %(inputxml)s.ring.xml  $ALIGNMENT_AFSDIR/%(directory)s
cp -f %(director)s.xml  $ALIGNMENT_AFSDIR/%(directory)s
cp -f %(director)s.db  $ALIGNMENT_AFSDIR/%(directory)s

# if it's last iteration, apply chamber motion policy
#if [ \"$ALIGNMENT_ITERATION\" == 2 ]; then
#  #nfiles=$(ls %(directory)splotting0*.root 2> /dev/null | wc -l)
#fi

""" % vars())
    os.system("chmod +x %salign.sh" % directory)

    bsubfile.append("echo %salign.sh" % directory)
    if options.big: queue = "cmscaf1nd"
    else: queue = "cmscaf1nh"
    bsubfile.append("bsub -R \"type==SLC5_64\" -q %s -J \"%s_align\" -w \"%s\" align.sh" % (queue, director, " && ".join(bsubnames)))
    bsubnames = []
    last_align = "%s_align" % director
    
    # after the last iteration do diagnostics run for putting into a browser
    if iteration == ITERATIONS:
        # do we have plotting files created?
        directory1 = "%s01/" % DIRNAME
        director1 = directory1[:-1]

        file("%svalidation.sh" % directory, "w").write("""#!/bin/sh
# %(commandline)s

export ALIGNMENT_CAFDIR=`pwd`
#mkdir files
mkdir out
mkdir tmp

cd %(pwd)s
eval `scramv1 run -sh`
ALIGNMENT_AFSDIR=`pwd`

# copy the scripts to CAFDIR
cd Alignment/MuonAlignmentAlgorithms/scripts/
cp -f plotscripts.py $ALIGNMENT_CAFDIR/
cp -f mutypes.py $ALIGNMENT_CAFDIR/
cp -f alignmentValidation.py $ALIGNMENT_CAFDIR/
cp -f phiedges_fitfunctions.C $ALIGNMENT_CAFDIR/
cp -f createTree.py $ALIGNMENT_CAFDIR/
cp -f signConventions.py $ALIGNMENT_CAFDIR/
cd -
cp Alignment/MuonAlignmentAlgorithms/test/browser/tree* $ALIGNMENT_CAFDIR/out/

# copy the results to CAFDIR
cp -f %(directory1)s%(director1)s_%(validationLabel)s.tgz $ALIGNMENT_CAFDIR/tmp/
cp -f %(directory)s%(director)s_%(validationLabel)s.tgz $ALIGNMENT_CAFDIR/tmp/

cd $ALIGNMENT_CAFDIR/
tar xzvf tmp/%(director1)s_%(validationLabel)s.tgz
mv tmp/out/* out/
mv out/iterN out/iter1
mv out/tmp_test_results_map__%(validationLabel)s.pkl out/tmp_test_results_map__%(validationLabel)s_1.pkl
tar xzvf tmp/%(director)s_%(validationLabel)s.tgz
mv tmp/out/* out/

echo \" ### Start running ###\"
date

# run simple diagnostic
./alignmentValidation.py -l %(validationLabel)s -i $ALIGNMENT_CAFDIR --i1 files --iN files --i1prefix %(director1)s --iNprefix %(director)s -o $ALIGNMENT_CAFDIR/out --csc --diagnostic

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

