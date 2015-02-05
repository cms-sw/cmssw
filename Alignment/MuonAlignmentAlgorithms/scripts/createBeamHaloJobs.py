#!/usr/bin/env python

import os, sys, re, optparse, math

copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""
    if copyargs[i].find(" ") != -1:
        copyargs[i] = "\"%s\"" % copyargs[i]
commandline = " ".join(copyargs)

usage = """./%prog DIRNAME PATTERN INITIALGEOM INPUTFILES [options]

Creates (overwrites) a directory for each word in PATTERN and creates
(overwrites) submitJobs.sh with the submission sequence and
dependencies.

DIRNAME        directories will be named DIRNAME01, DIRNAME02, etc.
PATTERN        a quoted combination of "phiy", "phipos", "phiz"
INITIALGEOM    SQLite file containing muon geometry with tag names
               CSCAlignmentRcd, CSCAlignmentErrorExtendedRcd
INPUTFILES     Python file defining 'fileNames', a list of input files as
               strings"""

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
parser.add_option("--globalTag",
                  help="GlobalTag for calibration conditions",
                  type="string",
                  default="GR_R_42_V14::All",
                  dest="globaltag")
parser.add_option("--photogrammetry",
                  help="if invoked, alignment will be constrained to photogrammetry",
                  action="store_true",
                  dest="photogrammetry")
parser.add_option("--photogrammetryOnlyholes",
                  help="if invoked, only missing data will be constrained to photogrammetry",
                  action="store_true",
                  dest="photogrammetryOnlyholes")
parser.add_option("--photogrammetryOnlyOnePerRing",
                  help="if invoked, only one chamber per ring will be constrained to photogrammetry",
                  action="store_true",
                  dest="photogrammetryOnlyOnePerRing")
parser.add_option("--photogrammetryScale",
                  help="scale factor for photogrammetry constraint: 1 is default and 10 *weakens* the constraint by a factor of 10",
                  type="string",
                  default="1.",
                  dest="photogrammetryScale")
parser.add_option("--slm",
                  help="if invoked, apply SLM constraint",
                  action="store_true",
                  dest="slm")
parser.add_option("--fillME11holes",
                  help="use CollisionsOct2010 data to fill holes in ME1/1",
                  action="store_true",
                  dest="fillME11holes")
parser.add_option("--disks",
                  help="align whole disks, rather than individual rings",
                  action="store_true",
                  dest="disks")

parser.add_option("--minP",
                  help="minimum track momentum (measured via radial component of fringe fields)",
                  type="string",
                  default="5",
                  dest="minP")
parser.add_option("--minHitsPerChamber",
                  help="minimum number of hits per chamber",
                  type="string",
                  default="5",
                  dest="minHitsPerChamber")
parser.add_option("--maxdrdz",
                  help="maximum dr/dz of tracklets (anti-cosmic cut)",
                  type="string",
                  default="0.2",
                  dest="maxdrdz")
parser.add_option("--maxRedChi2",
                  help="maximum reduced chi^2 of tracks",
                  type="string",
                  default="10",
                  dest="maxRedChi2")
parser.add_option("--fiducial",
                  help="if invoked, select only segments within the good region of the chamber (for all 6 layers)",
                  action="store_true",
                  dest="fiducial")
parser.add_option("--useHitWeights",
                  help="if invoked, use hit weights in tracklet fits",
                  action="store_true",
                  dest="useHitWeights")
parser.add_option("--truncateSlopeResid",
                  help="maximum allowed slope residual in mrad (like the histograms in a phiy job)",
                  type="string",
                  default="30.",
                  dest="truncateSlopeResid")
parser.add_option("--truncateOffsetResid",
                  help="maximum allowed offset residual in mm (like the histograms in a phipos or phiz job)",
                  type="string",
                  default="15.",
                  dest="truncateOffsetResid")
parser.add_option("--combineME11",
                  help="if invoked, combine ME1/1a and ME1/1b chambers",
                  action="store_true",
                  dest="combineME11")
parser.add_option("--useTrackWeights",
                  help="if invoked, weight residuals by track uncertainties",
                  action="store_true",
                  dest="useTrackWeights")
parser.add_option("--errorFromRMS",
                  help="if invoked, determine residuals uncertainties from the RMS of the residuals distribution",
                  action="store_true",
                  dest="errorFromRMS")
parser.add_option("--minTracksPerOverlap",
                  help="minimum number of tracks needed for an overlap constraint to be valid",
                  type="string",
                  default="10",
                  dest="minTracksPerOverlap")
parser.add_option("--slopeFromTrackRefit",
                  help="if invoked, determine direction of tracklets by refitting track to all other stations",
                  action="store_true",
                  dest="slopeFromTrackRefit")
parser.add_option("--minStationsInTrackRefits",
                  help="minimum number of stations in a full track refit (slopeFromTrackRefit)",
                  type="string",
                  default="2",
                  dest="minStationsInTrackRefits")
parser.add_option("--inputInBlocks",
                  help="if invoked, assume that INPUTFILES provides a list of files already groupped into job blocks, -j has no effect in that case",
                  action="store_true",
                  dest="inputInBlocks")
parser.add_option("--json",
                  help="If present with JSON file as argument, use JSON file for good lumi mask. "+\
                  "The latest JSON file is available at /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions10/7TeV/StreamExpress/",
                  type="string",
                  default="",
                  dest="json")

if len(sys.argv) < 5:
    raise SystemError, "Too few arguments.\n\n"+parser.format_help()

DIRNAME = sys.argv[1]
PATTERN = re.split("\s+", sys.argv[2])
INITIALGEOM = sys.argv[3]
INPUTFILES = sys.argv[4]

options, args = parser.parse_args(sys.argv[5:])
user_mail = options.user_mail
globaltag = options.globaltag
photogrammetry = options.photogrammetry
photogrammetryOnlyholes = options.photogrammetryOnlyholes
photogrammetryOnlyOnePerRing = options.photogrammetryOnlyOnePerRing
photogrammetryScale = options.photogrammetryScale
slm = options.slm
fillME11holes = options.fillME11holes
disks = options.disks
minP = options.minP
minHitsPerChamber = options.minHitsPerChamber
maxdrdz = options.maxdrdz
maxRedChi2 = options.maxRedChi2
fiducial = options.fiducial
useHitWeights = options.useHitWeights
truncateSlopeResid = options.truncateSlopeResid
truncateOffsetResid = options.truncateOffsetResid
combineME11 = options.combineME11
useTrackWeights = options.useTrackWeights
errorFromRMS = options.errorFromRMS
minTracksPerOverlap = options.minTracksPerOverlap
slopeFromTrackRefit = options.slopeFromTrackRefit
minStationsInTrackRefits = options.minStationsInTrackRefits

if options.inputInBlocks: inputInBlocks = "--inputInBlocks"
json_file = options.json


fileNames=[]
fileNamesBlocks=[]
execfile(INPUTFILES)
njobs = options.subjobs
if (options.inputInBlocks):
  njobs = len(fileNamesBlocks)
  if njobs==0:
    print "while --inputInBlocks is specified, the INPUTFILES has no blocks!"
    sys.exit()

stepsize = int(math.ceil(1.*len(fileNames)/options.subjobs))
pwd = str(os.getcwdu())

bsubfile = ["#!/bin/sh", ""]
bsubnames = []
last_align = None

directory = ""
for i, mode in enumerate(PATTERN):
    iteration = i+1
    if iteration == 1:
        inputdb = INITIALGEOM
        inputdbdir = directory[:]
    else:
        inputdb = director + ".db"
        inputdbdir = directory[:]

    directory = "%s%02d/" % (DIRNAME, iteration)
    director = directory[:-1]
    os.system("rm -rf %s; mkdir %s" % (directory, directory))
    os.system("cp gatherBH_cfg.py %s" % directory)
    os.system("cp alignBH_cfg.py %s" % directory)

    bsubfile.append("cd %s" % directory)

    constraints = """echo \"\" > constraints_cff.py
"""
    if photogrammetry and (mode == "phipos" or mode == "phiz"):
        diskswitch = ""
        if disks: diskswitch = "--disks "

        constraints += """export ALIGNMENT_CONVERTXML=%(inputdb)s
cmsRun $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/python/convertToXML_global_cfg.py 
python $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/scripts/relativeConstraints.py %(inputdb)s_global.xml $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/data/Photogrammetry2007.%(mode)s PGFrame --scaleErrors %(photogrammetryScale)s %(diskswitch)s>> constraints_cff.py
""" % vars()

    elif photogrammetryOnlyholes and (mode == "phipos" or mode == "phiz"):
        diskswitch = ""
        if disks: diskswitch = "--disks "

        constraints += """export ALIGNMENT_CONVERTXML=%(inputdb)s
cmsRun $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/python/convertToXML_global_cfg.py 
python $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/scripts/relativeConstraints.py %(inputdb)s_global.xml $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/data/Photogrammetry2007_onlyOct2010holes.%(mode)s PGFrame --scaleErrors %(photogrammetryScale)s %(diskswitch)s>> constraints_cff.py
""" % vars()

    elif photogrammetryOnlyOnePerRing and (mode == "phipos" or mode == "phiz"):
        diskswitch = ""
        if disks: diskswitch = "--disks "

        constraints += """export ALIGNMENT_CONVERTXML=%(inputdb)s
cmsRun $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/python/convertToXML_global_cfg.py 
python $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/scripts/relativeConstraints.py %(inputdb)s_global.xml $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/data/Photogrammetry2007_onlyOnePerRing.%(mode)s PGFrame --scaleErrors %(photogrammetryScale)s %(diskswitch)s>> constraints_cff.py
""" % vars()

    if slm and (mode == "phipos" or "phiz"):
        diskswitch = ""
        if disks: diskswitch = "--disks "

        constraints += """export ALIGNMENT_CONVERTXML=%(inputdb)s
cmsRun $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/python/convertToXML_global_cfg.py
python $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/scripts/relativeConstraints.py %(inputdb)s_global.xml $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/data/SLM_test.%(mode)s SLMFrame --scaleErrors 1.0 %(diskswitch)s>> constraints_cff.py
""" % vars()

    if fillME11holes and (mode == "phipos" or mode == "phiz"):
        diskswitch = ""
        if disks: diskswitch = "--disks "

        constraints += """export ALIGNMENT_CONVERTXML=%(inputdb)s
cmsRun $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/python/convertToXML_global_cfg.py 
python $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/scripts/relativeConstraints.py %(inputdb)s_global.xml $ALIGNMENT_AFSDIR/Alignment/MuonAlignmentAlgorithms/data/CollisionsOct2010_ME11holes.%(mode)s TKFrame --scaleErrors 1. %(diskswitch)s>> constraints_cff.py
""" % vars()

    for jobnumber in range(njobs):
        gather_fileName = "%sgather%03d.sh" % (directory, jobnumber)
        if not options.inputInBlocks:
          inputfiles = " ".join(fileNames[jobnumber*stepsize:(jobnumber+1)*stepsize])
        else:
          inputfiles = " ".join(fileNamesBlocks[jobnumber])

        if len(inputfiles) > 0:
            file(gather_fileName, "w").write("""#/bin/sh
# %(commandline)s

export ALIGNMENT_CAFDIR=`pwd`

cd %(pwd)s

export SCRAM_ARCH=slc5_amd64_gcc434
echo INFO: SCRAM_ARCH $SCRAM_ARCH

eval `scramv1 run -sh`

source /afs/cern.ch/cms/caf/setup.sh
echo INFO: CMS_PATH $CMS_PATH
echo INFO: STAGE_SVCCLASS $STAGE_SVCCLASS
echo INFO: STAGER_TRACE $STAGER_TRACE

export ALIGNMENT_AFSDIR=`pwd`

export ALIGNMENT_INPUTFILES='%(inputfiles)s'
export ALIGNMENT_ITERATION=%(iteration)d
export ALIGNMENT_MODE=%(mode)s
export ALIGNMENT_JOBNUMBER=%(jobnumber)d
export ALIGNMENT_INPUTDB=%(inputdb)s
export ALIGNMENT_GLOBALTAG=%(globaltag)s
export ALIGNMENT_PHOTOGRAMMETRY='%(photogrammetry)s or %(photogrammetryOnlyholes)s or %(photogrammetryOnlyOnePerRing)s'
export ALIGNMENT_SLM=%(slm)s
export ALIGNMENT_FILLME11HOLES='%(fillME11holes)s'
export ALIGNMENT_DISKS=%(disks)s
export ALIGNMENT_minP=%(minP)s
export ALIGNMENT_minHitsPerChamber=%(minHitsPerChamber)s
export ALIGNMENT_maxdrdz=%(maxdrdz)s
export ALIGNMENT_maxRedChi2=%(maxRedChi2)s
export ALIGNMENT_fiducial=%(fiducial)s
export ALIGNMENT_useHitWeights=%(useHitWeights)s
export ALIGNMENT_truncateSlopeResid=%(truncateSlopeResid)s
export ALIGNMENT_truncateOffsetResid=%(truncateOffsetResid)s
export ALIGNMENT_combineME11=%(combineME11)s
export ALIGNMENT_useTrackWeights=%(useTrackWeights)s
export ALIGNMENT_errorFromRMS=%(errorFromRMS)s
export ALIGNMENT_minTracksPerOverlap=%(minTracksPerOverlap)s
export ALIGNMENT_slopeFromTrackRefit=%(slopeFromTrackRefit)s
export ALIGNMENT_minStationsInTrackRefits=%(minStationsInTrackRefits)s

cp -f %(directory)sgatherBH_cfg.py %(inputdbdir)s%(inputdb)s inertGlobalPositionRcd.db $ALIGNMENT_CAFDIR/
cd $ALIGNMENT_CAFDIR/

%(constraints)s

ls -l
cmsRun gatherBH_cfg.py
ls -l
cp -f *.tmp *.root $ALIGNMENT_AFSDIR/%(directory)s
""" % vars())
            os.system("chmod +x %s" % gather_fileName)
            bsubfile.append("echo %sgather%03d.sh" % (directory, jobnumber))

            if last_align is None: waiter = ""
            else: waiter = "-w \"ended(%s)\"" % last_align
            if options.big: queue = "cmscaf1nd"
            else: queue = "cmscaf1nh"
            
	    if user_mail: bsubfile.append("bsub -R \"type==SLC5_64\" -q %s -J \"%s_gather%03d\" -u %s %s gather%03d.sh" % (queue, director, jobnumber, user_mail, waiter, jobnumber))
            else: bsubfile.append("bsub -R \"type==SLC5_64\" -q %s -J \"%s_gather%03d\" %s gather%03d.sh" % (queue, director, jobnumber, waiter, jobnumber))
	    
            bsubnames.append("ended(%s_gather%03d)" % (director, jobnumber))

    file("%sconvert-db-to-xml_cfg.py" % directory, "w").write("""from Alignment.MuonAlignment.convertSQLitetoXML_cfg import *
process.PoolDBESSource.connect = \"sqlite_file:%(directory)s%(director)s.db\"
process.MuonGeometryDBConverter.outputXML.fileName = \"%(directory)s%(director)s.xml\"
process.MuonGeometryDBConverter.outputXML.relativeto = \"ideal\"
process.MuonGeometryDBConverter.outputXML.suppressDTChambers = True
process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = True
process.MuonGeometryDBConverter.outputXML.suppressDTLayers = True
process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = False
process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = True

process.MuonGeometryDBConverter.getAPEs = True
process.PoolDBESSource.toGet = cms.VPSet(
    cms.PSet(record = cms.string(\"DTAlignmentRcd\"), tag = cms.string(\"DTAlignmentRcd\")),
    cms.PSet(record = cms.string(\"DTAlignmentErrorExtendedRcd\"), tag = cms.string(\"DTAlignmentErrorExtendedRcd\")),
    cms.PSet(record = cms.string(\"CSCAlignmentRcd\"), tag = cms.string(\"CSCAlignmentRcd\")),
    cms.PSet(record = cms.string(\"CSCAlignmentErrorExtendedRcd\"), tag = cms.string(\"CSCAlignmentErrorExtendedRcd\")),
      )
""" % vars())

    constraints += """\ncp -f constraints_cff.py $ALIGNMENT_AFSDIR/%(directory)sconstraints_cff.py""" % vars()

    file("%salign.sh" % directory, "w").write("""#!/bin/sh
# %(commandline)s

export ALIGNMENT_CAFDIR=`pwd`

cd %(pwd)s

export SCRAM_ARCH=slc5_amd64_gcc434
echo INFO: SCRAM_ARCH $SCRAM_ARCH

eval `scramv1 run -sh`

source /afs/cern.ch/cms/caf/setup.sh
echo INFO: CMS_PATH $CMS_PATH
echo INFO: STAGE_SVCCLASS $STAGE_SVCCLASS
echo INFO: STAGER_TRACE $STAGER_TRACE

export ALIGNMENT_AFSDIR=`pwd`

export ALIGNMENT_ITERATION=%(iteration)d
export ALIGNMENT_MODE=%(mode)s
export ALIGNMENT_INPUTDB=%(inputdb)s
export ALIGNMENT_GLOBALTAG=%(globaltag)s
export ALIGNMENT_PHOTOGRAMMETRY='%(photogrammetry)s or %(photogrammetryOnlyholes)s or %(photogrammetryOnlyOnePerRing)s'
export ALIGNMENT_SLM=%(slm)s
export ALIGNMENT_FILLME11HOLES='%(fillME11holes)s'
export ALIGNMENT_DISKS=%(disks)s
export ALIGNMENT_minP=%(minP)s
export ALIGNMENT_minHitsPerChamber=%(minHitsPerChamber)s
export ALIGNMENT_maxdrdz=%(maxdrdz)s
export ALIGNMENT_maxRedChi2=%(maxRedChi2)s
export ALIGNMENT_fiducial=%(fiducial)s
export ALIGNMENT_useHitWeights=%(useHitWeights)s
export ALIGNMENT_truncateSlopeResid=%(truncateSlopeResid)s
export ALIGNMENT_truncateOffsetResid=%(truncateOffsetResid)s
export ALIGNMENT_combineME11=%(combineME11)s
export ALIGNMENT_useTrackWeights=%(useTrackWeights)s
export ALIGNMENT_errorFromRMS=%(errorFromRMS)s
export ALIGNMENT_minTracksPerOverlap=%(minTracksPerOverlap)s
export ALIGNMENT_slopeFromTrackRefit=%(slopeFromTrackRefit)s
export ALIGNMENT_minStationsInTrackRefits=%(minStationsInTrackRefits)s

cp -f %(directory)salignBH_cfg.py %(directory)sconvert-db-to-xml_cfg.py %(inputdbdir)s%(inputdb)s %(directory)s*.tmp inertGlobalPositionRcd.db $ALIGNMENT_CAFDIR/
cd $ALIGNMENT_CAFDIR/
export ALIGNMENT_ALIGNMENTTMP=`ls alignment*.tmp`

%(constraints)s

ls -l
cmsRun alignBH_cfg.py
cp -f report.py $ALIGNMENT_AFSDIR/%(directory)s%(director)s_report.py
cp -f outputdb.db $ALIGNMENT_AFSDIR/%(directory)s%(director)s.db
cp -f plotting.root $ALIGNMENT_AFSDIR/%(directory)s%(director)s.root

cd $ALIGNMENT_AFSDIR
cmsRun %(directory)sconvert-db-to-xml_cfg.py

export ALIGNMENT_PLOTTINGTMP=`ls %(directory)splotting0*.root 2> /dev/null`
if [ \"zzz$ALIGNMENT_PLOTTINGTMP\" != \"zzz\" ]; then
  hadd -f1 %(directory)s%(director)s_plotting.root %(directory)splotting0*.root
  #if [ $? == 0 ] && [ \"$ALIGNMENT_CLEANUP\" == \"True\" ]; then rm %(directory)splotting0*.root; fi
fi

""" % vars())
    os.system("chmod +x %salign.sh" % directory)

    bsubfile.append("echo %salign.sh" % directory)
    
    if user_mail: bsubfile.append("bsub -R \"type==SLC5_64\" -q cmscaf1nd -J \"%s_align\" -u %s -w \"%s\" align.sh" % (director, user_mail, " && ".join(bsubnames)))
    else: bsubfile.append("bsub -R \"type==SLC5_64\" -q cmscaf1nd -J \"%s_align\" -w \"%s\" align.sh" % (director, " && ".join(bsubnames)))
    
    bsubfile.append("cd ..")
    bsubnames = []
    last_align = "%s_align" % director

bsubfile.append("")
file(options.submitJobs, "w").write("\n".join(bsubfile))
os.system("chmod +x %s" % options.submitJobs)
