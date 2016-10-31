from optparse import OptionParser
import sys, os, operator
from pprint import pprint

from dict2023Geometry import *

# define options
detectorVersionDefault = 999
parser = OptionParser()
for aDict in allDicts:
    parser.add_option("-"+aDict["abbrev"],"--"+aDict["name"],dest="v_"+aDict["name"],default=1,help="version for "+aDict["name"]+" (default = %default)")
parser.add_option("-D","--detector",dest="detectorVersionManual",default=detectorVersionDefault,help="manual detector version number (default = %default)")
parser.add_option("-L","--list",dest="doList",default=False,action="store_true",help="list known detector versions and exit (default = %default)")    
(options, args) = parser.parse_args()

# check options
if options.doList:
    pprint(sorted(detectorVersionDict.items(),key=operator.itemgetter(1)))
    sys.exit()

detectorTuple = tuple([aDict["abbrev"]+str(getattr(options,"v_"+aDict["name"])) for aDict in allDicts])

detectorVersion = "D"+str(options.detectorVersionManual)
if detectorTuple in detectorVersionDict.keys():
    detectorVersion = detectorVersionDict[detectorTuple]
else:
    print "Detector "+str(detectorTuple)+" not found in dictionary, using "+("default" if options.detectorVersionManual==detectorVersionDefault else "provided")+" version number "+str(detectorVersion)

# create output files
xmlName = "cmsExtendedGeometry2023"+detectorVersion+"XML_cfi.py"
simName = "GeometryExtended2023"+detectorVersion+"_cff.py"
recoName = "GeometryExtended2023"+detectorVersion+"Reco_cff.py"

# check directories
CMSSWBASE = os.getenv("CMSSW_BASE")
if CMSSWBASE is None: CMSSWBASE = ""
xmlDir = os.path.join(CMSSWBASE,"src","Geometry","CMSCommonData","python")
simrecoDir = os.path.join(CMSSWBASE,"src","Configuration","Geometry","python")
mvCommands = ""
if not os.path.isdir(xmlDir):
    mvCommands += "mv "+xmlName+" "+xmlDir+"/\n"
else:
    xmlName = os.path.join(xmlDir,xmlName)
if not os.path.isdir(simrecoDir):
    mvCommands += "mv "+simName+" "+simrecoDir+"/\n"
    mvCommands += "mv "+recoName+" "+simrecoDir+"/\n"
else:
    simName = os.path.join(simrecoDir,simName)
    recoName = os.path.join(simrecoDir,recoName)
if len(mvCommands)>0:
    print "Warning: some geometry packages not checked out.\nOnce they are available, please execute the following commands manually:\n"+mvCommands

# open files
xmlFile = open(xmlName,'w')
simFile = open(simName,'w')
recoFile = open(recoName,'w')

# common preamble
preamble = "import FWCore.ParameterSet.Config as cms"+"\n"+"\n"
preamble += "# This config was generated automatically using generate2023Geometry.py"+"\n"
preamble += "# If you notice a mistake, please update the generating script, not just this config"+"\n"+"\n"

# create XML config
xmlFile.write(preamble)
# extra preamble
xmlFile.write("XMLIdealGeometryESSource = cms.ESSource(\"XMLIdealGeometryESSource\","+"\n")
xmlFile.write("    geomXMLFiles = cms.vstring("+"\n")
for section in range(1,maxsections+1):
    # midamble
    if section==2:
        xmlFile.write("    )+"+"\n"+"    cms.vstring("+"\n")
    for iDict,aDict in enumerate(allDicts):
        if section in aDict[detectorTuple[iDict]].keys():
            xmlFile.write('\n'.join([ "        '"+aLine+"'," for aLine in aDict[detectorTuple[iDict]][section] ])+"\n")
# postamble
xmlFile.write("    ),"+"\n"+"    rootNodeName = cms.string('cms:OCMS')"+"\n"+")"+"\n")
xmlFile.close()

# create sim config
simFile.write(preamble)
# always need XML
simFile.write("from Geometry.CMSCommonData."+os.path.basename(xmlName).replace(".py","")+" import *"+"\n")
for iDict,aDict in enumerate(allDicts):
    if "sim" in aDict[detectorTuple[iDict]].keys():
        simFile.write('\n'.join([ aLine for aLine in aDict[detectorTuple[iDict]]["sim"] ])+"\n")
simFile.close()

# create reco config
recoFile.write(preamble)
# always need sim
recoFile.write("from Configuration.Geometry."+os.path.basename(simName).replace(".py","")+" import *"+"\n\n")
for iDict,aDict in enumerate(allDicts):
    if "reco" in aDict[detectorTuple[iDict]].keys():
        recoFile.write("# "+aDict["name"]+"\n")
        recoFile.write('\n'.join([ aLine for aLine in aDict[detectorTuple[iDict]]["reco"] ])+"\n\n")
recoFile.close()

# specify Era customizations
# must be checked manually in:
# Configuration/StandardSequences/python/Eras.py
# Configuration/Eras/python/
# Configuration/PyReleaseValidation/python/upgradeWorkflowComponents.py (workflow definitions)
eraLine = ""
eraLineItems = []
for iDict,aDict in enumerate(allDicts):
    if "era" in aDict[detectorTuple[iDict]].keys():
        eraLineItems.append(aDict[detectorTuple[iDict]]["era"])
eraLine += ", ".join([ eraLineItem for eraLineItem in eraLineItems ])
print "The Era for this detector should contain:"
print eraLine

# specify GeometryConf
from Configuration.StandardSequences.GeometryConf import GeometryConf
if not 'Extended2023'+detectorVersion in GeometryConf.keys():
    print "Please add this line in Configuration/StandardSequences/python/GeometryConf.py:"
    print "    'Extended2023"+detectorVersion+"' : 'Extended2023"+detectorVersion+",Extended2023"+detectorVersion+"Reco',"
