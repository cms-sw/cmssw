from optparse import OptionParser
import sys, os, operator
from pprint import pprint
import filecmp

from dict2023Geometry import *

# define global
detectorVersionDefault = 999

def generateGeom(detectorTuple, options):
    doTest = bool(options.doTest)

    detectorVersion = "D"+str(options.detectorVersionManual)
    # reverse dict search if overall D# specified
    if options.v_detector>0:
        detectorVersion = "D"+str(options.v_detector)
        if detectorVersion in detectorVersionDict.values():
            detectorTuple = detectorVersionDict.keys()[detectorVersionDict.values().index(detectorVersion)]
        else:
            print "Unknown detector "+detectorVersion
            sys.exit(1)
    elif detectorTuple in detectorVersionDict.keys():
        detectorVersion = detectorVersionDict[detectorTuple]
    else:
        if not doTest: print "Detector "+str(detectorTuple)+" not found in dictionary, using "+("default" if options.detectorVersionManual==detectorVersionDefault else "provided")+" version number "+str(detectorVersion)

    # check for deprecation
    if detectorVersion in deprecatedDets:
        print "Error: "+detectorVersion+" is deprecated and cannot be used."
        sys.exit(1)
    for subdet in detectorTuple:
        if subdet in deprecatedSubdets:
            print "Error: "+subdet+" is deprecated and cannot be used."
            sys.exit(1)
        
    # create output files
    xmlName = "cmsExtendedGeometry2023"+detectorVersion+"XML_cfi.py"
    simName = "GeometryExtended2023"+detectorVersion+"_cff.py"
    recoName = "GeometryExtended2023"+detectorVersion+"Reco_cff.py"

    # check directories
    CMSSWBASE = os.getenv("CMSSW_BASE")
    CMSSWRELBASE = os.getenv("CMSSW_RELEASE_BASE")
    if CMSSWBASE is None: CMSSWBASE = ""
    xmlDir = os.path.join(CMSSWBASE,"src","Geometry","CMSCommonData","python")
    simrecoDir = os.path.join(CMSSWBASE,"src","Configuration","Geometry","python")
    if doTest:
        if not os.path.isdir(xmlDir):
            xmlDir = os.path.join(CMSSWRELBASE,"src","Geometry","CMSCommonData","python")
    else:
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

    from Configuration.StandardSequences.GeometryConf import GeometryConf
    if not doTest: # todo: include these in unit test somehow
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
        if not 'Extended2023'+detectorVersion in GeometryConf.keys():
            print "Please add this line in Configuration/StandardSequences/python/GeometryConf.py:"
            print "    'Extended2023"+detectorVersion+"' : 'Extended2023"+detectorVersion+",Extended2023"+detectorVersion+"Reco',"

    errorList = []

    if doTest:
        # tests for Configuration/Geometry
        if not filecmp.cmp(simName,os.path.join(simrecoDir,simName)):
            errorList.append(simName+" differs");
        if not filecmp.cmp(recoName,os.path.join(simrecoDir,recoName)):
            errorList.append(recoName+" differs");
        # test for Configuration/StandardSequences
        if not 'Extended2023'+detectorVersion in GeometryConf.keys():
            errorList.append('Extended2023'+detectorVersion+" missing from GeometryConf")
        # test for Geometry/CMSCommonData
        if not filecmp.cmp(xmlName,os.path.join(xmlDir,xmlName)):
            errorList.append(xmlName+" differs");
    return errorList
        
if __name__ == "__main__":
    # define options
    parser = OptionParser()
    for aDict in allDicts:
        parser.add_option("-"+aDict["abbrev"],"--"+aDict["name"],dest="v_"+aDict["name"],default=aDict["default"],help="version for "+aDict["name"]+" (default = %default)")
    parser.add_option("-V","--version",dest="detectorVersionManual",default=detectorVersionDefault,help="manual detector version number (default = %default)")
    parser.add_option("-D","--detector",dest="v_detector",default=0,help="version for whole detector, ignored if 0, overrides subdet versions otherwise (default = %default)")
    parser.add_option("-l","--list",dest="doList",default=False,action="store_true",help="list known detector versions and exit (default = %default)")
    parser.add_option("-t","--test",dest="doTest",default=False,action="store_true",help="enable unit test mode (default = %default)")
    (options, args) = parser.parse_args()
    
    # check options
    if options.doList and not options.doTest:
        pprint(sorted(detectorVersionDict.items(),key=operator.itemgetter(1)))
        sys.exit(0)
    elif options.doTest:
        # list of errors
        errorList = []
        # run all known possibilities
        for detectorTuple in detectorVersionDict:
            errorTmp = generateGeom(detectorTuple,options)
            errorList.extend(errorTmp)
        if len(errorList)>0:
            print '\n'.join([anError for anError in errorList])
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        detectorTuple = tuple([aDict["abbrev"]+str(getattr(options,"v_"+aDict["name"])) for aDict in allDicts])
        generateGeom(detectorTuple,options)
        sys.exit(0)
    