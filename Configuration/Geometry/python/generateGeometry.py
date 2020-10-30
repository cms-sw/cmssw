from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter, RawDescriptionHelpFormatter
import sys, os, operator
from pprint import pprint
import filecmp

# convenience definition
# (from ConfigArgParse)
class ArgumentDefaultsRawHelpFormatter(
    ArgumentDefaultsHelpFormatter,
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter):
    """HelpFormatter that adds default values AND doesn't do line-wrapping"""
pass

class GeometryGenerator(object):
    def __init__(self, scriptName, detectorVersionDefault, detectorPrefix, detectorYear, maxSections, allDicts, detectorVersionDict, deprecatedDets = None, deprecatedSubdets = None, detectorVersionType = int):
        self.scriptName = scriptName
        self.detectorVersionDefault = detectorVersionDefault
        self.detectorPrefix = detectorPrefix
        self.detectorYear = detectorYear
        self.maxSections = maxSections
        self.allDicts = allDicts
        self.detectorVersionDict = detectorVersionDict
        self.deprecatedDets = deprecatedDets
        self.deprecatedSubdets = deprecatedSubdets
        self.detectorVersionType = detectorVersionType

    def generateGeom(self, detectorTuple, args):
        detectorVersion = self.detectorPrefix+str(args.detectorVersionManual)
        # reverse dict search if overall D# specified
        if args.v_detector>0:
            detectorVersion = self.detectorPrefix+str(args.v_detector)
            if detectorVersion in self.detectorVersionDict.values():
                detectorTuple = self.detectorVersionDict.keys()[self.detectorVersionDict.values().index(detectorVersion)]
            else:
                print("Unknown detector "+detectorVersion)
                sys.exit(1)
        elif detectorTuple in self.detectorVersionDict.keys():
            detectorVersion = self.detectorVersionDict[detectorTuple]
        else:
            if not args.doTest: print("Detector "+str(detectorTuple)+" not found in dictionary, using "+("default" if args.detectorVersionManual==self.detectorVersionDefault else "provided")+" version number "+str(detectorVersion))

        # check for deprecation
        if self.deprecatedDets is not None and detectorVersion in self.deprecatedDets:
            print("Error: "+detectorVersion+" is deprecated and cannot be used.")
            sys.exit(1)
        if self.deprecatedSubdets is not None:
            for subdet in detectorTuple:
                if subdet in self.deprecatedSubdets:
                    print("Error: "+subdet+" is deprecated and cannot be used.")
                    sys.exit(1)

        # create output files
        xmlName = "cmsExtendedGeometry"+self.detectorYear+detectorVersion+"XML_cfi.py"
        xmlDD4hepName = "cmsExtendedGeometry"+self.detectorYear+detectorVersion+".xml"
        simName = "GeometryExtended"+self.detectorYear+detectorVersion+"_cff.py"
        simDD4hepName = "GeometryDD4hepExtended"+self.detectorYear+detectorVersion+"_cff.py"
        recoName = "GeometryExtended"+self.detectorYear+detectorVersion+"Reco_cff.py"
        recoDD4hepName = "GeometryDD4hepExtended"+self.detectorYear+detectorVersion+"Reco_cff.py"

        # check directories
        CMSSWBASE = os.getenv("CMSSW_BASE")
        CMSSWRELBASE = os.getenv("CMSSW_RELEASE_BASE")
        if CMSSWBASE is None: CMSSWBASE = ""
        xmlDir = os.path.join(CMSSWBASE,"src","Geometry","CMSCommonData","python")
        xmlDD4hepDir = os.path.join(CMSSWBASE,"src","Geometry","CMSCommonData","data","dd4hep")
        simrecoDir = os.path.join(CMSSWBASE,"src","Configuration","Geometry","python")
        simrecoDD4hepDir = os.path.join(CMSSWBASE,"src","Configuration","Geometry","python")
        if args.doTest:
            if not os.path.isdir(xmlDir):
                xmlDir = os.path.join(CMSSWRELBASE,"src","Geometry","CMSCommonData","python")
                xmlDD4hepDir = os.path.join(CMSSWRELBASE,"src","Geometry","CMSCommonData","data","dd4hep")
        else:
            mvCommands = ""
            if not os.path.isdir(xmlDir):
                mvCommands += "mv "+xmlName+" "+xmlDir+"/\n"
            else:
                xmlName = os.path.join(xmlDir,xmlName)
            if not os.path.isdir(xmlDD4hepDir):
                mvCommands += "mv "+xmlDD4hepName+" "+xmlDD4hepDir+"/\n"
            else:
                xmlDD4hepName = os.path.join(xmlDD4hepDir,xmlDD4hepName)
            if not os.path.isdir(simrecoDir):
                mvCommands += "mv "+simName+" "+simrecoDir+"/\n"
                mvCommands += "mv "+recoName+" "+simrecoDir+"/\n"
            else:
                simName = os.path.join(simrecoDir,simName)
                recoName = os.path.join(simrecoDir,recoName)
            if not os.path.isdir(simrecoDD4hepDir):
                mvCommands += "mv "+simDD4hepName+" "+simrecoDD4hepDir+"/\n"
                mvCommands += "mv "+recoDD4hepName+" "+simrecoDD4hepDir+"/\n"
            else:
                simDD4hepName = os.path.join(simrecoDD4hepDir,simDD4hepName)
                recoDD4hepName = os.path.join(simrecoDD4hepDir,recoDD4hepName)
            if len(mvCommands)>0:
                print("Warning: some geometry packages not checked out.\nOnce they are available, please execute the following commands manually:\n"+mvCommands)

        # open files
        xmlFile = open(xmlName,'w')
        xmlDD4hepFile = open(xmlDD4hepName,'w')
        simFile = open(simName,'w')
        simDD4hepFile = open(simDD4hepName,'w')
        recoFile = open(recoName,'w')
        recoDD4hepFile = open(recoDD4hepName,'w')

        # common preamble
        preamble = "import FWCore.ParameterSet.Config as cms"+"\n"+"\n"
        preamble += "# This config was generated automatically using "+self.scriptName+"\n"
        preamble += "# If you notice a mistake, please update the generating script, not just this config"+"\n"+"\n"

        # create XML config
        xmlFile.write(preamble)
        # extra preamble
        xmlFile.write("XMLIdealGeometryESSource = cms.ESSource(\"XMLIdealGeometryESSource\","+"\n")
        xmlFile.write("    geomXMLFiles = cms.vstring("+"\n")
        for section in range(1,self.maxSections+1):
            # midamble
            if section==2:
                xmlFile.write("    )+"+"\n"+"    cms.vstring("+"\n")
            for iDict,aDict in enumerate(self.allDicts):
                if section in aDict[detectorTuple[iDict]].keys():
                    xmlFile.write('\n'.join([ "        '"+aLine+"'," for aLine in aDict[detectorTuple[iDict]][section] ])+"\n")
        # postamble
        xmlFile.write("    ),"+"\n"+"    rootNodeName = cms.string('cms:OCMS')"+"\n"+")"+"\n")
        xmlFile.close()

        # create DD4hep XML config
        xmlDD4hepFile.write("<?xml version=\"1.0\"?>\n"+
                            "<DDDefinition>\n"+
                            "  <open_geometry/>\n"+
                            "  <close_geometry/>\n"+
                            "\n"+
                            "  <IncludeSection>\n")
        for section in range(1,self.maxSections+1):
            # midamble
            for iDict,aDict in enumerate(self.allDicts):
                if section in aDict[detectorTuple[iDict]].keys():
                    xmlDD4hepFile.write('\n'.join([ "    <Include ref='"+aLine+"'/>" for aLine in aDict[detectorTuple[iDict]][section] ])+"\n")
        # postamble
        xmlDD4hepFile.write("  </IncludeSection>\n"+
                            "</DDDefinition>"+"\n")
        xmlDD4hepFile.close()

        # create sim config
        simFile.write(preamble)
        # always need XML
        simFile.write("from Geometry.CMSCommonData."+os.path.basename(xmlName).replace(".py","")+" import *"+"\n")
        for iDict,aDict in enumerate(self.allDicts):
            if "sim" in aDict[detectorTuple[iDict]].keys():
                simFile.write('\n'.join([ aLine for aLine in aDict[detectorTuple[iDict]]["sim"] ])+"\n")
        simFile.close()

        # create simDD4hep config
        simDD4hepFile.write(preamble)
        # always need XML
        simDD4hepFile.write("from Configuration.Geometry.GeometryDD4hep_cff"+" import *"+"\n")
        simDD4hepFile.write("DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath(\"Geometry/CMSCommonData/data/dd4hep/"+os.path.basename(xmlDD4hepName)+"\")\n\n")
        for iDict,aDict in enumerate(self.allDicts):
            if "sim" in aDict[detectorTuple[iDict]].keys():
                simDD4hepFile.write('\n'.join([ aLine for aLine in aDict[detectorTuple[iDict]]["sim"] ])+"\n")
        simDD4hepFile.close()

        # create reco config
        recoFile.write(preamble)
        # always need sim
        recoFile.write("from Configuration.Geometry."+os.path.basename(simName).replace(".py","")+" import *"+"\n\n")
        for iDict,aDict in enumerate(self.allDicts):
            if "reco" in aDict[detectorTuple[iDict]].keys():
               recoFile.write("# "+aDict["name"]+"\n")
               recoFile.write('\n'.join([ aLine for aLine in aDict[detectorTuple[iDict]]["reco"] ])+"\n\n")
        recoFile.close()

        # create recoDD4hep config
        recoDD4hepFile.write(preamble)
        # always need sim
        recoDD4hepFile.write("from Configuration.Geometry."+os.path.basename(simDD4hepName).replace(".py","")+" import *"+"\n\n")
        for iDict,aDict in enumerate(self.allDicts):
            if "reco" in aDict[detectorTuple[iDict]].keys():
                recoDD4hepFile.write("# "+aDict["name"]+"\n")
                recoDD4hepFile.write('\n'.join([ aLine for aLine in aDict[detectorTuple[iDict]]["reco"] ])+"\n\n")
        recoDD4hepFile.close()

        from Configuration.StandardSequences.GeometryConf import GeometryConf
        if not args.doTest: # todo: include these in unit test somehow
            # specify Era customizations
            # must be checked manually in:
            # Configuration/StandardSequences/python/Eras.py
            # Configuration/Eras/python/
            # Configuration/PyReleaseValidation/python/upgradeWorkflowComponents.py (workflow definitions)
            eraLine = ""
            eraLineItems = []
            for iDict,aDict in enumerate(self.allDicts):
                if "era" in aDict[detectorTuple[iDict]].keys():
                    eraLineItems.append(aDict[detectorTuple[iDict]]["era"])
            eraLine += ", ".join([ eraLineItem for eraLineItem in eraLineItems ])
            print("The Era for this detector should contain:")
            print(eraLine)

            # specify GeometryConf
            if not 'Extended'+self.detectorYear+detectorVersion in GeometryConf.keys():
                print("Please add this line in Configuration/StandardSequences/python/GeometryConf.py:")
                print("    'Extended"+self.detectorYear+detectorVersion+"' : 'Extended"+self.detectorYear+detectorVersion+",Extended"+self.detectorYear+detectorVersion+"Reco',")

        errorList = []

        if args.doTest:
            # tests for Configuration/Geometry
            simFile = os.path.join(simrecoDir,simName)
            if not os.path.isfile(simFile):
                errorList.append(simName+" missing")
            elif not filecmp.cmp(simName,simFile):
                errorList.append(simName+" differs")
            simDD4hepFile = os.path.join(simrecoDD4hepDir,simDD4hepName)
            if not os.path.isfile(simDD4hepFile):
                errorList.append(simDD4hepName+" missing")
            elif not filecmp.cmp(simDD4hepName,simDD4hepFile):
                errorList.append(simDD4hepName+" differs")
            recoFile = os.path.join(simrecoDir,recoName)
            if not os.path.isfile(recoFile):
                errorList.append(recoName+" missing")
            elif not filecmp.cmp(recoName,recoFile):
                errorList.append(recoName+" differs")
            recoDD4hepFile = os.path.join(simrecoDD4hepDir,recoDD4hepName)
            if not os.path.isfile(recoDD4hepFile):
                errorList.append(recoDD4hepName+" missing")
            elif not filecmp.cmp(recoDD4hepName,recoDD4hepFile):
                errorList.append(recoDD4hepName+" differs")
            # test for Configuration/StandardSequences
            if not 'Extended'+self.detectorYear+detectorVersion in GeometryConf.keys():
                errorList.append('Extended'+self.detectorYear+detectorVersion+" missing from GeometryConf")
            # test for Geometry/CMSCommonData
            xmlFile = os.path.join(xmlDir,xmlName)
            if not os.path.isfile(xmlFile):
                errorList.append(xmlName+" missing")
            elif not filecmp.cmp(xmlName,xmlFile):
                errorList.append(xmlName+" differs")
            # test for dd4hep xml
            xmlDD4hepFile = os.path.join(xmlDD4hepDir,xmlDD4hepName)
            if not os.path.exists(xmlDD4hepFile):
                errorList.append(xmlDD4hepName+" differs")
            elif not filecmp.cmp(xmlDD4hepName,xmlDD4hepFile):
                errorList.append(xmlDD4hepName+" differs")
        return errorList

    def run(self):
        # define options
        parser = ArgumentParser(formatter_class=ArgumentDefaultsRawHelpFormatter)
        for aDict in self.allDicts:
            parser.add_argument("-"+aDict["abbrev"], "--"+aDict["name"], dest="v_"+aDict["name"], default=aDict["default"], type=int, help="version for "+aDict["name"])
        parser.add_argument("-V", "--version", dest="detectorVersionManual", default=self.detectorVersionDefault, type=int, help="manual detector version number")
        parser.add_argument("-D", "--detector", dest="v_detector", default=0, type=self.detectorVersionType, help="version for whole detector, ignored if 0, overrides subdet versions otherwise")
        parser.add_argument("-l", "--list", dest="doList", default=False, action="store_true", help="list known detector versions and exit")
        parser.add_argument("-t", "--test", dest="doTest", default=False, action="store_true", help="enable unit test mode")
        args = parser.parse_args()

        # check options
        if args.doList and not args.doTest:
            pprint(sorted(self.detectorVersionDict.items(),key=operator.itemgetter(1)))
            sys.exit(0)
        elif args.doTest:
            # list of errors
            errorList = []
            # run all known possibilities
            for detectorTuple in self.detectorVersionDict:
                errorTmp = self.generateGeom(detectorTuple,args)
                errorList.extend(errorTmp)
            if len(errorList)>0:
                print('\n'.join([anError for anError in errorList]))
                sys.exit(1)
            else:
                sys.exit(0)
        else:
            detectorTuple = tuple([aDict["abbrev"]+str(getattr(args,"v_"+aDict["name"])) for aDict in self.allDicts])
            self.generateGeom(detectorTuple,args)
            sys.exit(0)
    
