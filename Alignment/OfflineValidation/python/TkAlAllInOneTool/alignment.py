import collections
import os
import re

import configTemplates
from helperFunctions import conddb, parsecolor, parsestyle, replaceByMap, clean_name
from TkAlExceptions import AllInOneError
import six

class Alignment(object):
    condShorts = {
        "TrackerAlignmentErrorExtendedRcd": {
            "zeroAPE_phase0": {
                "connectString":("frontier://FrontierProd"
                                         "/CMS_CONDITIONS"),
                "tagName": "TrackerIdealGeometryErrorsExtended210_mc",
                "labelName": ""
            },
            "zeroAPE_phase1": {
                "connectString":("frontier://FrontierProd"
                                         "/CMS_CONDITIONS"),
                "tagName": "TrackerAlignmentErrorsExtended_Upgrade2017_design_v0",
                "labelName": ""
            },
        },
        "TrackerSurfaceDeformationRcd": {
            "zeroDeformations": {
                "connectString":("frontier://FrontierProd"
                                         "/CMS_CONDITIONS"),
                "tagName": "TrackerSurfaceDeformations_zero",
                "labelName": ""
            },
        },
    }
    def __init__(self, name, config, runGeomComp = "1"):
        section = "alignment:%s"%name
        if not config.has_section( section ):
            raise AllInOneError("section %s not found. Please define the "
                                  "alignment!"%section)
        config.checkInput(section,
                          knownSimpleOptions = ['globaltag', 'style', 'color', 'title', 'mp', 'mp_alignments', 'mp_deformations', 'mp_APEs', 'hp', 'hp_alignments', 'hp_deformations', 'sm', 'sm_alignments', 'sm_deformations'],
                          knownKeywords = ['condition'])
        self.name = clean_name(name)
        if config.exists(section,"title"):
            self.title = config.get(section,"title")
        else:
            self.title = self.name
        if (int(runGeomComp) != 1):
            self.name += "_run" + runGeomComp
            self.title += " run " + runGeomComp
        if "|" in self.title or "," in self.title or '"' in self.title:
            msg = "The characters '|', '\"', and ',' cannot be used in the alignment title!"
            raise AllInOneError(msg)
        self.runGeomComp = runGeomComp
        self.globaltag = config.get( section, "globaltag" )
        self.conditions = self.__getConditions( config, section )

        self.color = config.get(section,"color")
        self.style = config.get(section,"style")

        self.color = str(parsecolor(self.color))
        self.style = str(parsestyle(self.style))

    def __shorthandExists(self, theRcdName, theShorthand):
        """Method which checks, if `theShorthand` is a valid shorthand for the
        given `theRcdName`.

        Arguments:
        - `theRcdName`: String which specifies the database record.
        - `theShorthand`: String which specifies the shorthand to check.
        """

        if (theRcdName in self.condShorts) and \
                (theShorthand in self.condShorts[theRcdName]):
            return True
        else:
            return False

    def __getConditions( self, theConfig, theSection ):
        conditions = []
        for option in theConfig.options( theSection ):
            if option in ("mp", "mp_alignments", "mp_deformations", "mp_APEs", "hp", "hp_alignments", "hp_deformations", "sm", "sm_alignments", "sm_deformations"):
                matches = [re.match(_, option) for _ in ("^(..)$", "^(..)_alignments$", "^(..)_deformations$", "^(..)_APEs$")]
                assert sum(bool(_) for _ in matches) == 1, option
                condPars = theConfig.get(theSection, option).split(",")
                condPars = [_.strip() for _ in condPars]
                if matches[0]:
                    alignments = True
                    deformations = True
                    APEs = {"hp": False, "mp": True}[option]
                elif matches[1]:
                    alignments = True
                    deformations = False
                    APEs = False
                    option = matches[1].group(1)
                elif matches[2]:
                    alignments = False
                    deformations = True
                    APEs = False
                    option = matches[2].group(1)
                elif matches[3]:
                    alignments = False
                    deformations = False
                    APEs = True
                    option = matches[3].group(1)
                else:
                    assert False

                if option == "mp":
                    if len(condPars) == 1:
                        number, = condPars
                        jobm = None
                    elif len(condPars) == 2:
                        number, jobm = condPars
                    else:
                        raise AllInOneError("Up to 2 arguments accepted for {} (job number, and optionally jobm index)".format(option))

                    folder = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/{}{}/".format(option, number)
                    if not os.path.exists(folder):
                        raise AllInOneError(folder+" does not exist.")
                    folder = os.path.join(folder, "jobData")
                    jobmfolders = set()
                    if jobm is None:
                        for filename in os.listdir(folder):
                            if re.match("jobm([0-9]*)", filename) and os.path.isdir(os.path.join(folder, filename)):
                                jobmfolders.add(filename)
                        if len(jobmfolders) == 0:
                            raise AllInOneError("No jobm or jobm(number) folder in {}".format(folder))
                        elif len(jobmfolders) == 1:
                            folder = os.path.join(folder, jobmfolders.pop())
                        else:
                            raise AllInOneError(
                                                "Multiple jobm or jobm(number) folders in {}\n".format(folder)
                                                + ", ".join(jobmfolders) + "\n"
                                                + "Please specify 0 for jobm, or a number for one of the others."
                                               )
                    elif jobm == "0":
                        folder = os.path.join(folder, "jobm")
                        if os.path.exists(folder + "0"):
                            raise AllInOneError("Not set up to handle a folder named jobm0")
                    else:
                        folder = os.path.join(folder, "jobm{}".format(jobm))

                    dbfile = os.path.join(folder, "alignments_MP.db")
                    if not os.path.exists(dbfile):
                        raise AllInOneError("No file {}.  Maybe your alignment folder is corrupted, or maybe you specified the wrong jobm?".format(dbfile))

                elif option in ("hp", "sm"):
                    if len(condPars) == 1:
                        number, = condPars
                        iteration = None
                    elif len(condPars) == 2:
                        number, iteration = condPars
                    else:
                        raise AllInOneError("Up to 2 arguments accepted for {} (job number, and optionally iteration)".format(option))
                    folder = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/{}{}".format(option, number)
                    if not os.path.exists(folder):
                        raise AllInOneError(folder+" does not exist.")
                    if iteration is None:
                        for filename in os.listdir(folder):
                            match = re.match("alignments_iter([0-9]*).db", filename)
                            if match:
                                if iteration is None or int(match.group(1)) > iteration:
                                    iteration = int(match.group(1))
                        if iteration is None:
                            raise AllInOneError("No alignments in {}".format(folder))
                    dbfile = os.path.join(folder, "alignments_iter{}.db".format(iteration))
                    if not os.path.exists(dbfile):
                        raise AllInOneError("No file {}.".format(dbfile))

                    if "\nDeformations" not in conddb("--db", dbfile, "listTags"):
                        deformations = False  #so that hp = XXXX works whether or not deformations were aligned
                        if not alignments:    #then it's specified with hp_deformations, which is a mistake
                            raise AllInOneError("{}{} has no deformations".format(option, number))

                else:
                    assert False, option

                if alignments:
                    conditions.append({"rcdName": "TrackerAlignmentRcd",
                                       "connectString": "sqlite_file:"+dbfile,
                                       "tagName": "Alignments",
                                       "labelName": ""})
                if deformations:
                    conditions.append({"rcdName": "TrackerSurfaceDeformationRcd",
                                       "connectString": "sqlite_file:"+dbfile,
                                       "tagName": "Deformations",
                                       "labelName": ""})
                if APEs:
                    conditions.append({"rcdName": "TrackerAlignmentErrorExtendedRcd",
                                       "connectString": "sqlite_file:"+dbfile,
                                       "tagName": "AlignmentErrorsExtended",
                                       "labelName": ""})

            elif option.startswith( "condition " ):
                rcdName = option.split( "condition " )[1]
                condPars = theConfig.get( theSection, option ).split( "," )
                if len(condPars) == 1:
                    if len(condPars[0])==0:
                        msg = ("In section [%s]: '%s' is used with too few "
                               "arguments. A connect_string and a tag are "
                               "required!"%(theSection, option))
                        raise AllInOneError(msg)
                    elif self.__shorthandExists(rcdName, condPars[0]):
                        shorthand = condPars[0]
                        condPars = [
                            self.condShorts[rcdName][shorthand]["connectString"],
                            self.condShorts[rcdName][shorthand]["tagName"],
                            self.condShorts[rcdName][shorthand]["labelName"]]
                    elif rcdName == "TrackerAlignmentErrorExtendedRcd" and condPars[0] == "zeroAPE":
                        raise AllInOneError("Please specify either zeroAPE_phase0 or zeroAPE_phase1")
                        #can probably make zeroAPE an alias of zeroAPE_phase1 at some point,
                        #but not sure if now is the time
                    else:
                        msg = ("In section [%s]: '%s' is used with '%s', "
                               "which is an unknown shorthand for '%s'. Either "
                               "provide at least a connect_string and a tag or "
                               "use a known shorthand.\n"
                               %(theSection, option, condPars[0], rcdName))
                        if rcdName in self.condShorts:
                            msg += "Known shorthands for '%s':\n"%(rcdName)
                            theShorts = self.condShorts[rcdName]
                            knownShorts = [("\t"+key+": "
                                            +theShorts[key]["connectString"]+","
                                            +theShorts[key]["tagName"]+","
                                            +theShorts[key]["labelName"]) \
                                               for key in theShorts]
                            msg+="\n".join(knownShorts)
                        else:
                            msg += ("There are no known shorthands for '%s'."
                                    %(rcdName))
                        raise AllInOneError(msg)
                if len( condPars ) == 2:
                    condPars.append( "" )
                if len(condPars) > 3:
                    msg = ("In section [%s]: '%s' is used with too many "
                           "arguments. A maximum of 3 arguments is allowed."
                           %(theSection, option))
                    raise AllInOneError(msg)
                conditions.append({"rcdName": rcdName.strip(),
                                   "connectString": condPars[0].strip(),
                                   "tagName": condPars[1].strip(),
                                   "labelName": condPars[2].strip()})

        rcdnames = collections.Counter(condition["rcdName"] for condition in conditions)
        if rcdnames and max(rcdnames.values()) >= 2:
            raise AllInOneError("Some conditions are specified multiple times (possibly through mp or hp options)!\n"
                                + ", ".join(rcdname for rcdname, count in six.iteritems(rcdnames) if count >= 2))

        for condition in conditions:
            self.__testDbExist(condition["connectString"], condition["tagName"])

        return conditions

    def __testDbExist(self, dbpath, tagname):
        if dbpath.startswith("sqlite_file:"):
            if not os.path.exists( dbpath.split("sqlite_file:")[1] ):
                raise AllInOneError("could not find file: '%s'"%dbpath.split("sqlite_file:")[1])
            elif "\n"+tagname not in conddb("--db", dbpath.split("sqlite_file:")[1], "listTags"):
                raise AllInOneError("{} does not exist in {}".format(tagname, dbpath))

    def restrictTo( self, restriction ):
        result = []
        if not restriction == None:
            for mode in self.mode:
                if mode in restriction:
                    result.append( mode )
            self.mode = result

    def getRepMap( self ):
        result = {
            "name": self.name,
            "title": self.title,
            "color": self.color,
            "style": self.style,
            "runGeomComp": self.runGeomComp,
            "GlobalTag": self.globaltag
            }
        return result

    def getConditions(self):
        """This function creates the configuration snippet to override
           global tag conditions.
           """
        if len( self.conditions ):
            loadCond = ("\nimport CalibTracker.Configuration."
                        "Common.PoolDBESSource_cfi\n")
            for cond in self.conditions:
                if not cond["labelName"] == "":
                    temp = configTemplates.conditionsTemplate.replace(
                        "tag = cms.string('.oO[tagName]Oo.')",
                        ("tag = cms.string('.oO[tagName]Oo.'),"
                         "\nlabel = cms.untracked.string('.oO[labelName]Oo.')"))
                else:
                    temp = configTemplates.conditionsTemplate
                loadCond += replaceByMap( temp, cond )
        else:
            loadCond = ""
        return loadCond
