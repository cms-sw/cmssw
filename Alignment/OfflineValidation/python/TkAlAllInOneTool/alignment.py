import os
import configTemplates
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError

class Alignment:
    def __init__(self, name, config, runGeomComp = "1"):
        section = "alignment:%s"%name
        if not config.has_section( section ):
            raise AllInOneError, ("section %s not found. Please define the "
                                  "alignment!"%section)
        config.checkInput(section,
                          knownSimpleOptions = ['globaltag', 'style', 'color'],
                          knownKeywords = ['condition'])
        self.name = name
        self.runGeomComp = runGeomComp
        self.globaltag = config.get( section, "globaltag" )
        self.conditions = self.__getConditions( config, section )
        self.color = config.get(section,"color")
        self.style = config.get(section,"style")

        # - removed backward compatibility
        # - keep the following lines, until the templates are adjusted
        #   to the new syntax
        self.dbpath = ""
        self.tag = ""
        self.errordbpath = "frontier://FrontierProd/CMS_COND_31X_FROM21X"
        self.errortag = "TrackerIdealGeometryErrors210_mc"
        self.kinksAndBows = ""
        self.kbdbpath = ""
        self.kbtag = ""
        
    def __getConditions( self, theConfig, theSection ):
        conditions = []
        for option in theConfig.options( theSection ):
            if option.startswith( "condition " ):
                rcdName = option.split( "condition " )[1]
                condParameters = theConfig.get( theSection, option ).split( "," )
                if len( condParameters ) < 2:
                    raise AllInOneError, ("'%s' is used with too few arguments."
                                          "A connect_string and a tag are "
                                          "required!"%option)
                if len( condParameters ) < 3:
                    condParameters.append( "" )
                conditions.append({"rcdName": rcdName.strip(),
                                   "connectString": condParameters[0].strip(),
                                   "tagName": condParameters[1].strip(),
                                   "labelName": condParameters[2].strip()})
        return conditions

    def __testDbExist(self, dbpath):
        #FIXME delete return to end train debuging
        return
        if not dbpath.startswith("sqlite_file:"):
            print "WARNING: could not check existence for",dbpath
        else:
            if not os.path.exists( dbpath.split("sqlite_file:")[1] ):
                raise "could not find file: '%s'"%dbpath.split("sqlite_file:")[1]
 
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
            "dbpath": self.dbpath,
            "errordbpath": self.errordbpath,
            "tag": self.tag,
            "errortag": self.errortag,
            "color": self.color,
            "style": self.style,
            "runGeomComp": self.runGeomComp,
            "kinksAndBows": self.kinksAndBows,
            "kbdbpath": self.kbdbpath,
            "kbtag": self.kbtag,
            "GlobalTag": self.globaltag
            }
        return result  

    def getLoadTemplate(self):
        """This function still exists only for historical reasons.
           Will be removed, when the templates are adjusted.
           """
        return ""

    def getAPETemplate(self):
        """This function still exists only for historical reasons.
           Will be removed, when the templates are adjusted.
           """
        return ""

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
