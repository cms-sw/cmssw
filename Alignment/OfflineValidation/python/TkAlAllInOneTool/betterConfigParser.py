import ConfigParser
import os
import copy
from TkAlExceptions import AllInOneError


class AdaptedDict(dict):
    """
    Dictionary which handles updates of values for already existing keys
    in a modified way.
    Instead of replacing the old value, the new value is appended to the
    value string separated by `self.getSep()`.
    This dictionary is used in the class `BetterConfigParser` instead of the
    default `dict_type` of the `ConfigParser` class.
    """

    def getSep(self):
        """
        This method returns the separator used to separate the values for 
        duplicate options in a config.
        """
        return " |/| "

    def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
        """
        od.__setitem__(i, y) <==> od[i]=y
        Updating an existing key appends the new value to the old value
        separated by `self.getSep()` instead of replacing it.

        Arguments:
        - `key`: key part of the key-value pair
        - `value`: value part of the key-value pair
        - `dict_item`: method which is used for finally setting the item
        """

        if "__name__" in self and self["__name__"]=="validation" \
                and key in self and value!=self[key][0]:
            the_value = [self[key][0]+self.getSep()+value[0]]
        else:
            the_value = value
        dict_setitem(self, key, the_value)


class BetterConfigParser(ConfigParser.ConfigParser):
    def __init__(self):
        ConfigParser.ConfigParser.__init__(self,dict_type=AdaptedDict)
        dummyDict = AdaptedDict()
        self._sep = dummyDict.getSep()
        del dummyDict

    def getSep(self):
        return self._sep

    def optionxform(self, optionstr):
        return optionstr
    
    def exists( self, section, option):
        try:
            items = self.items(section) 
        except ConfigParser.NoSectionError:
            return False
        for item in items:
            if item[0] == option:
                return True
        return False
        
    def __updateDict( self, dictionary, section ):
        result = dictionary
        try:
            for option in self.options( section ):
                result[option] = self.get( section, option )
            if "local"+section.title() in self.sections():
                for option in self.options( "local"+section.title() ):
                    result[option] = self.get( "local"+section.title(),
                                                   option )
        except ConfigParser.NoSectionError, section:
            msg = ("%s in configuration files. This section is mandatory."
                   %(str(section).replace(":", "", 1)))
            raise AllInOneError(msg)
        return result

    def getResultingSection( self, section, defaultDict = {}, demandPars = [] ):
        result = copy.deepcopy(defaultDict)
        for option in demandPars:
            try:
                result[option] = self.get( section, option )
            except ConfigParser.NoOptionError, globalSectionError:
                globalSection = str( globalSectionError ).split( "'" )[-2]
                splittedSectionName = section.split( ":" )
                if len( splittedSectionName ) > 1:
                    localSection = ("local"+section.split( ":" )[0].title()+":"
                                    +section.split(":")[1])
                else:
                    localSection = ("local"+section.split( ":" )[0].title())
                if self.has_section( localSection ):
                    try:
                        result[option] = self.get( localSection, option )
                    except ConfigParser.NoOptionError, option:
                        msg = ("%s. This option is mandatory."
                               %(str(option).replace(":", "", 1).replace(
                                    "section",
                                    "section '"+globalSection+"' or", 1)))
                        raise AllInOneError(msg)
                else:
                    msg = ("%s. This option is mandatory."
                           %(str(globalSectionError).replace(":", "", 1)))
                    raise AllInOneError(msg)
        result = self.__updateDict( result, section )
        return result

    def getAlignments( self ):
        alignments = []
        for section in self.sections():
            if "alignment:" in section:
                alignments.append( Alignment( section.split( "alignment:" )[1],
                                              self ) )
        return alignments

    def getCompares( self ):
        compares = {}
        for section in self.sections():
            if "compare:" in section:
                self.checkInput(section,
                                knownSimpleOptions = ["levels", "dbOutput",
                                                      "jobmode", "3DSubdetector1", "3Dubdetector2", "3DTranslationalScaleFactor"])
                levels = self.get( section, "levels" )
                dbOutput = self.get( section, "dbOutput" )
                compares[section.split(":")[1]] = ( levels, dbOutput )
        return compares

    def getGeneral( self ):
        defaults = {
            "jobmode":"interactive",
            "datadir":os.getcwd(),
            "logdir":os.getcwd(),
            "eosdir": "",
            "email":"true"
            }
        self.checkInput("general", knownSimpleOptions = defaults.keys())
        general = self.getResultingSection( "general", defaultDict = defaults )
        internal_section = "internals"
        if not self.has_section(internal_section):
            self.add_section(internal_section)
        if not self.has_option(internal_section, "workdir"):
            self.set(internal_section, "workdir", "/tmp/$USER")
        general["workdir"] = self.get(internal_section, "workdir")
        return general
    
    def checkInput(self, section, knownSimpleOptions=[], knownKeywords=[],
                   ignoreOptions=[]):
        """
        Method which checks, if the given options in `section` are in the
        list of `knownSimpleOptions` or match an item of `knownKeywords`.
        This is basically a check for typos and wrong parameters.
        
        Arguments:
        - `section`: Section of a configuration file
        - `knownSimpleOptions`: List of allowed simple options in `section`.
        - `knownKeywords`: List of allowed keywords in `section`.
        """

        for option in self.options( section ):
            if option in knownSimpleOptions:
                continue
            elif option.split()[0] in knownKeywords:
                continue
            elif option in ignoreOptions:
                print ("Ignoring option '%s' in section '[%s]'."
                       %(option, section))
            else:
                msg = ("Invalid or unknown parameter '%s' in section '%s'!"
                       %(option, section))
                raise AllInOneError(msg)

