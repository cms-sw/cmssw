import ConfigParser
import os
import re
import copy
import collections
from TkAlExceptions import AllInOneError


class AdaptedDict(collections.OrderedDict):
    """
    Dictionary which handles updates of values for already existing keys
    in a modified way.
    adapteddict[key] returns a list of all values associated with key
    This dictionary is used in the class `BetterConfigParser` instead of the
    default `dict_type` of the `ConfigParser` class.
    """

    def __init__(self, *args, **kwargs):
        self.validationslist = []
        collections.OrderedDict.__init__(self, *args, **kwargs)

    def __setitem__(self, key, value, dict_setitem=collections.OrderedDict.__setitem__):
        """
        od.__setitem__(i, y) <==> od[i]=y
        Updating an existing key appends the new value to the old value
        instead of replacing it.

        Arguments:
        - `key`: key part of the key-value pair
        - `value`: value part of the key-value pair
        - `dict_item`: method which is used for finally setting the item
        """

        if key != "__name__" and "__name__" in self and self["__name__"]=="validation":
            if isinstance(value, (str, unicode)):
                for index, item in enumerate(self.validationslist[:]):
                    if item == (key, value.split("\n")):
                        self.validationslist[index] = (key, value)
                        return
            self.validationslist.append((key, value))
        else:
            dict_setitem(self, key, value)

    def __getitem__(self, key):
        if key != "__name__" and "__name__" in self and self["__name__"]=="validation":
            return [validation[1] for validation in self.validationslist if validation[0] == key]
        else:
            return collections.OrderedDict.__getitem__(self, key)

    def items(self):
        if "__name__" in self and self["__name__"]=="validation":
            return self.validationslist
        else:
            return collections.OrderedDict.items(self)

class BetterConfigParser(ConfigParser.ConfigParser):
    def __init__(self):
        ConfigParser.ConfigParser.__init__(self,dict_type=AdaptedDict)
        self._optcre = self.OPTCRE_VALIDATION

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

    def items(self, section, raw=False, vars=None):
        if section == "validation":
            if raw or vars:
                raise NotImplementedError("'raw' and 'vars' do not work for betterConfigParser.items()!")
            items = self._sections["validation"].items()
            return items
        else:
            return ConfigParser.ConfigParser.items(self, section, raw, vars)

    def write(self, fp):
        """Write an .ini-format representation of the configuration state."""
        for section in self._sections:
            fp.write("[%s]\n" % section)
            for (key, value) in self._sections[section].items():
                if key == "__name__" or not isinstance(value, (str, unicode)):
                    continue
                if value is not None:
                    key = " = ".join((key, str(value).replace('\n', '\n\t')))
                fp.write("%s\n" % (key))
            fp.write("\n")


    #Preexisting validations in the validation section have syntax:
    #  preexistingoffline myoffline
    #with no = or :.  This regex takes care of that.
    OPTCRE_VALIDATION = re.compile(
        r'(?P<option>'
        r'(?P<preexisting>preexisting)?'
        r'[^:=\s][^:=]*)'                     # very permissive!
        r'\s*(?(preexisting)|'                # IF preexisting does not exist:
        r'(?P<vi>[:=])\s*'                    #   any number of space/tab,
                                              #   followed by separator
                                              #   (either : or =), followed
                                              #   by any # space/tab
        r'(?P<value>.*))$'                    #   everything up to eol
        )
