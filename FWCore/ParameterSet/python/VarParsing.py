import sys
import os
import re
from pprint import pprint
from FWCore.Utilities.Enumerate import Enumerate
from FWCore.Utilities.FileUtils import sectionNofTotal

class VarParsing (object):
    """Infrastructure to parse variable definitions passed to cmsRun
    configuration scripts"""


    multiplicity = Enumerate ("singleton list", "multiplicity")
    varType      = Enumerate ("bool int float string tagString")
    commaRE      = re.compile (r',')
    trueRE       = re.compile (r'^true$',  re.IGNORECASE)
    falseRE      = re.compile (r'^false$', re.IGNORECASE)


    def __init__ (self, *args):
        """Class initializer"""
        # Set everything up first
        self._singletons       = {}
        self._lists            = {}
        self._register         = {}
        self._beenSet          = {}
        self._info             = {}
        self._types            = {}
        self._maxLength        = 0
        self._tags             = {}
        self._tagOrder         = []
        self._noCommaSplit     = {}
        self._noDefaultClear   = {}
        self._setDuringParsing = {}
        self._currentlyParsing = False
        # now play with the rest
        for arg in args:
            lower = arg.lower()
            if lower == 'python':
                self.register ('storePrepend',
                               '',
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.string,
                               "Prepend location of files starting "
                               "with '/store'")
                # Don't continue because we want to process the next
                # piece, too.
            if lower == 'analysis' or lower == 'python':
                # Optionos for cmsRun or FWLite.Python
                self.register ('maxEvents',
                               -1,
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.int,
                               "Number of events to process (-1 for all)")
                self.register ('totalSections',
                               0,
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.int,
                               "Total number of sections")
                self.register ('section',
                               0,
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.int,
                               "This section (from 1..totalSections inclusive)")
                self.register ('inputFiles',
                               '',
                               VarParsing.multiplicity.list,
                               VarParsing.varType.string,
                               "Files to process")
                self.register ('secondaryInputFiles',
                               '',
                               VarParsing.multiplicity.list,
                               VarParsing.varType.string,
                               "Second group of files to process (if needed)")
                self.register ('outputFile',
                               'output.root',
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.tagString,
                               "Name of output file (if needed)")
                self.register ('secondaryOutputFile',
                               '',
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.tagString,
                               "Name of second output file (if needed)")
                self.setupTags (tag = 'numEvent%d',
                                ifCond = 'maxEvents > 0',
                                tagArg = 'maxEvents')
                continue
            # old, depricated, but here for compatibility of older code
            if lower == "standard":
                # load in old standard arguments and defaults
                self.register ('maxEvents',
                               -1,
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.int,
                               "Number of events to process (-1 for all)")
                self.register ('files',
                               '',
                               VarParsing.multiplicity.list,
                               VarParsing.varType.string,
                               "Files to process")
                self.register ('secondaryFiles',
                               '',
                               VarParsing.multiplicity.list,
                               VarParsing.varType.string,
                               "Second group of files to process (if needed)")
                self.register ('output',
                               'output.root',
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.tagString,
                               "Name of output file (if needed)")
                self.register ('secondaryOutput',
                               '',
                               VarParsing.multiplicity.singleton,
                               VarParsing.varType.tagString,
                               "Name of second output file (if needed)")
                self.setupTags (tag = 'numEvent%d',
                                ifCond = 'maxEvents > 0',
                                tagArg = 'maxEvents')
                continue
            # if we're still here, then we've got a rogue arument
            print "Error: VarParsing.__init__ doesn't understand '%s'" \
                  % arg
            raise RuntimeError, "Failed to create VarParsing object"


    def setupTags (self, **kwargs):
        """Sets up information for tags for output names"""
        necessaryKeys = set (['ifCond', 'tag'])
        allowedKeys   = set (['tagArg'])
        for key in kwargs.keys():
            if key in allowedKeys:
                continue
            if key in necessaryKeys:
                necessaryKeys.remove (key)
                continue
            # if we're here, then we have a key that's not understood
            print "Unknown option '%s'" % key
            raise RuntimeError, "Unknown option"
        if necessaryKeys:
            # if this is not empty, then we didn't have a key that was
            # necessary.
            print "Missing keys: %s" % necessaryKeys
            raise runtimeError, "Missing keys"
        tag = kwargs.get('tag')
        del kwargs['tag']
        self._tags[tag] = kwargs
        self._tagOrder.append (tag)


    def parseArguments (self):
        """Parses command line arguments.  Parsing starts just after
        the name of the configuration script.  Parsing will fail if
        there is not 'xxxx.py'"""
        self._currentlyParsing = True
        foundPy      = False
        printStatus  = False
        help         = False
        singleAssign = True 
        for arg in sys.argv:
            if not foundPy and arg.endswith ('.py'):
                foundPy = True
                continue
            if not foundPy:
                continue
            # If we're here, then we can parse to our hearts content.
            # So, is this a command or a declaration?
            if arg.count('='):
                # declaration
                name, value = arg.split ('=', 1)
                if name.count('_'):
                    # name with command
                    name, command = name.split ('_', 1)
                    command = command.lower()
                    if command == 'load':
                        self.loadFromFile (name, value)
                        continue
                    if command == 'clear':
                        self.clearList (name)
                        continue
                    # If we're here, then I don't recognize this command
                    print "Unknown command '%s' in '%s_%s" % \
                          (command, name, command)
                    raise RuntimeError, "Illegal parsing command"
                else:
                    # just a name and value
                    if not self._register.has_key (name):
                        print "Error:  '%s' not registered." \
                              % name
                        raise RuntimeError, "Unknown variable"
                    if VarParsing.multiplicity.singleton == \
                           self._register[name]:
                        # singleton
                        if self._beenSet.get (name) and singleAssign:
                            print "Variable '%s' assigned multiple times. Use" \
                                  , "'multipleAssign' command to avoid"
                            raise RuntimeError, "Multiple assignment"
                        self._beenSet[name] = True
                        self.setDefault (name, value)
                    else:
                        # list
                        self.setDefault (name, value)
            else:
                # commands
                if arg.count('_'):
                    # name modifier
                    name, command = arg.split ('_', 1)
                    command = command.lower()
                    if not self._register.has_key (name):
                        print "Error:  '%s' not registered." \
                              % name
                        raise RuntimeError, "Unknown variable"
                    if command == 'clear':
                        self.clearList (name)
                        continue
                    # if we're still here, complain that we don't
                    # understand this command:
                    print "Do not understand '%s' in '%s'" % (command, arg)
                    raise RuntimeError, "Unknown command"
                else:
                    # simple command
                    command = arg.lower()
                    if command == 'help' or command == '--help':
                        help = True
                    elif command == 'print' or command == '--print':
                        printStatus = True
                    elif command == 'noprint' or command == '--noprint':
                        printStatus = False
                    else:
                        # We don't understand this command
                        print "Do not understand command '%s'" % (arg)
                        raise RuntimeError, "Unknown command"
            # else if declaration
        ###########################
        # Post-loading processing #
        ###########################
        # sections
        if self._register.has_key ('totalSections') and \
           self._register.has_key ('section') and \
           self._register.has_key ('inputFiles') and \
           self.totalSections and self.section:
            # copy list
            oldInputFiles = self.inputFiles
            # clear list
            self.clearList ('inputFiles')
            # used old list to make list
            self.inputFiles = sectionNofTotal (oldInputFiles,
                                               self.section,
                                               self.totalSections)
        # storePrepend
        if self._register.has_key ('storePrepend') and \
           self._register.has_key ('inputFiles') and \
           self.storePrepend:
            storeRE = re.compile (r'^/store/')
            newFileList = []
            for filename in self.inputFiles:
                if storeRE.match (filename):
                    filename = self.storePrepend + filename
                newFileList.append (filename)
            # clear old list
            self.clearList ('inputFiles')
            # set new list as list
            self.inputFiles = newFileList
        # make sure found the py file
        if not foundPy:
            print "VarParsing.parseArguments() Failure: No configuration " + \
                  "file found ending in .py."
            raise RuntimeError, "Invalid configuration ending"
        if help:
            self.help()
        if printStatus:
            print self
        self._currentlyParsing = False


    def clearList (self, name):
        """Empties all entries from list"""
        if not self._register.has_key (name):
            print "Error:  '%s' not registered." \
                  % name
            raise RuntimeError, "Unknown variable"
        if self._register[name] != VarParsing.multiplicity.list:
            print "Error: '%s' is not a list" % name
            raise RuntimeError, "Faulty 'clear' command"
        # if we're still here, do what we came to do
        self._lists[name] = []


    def setNoDefaultClear (self, name, value=True):
        """Tells lists to not clear default list values when set from
        command line."""
        if not self._register.has_key (name):
            print "Error:  '%s' not registered." \
                  % name
            raise RuntimeError, "Unknown variable"
        if self._register[name] != VarParsing.multiplicity.list:
            print "Error: '%s' is not a list" % name
            raise RuntimeError, "Faulty 'setNoDefaultClear' command"
        self._noDefaultClear[name] = bool (value)


    def setNoCommaSplit (self, name, value=True):
        """Tells lists to not split up values by commas."""
        if not self._register.has_key (name):
            print "Error:  '%s' not registered." \
                  % name
            raise RuntimeError, "Unknown variable"
        if self._register[name] != VarParsing.multiplicity.list:
            print "Error: '%s' is not a list" % name
            raise RuntimeError, "Faulty 'setNoCommaSplit' command"
        self._noCommaSplit[name] = bool (value)


    def loadFromFile (self, name, filename):
        """Loads a list from file"""
        if not self._register.has_key (name):
            print "Error:  '%s' not registered." \
                  % name
            raise RuntimeError, "Unknown variable"
        if self._register[name] != VarParsing.multiplicity.list:
            print "Error: '%s' is not a list" % name
            raise RuntimeError, "'load' only works for lists"
        filename = os.path.expanduser (filename)
        if not os.path.exists (filename):
            print "Error: '%s' file does not exist."
            raise RuntimeError, "Bad filename"
        source = open (filename, 'r')        
        for line in source.readlines():
            line = re.sub (r'#.+$', '', line) # remove comment characters
            line = line.strip()
            if len (line):
                self._lists[name].append( self._convert (name, line ) )
        source.close()


    def help (self):
        """Prints out help information and exits"""
        print self
        print """Options:
        help           : This screen
        multipleAssign : Allows singletons to have multiple assigments
        print          : Prints out current values
        XXX_clear      : Clears list named 'XXX'
        """    
        sys.exit (0)


    def register (self, name,
                  default = "",
                  mult    = multiplicity.singleton,
                  mytype  = varType.int,
                  info    = "",
                  **kwargs):
        """Register a variable"""
        # is type ok?
        if not VarParsing.multiplicity.isValidValue (mult):
            print "Error: VarParsing.register() must use ",\
                  "VarParsing.multiplicity."
            raise RuntimeError, "Improper 'mult' value"
        if not VarParsing.varType.isValidValue (mytype):
            print "Error: VarParsing.register() must use ",\
                  "VarParsing.varType."
            raise RuntimeError, "Improper 'type' value %s" % mytype
        if VarParsing.multiplicity.list == mult and \
           VarParsing.varType.tagString == mytype:
            print "Error: 'tagString' can only be used with 'singleton'"
            raise RuntimeError, "Improper registration"
        # is the name ok
        if name.count ("_"):
            print "Error: Name can not contain '_': %s" % name
            raise RuntimeError, "Improper 'name'"
        # has this been registered before?
        if self._register.has_key (name):
            # Uh oh
            print "Error: You can not register a name twice, '%s'" \
                  % name
            raise RuntimeError, "Attempt to re-register variable"
        self._register[name] = mult
        self._beenSet[name]  = False
        self._info[name]     = info
        self._types[name]    = mytype
        if len (name) > self._maxLength:
            self._maxLength = len (name)
        if VarParsing.multiplicity.singleton == mult:
            self._singletons[name] = default
        else:
            self._lists[name] = []
            # if it's a list, we only want to use the default if it
            # does exist.
            if len (default):
                self._lists[name].append (default)
        #######################################
        ## Process any additional directives ##
        #######################################
        # do we want to tell the list to not split command line
        # arguments by commas?
        if kwargs.get ('noCommaSplit'):
            self._noCommaSplit[name] = bool( kwargs['noCommaSplit'] )
            del kwargs['noCommaSplit']
        if kwargs.get ('noDefaultClear'):
            self._noDefaultClear[name] = bool( kwargs['noDefaultClear'] )
            del kwargs['noDefaultClear']
        if len (kwargs):
            raise RuntimeError, "register() Unknown arguments %s" % kwargs


    def has_key (self, key):
        """Returns true if a key is registered"""
        return self._register.has_key (key)


    def setType (self, name, mytype):
        """Change the type of 'name' to 'mytype'"""
        if not VarParsing.varType.isValidValue (mytype):
            print "Error: VarParsing.setType() must use ",\
                  "VarParsing.varType."
            raise RuntimeError, "Improper 'type' value"
        oldVal = self.__getattr__ (name, noTags = True)
        self._types[name] = mytype
        self.setDefault (name, oldVal)
        

    def setDefault (self, name, *args):
        """Used to set or change the default of an already registered
        name"""
        # has this been registered?
        if not self._register.has_key (name):
            print "Error: VarParsing.setDefault '%s' not already registered." \
                  % name
            raise RuntimeError, "setDefault without registration"
        if VarParsing.multiplicity.singleton == self._register[name]:
            # make sure we only have one value
            if len (args) != 1:
                print "Error: VarParsing.setDefault needs exactly 1 ",\
                      "value for '%s'" % name
                raise RuntimeError, "setDefault args problem"
            self._singletons[name] = self._convert (name, args[0])
        else:
            # If:
            #   - We have a list (which we do)
            #   - We are now processing command line parsing
            #   - It has not yet been set after parsing
            #   - We have not explicitly asked for it to be NOT cleared
            # Then:
            #   - We clear the list
            if self._currentlyParsing and \
               not self._setDuringParsing.get(name) and \
               not self._noDefaultClear.get(name):
                # All four conditions have been satisfied, so let's go
                # ahead and clear this.
                self.clearList (name)
            # IF we are currently parsing, then tell people I've set
            # this:
            if self._currentlyParsing:
                self._setDuringParsing[name] = True
            # if args is a tuple and it only has one entry, get rid of
            # the first level of tupleness:
            if isinstance (args, tuple) and len (args) == 1:
                args = args[0]
            # is this still a tuple
            if isinstance (args, tuple):
                mylist = list (args)
            elif isinstance (args, list):
                mylist = args
            else:
                mylist = []
                mylist.append (args)
            if not self._noCommaSplit.get (name):
                oldList = mylist
                mylist = []
                for item in oldList:
                    mylist.extend( VarParsing.commaRE.split( item ) )
            for item in mylist:
                self._lists[name].append( self._convert (name, item ) )


    def _convert (self, name, inputVal):
        """Converts inputVal to the type required by name"""
        inputVal = str (inputVal)
        if self._types[name] == VarParsing.varType.bool:
            if VarParsing.trueRE.match (inputVal) or '1' == inputVal:
                return True
            elif VarParsing.falseRE.match (inputVal) or '0' == inputVal:
                return False
            # if we're still here, then we don't have 'true' or 'false'
            raise RuntimeError, "Unknown bool value '%s'.  Must be 'true' or 'false'" % inputVal
        if self._types[name] == VarParsing.varType.string or \
           self._types[name] == VarParsing.varType.tagString:
            return inputVal
        elif self._types[name] == VarParsing.varType.int:
            return int (inputVal, 0)
        elif self._types[name] == VarParsing.varType.float:
            return float (inputVal)
        else:
            raise RuntimeError, "Unknown varType"
        

    def _withTags (self, name):
        if not self._register.has_key (name):
            print "Error:  '%s' not registered." \
                  % name
            raise RuntimeError, "Unknown variable"
        if self._register[name] == VarParsing.multiplicity.list:
            print "Error: '%s' is a list" % name
            raise RuntimeError, "withTags() only works on singletons"
        retval = self._singletons[name]
        if retval.endswith ('.root'):
            retval, garbage = os.path.splitext (retval)
        reverseOrder = self._tagOrder
        reverseOrder.reverse()
        for tag in reverseOrder:
            tagDict = self._tags[tag]
            ifCond = tagDict['ifCond']
            if ifCond.count('%'):
                pass
            else:
                ifCond = "self." + ifCond
            boolValue = eval (ifCond)
            tagArg = tagDict.get ('tagArg')
            if tagArg:
                evalString = "'%s' %% self.%s" % (tag, tagArg)
                tag = eval (evalString)
            if boolValue:
                retval = retval + "_" + tag        
        return retval + ".root"
            

    def __str__ (self):
        """String form of self"""
        maxLen = min (self._maxLength, 20)
        form     = "  %%-%ds: %%s" % maxLen
        formInfo = "  %%%ds  - %%s" % (maxLen - 2)
        formItem = "  %%%ds    %%s" % (maxLen - 1)
        retval = ""
        if len (self._singletons.keys()):
            retval = retval + "Singletons:\n"
        for varName, value in sorted (self._singletons.iteritems()):
            retval = retval + form % (varName, value) + "\n";
            if self._info.get(varName):
                retval = retval + formInfo % ('', self._info[varName]) + "\n"
        if len (self._singletons.keys()):
            retval = retval +  "Lists:\n"
        for varName, value in sorted (self._lists.iteritems()):
            stringValue = "%s" % value
            if len (stringValue) < 76 - maxLen:
                retval = retval + form % (varName, value) + "\n"
            else:
                varLength = len (value)
                for index, item in enumerate (value):
                    if index == 0:
                        retval = retval + form % (varName, "['" + item)
                    else:
                        retval = retval + formItem % ('',"'" + item)
                    if index == varLength - 1:
                        retval = retval + "' ]\n"
                    else:
                        retval = retval + "',\n"
            if self._info.get(varName):
                retval = retval + formInfo % ('', self._info[varName]) + "\n"
        return retval


    def __setattr__ (self, name, value, *extras):
        """Lets me set internal values, or uses setDefault"""
        if not name.startswith ("_"):
            mylist = list (extras)
            mylist.insert (0, value)
            self.setDefault (name, *mylist)
        else:
            object.__setattr__ (self, name, value)


    def __getattr__ (self, name, noTags = False):
        """Lets user get the info they want with obj.name"""
        if name.startswith ("_"):
            # internal use
            return object.__getattribute__ (self, name)
        else:
            # user variable
            if not self._register.has_key (name):
                print "Error:  '%s' not already registered." \
                      % name
                raise RuntimeError, "Unknown variable"
            if VarParsing.multiplicity.singleton == self._register[name]:
                if VarParsing.varType.tagString == self._types[name] \
                       and not noTags:
                    return self._withTags (name)
                else:
                    return self._singletons[name]
            else:
                return self._lists[name]
 

##############################################################################
## ######################################################################## ##
## ##                                                                    ## ##
## ######################################################################## ##
##############################################################################

    
if __name__ == "__main__":
    #############################################
    ## Load and save command line history when ##
    ## running interactively.                  ##
    #############################################
    import os, readline
    import atexit
    historyPath = os.path.expanduser("~/.pyhistory")


    def save_history(historyPath=historyPath):
        import readline
        readline.write_history_file(historyPath)
        if os.path.exists(historyPath):
            readline.read_history_file(historyPath)


    atexit.register(save_history)
    readline.parse_and_bind("set show-all-if-ambiguous on")
    readline.parse_and_bind("tab: complete")
    if os.path.exists (historyPath) :
        readline.read_history_file(historyPath)
        readline.set_history_length(-1)


    ############################
    # Example code starts here #
    ############################

    obj = VarParsing ('standard')
    obj.register ('someVar',
                  mult=VarParsing.multiplicity.singleton,
                  info="for testing")
    obj.setupTags (tag    = "someCondition",
                   ifCond = "someVar")
    ## obj.register ('numbers',
    ##               mult=VarParsing.multiplicity.list,
    ##               info="Numbers")
    ## obj.register ('singleNumber',
    ##               mult=VarParsing.multiplicity.singleton,
    ##               info="A single number")
    obj.parseArguments()
    
