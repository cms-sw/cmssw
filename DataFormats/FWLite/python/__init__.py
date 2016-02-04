#! /usr/bin/env python

import ROOT
import inspect
import sys
import optparse
from FWCore.ParameterSet.VarParsing import VarParsing


ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.AutoLibraryLoader.enable()

# Whether warn() should print anythingg
quietWarn = False

def setQuietWarn (quiet = True):
    global quietWarn
    quietWarn = quiet

def warn (*args, **kwargs):
    """print out warning with line number and rest of arguments"""
    if quietWarn: return
    frame = inspect.stack()[1]
    filename = frame[1]
    lineNum  = frame[2]
    #print "after '%s'" % filename
    blankLines = kwargs.get('blankLines', 0)
    if blankLines:
        print '\n' * blankLines
    spaces = kwargs.get('spaces', 0)
    if spaces:
        print ' ' * spaces,
    if len (args):
        print "%s (%s): " % (filename, lineNum),
        for arg in args:
            print arg,
        print
    else:
        print "%s (%s):" % (filename, lineNum)

########################
## ################## ##
## ## ############ ## ##
## ## ## Handle ## ## ##
## ## ############ ## ##
## ################## ##
########################

class Handle:
    """Python interface to FWLite Handle class"""

    def __init__ (self,
                  typeString,
                  **kwargs):
        """Initialize python handle wrapper """
        self._type      = typeString
        self._wrapper   = ROOT.edm.Wrapper (self._type)()
        self._typeInfo  = self._wrapper.typeInfo()
        self._exception = RuntimeError ("getByLabel not called for '%s'", self)
        global options
        ROOT.SetOwnership (self._wrapper, False)
        # O.k.  This is a little weird.  We want a pointer to an EDM
        # wrapper, but we don't want the memory it is pointing to.
        # So, we've created it and grabbed the type info.  Since we
        # don't want a memory leak, we destroy it.
        if kwargs.get ('noDelete'):
            print "Not deleting wrapper"
            del kwargs['noDelete']
        else:
            self._wrapper.IsA().Destructor( self._wrapper )
        # Since we deleted the options as we used them, that means
        # that kwargs should be empty.  If it's not, that means that
        # somebody passed in an argument that we're not using and we
        # should complain.
        if len (kwargs):
            raise RuntimeError, "Unknown arguments %s" % kwargs


    def isValid (self):
        """Returns true if getByLabel call was successful and data is
        present in handle."""
        return not self._exception


    def product (self):
        """Returns product stored in handle."""
        if self._exception:
            raise self._exception
        return self._wrapper.product()


    def __str__ (self):
        return "%s" % (self._type)

                                          
    ## Private member functions ##

    def _typeInfoGetter (self):
        """(Internal) Return the type info"""
        return self._typeInfo


    def _addressOf (self):
        """(Internal) Return address of edm wrapper"""
        return ROOT.AddressOf (self._wrapper)


    def _setStatus (self, getByLabelSuccess, labelString):
        """(Internal) To be called by Events.getByLabel"""
        if not getByLabelSuccess:
            self._exception = RuntimeError ("getByLabel (%s, %s) failed" \
                                            % (self, labelString))
            print "one", self._exception
            return
        if not self._wrapper.isPresent():
            self._exception = RuntimeError ("getByLabel (%s, %s) not present this event" \
                                            % (self, labelString))
            print "two", self._exception
            return
        # if we're still here, then everything is happy.  Clear the exception
        self._exception = None


#######################
## ################# ##
## ## ########### ## ##
## ## ## Lumis ## ## ##
## ## ########### ## ##
## ################# ##
#######################

class Lumis:
    """Python interface to FWLite LuminosityBlock"""
    def __init__ (self, inputFiles = '', **kwargs):
        self._lumi = None
        self._lumiCounts = 0
        self._tfile = None
        self._maxLumis = 0
        if isinstance (inputFiles, list):
            # it's a list
            self._filenames = inputFiles[:]
        elif isinstance (inputFiles, VarParsing):
            # it's a VarParsing object
            options = inputFiles
            self._maxLumis           = options.maxEvents
            self._filenames           = options.inputFiles
        else:
            # it's probably a single string
            self._filenames = [inputFiles]
        ##############################
        ## Parse optional arguments ##
        ##############################
        if kwargs.has_key ('maxEvents'):
            self._maxLumis = kwargs['maxEvents']
            del kwargs['maxEvents']
        if kwargs.has_key ('options'):
            options = kwargs ['options']
            self._maxLumis           = options.maxEvents
            self._filenames           = options.inputFiles
            self._secondaryFilenames  = options.secondaryInputFiles
            del kwargs['options']
        # Since we deleted the options as we used them, that means
        # that kwargs should be empty.  If it's not, that means that
        # somebody passed in an argument that we're not using and we
        # should complain.
        if len (kwargs):
            raise RuntimeError, "Unknown arguments %s" % kwargs
        if not self._filenames:
            raise RuntimeError, "No input files given"
        if not self._createFWLiteLumi():
            # this shouldn't happen as you are getting nothing the
            # very first time out, but let's at least check to
            # avoid problems.
            raise RuntimeError, "Never and information about Lumi"


    def __del__ (self):
        """(Internal) Destructor"""
        # print "Goodbye cruel world, I'm leaving you today."
        del self._lumi
        # print "Goodbye, goodbye, goodbye."


    def __iter__ (self):
        return self._next()


    def aux (self):
        try:
            return self._lumi.luminosityBlockAuxiliary()
        except:
            raise RuntimeError, "Lumis.aux() called on object in invalid state"


    def luminosityBlockAuxiliary (self):
        try:
            return self._lumi.luminosityBlockAuxiliary()
        except:
            raise RuntimeError, "Lumis.luminosityBlockAuxiliary() called on object in invalid state"
        

    def getByLabel (self, *args):
        """Calls FWLite's getByLabel.  Called:
        getByLabel (moduleLabel, handle)
        getByLabel (moduleLabel, productInstanceLabel, handle),
        getByLabel (moduleLabel, productInstanceLabel, processLabel, handle),
        or
        getByLabel ( (mL, pIL,pL), handle)
        """
        length = len (args)
        if length < 2 or length > 4:
            # not called correctly
            raise RuntimeError, "Incorrect number of arguments"
        # handle is always the last argument
        argsList = list (args)
        handle = argsList.pop()
        if len(argsList)==1 and \
               ( isinstance (argsList[0], tuple) or
                 isinstance (argsList[0], list) ) :
            if len (argsList) > 3:
                raise RuntimeError, "getByLabel Error: label tuple has too " \
                      "many arguments '%s'" % argsList[0]
            argsList = list(argsList[0])
        while len(argsList) < 3:
            argsList.append ('')
        (moduleLabel, productInstanceLabel, processLabel) = argsList
        labelString = "'" + "', '".join(argsList) + "'"
        handle._setStatus ( self._lumi.getByLabel( handle._typeInfoGetter(),
                                                   moduleLabel,
                                                   productInstanceLabel,
                                                   processLabel,
                                                   handle._addressOf() ),
                            labelString )
        return handle.isValid()


    ##############################
    ## Private Member Functions ##
    ##############################

    def _createFWLiteLumi (self):
        """(Internal) Creates an FWLite Lumi"""
        # are there any files left?
        if not self._filenames:
            return False
        if self._lumi:
            del self._lumi
            self._lumi = None
        self._veryFirstTime = False
        self._currFilename = self._filenames.pop(0)
        #print "Opening file", self._currFilename
        if self._tfile:
            del self._tfile
        self._tfile = ROOT.TFile.Open (self._currFilename)
        self._lumi = ROOT.fwlite.LuminosityBlock (self._tfile);
        self._lumi.toBegin()
        return True


    def _next (self):
        """(Internal) Iterator internals"""
        while True:
            if self._lumi.atEnd():
                if not self._createFWLiteLumi():
                    # there are no more files here, so we are done
                    break
            yield self
            self._lumiCounts += 1
            if self._maxLumis > 0 and self._lumiCounts >= self._maxLumis:
                break
            self._lumi.__preinc__()
            
                    
        
######################
## ################ ##
## ## ########## ## ##
## ## ## Runs ## ## ##
## ## ########## ## ##
## ################ ##
######################

class Runs:
    """Python interface to FWLite LuminosityBlock"""
    def __init__ (self, inputFiles = '', **kwargs):
        self._run = None
        self._runCounts = 0
        self._tfile = None
        self._maxRuns = 0
        if isinstance (inputFiles, list):
            # it's a list
            self._filenames = inputFiles[:]
        elif isinstance (inputFiles, VarParsing):
            # it's a VarParsing object
            options = inputFiles
            self._maxRuns           = options.maxEvents
            self._filenames           = options.inputFiles
        else:
            # it's probably a single string
            self._filenames = [inputFiles]
        ##############################
        ## Parse optional arguments ##
        ##############################
        if kwargs.has_key ('maxEvents'):
            self._maxRuns = kwargs['maxEvents']
            del kwargs['maxEvents']
        if kwargs.has_key ('options'):
            options = kwargs ['options']
            self._maxRuns           = options.maxEvents
            self._filenames           = options.inputFiles
            self._secondaryFilenames  = options.secondaryInputFiles
            del kwargs['options']
        # Since we deleted the options as we used them, that means
        # that kwargs should be empty.  If it's not, that means that
        # somebody passed in an argument that we're not using and we
        # should complain.
        if len (kwargs):
            raise RuntimeError, "Unknown arguments %s" % kwargs
        if not self._filenames:
            raise RuntimeError, "No input files given"
        if not self._createFWLiteRun():
            # this shouldn't happen as you are getting nothing the
            # very first time out, but let's at least check to
            # avoid problems.
            raise RuntimeError, "Never and information about Run"


    def __del__ (self):
        """(Internal) Destructor"""
        # print "Goodbye cruel world, I'm leaving you today."
        del self._run
        # print "Goodbye, goodbye, goodbye."


    def __iter__ (self):
        return self._next()


    def aux (self):
        try:
            return self._run.runAuxiliary()
        except:
            raise RuntimeError, "Runs.aux() called on object in invalid state"


    def runAuxiliary (self):
        try:
            return self._run.runAuxiliary()
        except:
            raise RuntimeError, "Runs.runAuxiliary() called on object in invalid state"
        

    def getByLabel (self, *args):
        """Calls FWLite's getByLabel.  Called:
        getByLabel (moduleLabel, handle)
        getByLabel (moduleLabel, productInstanceLabel, handle),
        getByLabel (moduleLabel, productInstanceLabel, processLabel, handle),
        or
        getByLabel ( (mL, pIL,pL), handle)
        """
        length = len (args)
        if length < 2 or length > 4:
            # not called correctly
            raise RuntimeError, "Incorrect number of arguments"
        # handle is always the last argument
        argsList = list (args)
        handle = argsList.pop()
        if len(argsList)==1 and \
               ( isinstance (argsList[0], tuple) or
                 isinstance (argsList[0], list) ) :
            if len (argsList) > 3:
                raise RuntimeError, "getByLabel Error: label tuple has too " \
                      "many arguments '%s'" % argsList[0]
            argsList = list(argsList[0])
        while len(argsList) < 3:
            argsList.append ('')
        (moduleLabel, productInstanceLabel, processLabel) = argsList
        labelString = "'" + "', '".join(argsList) + "'"
        handle._setStatus ( self._run.getByLabel( handle._typeInfoGetter(),
                                                   moduleLabel,
                                                   productInstanceLabel,
                                                   processLabel,
                                                   handle._addressOf() ),
                            labelString )
        return handle.isValid()

                    
       

    ##############################
    ## Private Member Functions ##
    ##############################

    def _createFWLiteRun (self):
        """(Internal) Creates an FWLite Run"""
        # are there any files left?
        if not self._filenames:
            return False
        if self._run:
            del self._run
            self._run = None
        self._veryFirstTime = False
        self._currFilename = self._filenames.pop(0)
        #print "Opening file", self._currFilename
        if self._tfile:
            del self._tfile
        self._tfile = ROOT.TFile.Open (self._currFilename)
        self._run = ROOT.fwlite.Run (self._tfile);
        self._run.toBegin()
        return True


    def _next (self):
        """(Internal) Iterator internals"""
        while True:
            if self._run.atEnd():
                if not self._createFWLiteRun():
                    # there are no more files here, so we are done
                    break
            yield self
            self._runCounts += 1
            if self._maxRuns > 0 and self._runCounts >= self._maxRuns:
                break
            self._run.__preinc__()
            

########################
## ################## ##
## ## ############ ## ##
## ## ## Events ## ## ##
## ## ############ ## ##
## ################## ##
########################

class Events:
    """Python interface to FWLite ChainEvent class"""

    def __init__(self, inputFiles = '', **kwargs):
        """inputFiles    => Either a single filename or a list of filenames
        Optional arguments:
        forceEvent  => Use fwlite::Event IF there is only one file
        maxEvents   => Maximum number of events to process
        """        
        self._veryFirstTime      = True
        self._event              = 0
        self._eventCounts        = 0
        self._maxEvents          = 0
        self._forceEvent         = False
        self._mode               = None
        self._secondaryFilenames = None
        if isinstance (inputFiles, list):
            # it's a list
            self._filenames = inputFiles[:]
        elif isinstance (inputFiles, VarParsing):
            # it's a VarParsing object
            options = inputFiles
            self._maxEvents           = options.maxEvents
            self._filenames           = options.inputFiles
            self._secondaryFilenames  = options.secondaryInputFiles
        else:
            # it's probably a single string
            self._filenames = [inputFiles]
        ##############################
        ## Parse optional arguments ##
        ##############################
        if kwargs.has_key ('maxEvents'):
            self._maxEvents = kwargs['maxEvents']
            del kwargs['maxEvents']
        if kwargs.has_key ('forceEvent'):
            self._forceEvent = kwargs['forceEvent']
            del kwargs['forceEvent']
        if kwargs.has_key ('options'):
            options = kwargs ['options']
            self._maxEvents           = options.maxEvents
            self._filenames           = options.inputFiles
            self._secondaryFilenames  = options.secondaryInputFiles
            del kwargs['options']
        # Since we deleted the options as we used them, that means
        # that kwargs should be empty.  If it's not, that means that
        # somebody passed in an argument that we're not using and we
        # should complain.
        if len (kwargs):
            raise RuntimeError, "Unknown arguments %s" % kwargs
        if not self._filenames:
            raise RuntimeError, "No input files given"


    def to (self, entryIndex):
        """Jumps to event entryIndex"""
        if self._veryFirstTime:
            self._createFWLiteEvent()
        return self._event.to ( long(entryIndex) )

        
    def toBegin (self):
        """Called to reset event loop to first event."""
        self._toBegin = True


    def size (self):
        """Returns number of events"""
        if self._veryFirstTime:
            self._createFWLiteEvent()
        return self._event.size()


    def eventAuxiliary (self):
        """Returns eventAuxiliary object"""
        if self._veryFirstTime:
            raise RuntimeError, "eventAuxiliary() called before "\
                  "toBegin() or to()"
        return self._event.eventAuxiliary()


    def object (self):
        """Returns event object"""
        return self._event


    def getByLabel (self, *args):
        """Calls FWLite's getByLabel.  Called:
        getByLabel (moduleLabel, handle)
        getByLabel (moduleLabel, productInstanceLabel, handle),
        getByLabel (moduleLabel, productInstanceLabel, processLabel, handle),
        or
        getByLabel ( (mL, pIL,pL), handle)
        """
        if self._veryFirstTime:
            self._createFWLiteEvent()        
        length = len (args)
        if length < 2 or length > 4:
            # not called correctly
            raise RuntimeError, "Incorrect number of arguments"
        # handle is always the last argument
        argsList = list (args)
        handle = argsList.pop()
        if len(argsList)==1 and \
               ( isinstance (argsList[0], tuple) or
                 isinstance (argsList[0], list) ) :
            if len (argsList) > 3:
                raise RuntimeError, "getByLabel Error: label tuple has too " \
                      "many arguments '%s'" % argsList[0]
            argsList = list(argsList[0])
        while len(argsList) < 3:
            argsList.append ('')
        (moduleLabel, productInstanceLabel, processLabel) = argsList
        labelString = "'" + "', '".join(argsList) + "'"
        handle._setStatus ( self._event.getByLabel( handle._typeInfoGetter(),
                                                    moduleLabel,
                                                    productInstanceLabel,
                                                    processLabel,
                                                    handle._addressOf() ),
                            labelString )
        return handle.isValid()

                    
    def __iter__ (self):
        return self._next()


    def fileIndex (self):
        if self._event:
            return self._event.fileIndex()
        else:
            # default non-existant value is -1.  Return something else
            return -2


    def secondaryFileIndex (self):
        if self._event:
            return self._event.secondaryFileIndex()
        else:
            # default non-existant value is -1.  Return something else
            return -2


    def fileIndicies (self):
        return (self.fileIndex(), self.secondaryFileIndex())


    ## Private Member Functions ##


    def _parseOptions (self, options):
        """(Internal) Parse options"""


    def _toBeginCode (self):
        """(Internal) Does actual work of toBegin() call"""
        self._toBegin = False
        self._event.toBegin()
        self._eventCounts = 0


    def __del__ (self):
        """(Internal) Destructor"""
        # print "Goodbye cruel world, I'm leaving you today."
        del self._event
        # print "Goodbye, goodbye, goodbye."


    def _createFWLiteEvent (self):
        """(Internal) Creates an FWLite Event"""
        self._veryFirstTime = False
        self._toBegin = True
        if isinstance (self._filenames[0], ROOT.TFile):
            self._event = ROOT.fwlite.Event (self._filenames[0])
            self._mode = 'single'
            return self._mode
        if len (self._filenames) == 1 and self._forceEvent:
            self._tfile = ROOT.TFile.Open (self._filenames[0])
            self._event = ROOT.fwlite.Event (self._tfile)
            self._mode = 'single'
            return self._mode
        filenamesSVec = ROOT.vector("string") ()
        for name in self._filenames:
            filenamesSVec.push_back (name)
        if self._secondaryFilenames:
            secondarySVec =  ROOT.vector("string") ()
            for name in self._secondaryFilenames:
                secondarySVec.push_back (name)
            self._event = ROOT.fwlite.MultiChainEvent (filenamesSVec,
                                                       secondarySVec)
            self._mode = 'multi'
        else:
            self._event = ROOT.fwlite.ChainEvent (filenamesSVec)
            self._mode = 'chain'
        return self._mode


    def _next (self):
        """(Internal) Iterator internals"""
        if self._veryFirstTime:
            self._createFWLiteEvent()
        if self._toBegin:
            self._toBeginCode()
        while not self._event.atEnd() :
            yield self
            self._eventCounts += 1
            if self._maxEvents > 0 and self._eventCounts >= self._maxEvents:
                break
            # Have we been asked to go to the first event?
            if self._toBegin:
                self._toBeginCode()
            else:
                # if not, lets go to the next event
                self._event.__preinc__()
            


if __name__ == "__main__":
    # test code can go here
    pass
