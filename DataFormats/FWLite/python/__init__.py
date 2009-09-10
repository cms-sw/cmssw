#! /usr/bin/env python

import ROOT
import inspect
import sys

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
            return
        if not self._wrapper.isPresent():
            self._exception = RuntimeError ("getByLabel (%s, %s) not present this event" \
                                            % (self, labelString))
            return
        # if we're still here, then everything is happy.  Clear the exception
        self._exception = None



class Events:
    """Python interface to FWLite ChainEvent class"""

    def __init__(self, inputFiles = '', **kwargs):
        """inputFiles    => Either a single filename or a list of filenames
        Optional arguments:
        forceEvent  => Use fwlite::Event IF there is only one file
        maxEvents   => Maximum number of events to process
        """
        self._veryFirstTime = True
        self._event         = 0
        self._eventCounts   = 0
        self._maxEvents     = 0
        self._forceEvent    = True
        self._mode          = None
        if kwargs.has_key ('maxEvents'):
            self._maxEvents = kwargs['maxEvents']
            del kwargs['maxEvents']
        if kwargs.has_key ('forceEvent'):
            self._forceEvent = kwargs['forceEvent']
            del kwargs['forceEvent']
        # Since we deleted the options as we used them, that means
        # that kwargs should be empty.  If it's not, that means that
        # somebody passed in an argument that we're not using and we
        # should complain.
        if len (kwargs):
            raise RuntimeError, "Unknown arguments %s" % kwargs
        if isinstance (inputFiles, list):
            self._filenames = inputFiles[:]
        else:
            self._filenames = [inputFiles]
        if not self._filenames:
            raise RuntimeError, "No input files given"

        
    def toBegin (self):
        """Called to reset event loop to first event."""
        self._toBegin = True


    def getByLabel (self, *args):
        """Calls FWLite's getByLabel.  Called:
        getByLabel (moduleLabel, handle)
        getByLabel (moduleLabel, productInstanceLabel, handle), or
        getByLabel (moduleLabel, productInstanceLabel, processLabel, handle)
        """
        length = len (args)
        if length < 2 or length > 4:
            # not called correctly
            raise RuntimeError, "Incorrect number of arguments"
        # handle is always the last argument
        argsList = list (args)
        handle = argsList.pop()
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


    def eventIndex (self):
        if 'chain' == self._mode:
            return self._event.eventIndex()
        else:
            return 0


    ## Private Member Functions ##


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
        if len (self._filenames) == 1 and self._forceEvent:
            self._tfile = ROOT.TFile.Open (self._filenames[0])
            self._event = ROOT.fwlite.Event (self._tfile)
            self._mode = 'single'
            return
        filenamesSVec = ROOT.vector("string") ()
        for name in self._filenames:
            filenamesSVec.push_back (name)
        self._event = ROOT.fwlite.ChainEvent (filenamesSVec)
        self._mode = 'chain'


    def _next (self):
        """(Internal) Iterator internals"""
        if self._veryFirstTime:
            self._createFWLiteEvent()
        if self._toBegin:
            self._toBeginCode()
        while not self._event.atEnd() :
            yield self
            self._eventCounts += 1
            if self._maxEvents and self._eventCounts >= self._maxEvents:
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
