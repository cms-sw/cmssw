"""Python helper tools for CMS FWLite

benedikt.hegner@cern.ch

"""
import re
import ROOT
import exceptions
### define tab completion
try:
  import readline, cmscompleter
  readline.parse_and_bind('tab: complete')
except:
  print 'WARNING: Could not load tab completion'


### workaround iterator generators for ROOT classes
def all(container):

  # loop over ROOT::TTree and similar
  if hasattr(container,'GetEntries'):
    try:
      entries = container.GetEntries()
      for entry in xrange(entries):
        yield entry
    except:
      pass

  # loop over std::vectors and similar
  elif hasattr(container, 'size'):
    try:
      entries = container.size()
      for entry in xrange(entries):
        yield container[entry]
    except:
      pass

 # loop over containers with begin and end iterators
def loop(begin, end):
    """Convert a pair of C++ iterators into a python generator"""
    while (begin != end):
        yield begin.__deref__()  #*b
        begin.__preinc__()       #++b 

### auto branch types (Chris Jones)
def createBranchBuffer(branch):
    reColons = re.compile(r'::')
    reCloseTemplate =re.compile(r'>')
    reOpenTemplate =re.compile(r'<')
    branchType = ROOT.branchToClass(branch)
    buffer = eval ('ROOT.'+reColons.sub(".",reOpenTemplate.sub("(ROOT.",reCloseTemplate.sub(")",branchType.GetName())))+'()')
    if( branch.GetName()[-1] != '.'):
        branch.SetAddress(buffer)
    else:
        branch.SetAddress(ROOT.AddressOf(buffer))
    return buffer

class EventTree(object):
      def __init__(self,ttree):
          self._tree = ttree
          self._usedBranches = dict()
          self._index = -1
          self._aliases = ttree.GetListOfAliases()
      def branch(self,name):
          # support for aliases
          alias = self._tree.GetAlias(name)
          if alias != '': name = alias 
          # access the branch in ttree
          if name in self._usedBranches:
              return self._usedBranches[name]
          self._usedBranches[name]=EventBranch(self,name)
          return self._usedBranches[name]
      def getListOfAliases(self):
          return self._aliases
      def tree(self):
          return self._tree
      def index(self):
          return self._index
      def __setBranchIndicies(self):
          for branch in self._usedBranches.itervalues():
              branch.setIndex(self._index)
      def __getitem__(self,key):
          if key <0 or key > self._tree.GetEntries():
              raise IndexError
          self._index = key
          self.__setBranchIndicies()
      def __iter__(self):
          for entry in xrange(self._tree.GetEntries()):
              self._index = entry
              self.__setBranchIndicies()
              yield entry


class EventBranch(object):
    def __init__(self,parent,name):
        self._branch = parent.tree().GetBranch(name)
        if self._branch == None:
            raise cmserror("Unknown branch "+name)
        self._buffer = createBranchBuffer(self._branch)
        self._index = parent.index()
        self._readData = False
    def setIndex(self,index):
        self._index = index
        self._readData = False
    def __readData(self):
        self._branch.GetEntry(self._index)
        self._readData = True

    # replace this by __getattr__ to allow branch.attr instead of branch().attr
    def __call__(self):
        if not self._readData:
            self.__readData()
        return self._buffer

class cmserror(exceptions.StandardError):
    def __init__(self, message):
          print "========================================"
          print "ERROR:", message
          print "========================================"
