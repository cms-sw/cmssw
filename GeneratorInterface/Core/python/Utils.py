import FWCore.ParameterSet.Config as cms

class LoadPrerequisiteSource(cms.Source):
  """The class is a Source which loads prerequisites libraries in advance. 
     This is done to make sure we can load libraries containing common blocks in
     the correct order.
  """
  def setPrerequisites(self, *libs):
    self.__dict__["libraries"] = libs

  def insertInto(self, parameterSet, myname):
    from ctypes import LibraryLoader, CDLL
    import platform
    loader = LibraryLoader(CDLL)
    ext = platform.uname()[0] == "Darwin" and "dylib" or "so"
    [loader.LoadLibrary("lib%s.%s" % (l, ext)) for l in self.libraries]
    super(LoadPrerequisiteSource,self).insertInto(parameterSet,myname)
