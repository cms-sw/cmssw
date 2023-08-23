from __future__ import print_function
from ROOT import *
from array import array

#-------------------------------------------------------------------------------
def tfile_cd(dirname, tfile, debug=False):

  """ Safely re-build and navigate the directory structure. dirname is
  considered to be an absolute path."""

  gDirectory.cd("/")
  if tfile.GetDirectory(dirname):
    gDirectory.cd(dirname)
  else:
    path=""
    for component in dirname.split('/'):
      path += "/%s" % component
      if not tfile.GetDirectory(path):
        gDirectory.mkdir(component)
      gDirectory.cd(component)

  if debug:
    print("Current dir %s" % gDirectory.pwd())

def loadStreamerInfo(literal, debug):

  """Decode a literal made of TStreamerInfo informations and load
  streamers that are not part of the currently used version of
  ROOT. The implementation is a back-to-bone and simplified version of
  the one contained in the DQM GUI source code."""

  bitsarray = array('B')
  bitsarray.frombytes(bytes.fromhex(literal))

  tbuffer = TBufferFile(TBufferFile.kRead)
  tbuffer.Reset();
  tbuffer.SetBuffer(bitsarray, len(bitsarray), False)
  while tbuffer.Length() != tbuffer.BufferSize():
    obj = tbuffer.ReadObject(eval("TStreamerInfo.Class()"))
    v = obj.GetClassVersion()
    c = TClass.GetClass(obj.GetName(), kTRUE)
    if c:
      c.GetStreamerInfo();
      if c.GetStreamerInfos().At(v):
        if debug:
          print("skipping already present streamer info version %d for %s" % (v, obj.GetName()))
        continue
    if debug:
      print("Importing streamer info version %d for %s" % (v, obj.GetName()))
    obj.BuildCheck();

#-------------------------------------------------------------------------------
def literal2root (literal, rootType, debug=False):

  """Convert an hexadecimal string into a root-object. In case a
  TStreamerInfo object is passed, this will be decoded by the
  loadStreamerInfo function to handle it properly and a None object
  will be returned. It is the responsibility of the user not the use
  the returned object in this very case."""

  if rootType == "TStreamerInfo":
    loadStreamerInfo(literal, debug)
    return None

  bitsarray = array('B')
  bitsarray.frombytes(bytes.fromhex(literal))

  tbuffer = TBufferFile(TBufferFile.kRead)
  tbuffer.SetBuffer(bitsarray,len(bitsarray),False)

  # replace a couple of shortcuts with the real root class name
  if rootType == 'TPROF':
      rootType = 'TProfile'
  if rootType == 'TPROF2D':
      rootType = 'TProfile2D'

  root_class = eval(rootType+'.Class()')

  return tbuffer.ReadObject(root_class)

