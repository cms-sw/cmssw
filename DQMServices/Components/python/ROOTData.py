from ROOT import *
from array import array

#-------------------------------------------------------------------------------
def tfile_cd(dirname,tfile,debug=False):
  """
  Safely re-build and navigate the directory structure.
  """

  if not tfile.GetDirectory(dirname):
    gDirectory.cd("/")
    path=""
    for component in dirname.split('/'):
      path+="/%s" %component
      if not tfile.GetDirectory(path):
        gDirectory.mkdir(component)
      gDirectory.cd(component)

      if debug:
        print "Current dir"
        gDirectory.pwd()

#-------------------------------------------------------------------------------
def literal2root (literal,rootType,tstreamerinfo=None):
  """
  Convert an hexadecimal string into a root-object. The correct TStreamerInfo
  is passed in order to avoid inconsistencies.
  """
  bitsarray = array('B')
  bitsarray.fromstring(literal.decode('hex'))

  tbuffer = TBufferFile(TBufferFile.kRead)
  if tstreamerinfo:
    tbuffer.IncrementLevel(tstreamerinfo)
  tbuffer.SetBuffer(bitsarray,len(bitsarray),False)

  # replace a couple of shortcuts with the real root class name
  if rootType == 'TPROF':
      rootType = 'TProfile'
  if rootType == 'TPROF2D':
      rootType = 'TProfile2D'

  root_class=eval(rootType+'.Class()')

  return tbuffer.ReadObject(root_class)

