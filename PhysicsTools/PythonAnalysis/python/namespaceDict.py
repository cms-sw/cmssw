"""parsing the LCGDict namespace and build dictionary
(quick'n'dirty solution)

benedikt.hegner@cern.ch

"""

import popen2
import string

def getNamespaceDict():

  # execute SealPluginDump to get list of plugins
  process = popen2.Popen3('SealPluginDump')
  output = process.fromchild.read()

  namespaceDict = {}

  for line in output.split('\n'):
    # search for Reflex, global namespace and non templates
    if line.find('Reflex') !=-1 and line.find(':') ==-1 and line.find('<') ==-1:
      className = line.replace('LCGReflex/', '').strip()
      namespaceDict[className] = line.strip()
    
  return namespaceDict
