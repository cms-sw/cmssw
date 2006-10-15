"""Python helper tools for CMS FWLite

benedikt.hegner@cern.ch

"""

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
