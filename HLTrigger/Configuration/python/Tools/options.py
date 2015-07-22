# available "type"s and relative global tags
globalTag = {
  'Fake': 'auto:run1_mc_Fake',
  'FULL': 'auto:run2_mc_FULL',
  'GRun': 'auto:run2_mc_GRun',       # used as default
  '25ns14e33_v2': 'auto:run2_mc_GRun',
  '50ns_5e33_v2': 'auto:run2_mc_50nsGRun',
  '25ns14e33_v1': 'auto:run2_mc_GRun',
  '50ns_5e33_v1': 'auto:run2_mc_50nsGRun',
  '50nsGRun': 'auto:run2_mc_50nsGRun',
  '50ns' : 'auto:run2_mc_50nsGRun',
  'HIon' : 'auto:run2_mc_HIon',
  'PIon' : 'auto:run2_mc_PIon',
  'LowPU': 'auto:run2_mc_LowPU',
  'data' : 'auto:run1_hlt',
}


# type used to store a reference to an L1 menu
class ConnectionL1TMenu(object):
  def __init__(self, value):
    self.override = None
    self.connect  = None

    # extract the override tag and the connection string
    if value:
      if ',' in value:
        self.override = value.split(',')[0]
        self.connect  = value.split(',')[1]
      else:
        self.override = value
        self.connect  = None


# type used to store a reference to an L1 menu
class ConnectionL1TMenuXml(object):
  def __init__(self, value):
    self.XmlFile = None
    self.LumiDir = None

    # extract the override tag and the connection string
    if value:
      if ',' in value:
        self.XmlFile = value.split(',')[0]
        self.LumiDir = value.split(',')[1]
      else:
        self.XmlFile = value
        self.LumiDir = "startup"


# type used to store a reference to an HLT configuration
class ConnectionHLTMenu(object):
  def __init__(self, value):
    self.value = value
    self.db    = None
    self.name  = None
    self.run   = None

    # extract the database and configuration name
    if value:
      if ':' in self.value:
        (db, name) = self.value.split(':')
        if db == 'run':
          self.db   = 'orcoff'
          self.run  = name
        elif db in ('hltdev', 'orcoff'):
          self.db   = db
          self.name = name
        else:
          raise Exception('Unknown ConfDB database "%s", valid values are "hltdev" (default) and "orcoff")' % db)
      else:
        self.db   = 'hltdev'
        self.name = self.value


# options marked with a (*) only apply when creating a whole process configuration
class HLTProcessOptions(object):
  def __init__(self):
    self.menu       = None        #     hlt menu
    self.name       = 'HLTX'      # (*) if set, override the process name
    self.type       = 'GRun'      #     defines global options for 'GRun', 'HIon', 'PIon' or 'online' menus
    self.data       = True        #     run on data (true) or mc (false)
    self.online     = False       # (*) run online (true) or offline (false)
    self.globaltag  = None        # (*) if set, override the GlobalTag
    self.l1         = None        # (*) if set, override the L1 menu
    self.l1Xml      = None        # (*) if set, override the L1 menu Xml
    self.l1skim     = False       # (*) if set, add snippet to process L1 skim files done with new L1, ignoring old L1
    self.emulator   = None        # (*) if set, run (part of) the L1 emulator instead of taking the L1 results from the data
    self.prescale   = None        # (*) if set, force the use of a specific prescale column. If set to "none", unprescale all paths
    self.open       = False       #     if set, cms.ignore all filters, making all paths run on and accept all events
    self.errortype  = False       #     if set, change all HLTTriggerTypeFilter EDFilters to accept only error events (SelectedTriggerType = 0)
    self.profiling  = False       #     if set, instrument the menu for profiling measurements
    self.timing     = False       #     if set, instrument the menu for timing measurements (implies profiling)
    self.paths      = None        #     if set, include in the dump only the given paths (wildcards are supported)
    self.input      = None        # (*) if set, specify the input file(s) or dataset
    self.parent     = None        # (*) if set, specify the parent input file(s) or dataset
    self.events     = 100         # (*) run on these many events
    self.output     = 'all'       # (*) output 'all', 'minimal' or 'none' output modules
    self.fragment   = False       #     prepare a configuration fragment (true) or a whole process (false)
    self.hilton     = False       #     prepare a configuration for running with hilton-like modules


  # convert HLT and L1 menus to a dedicated object representation on the fly
  def __setattr__(self, name, value):
    if name is 'menu' and type(value) is not ConnectionHLTMenu:
      # format 'menu' as needed
      object.__setattr__(self, name, ConnectionHLTMenu(value))
    elif name is 'l1' and type(value) is not ConnectionL1TMenu:
      # format '--l1' as needed
      object.__setattr__(self, name, ConnectionL1TMenu(value))
    elif name is 'l1Xml' and type(value) is not ConnectionL1TMenuXml:
      # format '--l1Xml' as needed
      object.__setattr__(self, name, ConnectionL1TMenuXml(value))
    elif name is 'open' and value:
      # '--open' implies '--unprescale'
      object.__setattr__(self, 'open',      True)
      object.__setattr__(self, 'prescale',  "none")
    elif name is 'prescale' and value is not None:
      # '--open' overrides '--prescale', set the prescale value only if '--open' is not set
      if not self.open:
        object.__setattr__(self, 'prescale', value)
    elif name is 'profiling' and value:
      # '--profiling'
      object.__setattr__(self, 'profiling', True)
    elif name is 'timing' and value:
      # '--timing' implies '--profiling'
      object.__setattr__(self, 'timing',    True)
      object.__setattr__(self, 'profiling', True)
    else:
      object.__setattr__(self, name, value)
