# available "type"s and relative global tags
globalTag = {
  'Fake' : 'auto:run1_mc_Fake',
  'Fake1': 'auto:run2_mc_Fake1',
  'Fake2': 'auto:run2_mc_Fake2',
  'FULL' : 'auto:run3_mc_FULL',
  'GRun' : 'auto:run3_mc_GRun',       # used as default
  'HIon' : 'auto:run3_mc_HIon',
  'PIon' : 'auto:run3_mc_PIon',
  'PRef' : 'auto:run3_mc_PRef',
  'data' : 'auto:run3_hlt_relval',
  '2022v15' : 'auto:run3_mc_2022v15',
  '2023v11' : 'auto:run3_mc_2023v11',
}


# type used to store a reference to an L1 menu
class ConnectionL1TMenu(object):
  def __init__(self, value):
    self.override = None
    self.snapshotTime = None

    # extract the override tag and the connection string
    if value:
      if ',' in value:
        self.override = value.split(',')[0]
        self.snapshotTime = value.split(',')[1]
      else:
        self.override = value
        self.smapshotTime = None


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
  valid_versions  = 'v1', 'v2', 'v3', 'v3-beta', 'v3-test'
  valid_databases = 'online', 'run3', 'adg','dev','run2'
  compatibility   = { 'hltdev': ('v3', 'run3'), 'orcoff': ('v3', 'adg') }

  def __init__(self, value):
    self.version    = None
    self.database   = None
    self.name       = None
    self.run        = None

    if not value:
      return

    if not ':' in value:
      # default to 'v3/run3'
      self.version    = 'v3'
      self.database   = 'run3'
      self.name       = value
      return

    # extract the version, database and configuration name
    tokens = value.split(':')
    if len(tokens) != 2:
      raise Exception('Invalid HLT menu specification "%s"' % value)
    (db, name) = tokens
    # check if the menu should be automatically determined based on the run number
    if db == 'run':
      self.version  = 'v3'
      self.database = 'adg'
      self.run      = name
    # check for backward compatibility names
    elif db in self.compatibility:
      self.version, self.database = self.compatibility[db]
      self.name = name
    else:
      if '/' in db:
        # extract the version and database
        tokens = db.split('/')
        if len(tokens) != 2:
          raise Exception('Invalid HLT menu specification "%s"' % value)
        (v, db) = tokens
        if v not in self.valid_versions:
          raise Exception('Invalid HLT database version "%s", valid values are "%s"' % (v, '", "'.join(self.valid_versions)))
        if db not in self.valid_databases:
          raise Exception('Invalid HLT database "%s", valid values are "%s"' % (db, '", "'.join(self.valid_databases)))
        self.version  = v
        self.database = db
        self.name     = name
      else:
        # use the confdb v3 by default
        if db not in self.valid_databases:
          raise Exception('Invalid HLT database "%s", valid values are "%s"' % (db, '", "'.join(self.valid_databases)))
        self.database = db
        self.version  = 'v3'
        self.name     = name

# options marked with a (*) only apply when creating a whole process configuration
class HLTProcessOptions(object):
  def __init__(self):
    self.menu       = None        #     hlt menu
    self.name       = 'HLTX'      # (*) if set, override the process name
    self.type       = 'GRun'      #     defines global options for 'GRun', 'HIon', 'PIon', 'PRef' or 'online' menus
    self.data       = True        #     run on data (true) or mc (false)
    self.globaltag  = None        # (*) if set, override the GlobalTag
    self.l1         = None        # (*) if set, override the L1 menu
    self.l1Xml      = None        # (*) if set, override the L1 menu Xml
    self.emulator   = None        # (*) if set, run (part of) the L1 emulator instead of taking the L1 results from the data
    self.prescale   = None        # (*) if set, force the use of a specific prescale column. If set to "none", unprescale all paths
    self.open       = False       #     if set, cms.ignore all filters, making all paths run on and accept all events
    self.eras       = None        #     if set, select the defined Eras into the HLT configuration
    self.customise  = None        #     if set, apply the user-defined customization functions using the format HLTrigger/Configuration/customizeHLTTrackingForPhaseI2017.customizeHLTForPFTrackingPhaseI2017
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
    self.setup      = None        #     if set, downlad the setup_cff from the specified configuration and load it.
    self.proxy      = False       #     use a socks proxy to connect
    self.proxy_host = 'localhost' #     host of the proxy server
    self.proxy_port = '8080'      #     port of the proxy server
    self.tunnel     = False       #     use a direct tunnel on localhost to connect
    self.tunnel_port = '10121'    #     port to connect to on localhost when tunneling

  # convert HLT and L1 menus to a dedicated object representation on the fly
  def __setattr__(self, name, value):
    if name == 'menu' and not isinstance(value, ConnectionHLTMenu):
      # format 'menu' as needed
      object.__setattr__(self, name, ConnectionHLTMenu(value))
    elif name == 'l1' and not isinstance(value, ConnectionL1TMenu):
      # format '--l1' as needed
      object.__setattr__(self, name, ConnectionL1TMenu(value))
    elif name == 'l1Xml' and not isinstance(value, ConnectionL1TMenuXml):
      # format '--l1Xml' as needed
      object.__setattr__(self, name, ConnectionL1TMenuXml(value))
    elif name == 'open' and value:
      # '--open' implies '--unprescale'
      object.__setattr__(self, 'open',      True)
      object.__setattr__(self, 'prescale',  "none")
    elif name == 'prescale' and value is not None:
      # '--open' overrides '--prescale', set the prescale value only if '--open' is not set
      if not self.open:
        object.__setattr__(self, 'prescale', value)
    elif name == 'profiling' and value:
      # '--profiling'
      object.__setattr__(self, 'profiling', True)
    elif name == 'timing' and value:
      # '--timing' implies '--profiling'
      object.__setattr__(self, 'timing',    True)
      object.__setattr__(self, 'profiling', True)
    elif name == 'setup' and value and value.find(":")!=-1:
      raise Exception('you can not specify a converter/database in the setup option.\nIt takes the converter database specified by the primary config.\nPlease remove the text upto and including the ":" in\n  {} '.format(value))      
    else:
      object.__setattr__(self, name, value)
