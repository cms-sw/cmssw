#!/usr/bin/env python

# TODO
# --timing implies --no-output, unless overridden
# fix prescaler modules with --open

import sys
import re
import shlex, subprocess


# available "type"s and relative global tags
globalTag = {
  'FULL': 'auto:startup',
  'GRun': 'auto:startup',       # use as default
  'data': 'auto:hltonline',
  'HIon': 'auto:starthi',
}


# type used to store a reference to an L1 menu
class ConnectionL1TMenu(object):
  def __init__(self, value):
    self.override = None
    self.connect  = None

    # extract the connection string and configuration name
    if value:
      if ':' in value:
        self.override = "L1GtTriggerMenu_%s_mc" % value.rsplit(':', 1)[1]
        self.connect  = '"%s"' % value.rsplit(':', 1)[0]
      else:
        self.override = "L1GtTriggerMenu_%s_mc" % value
        self.connect  = None


# type used to store a reference to an HLT configuration
class ConnectionHLTMenu(object):
  def __init__(self, value):
    self.value  = value
    self.db     = None
    self.name   = None
    self.run    = None

    # extract the database and configuration name
    if value:
      if ':' in self.value:
        (db, name) = self.value.split(':')
        if db == 'run':
          self.run  = name
        elif db in ('hltdev', 'orcoff'):
          self.db   = db
          self.name = name
        else:
          print 'Unknown ConfDB database "%s", valid values are "hltdev" (default) and "orcoff")' % db
          sys.exit(1)
      else:
        self.db   = 'hltdev'
        self.name = self.value


# wrapper around subprocess to simplify te interface
def _pipe(cmdline, input = None):
  args = shlex.split(cmdline)
  if input is not None:
    command = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
  else:
    command = subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE, stderr=None)
  (out, err) = command.communicate(input)
  return out


class HLTProcess(object):
  def __init__(self):
    self.data     = None
    self.labels   = {}

  @staticmethod
  def _build_query(config):
    if config.run:
      return '--runNumber %s' % config.run
    else:
      return '--%s --configName %s' % (config.db, config.name)

  @staticmethod
  def _build_source(src):
    if src is None:
      return '--noedsources'
    else:
      return '--input ' + src

  @staticmethod
  def _build_options(opts):
    return ' '.join(['--%s %s' % (key, ','.join(vals)) for key, vals in opts.iteritems() if vals])

  def _build_cmdline(self, config, src, opts, fragment):
    if not fragment:
      return 'edmConfigFromDB %s %s %s'       % (self._build_query(config), self._build_source(src), self._build_options(opts))
    else:
      return 'edmConfigFromDB --cff %s %s %s' % (self._build_query(config), self._build_source(src), self._build_options(opts))

  def getRawConfigurationFromDB(self, config, confdb, fragment):
    if fragment:
      self.labels['process'] = ''
      self.labels['dict']    = 'locals()'
    else:
      self.labels['process'] = 'process.'
      self.labels['dict']    = 'process.__dict__'

    cmdline = self._build_cmdline(config, confdb.source, confdb.options, fragment)
    data = _pipe(cmdline)
    
    if 'Exhausted Resultset' in data or 'CONFIG_NOT_FOUND' in data:
      raise ImportError('%s is not a valid HLT menu' % configuration.menuConfig.value)

    self.data = data

  # dump the final configuration
  def dump(self):
    return self.data % self.labels


  # customize the configuration according to the options
  def customize(self, configuration):
    if configuration.doCff:

      # if running on MC, adapt the configuration accordingly
      self.fixForMC(configuration)

      # if requested, remove the HLT prescales
      self.unprescale(configuration)

      # if requested, override all ED/HLTfilters to always pass ("open" mode)
      self.instrumentOpenMode(configuration)

      # if requested or necessary, override the GlobalTag and connection strings
      #self.overrideGlobalTag(configuration)

      # if requested, override the L1 self from the GlobalTag (using the same connect as the GlobalTag itself)
      #self.overrideL1Menu(configuration)

      # if requested, instrument the self with the modules and EndPath needed for timing studies
      self.instrumentTiming(configuration)

    else:

      # if running on MC, adapt the configuration accordingly
      self.fixForMC(configuration)

      # override the process name and adapt the relevant filters
      self.overrideProcessName(configuration)

      # if required, remove the HLT prescales
      self.unprescale(configuration)

      # if requested, override all ED/HLTfilters to always pass ("open" mode)
      self.instrumentOpenMode(configuration)

      # manual override some Heavy Ion parameters
      if configuration.processType in ('HIon', ):
        self.data += """
# HIon paths in smart prescalers
if 'hltPreHLTDQMSmart' in %(dict)s:
    %(process)shltPreHLTDQMSmart.throw  = cms.bool( False )
if 'hltPreHLTMONSmart' in %(dict)s:
    %(process)shltPreHLTMONSmart.throw  = cms.bool( False )
if 'hltPreExpressSmart' in %(dict)s:
    %(process)shltPreExpressSmart.throw = cms.bool( False )
if 'hltPreDQMSmart' in %(dict)s:
    %(process)shltPreDQMSmart.throw     = cms.bool( False )
"""        

      # override the output modules to output root files
      self.overrideOutput(configuration)

      # add global options
      self.addGlobalOptions()

      # if requested or necessary, override the GlobalTag and connection strings
      self.overrideGlobalTag(configuration)

      # if requested, override the L1 self from the GlobalTag (using the same connect as the GlobalTag itself)
      self.overrideL1Menu(configuration)

      # request summary informations from the MessageLogger
      self.updateMessageLogger()

      # if requested, instrument the self with the modules and EndPath needed for timing studies
      self.instrumentTiming(configuration)


  def addGlobalOptions(self):
    # add global options
    self.data += """
# add global options
%(process)smaxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)
%(process)soptions = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)
"""


  def fixForMC(self, configuration):
    if not configuration.runOnData:
      # override the raw data collection label
      self.data = re.sub( r'cms\.InputTag\( "source" \)',            r'cms.InputTag( "rawDataCollector" )',           self.data)
      self.data = re.sub( r'cms\.untracked\.InputTag\( "source" \)', r'cms.untracked.InputTag( "rawDataCollector" )', self.data)
      self.data = re.sub( r'cms\.string\( "source" \)',              r'cms.string( "rawDataCollector" )',             self.data)


  def unprescale(self, configuration):
    if configuration.menuUnprescale:
      self.data += """
# remove the HLT prescales
if 'PrescaleService' in %(dict)s:
    %(process)sPrescaleService.lvl1DefaultLabel = cms.untracked.string( '0' )
    %(process)sPrescaleService.lvl1Labels = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    %(process)sPrescaleService.prescaleTable = cms.VPSet( )
"""


  def instrumentOpenMode(self, configuration):
    if configuration.menuOpen:
      # find all EDfilters
      filters = [ match[1] for match in re.findall(r'(process\.)?\b(\w+) = cms.EDFilter', self.data) ] 
      # wrap all EDfilters with "cms.ignore( ... )"
      re_filters  = re.compile( r'\b((process\.)?(' + r'|'.join(filters) + r'))\b' )
      re_sequence = re.compile( r'cms\.(Path|Sequence)\((.*)\)' )
      self.data = re_sequence.sub( lambda line: re_filters.sub( r'cms.ignore( \1 )', line.group(0) ), self.data )


  def overrideGlobalTag(self, configuration):
    # overwrite GlobalTag
    # the logic is:
    #   - for running online, do nothing, unless a globaltag has been specified on the command line
    #   - for running offline on data, only add the pfnPrefix
    #   - for running offline on mc, take the GT from the command line of the configuration.processType
    #      - if the GT is "auto:...", insert the code to read it from Configuration.PyReleaseValidation.autoCond
    text = ''
    if configuration.runOnline:
      if configuration.menuGlobalTag:
        text += """
# override the GlobalTag 
if 'GlobalTag' in %%(dict)s:
    %%(process)sGlobalTag.globaltag = '%(menuGlobalTag)s'
"""

    else:
      text += """
# override the GlobalTag connection string and pfnPrefix
if 'GlobalTag' in %%(dict)s:
"""

      # override the GlobalTag connection string and pfnPrefix
      text += "    %%(process)sGlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'\n"
      text += "    %%(process)sGlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')\n"

      if configuration.runOnData:
        # do not override the GlobalTag unless one was specified on the command line 
        pass
      else:
        # check if a specific GlobalTag was specified on the command line, or choose one from the configuration.processType
        if not configuration.menuGlobalTag:
          if configuration.processType in globalTag:
            configuration.menuGlobalTag = globalTag[configuration.processType]
          else:
            configuration.menuGlobalTag = globalTag['GRun']

      # check if the GlobalTag is an autoCond or an explicit tag
      if not configuration.menuGlobalTag:
        # when running on data, do not override the GlobalTag unless one was specified on the command line
        pass
      elif configuration.menuGlobalTag[0:5] == 'auto:':
        configuration.menuGlobalTagAuto = configuration.menuGlobalTag[5:]
        text += "    from Configuration.PyReleaseValidation.autoCond import autoCond\n"
        text += "    %%(process)sGlobalTag.globaltag = autoCond['%(menuGlobalTagAuto)s']\n"
      else:
        text += "    %%(process)sGlobalTag.globaltag = '%(menuGlobalTag)s'\n"

    self.data += text % configuration.__dict__


  def overrideL1Menu(self, configuration):
    # if requested, override the L1 menu from the GlobalTag (using the same connect as the GlobalTag itself)
    if configuration.menuL1.override:
      if not configuration.menuL1.connect:
        configuration.menuL1.connect = "%(process)sGlobalTag.connect.value().replace('CMS_COND_31X_GLOBALTAG', 'CMS_COND_31X_L1T')"
      self.data += """
# override the L1 menu
if 'GlobalTag' in %%(dict)s:
    %%(process)sGlobalTag.toGet.append(
        cms.PSet(  
            record  = cms.string( "L1GtTriggerMenuRcd" ),
            tag     = cms.string( "%(override)s" ),
            connect = cms.untracked.string( %(connect)s )
        )
    )
""" % configuration.menuL1.__dict__


  def overrideOutput(self, configuration):
    reOutputModuleDef = re.compile(r'\b(process\.)?hltOutput(\w+) *= *cms\.OutputModule\(.*\n([^)].*\n)*\) *\n')
    reOutputModuleRef = re.compile(r' *[+*]? *\b(process\.)?hltOutput(\w+)')    # FIXME this does not cover "hltOutputX + something"
    if configuration.menuOutput == 'none':
      # drop all output modules
      self.data = reOutputModuleDef.sub('', self.data)
      self.data = reOutputModuleRef.sub('', self.data)

    elif configuration.menuOutput == 'minimal':
      # drop all output modules except "HLTDQMResults"
      repl = lambda match: (match.group(2) == 'HLTDQMResults') and match.group() or ''
      self.data = reOutputModuleDef.sub(repl, self.data)
      self.data = reOutputModuleRef.sub(repl, self.data)

    # override the "online" ShmStreamConsumer output modules with "offline" PoolOutputModule's
    self.data = re.sub( 
      r'\b(process\.)?hltOutput(\w+) *= *cms\.OutputModule\( *"ShmStreamConsumer" *,',
      r'%(process)shltOutput\2 = cms.OutputModule( "PoolOutputModule",\n    fileName = cms.untracked.string( "output\2.root" ),\n    fastCloning = cms.untracked.bool( False ),', 
      self.data
    )


  # override the process name and adapt the relevant filters
  def overrideProcessName(self, configuration):
    # the following was stolen and adapted from HLTrigger.Configuration.customL1THLT_Options
    self.data += """
# override the process name
%%(process)ssetName_('%(processName)s')

# adapt HLT modules to the correct process name
if 'hltTrigReport' in %%(dict)s:
    %%(process)shltTrigReport.HLTriggerResults       = cms.InputTag( 'TriggerResults', '', '%(processName)s' )

if 'hltDQMHLTScalers' in %%(dict)s:
    %%(process)shltDQMHLTScalers.triggerResults      = cms.InputTag( 'TriggerResults', '', '%(processName)s' )

if 'hltPreExpressSmart' in %%(dict)s:
    %%(process)shltPreExpressSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', '%(processName)s' )

if 'hltPreHLTMONSmart' in %%(dict)s:
    %%(process)shltPreHLTMONSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', '%(processName)s' )

if 'hltPreDQMSmart' in %%(dict)s:
    %%(process)shltPreDQMSmart.TriggerResultsTag     = cms.InputTag( 'TriggerResults', '', '%(processName)s' )

if 'hltDQML1SeedLogicScalers' in %%(dict)s:
    %%(process)shltDQML1SeedLogicScalers.processname = '%(processName)s'
""" % configuration.__dict__


  def updateMessageLogger(self):
    # request summary informations from the MessageLogger
    self.data += """
if 'MessageLogger' in %(dict)s:
    %(process)sMessageLogger.categories.append('TriggerSummaryProducerAOD')
    %(process)sMessageLogger.categories.append('L1GtTrigReport')
    %(process)sMessageLogger.categories.append('HLTrigReport')
"""



  def instrumentTiming(self, configuration):
    if configuration.menuTiming:
      # instrument the menu with the modules and EndPath needed for timing studies
      self.data += """
# instrument the menu with the modules and EndPath needed for timing studies
%(process)sPathTimerService = cms.Service( "PathTimerService",
)
%(process)shltTimer = cms.EDProducer( "PathTimerInserter",
)
%(process)shltOutputTiming = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputTiming.root" ),
    fastCloning = cms.untracked.bool( False ),
    splitLevel = cms.untracked.int32( 0 ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string( 'RECO' ),
        filterName = cms.untracked.string( '' )
    ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep HLTPerformanceInfo_*_*_*' )
)

%(process)sTimingOutput = cms.EndPath( %(process)shltTimer + %(process)shltOutputTiming )
"""


class ConfDB(object):
  def __init__(self, config):
    self.options = {
      'essources' : [],
      'esmodules' : [],
      'modules'   : [],
      'services'  : [],
      'paths'     : [],
      'psets'     : [],
    }
    self.source = None

    # common configuration for all scenarios
    self.options['services'].append( "-FUShmDQMOutputService" )
    self.options['paths'].append( "-OfflineOutput" )

    # adapt source and options to the current scenario
    self.build_source(config)
    self.build_options(config)


  def build_options(self, config):
    if config.doCff:
      # extract a configuration file fragment
      self.options['essources'].append( "-GlobalTag" )
      self.options['essources'].append( "-Level1MenuOverride" )
      self.options['essources'].append( "-HepPDTESSource" )
      self.options['essources'].append( "-XMLIdealGeometryESSource" )
      self.options['essources'].append( "-eegeom" )
      self.options['essources'].append( "-es_hardcode" )
      self.options['essources'].append( "-magfield" )

      self.options['esmodules'].append( "-AutoMagneticFieldESProducer" )
      self.options['esmodules'].append( "-SlaveField0" )
      self.options['esmodules'].append( "-SlaveField20" )
      self.options['esmodules'].append( "-SlaveField30" )
      self.options['esmodules'].append( "-SlaveField35" )
      self.options['esmodules'].append( "-SlaveField38" )
      self.options['esmodules'].append( "-SlaveField40" )
      self.options['esmodules'].append( "-VBF0" )
      self.options['esmodules'].append( "-VBF20" )
      self.options['esmodules'].append( "-VBF30" )
      self.options['esmodules'].append( "-VBF35" )
      self.options['esmodules'].append( "-VBF38" )
      self.options['esmodules'].append( "-VBF40" )
      self.options['esmodules'].append( "-CSCGeometryESModule" )
      self.options['esmodules'].append( "-CaloGeometryBuilder" )
      self.options['esmodules'].append( "-CaloTowerGeometryFromDBEP" )
      self.options['esmodules'].append( "-CaloTowerHardcodeGeometryEP" )
      self.options['esmodules'].append( "-CastorGeometryFromDBEP" )
      self.options['esmodules'].append( "-CastorHardcodeGeometryEP" )
      self.options['esmodules'].append( "-DTGeometryESModule" )
      self.options['esmodules'].append( "-EcalBarrelGeometryEP" )
      self.options['esmodules'].append( "-EcalBarrelGeometryFromDBEP" )
      self.options['esmodules'].append( "-EcalElectronicsMappingBuilder" )
      self.options['esmodules'].append( "-EcalEndcapGeometryEP" )
      self.options['esmodules'].append( "-EcalEndcapGeometryFromDBEP" )
      self.options['esmodules'].append( "-EcalLaserCorrectionService" )
      self.options['esmodules'].append( "-EcalPreshowerGeometryEP" )
      self.options['esmodules'].append( "-EcalPreshowerGeometryFromDBEP" )
      self.options['esmodules'].append( "-HcalGeometryFromDBEP" )
      self.options['esmodules'].append( "-HcalHardcodeGeometryEP" )
      self.options['esmodules'].append( "-HcalTopologyIdealEP" )
      self.options['esmodules'].append( "-MuonNumberingInitialization" )
      self.options['esmodules'].append( "-ParametrizedMagneticFieldProducer" )
      self.options['esmodules'].append( "-RPCGeometryESModule" )
      self.options['esmodules'].append( "-SiStripGainESProducer" )
      self.options['esmodules'].append( "-SiStripRecHitMatcherESProducer" )
      self.options['esmodules'].append( "-SiStripQualityESProducer" )
      self.options['esmodules'].append( "-StripCPEfromTrackAngleESProducer" )
      self.options['esmodules'].append( "-TrackerDigiGeometryESModule" )
      self.options['esmodules'].append( "-TrackerGeometricDetESModule" )
      self.options['esmodules'].append( "-VolumeBasedMagneticFieldESProducer" )
      self.options['esmodules'].append( "-XMLFromDBSource" )
      self.options['esmodules'].append( "-ZdcGeometryFromDBEP" )
      self.options['esmodules'].append( "-ZdcHardcodeGeometryEP" )
      self.options['esmodules'].append( "-hcal_db_producer" )
      self.options['esmodules'].append( "-l1GtTriggerMenuXml" )
      self.options['esmodules'].append( "-L1GtTriggerMaskAlgoTrigTrivialProducer" )
      self.options['esmodules'].append( "-L1GtTriggerMaskTechTrigTrivialProducer" )
      self.options['esmodules'].append( "-sistripconn" )

      self.options['esmodules'].append( "-hltESPEcalTrigTowerConstituentsMapBuilder" )
      self.options['esmodules'].append( "-hltESPGlobalTrackingGeometryESProducer" )
      self.options['esmodules'].append( "-hltESPMuonDetLayerGeometryESProducer" )
      self.options['esmodules'].append( "-hltESPTrackerRecoGeometryESProducer" )

      self.options['services'].append( "-PrescaleService" )
      self.options['services'].append( "-MessageLogger" )
      self.options['services'].append( "-DQM" )
      self.options['services'].append( "-DQMStore" )
      self.options['services'].append( "-MicroStateService" )
      self.options['services'].append( "-ModuleWebRegistry" )
      self.options['services'].append( "-TimeProfilerService" )

      self.options['paths'].append( "-HLTOutput" )
      self.options['paths'].append( "-ExpressOutput" )
      self.options['paths'].append( "-EventDisplayOutput" )
      self.options['paths'].append( "-AlCaOutput" )
      self.options['paths'].append( "-AlCaPPOutput" )
      self.options['paths'].append( "-AlCaHIOutput" )
      self.options['paths'].append( "-DQMOutput" )
      self.options['paths'].append( "-HLTDQMOutput" )
      self.options['paths'].append( "-HLTDQMResultsOutput" )
      self.options['paths'].append( "-HLTMONOutput" )
      self.options['paths'].append( "-NanoDSTOutput" )

      self.options['psets'].append( "-maxEvents" )
      self.options['psets'].append( "-options" )

    else:
      # extract a *full* configuration file
      if not config.runOnData or config.menuL1.override:
        # remove any eventual L1 override from the table
        self.options['essources'].append( "-Level1MenuOverride" )
        self.options['esmodules'].append( "-l1GtTriggerMenuXml" )


  def build_source(self, configuration):
    if configuration.runOnline:
      # online we always run on data
      self.source =  "file:/tmp/InputCollection.root"
    else:
      # offline we can run on data, on mc, or on a user-specified dataset
      if configuration.menuDataset:
        # query DBS and extract the files for the specified dataset
        files = _pipe("dbsql 'find file where dataset like %s'" % configuration.menuDataset)
        files = [ f for f in files.split('\n') if 'store' in f ][0:10]
        self.source = ','.join(files)
      elif configuration.runOnData:
        self.source = "/store/data/Run2010A/MinimumBias/RAW/v1/000/144/011/140DA3FD-AAB1-DF11-8932-001617E30E28.root"
      else:
        self.source =  "file:RelVal_DigiL1Raw_%s.root" % configuration.processType

