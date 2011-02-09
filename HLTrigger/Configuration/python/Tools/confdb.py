#!/usr/bin/env python

import sys
import re
from pipe import pipe as _pipe
from options import globalTag


class HLTProcess(object):
  def __init__(self, configuration):
    self.config = configuration
    self.data   = None
    self.source = None

    self.options = {
      'essources' : [],
      'esmodules' : [],
      'modules'   : [],
      'sequences' : [],
      'services'  : [],
      'paths'     : [],
      'psets'     : [],
      'blocks'    : [],
    }

    self.labels = {}
    if self.config.fragment:
      self.labels['process'] = ''
      self.labels['dict']    = 'locals()'
    else:
      self.labels['process'] = 'process.'
      self.labels['dict']    = 'process.__dict__'

    if self.config.online:
      self.labels['connect'] = 'frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)'
    else:
      self.labels['connect'] = 'frontier://FrontierProd'

    # get the configuration from ConfdB
    self.buildOptions()
    self.expandWildcardOptions()
    self.getRawConfigurationFromDB()
    self.customize()


  def _build_query(self):
    if self.config.menu.run:
      return '--runNumber %s' % self.config.menu.run
    else:
      return '--%s --configName %s' % (self.config.menu.db, self.config.menu.name)

  def _build_source(self):
    if self.source is None:
      return '--noedsources'
    else:
      return '--input ' + self.source

  def _build_options(self):
    return ' '.join(['--%s %s' % (key, ','.join(vals)) for key, vals in self.options.iteritems() if vals])

  def _build_cmdline(self):
    if not self.config.fragment:
      return 'edmConfigFromDB %s %s %s'       % (self._build_query(), self._build_source(), self._build_options())
    else:
      return 'edmConfigFromDB --cff %s %s %s' % (self._build_query(), self._build_source(), self._build_options())


  def getRawConfigurationFromDB(self):
    cmdline = self._build_cmdline()
    data = _pipe(cmdline)
    if 'Exhausted Resultset' in data or 'CONFIG_NOT_FOUND' in data:
      raise ImportError('%s is not a valid HLT menu' % self.config.menuConfig.value)
    self.data = data


  def getPathList(self):
    cmdline = 'edmConfigFromDB --cff %s --noedsources --noes --noservices --nosequences --nomodules' % self._build_query()
    data = _pipe(cmdline)
    if 'Exhausted Resultset' in data or 'CONFIG_NOT_FOUND' in data:
      raise ImportError('%s is not a valid HLT menu' % self.config.menuConfig.value)
    filter = re.compile(r' *= *cms.(End)?Path.*')
    paths  = [ filter.sub('', line) for line in data.splitlines() if filter.search(line) ]
    return paths


  def expandWildcardOptions(self):
    # for the time being, this is limited only to the --paths option
    self.options['paths'] = self.expandWildcards(self.options['paths'], self.getPathList())


  @staticmethod
  def expandWildcards(globs, collection):
    # expand a list of unix-style wildcards matching a given collection
    # wildcards with no matches are silently discarded
    matches = []
    for glob in globs:
      negate = ''
      if glob[0] == '-':
        negate = '-'
        glob   = glob[1:]
      # translate a unix-style glob expression into a regular expression
      filter = re.compile(r'^' + glob.replace('?', '.').replace('*', '.*').replace('[!', '[^') + r'$')
      matches.extend( negate + element for element in collection if filter.match(element) )
    return matches


  # dump the final configuration
  def dump(self):
    return self.data % self.labels


  # customize the configuration according to the options
  def customize(self):
    if self.config.fragment:
      # if running on MC, adapt the configuration accordingly
      self.fixForMC()

      # if requested, adapt the configuration for FastSim
      self.fixForFastSim()

      # if requested, remove the HLT prescales
      self.unprescale()

      # if requested, override all ED/HLTfilters to always pass ("open" mode)
      self.instrumentOpenMode()

      # if requested, instrument the self with the modules and EndPath needed for timing studies
      self.instrumentTiming()

    else:
      # if running on MC, adapt the configuration accordingly
      self.fixForMC()

      # override the process name and adapt the relevant filters
      self.overrideProcessName()

      # if required, remove the HLT prescales
      self.unprescale()

      # if requested, override all ED/HLTfilters to always pass ("open" mode)
      self.instrumentOpenMode()

      # manual override some Heavy Ion parameters
      if self.config.type in ('HIon', ):
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
      self.overrideOutput()

      # add global options
      self.addGlobalOptions()

      # if requested or necessary, override the GlobalTag and connection strings
      self.overrideGlobalTag()

      # if requested, override the L1 self from the GlobalTag (using the same connect as the GlobalTag itself)
      self.overrideL1Menu()

      # request summary informations from the MessageLogger
      self.updateMessageLogger()

      # if requested, instrument the self with the modules and EndPath needed for timing studies
      self.instrumentTiming()


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


  def _fix_parameter(self, **args):
    """arguments:
        name:     parameter name (optional)
        type:     parameter type (look for tracked and untracked variants)
        value:    original value
        replace:  replacement value
    """
    if 'name' in args:
      self.data = re.sub(
          r'%(name)s = cms(?P<tracked>(?:\.untracked)?)\.%(type)s\( (?P<quote>["\']?)%(value)s(?P=quote)' % args,
          r'%(name)s = cms\g<tracked>.%(type)s( \g<quote>%(replace)s\g<quote>' % args,
          self.data)
    else:
      self.data = re.sub(
          r'cms(?P<tracked>(?:\.untracked)?)\.%(type)s\( (?P<quote>["\']?)%(value)s(?P=quote)' % args,
          r'cms\g<tracked>.%(type)s( \g<quote>%(replace)s\g<quote>' % args,
          self.data)


  def fixForMC(self):
    if not self.config.data:
      # override the raw data collection label
      self._fix_parameter(type = 'InputTag', value = 'source', replace = 'rawDataCollector')
      self._fix_parameter(type = 'string',   value = 'source', replace = 'rawDataCollector')


  def fixForFastSim(self):
    if self.config.fastsim:
      # adapt the hle configuration (fragment) to run under fastsim
      self.data = re.sub( r'import FWCore.ParameterSet.Config as cms', r'\g<0>\nfrom FastSimulation.HighLevelTrigger.HLTSetup_cff import *', self.data)

      # remove the definition of streams and datasets
      self.data = re.compile( r'^streams.*\n(.*\n)*?^\)\s*\n',  re.MULTILINE ).sub( '', self.data )
      self.data = re.compile( r'^datasets.*\n(.*\n)*?^\)\s*\n', re.MULTILINE ).sub( '', self.data )

      # fix the definition of module
      self._fix_parameter(                               type = 'InputTag', value = 'hltL1extraParticles',  replace = 'l1extraParticles')
      self._fix_parameter(name = 'GMTReadoutCollection', type = 'InputTag', value = 'hltGtDigis',           replace = 'gmtDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltGtDigis',           replace = 'gtDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltL1GtObjectMap',     replace = 'gtDigis')
      self._fix_parameter(name = 'initialSeeds',         type = 'InputTag', value = 'noSeedsHere',          replace = 'globalPixelSeeds:GlobalPixel')
      self._fix_parameter(name = 'preFilteredSeeds',     type = 'bool',     value = 'True',                 replace = 'False')
      self._fix_parameter(                               type = 'InputTag', value = 'hltOfflineBeamSpot',   replace = 'offlineBeamSpot')
      self._fix_parameter(                               type = 'InputTag', value = 'hltMuonCSCDigis',      replace = 'simMuonCSCDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltMuonDTDigis',       replace = 'simMuonDTDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltMuonRPCDigis',      replace = 'simMuonRPCDigis')

      # fix the definition of sequences and paths
      self.data = re.sub( r'hltMuonCSCDigis', r'cms.SequencePlaceholder( "simMuonCSCDigis" )',  self.data )
      self.data = re.sub( r'hltMuonDTDigis',  r'cms.SequencePlaceholder( "simMuonDTDigis" )',   self.data )
      self.data = re.sub( r'hltMuonRPCDigis', r'cms.SequencePlaceholder( "simMuonRPCDigis" )',  self.data )
      self.data = re.sub( r'HLTEndSequence',  r'cms.SequencePlaceholder( "HLTEndSequence" )',   self.data )
      self.data = re.sub( r'hltGtDigis',      r'HLTBeginSequence',                              self.data )


  def unprescale(self):
    if self.config.unprescale:
      self.data += """
# remove the HLT prescales
if 'PrescaleService' in %(dict)s:
    %(process)sPrescaleService.lvl1DefaultLabel = cms.untracked.string( '0' )
    %(process)sPrescaleService.lvl1Labels = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    %(process)sPrescaleService.prescaleTable = cms.VPSet( )
"""


  def instrumentOpenMode(self):
    if self.config.open:
      # find all EDfilters
      filters = [ match[1] for match in re.findall(r'(process\.)?\b(\w+) = cms.EDFilter', self.data) ]
      # wrap all EDfilters with "cms.ignore( ... )"
      re_filters  = re.compile( r'\b((process\.)?(' + r'|'.join(filters) + r'))\b' )
      re_sequence = re.compile( r'cms\.(Path|Sequence)\((.*)\)' )
      self.data = re_sequence.sub( lambda line: re_filters.sub( r'cms.ignore( \1 )', line.group(0) ), self.data )


  def overrideGlobalTag(self):
    # overwrite GlobalTag
    # the logic is:
    #   - for running online, do nothing, unless a globaltag has been specified on the command line
    #   - for running offline on data, only add the pfnPrefix
    #   - for running offline on mc, take the GT from the command line of the configuration.type
    #      - if the GT is "auto:...", insert the code to read it from Configuration.PyReleaseValidation.autoCond
    text = ''
    if self.config.online:
      if self.config.globaltag:
        # override the GlobalTag
        text += """
# override the GlobalTag
if 'GlobalTag' in %%(dict)s:
    %%(process)sGlobalTag.globaltag = '%(globaltag)s'
"""

    else:
      # override the GlobalTag connection string and pfnPrefix
      text += """
# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in %%(dict)s:
    %%(process)sGlobalTag.connect   = '%%(connect)s/CMS_COND_31X_GLOBALTAG'
    %%(process)sGlobalTag.pfnPrefix = cms.untracked.string('%%(connect)s/')
"""

      if self.config.data:
        # do not override the GlobalTag unless one was specified on the command line
        pass
      else:
        # check if a specific GlobalTag was specified on the command line, or choose one from the configuration.type
        if not self.config.globaltag:
          if self.config.type in globalTag:
            self.config.globaltag = globalTag[self.config.type]
          else:
            self.config.globaltag = globalTag['GRun']

      # check if the GlobalTag is an autoCond or an explicit tag
      if not self.config.globaltag:
        # when running on data, do not override the GlobalTag unless one was specified on the command line
        pass
      elif self.config.globaltag.startswith('auto:'):
        self.config.menuGlobalTagAuto = self.config.globaltag[5:]
        text += "    from Configuration.PyReleaseValidation.autoCond import autoCond\n"
        text += "    %%(process)sGlobalTag.globaltag = autoCond['%(menuGlobalTagAuto)s']\n"
      else:
        text += "    %%(process)sGlobalTag.globaltag = '%(globaltag)s'\n"

    self.data += text % self.config.__dict__


  def overrideL1Menu(self):
    # if requested, override the L1 menu from the GlobalTag (using the same connect as the GlobalTag itself)
    if self.config.l1.override:
      self.config.l1.record = 'L1GtTriggerMenuRcd'
      self.config.l1.label  = ''
      self.config.l1.tag    = self.config.l1.override
      if not self.config.l1.connect:
        self.config.l1.connect = '%(connect)s/CMS_COND_31X_L1T'
      self.loadAdditionalConditions( 'override the L1 menu', self.config.l1.__dict__ )


  def overrideOutput(self):
    reOutputModuleDef = re.compile(r'\b(process\.)?hltOutput(\w+) *= *cms\.OutputModule\(.*\n([^)].*\n)*\) *\n')
    reOutputModuleRef = re.compile(r' *[+*]? *\b(process\.)?hltOutput(\w+)')    # FIXME this does not cover "hltOutputX + something"
    if self.config.output == 'none':
      # drop all output modules
      self.data = reOutputModuleDef.sub('', self.data)
      self.data = reOutputModuleRef.sub('', self.data)

    elif self.config.output == 'minimal':
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
  def overrideProcessName(self):
    # the following was stolen and adapted from HLTrigger.Configuration.customL1THLT_Options
    self.data += """
# override the process name
%%(process)ssetName_('%(name)s')

# adapt HLT modules to the correct process name
if 'hltTrigReport' in %%(dict)s:
    %%(process)shltTrigReport.HLTriggerResults       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltDQMHLTScalers' in %%(dict)s:
    %%(process)shltDQMHLTScalers.triggerResults      = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressSmart' in %%(dict)s:
    %%(process)shltPreExpressSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTMONSmart' in %%(dict)s:
    %%(process)shltPreHLTMONSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMSmart' in %%(dict)s:
    %%(process)shltPreDQMSmart.TriggerResultsTag     = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltDQML1SeedLogicScalers' in %%(dict)s:
    %%(process)shltDQML1SeedLogicScalers.processname = '%(name)s'
""" % self.config.__dict__


  def updateMessageLogger(self):
    # request summary informations from the MessageLogger
    self.data += """
if 'MessageLogger' in %(dict)s:
    %(process)sMessageLogger.categories.append('TriggerSummaryProducerAOD')
    %(process)sMessageLogger.categories.append('L1GtTrigReport')
    %(process)sMessageLogger.categories.append('HLTrigReport')
"""


  def loadAdditionalConditions(self, comment, *conditions):
    # load additional conditions
    self.data += """
# %s
if 'GlobalTag' in %%(dict)s:
""" % comment
    for condition in conditions:
      self.data += """    %%(process)sGlobalTag.toGet.append(
        cms.PSet(
            record  = cms.string( '%(record)s' ),
            tag     = cms.string( '%(tag)s' ),
            label   = cms.untracked.string( '%(label)s' ),
            connect = cms.untracked.string( '%(connect)s' )
        )
    )
""" % condition

  def instrumentTiming(self):
    if self.config.timing:
      # instrument the menu with the modules and EndPath needed for timing studies
      text = ''

      if 'HLTriggerFirstPath' in self.data:
        # remove HLTriggerFirstPath
        self.data = re.sub(r'.*\bHLTriggerFirstPath\s*=.*\n', '', self.data)

      if not 'hltGetRaw' in self.data:
        # add hltGetRaw
        text += """
%%(process)shltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "%s" )
)
""" % ( self.config.data and 'source' or 'rawDataCollector' )

      if not 'hltGetConditions' in self.data:
        # add hltGetConditions
        text += """
%(process)shltGetConditions = cms.EDAnalyzer( 'EventSetupRecordDataGetter',
    verbose = cms.untracked.bool( False ),
    toGet = cms.VPSet( )
)
"""

      # add the definition of HLTriggerFirstPath
      text += """
%(process)sHLTriggerFirstPath = cms.Path( %(process)shltGetRaw + %(process)shltGetConditions + %(process)shltBoolFalse )
"""
      self.data = re.sub(r'.*cms\.(End)?Path.*', text + r'\g<0>', self.data, 1)

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
      self.loadAdditionalConditions('add XML geometry to keep hltGetConditions happy',
        {
          'record'  : 'GeometryFileRcd',
          'tag'     : 'XMLFILE_Geometry_380V3_Ideal_mc',
          'label'   : 'Ideal',
          'connect' : '%(connect)s/CMS_COND_34X_GEOMETRY'
        }, {
          'record'  : 'GeometryFileRcd',
          'tag'     : 'XMLFILE_Geometry_380V3_Extended_mc',
          'label'   : 'Extended',
          'connect' : '%(connect)s/CMS_COND_34X_GEOMETRY'
        }
      )


  def buildOptions(self):
    # common configuration for all scenarios
    self.options['services'].append( "-FUShmDQMOutputService" )
    self.options['paths'].append( "-OfflineOutput" )

    # adapt source and options to the current scenario
    if not self.config.fragment:
      self.build_source()

    if self.config.fragment:
      # extract a configuration file fragment
      self.options['essources'].append( "-GlobalTag" )
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
      self.options['esmodules'].append( "-CaloTowerHardcodeGeometryEP" )
      self.options['esmodules'].append( "-CastorHardcodeGeometryEP" )
      self.options['esmodules'].append( "-DTGeometryESModule" )
      self.options['esmodules'].append( "-EcalBarrelGeometryEP" )
      self.options['esmodules'].append( "-EcalElectronicsMappingBuilder" )
      self.options['esmodules'].append( "-EcalEndcapGeometryEP" )
      self.options['esmodules'].append( "-EcalLaserCorrectionService" )
      self.options['esmodules'].append( "-EcalPreshowerGeometryEP" )
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
      self.options['esmodules'].append( "-ZdcHardcodeGeometryEP" )
      self.options['esmodules'].append( "-hcal_db_producer" )
      self.options['esmodules'].append( "-L1GtTriggerMaskAlgoTrigTrivialProducer" )
      self.options['esmodules'].append( "-L1GtTriggerMaskTechTrigTrivialProducer" )
      self.options['esmodules'].append( "-hltESPEcalTrigTowerConstituentsMapBuilder" )
      self.options['esmodules'].append( "-hltESPGlobalTrackingGeometryESProducer" )
      self.options['esmodules'].append( "-hltESPMuonDetLayerGeometryESProducer" )
      self.options['esmodules'].append( "-hltESPTrackerRecoGeometryESProducer" )
      if not self.config.fastsim:
        self.options['esmodules'].append( "-CaloTowerGeometryFromDBEP" )
        self.options['esmodules'].append( "-CastorGeometryFromDBEP" )
        self.options['esmodules'].append( "-EcalBarrelGeometryFromDBEP" )
        self.options['esmodules'].append( "-EcalEndcapGeometryFromDBEP" )
        self.options['esmodules'].append( "-EcalPreshowerGeometryFromDBEP" )
        self.options['esmodules'].append( "-HcalGeometryFromDBEP" )
        self.options['esmodules'].append( "-ZdcGeometryFromDBEP" )
        self.options['esmodules'].append( "-XMLFromDBSource" )
        self.options['esmodules'].append( "-sistripconn" )

      self.options['services'].append( "-PrescaleService" )
      self.options['services'].append( "-MessageLogger" )
      self.options['services'].append( "-DQM" )
      self.options['services'].append( "-MicroStateService" )
      self.options['services'].append( "-ModuleWebRegistry" )
      self.options['services'].append( "-TimeProfilerService" )
      if not self.config.fastsim:
        self.options['services'].append( "-DQMStore" )

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

    if self.config.fastsim:
      # remove components not supported or needed by fastsim
      self.options['essources'].append( "-BTagRecord" )

      self.options['esmodules'].append( "-SiPixelTemplateDBObjectESProducer" )
      self.options['esmodules'].append( "-TTRHBuilderPixelOnly" )
      self.options['esmodules'].append( "-WithTrackAngle" )
      self.options['esmodules'].append( "-trajectoryCleanerBySharedHits" )
      self.options['esmodules'].append( "-trackCounting3D2nd" )
      self.options['esmodules'].append( "-navigationSchoolESProducer" )
      self.options['esmodules'].append( "-muonCkfTrajectoryFilter" )
      self.options['esmodules'].append( "-ckfBaseTrajectoryFilter" )
      self.options['esmodules'].append( "-TransientTrackBuilderESProducer" )
      self.options['esmodules'].append( "-TrackerRecoGeometryESProducer" )
      self.options['esmodules'].append( "-SteppingHelixPropagatorOpposite" )
      self.options['esmodules'].append( "-SteppingHelixPropagatorAny" )
      self.options['esmodules'].append( "-SteppingHelixPropagatorAlong" )
      self.options['esmodules'].append( "-SmootherRK" )
      self.options['esmodules'].append( "-SmartPropagatorRK" )
      self.options['esmodules'].append( "-SmartPropagatorOpposite" )
      self.options['esmodules'].append( "-SmartPropagatorAnyRK" )
      self.options['esmodules'].append( "-SmartPropagatorAnyOpposite" )
      self.options['esmodules'].append( "-SmartPropagatorAny" )
      self.options['esmodules'].append( "-SmartPropagator" )
      self.options['esmodules'].append( "-RungeKuttaTrackerPropagator" )
      self.options['esmodules'].append( "-OppositeMaterialPropagator" )
      self.options['esmodules'].append( "-MuonTransientTrackingRecHitBuilderESProducer" )
      self.options['esmodules'].append( "-MuonDetLayerGeometryESProducer" )
      self.options['esmodules'].append( "-MuonCkfTrajectoryBuilder" )
      self.options['esmodules'].append( "-hltMeasurementTracker" )
      self.options['esmodules'].append( "-MaterialPropagator" )
      self.options['esmodules'].append( "-L3MuKFFitter" )
      self.options['esmodules'].append( "-KFUpdatorESProducer" )
      self.options['esmodules'].append( "-KFSmootherForRefitInsideOut" )
      self.options['esmodules'].append( "-KFSmootherForMuonTrackLoader" )
      self.options['esmodules'].append( "-KFFitterForRefitInsideOut" )
      self.options['esmodules'].append( "-GroupedCkfTrajectoryBuilder" )
      self.options['esmodules'].append( "-GlobalTrackingGeometryESProducer" )
      self.options['esmodules'].append( "-FittingSmootherRK" )
      self.options['esmodules'].append( "-FitterRK" )
      self.options['esmodules'].append( "-hltCkfTrajectoryBuilder" )
      self.options['esmodules'].append( "-Chi2MeasurementEstimator" )
      self.options['esmodules'].append( "-Chi2EstimatorForRefit" )
      self.options['esmodules'].append( "-CaloTowerConstituentsMapBuilder" )
      self.options['esmodules'].append( "-CaloTopologyBuilder" )

      self.options['services'].append( "-UpdaterService" )

      self.options['blocks'].append( "hltL1NonIsoLargeWindowElectronPixelSeeds::SeedConfiguration" )
      self.options['blocks'].append( "hltL1IsoLargeWindowElectronPixelSeeds::SeedConfiguration" )
      self.options['blocks'].append( "hltL1NonIsoStartUpElectronPixelSeeds::SeedConfiguration" )
      self.options['blocks'].append( "hltL1IsoStartUpElectronPixelSeeds::SeedConfiguration" )

      self.options['modules'].append( "hltL3MuonIsolations" )
      self.options['modules'].append( "hltPixelVertices" )
      self.options['modules'].append( "-hltCkfL1IsoTrackCandidates" )
      self.options['modules'].append( "-hltCtfL1IsoWithMaterialTracks" )
      self.options['modules'].append( "-hltCkfL1NonIsoTrackCandidates" )
      self.options['modules'].append( "-hltCtfL1NonIsoWithMaterialTracks" )
      self.options['modules'].append( "hltPixelMatchLargeWindowElectronsL1Iso" )
      self.options['modules'].append( "hltPixelMatchLargeWindowElectronsL1NonIso" )
      self.options['modules'].append( "-hltESRegionalEgammaRecHit" )
      self.options['modules'].append( "-hltEcalRegionalJetsFEDs" )
      self.options['modules'].append( "-hltEcalRegionalJetsRecHitTmp" )
      self.options['modules'].append( "-hltEcalRegionalMuonsFEDs" )
      self.options['modules'].append( "-hltEcalRegionalMuonsRecHitTmp" )
      self.options['modules'].append( "-hltEcalRegionalEgammaFEDs" )
      self.options['modules'].append( "-hltEcalRegionalEgammaRecHitTmp" )
      self.options['modules'].append( "-hltFEDSelector" )
      self.options['modules'].append( "-hltL3TrajSeedOIHit" )
      self.options['modules'].append( "-hltL3TrajSeedIOHit" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2OIState" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2OIHit" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2IOHit" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2NoVtx" )
      self.options['modules'].append( "-hltHcalDigis" )
      self.options['modules'].append( "-hltHoreco" )
      self.options['modules'].append( "-hltHfreco" )
      self.options['modules'].append( "-hltHbhereco" )
      self.options['modules'].append( "-hltEcalRegionalRestFEDs" )
      self.options['modules'].append( "-hltEcalRegionalESRestFEDs" )
      self.options['modules'].append( "-hltEcalRawToRecHitFacility" )
      self.options['modules'].append( "-hltESRawToRecHitFacility" )
      self.options['modules'].append( "-hltEcalRegionalJetsRecHit" )
      self.options['modules'].append( "-hltEcalRegionalMuonsRecHit" )
      self.options['modules'].append( "-hltEcalRegionalEgammaRecHit" )
      self.options['modules'].append( "-hltEcalRecHitAll" )
      self.options['modules'].append( "-hltESRecHitAll" )
      self.options['modules'].append( "-hltL3TauPixelSeeds" )
      self.options['modules'].append( "-hltL3TauHighPtPixelSeeds" )
      self.options['modules'].append( "-hltL3TauCkfTrackCandidates" )
      self.options['modules'].append( "-hltL3TauCkfHighPtTrackCandidates" )
      self.options['modules'].append( "-hltL3TauCtfWithMaterialTracks" )
      self.options['modules'].append( "-hltL25TauPixelSeeds" )
      self.options['modules'].append( "-hltL25TauCkfTrackCandidates" )
      self.options['modules'].append( "-hltL25TauCtfWithMaterialTracks" )
      self.options['modules'].append( "-hltL3TauSingleTrack15CtfWithMaterialTracks" )
      self.options['modules'].append( "-hltPFJetCtfWithMaterialTracks" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorStartup" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesStartup" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksStartup" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorStartupU" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesStartupU" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksStartupU" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGenerator" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidates" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracks" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorRelaxed" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesRelaxed" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksRelaxed" )
      self.options['modules'].append( "-hltPixelTracksForMinBias" )
      self.options['modules'].append( "-hltPixelTracksForHighMult" )
      self.options['modules'].append( "-hltMuonCSCDigis" )
      self.options['modules'].append( "-hltMuonDTDigis" )
      self.options['modules'].append( "-hltMuonRPCDigis" )
      self.options['modules'].append( "-hltGtDigis" )
      self.options['modules'].append( "-hltL1GtTrigReport" )
      self.options['modules'].append( "hltCsc2DRecHits" )
      self.options['modules'].append( "hltDt1DRecHits" )
      self.options['modules'].append( "hltRpcRecHits" )

      self.options['sequences'].append( "-HLTL1IsoEgammaRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTL1NonIsoEgammaRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTL1IsoElectronsRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTL1NonIsoElectronsRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTPixelMatchLargeWindowElectronL1IsoTrackingSequence" )
      self.options['sequences'].append( "-HLTPixelMatchLargeWindowElectronL1NonIsoTrackingSequence" )
      self.options['sequences'].append( "-HLTPixelTrackingForMinBiasSequence" )
      self.options['sequences'].append( "-HLTDoLocalStripSequence" )
      self.options['sequences'].append( "-HLTDoLocalPixelSequence" )
      self.options['sequences'].append( "-HLTRecopixelvertexingSequence" )
      self.options['sequences'].append( "-HLTL3TauTrackReconstructionSequence" )
      self.options['sequences'].append( "-HLTL3TauHighPtTrackReconstructionSequence" )
      self.options['sequences'].append( "-HLTL25TauTrackReconstructionSequence" )
      self.options['sequences'].append( "-HLTL3TauSingleTrack15ReconstructionSequence" )
      self.options['sequences'].append( "-HLTTrackReconstructionForJets" )
      self.options['sequences'].append( "-HLTEndSequence" )
      self.options['sequences'].append( "-HLTBeginSequence" )
      self.options['sequences'].append( "-HLTBeginSequenceNZS" )
      self.options['sequences'].append( "-HLTBeginSequenceBPTX" )
      self.options['sequences'].append( "-HLTBeginSequenceAntiBPTX" )
      self.options['sequences'].append( "-HLTL2HcalIsolTrackSequence" )
      self.options['sequences'].append( "-HLTL2HcalIsolTrackSequenceHB" )
      self.options['sequences'].append( "-HLTL2HcalIsolTrackSequenceHE" )
      self.options['sequences'].append( "-HLTL3HcalIsolTrackSequence" )

      # remove unsupported paths
      self.options['paths'].append( "-AlCa_EcalEta" )
      self.options['paths'].append( "-AlCa_EcalPhiSym" )
      self.options['paths'].append( "-AlCa_EcalPi0" )
      self.options['paths'].append( "-AlCa_RPCMuonNoHits" )
      self.options['paths'].append( "-AlCa_RPCMuonNoTriggers" )
      self.options['paths'].append( "-AlCa_RPCMuonNormalisation" )
      self.options['paths'].append( "-DQM_FEDIntegrity" )
      self.options['paths'].append( "-DQM_FEDIntegrity_v*" )
      self.options['paths'].append( "-HLT_Activity_DT" )
      self.options['paths'].append( "-HLT_Activity_DT_Tuned" )
      self.options['paths'].append( "-HLT_Activity_Ecal" )
      self.options['paths'].append( "-HLT_Activity_EcalREM" )
      self.options['paths'].append( "-HLT_Activity_Ecal_SC15" )
      self.options['paths'].append( "-HLT_Activity_Ecal_SC17" )
      self.options['paths'].append( "-HLT_Activity_Ecal_SC7" )
      self.options['paths'].append( "-HLT_Activity_L1A" )
      self.options['paths'].append( "-HLT_Activity_PixelClusters" )
      self.options['paths'].append( "-HLT_Calibration" )
      self.options['paths'].append( "-HLT_DTErrors" )
      self.options['paths'].append( "-HLT_DoubleEle4_SW_eeRes_L1R" )
      self.options['paths'].append( "-HLT_DoubleEle4_SW_eeRes_L1R_v*" )
      self.options['paths'].append( "-HLT_DoubleEle5_SW_Upsilon_L1R_v*" )
      self.options['paths'].append( "-HLT_DoublePhoton4_Jpsi_L1R" )
      self.options['paths'].append( "-HLT_DoublePhoton4_Upsilon_L1R" )
      self.options['paths'].append( "-HLT_DoublePhoton4_eeRes_L1R" )
      self.options['paths'].append( "-HLT_EcalCalibration" )
      self.options['paths'].append( "-HLT_EgammaSuperClusterOnly_L1R" )
      self.options['paths'].append( "-HLT_Ele15_SiStrip_L1R" )
      self.options['paths'].append( "-HLT_Ele20_SiStrip_L1R" )
      self.options['paths'].append( "-HLT_HFThreshold10" )
      self.options['paths'].append( "-HLT_HFThreshold3" )
      self.options['paths'].append( "-HLT_HcalCalibration" )
      self.options['paths'].append( "-HLT_HcalNZS" )
      self.options['paths'].append( "-HLT_HcalPhiSym" )
      self.options['paths'].append( "-HLT_IsoTrackHB_v*" )
      self.options['paths'].append( "-HLT_IsoTrackHE_v*" )
      self.options['paths'].append( "-HLT_Jet15U_HcalNoiseFiltered" )
      self.options['paths'].append( "-HLT_Jet15U_HcalNoiseFiltered_v*" )
      self.options['paths'].append( "-HLT_L1DoubleMuOpen_Tight" )
      self.options['paths'].append( "-HLT_L1MuOpen_AntiBPTX" )
      self.options['paths'].append( "-HLT_L1MuOpen_AntiBPTX_v*" )
      self.options['paths'].append( "-HLT_Mu0_TkMu0_OST_Jpsi" )
      self.options['paths'].append( "-HLT_Mu0_TkMu0_OST_Jpsi_Tight_v*" )
      self.options['paths'].append( "-HLT_Mu0_Track0_Jpsi" )
      self.options['paths'].append( "-HLT_Mu3_TkMu0_OST_Jpsi" )
      self.options['paths'].append( "-HLT_Mu3_TkMu0_OST_Jpsi_Tight_v*" )
      self.options['paths'].append( "-HLT_Mu3_Track0_Jpsi" )
      self.options['paths'].append( "-HLT_Mu3_Track3_Jpsi" )
      self.options['paths'].append( "-HLT_Mu3_Track3_Jpsi_v*" )
      self.options['paths'].append( "-HLT_Mu3_Track5_Jpsi_v*" )
      self.options['paths'].append( "-HLT_Mu5_TkMu0_OST_Jpsi" )
      self.options['paths'].append( "-HLT_Mu5_TkMu0_OST_Jpsi_Tight_v*" )
      self.options['paths'].append( "-HLT_Mu5_Track0_Jpsi" )
      self.options['paths'].append( "-HLT_Mu5_Track0_Jpsi_v*" )
      self.options['paths'].append( "-HLT_Random" )
      self.options['paths'].append( "-HLT_SelectEcalSpikesHighEt_L1R" )
      self.options['paths'].append( "-HLT_SelectEcalSpikes_L1R" )

      # remove HLTAnalyzerEndpath from fastsim cff's
      if self.config.fragment:
        self.options['paths'].append( "-HLTAnalyzerEndpath" )


  def build_source(self):
    if self.config.online:
      # online we always run on data
      self.source = "file:/tmp/InputCollection.root"
    elif self.config.data:
      # offline we can run on data...
      self.source = "/store/data/Run2010B/MinimumBias/RAW/v1/000/149/291/DC6C917A-0EE3-DF11-867B-001617C3B654.root"
    else:
      # ...or on mc
      self.source = "file:RelVal_DigiL1Raw_%s.root" % self.config.type

