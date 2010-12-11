#!/usr/bin/env python

# TODO
# --timing implies --no-output, unless overridden
# fix prescaler modules with --open

import sys
import re
from pipe import pipe as _pipe
from options import globalTag


class HLTProcess(object):
  def __init__(self, menu, configuration, fragment = False):
    self.menu    = menu
    self.config  = configuration
    self.data    = None
    self.labels  = {}
    self.source  = None
    self.options = {
      'essources' : [],
      'esmodules' : [],
      'modules'   : [],
      'services'  : [],
      'paths'     : [],
      'psets'     : [],
    }
    self.fragment = fragment

    # get the configuration from ConfdB
    self.buildOptions()
    self.getRawConfigurationFromDB()
    self.customize()


  def _build_query(self):
    if self.menu.run:
      return '--runNumber %s' % self.menu.run
    else:
      return '--%s --configName %s' % (self.menu.db, self.menu.name)

  def _build_source(self):
    if self.source is None:
      return '--noedsources'
    else:
      return '--input ' + self.source

  def _build_options(self):
    return ' '.join(['--%s %s' % (key, ','.join(vals)) for key, vals in self.options.iteritems() if vals])

  def _build_cmdline(self):
    if not self.fragment:
      return 'edmConfigFromDB %s %s %s'       % (self._build_query(), self._build_source(), self._build_options())
    else:
      return 'edmConfigFromDB --cff %s %s %s' % (self._build_query(), self._build_source(), self._build_options())

  def getRawConfigurationFromDB(self):
    if self.fragment:
      self.labels['process'] = ''
      self.labels['dict']    = 'locals()'
    else:
      self.labels['process'] = 'process.'
      self.labels['dict']    = 'process.__dict__'

    cmdline = self._build_cmdline()
    data = _pipe(cmdline)
    
    if 'Exhausted Resultset' in data or 'CONFIG_NOT_FOUND' in data:
      raise ImportError('%s is not a valid HLT menu' % self.config.menuConfig.value)

    self.data = data

  # dump the final configuration
  def dump(self):
    return self.data % self.labels


  # customize the configuration according to the options
  def customize(self):
    if self.fragment:
      # if running on MC, adapt the configuration accordingly
      self.fixForMC()

      # if requested, remove the HLT prescales
      self.unprescale()

      # if requested, override all ED/HLTfilters to always pass ("open" mode)
      self.instrumentOpenMode()

      # if requested or necessary, override the GlobalTag and connection strings
      #self.overrideGlobalTag()

      # if requested, override the L1 self from the GlobalTag (using the same connect as the GlobalTag itself)
      #self.overrideL1Menu()

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


  def fixForMC(self):
    if not self.config.data:
      # override the raw data collection label
      self.data = re.sub( r'cms\.InputTag\( "source" \)',            r'cms.InputTag( "rawDataCollector" )',           self.data)
      self.data = re.sub( r'cms\.untracked\.InputTag\( "source" \)', r'cms.untracked.InputTag( "rawDataCollector" )', self.data)
      self.data = re.sub( r'cms\.string\( "source" \)',              r'cms.string( "rawDataCollector" )',             self.data)


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
        text += """
# override the GlobalTag 
if 'GlobalTag' in %%(dict)s:
    %%(process)sGlobalTag.globaltag = '%(globaltag)s'
"""

    else:
      text += """
# override the GlobalTag connection string and pfnPrefix
if 'GlobalTag' in %%(dict)s:
"""

      # override the GlobalTag connection string and pfnPrefix
      text += "    %%(process)sGlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'\n"
      text += "    %%(process)sGlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')\n"

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
      if not self.config.l1.connect:
        self.config.l1.connect = "%(process)sGlobalTag.connect.value().replace('CMS_COND_31X_GLOBALTAG', 'CMS_COND_31X_L1T')"
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
""" % self.config.l1.__dict__


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



  def instrumentTiming(self):
    if self.config.timing:
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


  def buildOptions(self):
    # common configuration for all scenarios
    self.options['services'].append( "-FUShmDQMOutputService" )
    self.options['paths'].append( "-OfflineOutput" )

    # adapt source and options to the current scenario
    if not self.fragment:
      self.build_source()

    if self.fragment:
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


  def build_source(self):
    if self.config.online:
      # online we always run on data
      self.source = "file:/tmp/InputCollection.root"
    elif self.config.data:
      # offline we can run on data...
      self.source = "/store/data/Run2010A/MinimumBias/RAW/v1/000/144/011/140DA3FD-AAB1-DF11-8932-001617E30E28.root"
    else:
      # ...or on mc 
      self.source = "file:RelVal_DigiL1Raw_%s.root" % self.config.type

