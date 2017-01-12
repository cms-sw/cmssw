#!/usr/bin/env python

import sys
import re
import os
import urllib, urllib2
from pipe import pipe as _pipe
from options import globalTag
from itertools import islice

def splitter(iterator, n):
  i = iterator.__iter__()
  while True:
    l = list(islice(i, n))
    if l:
      yield l
    else:
      break


class HLTProcess(object):

  def __init__(self, configuration):
    self.config = configuration
    self.data   = None
    self.source = []
    self.parent = []

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
      self.labels['process'] = 'fragment'
      self.labels['dict']    = 'fragment.__dict__'
    else:
      self.labels['process'] = 'process'
      self.labels['dict']    = 'process.__dict__'

    if self.config.online:
      self.labels['connect'] = 'frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)'
    else:
      self.labels['connect'] = 'frontier://FrontierProd'

    if self.config.prescale and (self.config.prescale.lower() != 'none'):
      self.labels['prescale'] = self.config.prescale

    # get the configuration from ConfdB
    from confdbOfflineConverter import OfflineConverter
    self.converter = OfflineConverter(version = self.config.menu.version, database = self.config.menu.database)
    self.buildPathList()
    self.buildOptions()
    self.getRawConfigurationFromDB()
    self.customize()


  def getRawConfigurationFromDB(self):
    if self.config.menu.run:
      args = ['--runNumber', self.config.menu.run]
    else:
      args = ['--configName', self.config.menu.name ]
    args.append('--noedsources')
    for key, vals in self.options.iteritems():
      if vals:
        args.extend(('--'+key, ','.join(vals)))

    data, err = self.converter.query( *args )
    if 'ERROR' in err or 'Exhausted Resultset' in err or 'CONFIG_NOT_FOUND' in err:
        sys.stderr.write("%s: error while retrieving the HLT menu\n\n" % os.path.basename(sys.argv[0]))
        sys.stderr.write(err + "\n\n")
        sys.exit(1)
    self.data = data


  def getPathList(self):
    if self.config.menu.run:
      args = ['--runNumber', self.config.menu.run]
    else:
      args = ['--configName', self.config.menu.name]
    args.extend( (
      '--cff',
      '--noedsources',
      '--noes',
      '--noservices',
      '--nosequences',
      '--nomodules'
    ) )

    data, err = self.converter.query( *args )
    if 'ERROR' in err or 'Exhausted Resultset' in err or 'CONFIG_NOT_FOUND' in err:
        sys.stderr.write("%s: error while retrieving the list of paths from the HLT menu\n\n" % os.path.basename(sys.argv[0]))
        sys.stderr.write(err + "\n\n")
        sys.exit(1)
    filter = re.compile(r' *= *cms.(End)?Path.*')
    paths  = [ filter.sub('', line) for line in data.splitlines() if filter.search(line) ]
    return paths


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


  @staticmethod
  def consolidateNegativeList(elements):
    # consolidate a list of path exclusions and re-inclusions
    # the result is the list of paths to be removed from the dump
    result = set()
    for element in elements:
      if element[0] == '-':
        result.add( element )
      else:
        result.discard( '-' + element )
    return sorted( element for element in result )

  @staticmethod
  def consolidatePositiveList(elements):
    # consolidate a list of path selection and re-exclusions
    # the result is the list of paths to be included in the dump
    result = set()
    for element in elements:
      if element[0] == '-':
        result.discard( element[1:] )
      else:
        result.add( element )
    return sorted( element for element in result )


  # dump the final configuration
  def dump(self):
    self.data = self.data % self.labels
    if self.config.fragment:
      self.data = re.sub( r'\bprocess\b', 'fragment', self.data )
      self.data = re.sub( r'\bProcess\b', 'ProcessFragment', self.data )
    return self.data


  # add specific customizations
  def specificCustomize(self):
    # specific customizations now live in HLTrigger.Configuration.customizeHLTforALL.customizeHLTforAll(.,.)
    if self.config.fragment:
      self.data += """
# add specific customizations
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
fragment = customizeHLTforAll(fragment,"%s")
""" % (self.config.type)
    else:
      if self.config.type=="Fake":
        prefix = "run1"
      else:
        prefix = "run2"
      _gtData = "auto:"+prefix+"_hlt_"+self.config.type
      _gtMc   = "auto:"+prefix+"_mc_" +self.config.type
      self.data += """
# add specific customizations
_customInfo = {}
_customInfo['menuType'  ]= "%s"
_customInfo['globalTags']= {}
_customInfo['globalTags'][True ] = "%s"
_customInfo['globalTags'][False] = "%s"
_customInfo['inputFiles']={}
_customInfo['inputFiles'][True]  = "file:RelVal_Raw_%s_DATA.root"
_customInfo['inputFiles'][False] = "file:RelVal_Raw_%s_MC.root"
_customInfo['maxEvents' ]=  %s
_customInfo['globalTag' ]= "%s"
_customInfo['inputFile' ]=  %s
_customInfo['realData'  ]=  %s
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
%%(process)s = customizeHLTforAll(%%(process)s,"%s",_customInfo)
""" % (self.config.type,_gtData,_gtMc,self.config.type,self.config.type,self.config.events,self.config.globaltag,self.source,self.config.data,self.config.type)

    self.data += """
from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
%%(process)s = customizeHLTforCMSSW(%%(process)s,"%s")
""" % (self.config.type)

  # customize the configuration according to the options
  def customize(self):

    # adapt the source to the current scenario
    if not self.config.fragment:
      self.build_source()

    # manual override some parameters
    if self.config.type in ('HIon', ):
      if self.config.data:
        if not self.config.fragment:
          self._fix_parameter( type = 'InputTag', value = 'rawDataCollector',  replace = 'rawDataRepacker')

    # if requested, remove the HLT prescales
    self.fixPrescales()

    # if requested, override all ED/HLTfilters to always pass ("open" mode)
    self.instrumentOpenMode()

    # if requested, change all HLTTriggerTypeFilter EDFilters to accept only error events (SelectedTriggerType = 0)
    self.instrumentErrorEventType()

    # if requested, instrument the self with the modules and EndPath needed for timing studies
    self.instrumentTiming()

    # if requested, override the L1 self from the GlobalTag (Xml)
    self.overrideL1MenuXml()

    # if requested, run the L1 emulator
    self.runL1Emulator()

    if self.config.fragment:
      self.data += """
# dummyfy hltGetConditions in cff's
if 'hltGetConditions' in %(dict)s and 'HLTriggerFirstPath' in %(dict)s :
    %(process)s.hltDummyConditions = cms.EDFilter( "HLTBool",
        result = cms.bool( True )
    )
    %(process)s.HLTriggerFirstPath.replace(%(process)s.hltGetConditions,%(process)s.hltDummyConditions)
"""

    else:

      # override the process name and adapt the relevant filters
      self.overrideProcessName()

      # override the output modules to output root files
      self.overrideOutput()

      # add global options
      self.addGlobalOptions()

      # if requested or necessary, override the GlobalTag and connection strings (incl. L1!)
      self.overrideGlobalTag()

      # request summary informations from the MessageLogger
      self.updateMessageLogger()

      # replace DQMStore and DQMRootOutputModule with a configuration suitable for running offline
      self.instrumentDQM()

      # load 5.2.x JECs, until they are in the GlobalTag
#      self.loadAdditionalConditions('load 5.2.x JECs',
#        {
#          'record'  : 'JetCorrectionsRecord',
#          'tag'     : 'JetCorrectorParametersCollection_AK5Calo_2012_V8_hlt_mc',
#          'label'   : 'AK5CaloHLT',
#          'connect' : '%(connect)s/CMS_CONDITIONS'
#        }, {
#          'record'  : 'JetCorrectionsRecord',
#          'tag'     : 'JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc',
#          'label'   : 'AK5PFHLT',
#          'connect' : '%(connect)s/CMS_CONDITIONS'
#        }, {
#          'record'  : 'JetCorrectionsRecord',
#          'tag'     : 'JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc',
#          'label'   : 'AK5PFchsHLT',
#          'connect' : '%(connect)s/CMS_CONDITIONS'
#        }
#      )

    # add specific customisations
    self.specificCustomize()


  def addGlobalOptions(self):
    # add global options
    self.data += """
# limit the number of events to be processed
%%(process)s.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( %d )
)
""" % self.config.events

    if not self.config.profiling:
      self.data += """
# enable the TrigReport and TimeReport
%(process)s.options = cms.untracked.PSet(
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


  def fixPrescales(self):
    # update the PrescaleService to match the new list of paths
    if self.options['paths']:
      if self.options['paths'][0][0] == '-':
        # drop requested paths
        for minuspath in self.options['paths']:
          path = minuspath[1:]
          self.data = re.sub(r'      cms.PSet\(  pathName = cms.string\( "%s" \),\n        prescales = cms.vuint32\( .* \)\n      \),?\n' % path, '', self.data)
      else:
        # keep requested paths
        for path in self.all_paths:
          if path not in self.options['paths']:
            self.data = re.sub(r'      cms.PSet\(  pathName = cms.string\( "%s" \),\n        prescales = cms.vuint32\( .* \)\n      \),?\n' % path, '', self.data)

    if self.config.prescale and (self.config.prescale.lower() != 'none'):
      # TO DO: check that the requested prescale column is valid
      self.data += """
# force the use of a specific HLT prescale column
if 'PrescaleService' in %(dict)s:
    %(process)s.PrescaleService.forceDefault     = True
    %(process)s.PrescaleService.lvl1DefaultLabel = '%(prescale)s'
"""


  def instrumentOpenMode(self):
    if self.config.open:
      # find all EDfilters
      filters = [ match[1] for match in re.findall(r'(process\.)?\b(\w+) = cms.EDFilter', self.data) ]
      re_sequence = re.compile( r'cms\.(Path|Sequence)\((.*)\)' )
      # remove existing 'cms.ignore' and '~' modifiers
      self.data = re_sequence.sub( lambda line: re.sub( r'cms\.ignore *\( *((process\.)?\b(\w+)) *\)', r'\1', line.group(0) ), self.data )
      self.data = re_sequence.sub( lambda line: re.sub( r'~', '', line.group(0) ), self.data )
      # wrap all EDfilters with "cms.ignore( ... )", 1000 at a time (python 2.6 complains for too-big regular expressions)
      for some in splitter(filters, 1000):
        re_filters  = re.compile( r'\b((process\.)?(' + r'|'.join(some) + r'))\b' )
        self.data = re_sequence.sub( lambda line: re_filters.sub( r'cms.ignore( \1 )', line.group(0) ), self.data )


  def instrumentErrorEventType(self):
    if self.config.errortype:
      # change all HLTTriggerTypeFilter EDFilters to accept only error events (SelectedTriggerType = 0)
      self._fix_parameter(name = 'SelectedTriggerType', type ='int32', value = '1', replace = '0')
      self._fix_parameter(name = 'SelectedTriggerType', type ='int32', value = '2', replace = '0')
      self._fix_parameter(name = 'SelectedTriggerType', type ='int32', value = '3', replace = '0')


  def overrideGlobalTag(self):
    # overwrite GlobalTag
    # the logic is:
    #   - always set the correct connection string and pfnPrefix
    #   - if a GlobalTag is specified on the command line:
    #      - override the global tag
    #      - if the GT is "auto:...", insert the code to read it from Configuration.AlCa.autoCond
    #   - if a GlobalTag is NOT  specified on the command line:
    #      - when running on data, do nothing, and keep the global tag in the menu
    #      - when running on mc, take the GT from the configuration.type

    # override the GlobalTag connection string and pfnPrefix
    text = """
# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in %(dict)s:
"""

    # when running on MC, override the global tag even if not specified on the command line
    if not self.config.data and not self.config.globaltag:
      if self.config.type in globalTag:
        self.config.globaltag = globalTag[self.config.type]
      else:
        self.config.globaltag = globalTag['GRun']

    # if requested, override the L1 menu from the GlobalTag (using the same connect as the GlobalTag itself)
    if self.config.l1.override:
      self.config.l1.tag    = self.config.l1.override
      self.config.l1.record = 'L1TUtmTriggerMenuRcd'
      self.config.l1.connect = '%(connect)s/CMS_CONDITIONS'
      self.config.l1.label  = ''
      if not self.config.l1.snapshotTime:
        self.config.l1.snapshotTime = '9999-12-31 23:59:59.000'
      self.config.l1cond = '%(tag)s,%(record)s,%(connect)s,%(label)s,%(snapshotTime)s' % self.config.l1.__dict__
    else:
      self.config.l1cond = None

    if self.config.globaltag or self.config.l1cond:
      text += "    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag\n"
      text += "    %(process)s.GlobalTag = customiseGlobalTag(%(process)s.GlobalTag"
      if self.config.globaltag:
        text += ", globaltag = %s"  % repr(self.config.globaltag)
      if self.config.l1cond:
        text += ", conditions = %s" % repr(self.config.l1cond)
      text += ")\n"

    text += """    %(process)s.GlobalTag.connect   = '%(connect)s/CMS_CONDITIONS'
    %(process)s.GlobalTag.pfnPrefix = cms.untracked.string('%(connect)s/')
    for pset in %(process)s.GlobalTag.toGet.value():
        pset.connect = pset.connect.value().replace('frontier://FrontierProd/', '%(connect)s/')
    # fix for multi-run processing
    %(process)s.GlobalTag.RefreshEachRun = cms.untracked.bool( False )
    %(process)s.GlobalTag.ReconnectEachRun = cms.untracked.bool( False )
"""
    self.data += text

  def overrideL1MenuXml(self):
    # if requested, override the GlobalTag's L1T menu from an Xml file
    if self.config.l1Xml.XmlFile:
      text = """
# override the GlobalTag's L1T menu from an Xml file
from HLTrigger.Configuration.CustomConfigs import L1XML
%%(process)s = L1XML(%%(process)s,"%s")
""" % (self.config.l1Xml.XmlFile) 
      self.data += text

  def runL1Emulator(self):
    # if requested, run the Full L1T emulator, then repack the data into a new RAW collection, to be used by the HLT
    if self.config.emulator:
      text = """
# run the Full L1T emulator, then repack the data into a new RAW collection, to be used by the HLT
from HLTrigger.Configuration.CustomConfigs import L1REPACK
%%(process)s = L1REPACK(%%(process)s,"%s")
""" % (self.config.emulator)
      self.data += text

  def overrideOutput(self):
    # override the "online" ShmStreamConsumer output modules with "offline" PoolOutputModule's
    self.data = re.sub(
      r'\b(process\.)?hltOutput(\w+) *= *cms\.OutputModule\( *"ShmStreamConsumer" *,',
      r'%(process)s.hltOutput\2 = cms.OutputModule( "PoolOutputModule",\n    fileName = cms.untracked.string( "output\2.root" ),\n    fastCloning = cms.untracked.bool( False ),\n    dataset = cms.untracked.PSet(\n        filterName = cms.untracked.string( "" ),\n        dataTier = cms.untracked.string( "RAW" )\n    ),',
      self.data
    )

    if not self.config.fragment and self.config.output == 'full':
      # add a single "keep *" output
      self.data += """
# add a single "keep *" output
%(process)s.hltOutputFULL = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputFULL.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string( 'RECO' ),
        filterName = cms.untracked.string( '' )
    ),
    outputCommands = cms.untracked.vstring( 'keep *' )
)
%(process)s.FULLOutput = cms.EndPath( %(process)s.hltOutputFULL )
"""


  # override the process name and adapt the relevant filters
  def overrideProcessName(self):
    if self.config.name is None:
      return

    # sanitise process name
    self.config.name = self.config.name.replace("_","")
    # override the process name
    quote = '[\'\"]'
    self.data = re.compile(r'^(process\s*=\s*cms\.Process\(\s*' + quote + r')\w+(' + quote + r'\s*\).*)$', re.MULTILINE).sub(r'\1%s\2' % self.config.name, self.data, 1)

    # the following was stolen and adapted from HLTrigger.Configuration.customL1THLT_Options
    self.data += """
# adapt HLT modules to the correct process name
if 'hltTrigReport' in %%(dict)s:
    %%(process)s.hltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressCosmicsOutputSmart' in %%(dict)s:
    %%(process)s.hltPreExpressCosmicsOutputSmart.hltResults = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressOutputSmart' in %%(dict)s:
    %%(process)s.hltPreExpressOutputSmart.hltResults        = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMForHIOutputSmart' in %%(dict)s:
    %%(process)s.hltPreDQMForHIOutputSmart.hltResults       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMForPPOutputSmart' in %%(dict)s:
    %%(process)s.hltPreDQMForPPOutputSmart.hltResults       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTDQMResultsOutputSmart' in %%(dict)s:
    %%(process)s.hltPreHLTDQMResultsOutputSmart.hltResults  = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTDQMOutputSmart' in %%(dict)s:
    %%(process)s.hltPreHLTDQMOutputSmart.hltResults         = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTMONOutputSmart' in %%(dict)s:
    %%(process)s.hltPreHLTMONOutputSmart.hltResults         = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltDQMHLTScalers' in %%(dict)s:
    %%(process)s.hltDQMHLTScalers.triggerResults                   = cms.InputTag( 'TriggerResults', '', '%(name)s' )
    %%(process)s.hltDQMHLTScalers.processname                      = '%(name)s'

if 'hltDQML1SeedLogicScalers' in %%(dict)s:
    %%(process)s.hltDQML1SeedLogicScalers.processname              = '%(name)s'
""" % self.config.__dict__


  def updateMessageLogger(self):
    # request summary informations from the MessageLogger
    self.data += """
if 'MessageLogger' in %(dict)s:
    %(process)s.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    %(process)s.MessageLogger.categories.append('L1GtTrigReport')
    %(process)s.MessageLogger.categories.append('L1TGlobalSummary')
    %(process)s.MessageLogger.categories.append('HLTrigReport')
    %(process)s.MessageLogger.categories.append('FastReport')
"""


  def loadAdditionalConditions(self, comment, *conditions):
    # load additional conditions
    self.data += """
# %s
if 'GlobalTag' in %%(dict)s:
""" % comment
    for condition in conditions:
      self.data += """    %%(process)s.GlobalTag.toGet.append(
        cms.PSet(
            record  = cms.string( '%(record)s' ),
            tag     = cms.string( '%(tag)s' ),
            label   = cms.untracked.string( '%(label)s' ),
            connect = cms.untracked.string( '%(connect)s' )
        )
    )
""" % condition


  def loadCffCommand(self, module):
    # load a cfi or cff module
    if self.config.fragment:
      return 'from %s import *\n' % module
    else:
      return 'process.load( "%s" )\n' % module

  def loadCff(self, module):
    self.data += self.loadCffCommand(module)


  def overrideParameters(self, module, parameters):
    # override a module's parameter if the module is present in the configuration
    self.data += "if '%s' in %%(dict)s:\n" % module
    for (parameter, value) in parameters:
      self.data += "    %%(process)s.%s.%s = %s\n" % (module, parameter, value)
    self.data += "\n"


  def instrumentTiming(self):
    if self.config.profiling:
      # instrument the menu for profiling: remove the HLTAnalyzerEndpath, add/override the HLTriggerFirstPath, with hltGetRaw and hltGetConditions
      text = ''

      if not 'hltGetRaw' in self.data:
        # add hltGetRaw
        text += """
%(process)s.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
"""

      if not 'hltGetConditions' in self.data:
        # add hltGetConditions
        text += """
%(process)s.hltGetConditions = cms.EDAnalyzer( 'EventSetupRecordDataGetter',
    verbose = cms.untracked.bool( False ),
    toGet = cms.VPSet( )
)
"""

      if not 'hltBoolFalse' in self.data:
        # add hltBoolFalse
        text += """
%(process)s.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
"""

      # add the definition of HLTriggerFirstPath
      # FIXME in a cff, should also update the HLTSchedule
      text += """
%(process)s.HLTriggerFirstPath = cms.Path( %(process)s.hltGetRaw + %(process)s.hltGetConditions + %(process)s.hltBoolFalse )
"""
      self.data = re.sub(r'.*cms\.(End)?Path.*', text + r'\g<0>', self.data, 1)


    # instrument the menu with the Service, EDProducer and EndPath needed for timing studies
    # FIXME in a cff, should also update the HLTSchedule
    if self.config.timing:
      self.data += """
# instrument the menu with the modules and EndPath needed for timing studies
"""

      if not 'FastTimerService' in self.data:
        self.data += '\n# configure the FastTimerService\n'
        self.loadCff('HLTrigger.Timer.FastTimerService_cfi')
      else:
        self.data += '\n# configure the FastTimerService\n'

      self.data += """# this is currently ignored in CMSSW 7.x, always using the real time clock
%(process)s.FastTimerService.useRealTimeClock          = True
# enable specific features
%(process)s.FastTimerService.enableTimingPaths         = True
%(process)s.FastTimerService.enableTimingModules       = True
%(process)s.FastTimerService.enableTimingExclusive     = True
# print a text summary at the end of the job
%(process)s.FastTimerService.enableTimingSummary       = True
# skip the first path (disregard the time spent loading event and conditions data)
%(process)s.FastTimerService.skipFirstPath             = True
# enable DQM plots
%(process)s.FastTimerService.enableDQM                 = True
# enable most per-path DQM plots
%(process)s.FastTimerService.enableDQMbyPathActive     = True
%(process)s.FastTimerService.enableDQMbyPathTotal      = True
%(process)s.FastTimerService.enableDQMbyPathOverhead   = False
%(process)s.FastTimerService.enableDQMbyPathDetails    = True
%(process)s.FastTimerService.enableDQMbyPathCounters   = True
%(process)s.FastTimerService.enableDQMbyPathExclusive  = True
# disable per-module DQM plots
%(process)s.FastTimerService.enableDQMbyModule         = False
%(process)s.FastTimerService.enableDQMbyModuleType     = False
# enable per-event DQM sumary plots
%(process)s.FastTimerService.enableDQMSummary          = True
# enable per-event DQM plots by lumisection
%(process)s.FastTimerService.enableDQMbyLumiSection    = True
%(process)s.FastTimerService.dqmLumiSectionsRange      = 2500
# set the time resolution of the DQM plots
%(process)s.FastTimerService.dqmTimeRange              = 1000.
%(process)s.FastTimerService.dqmTimeResolution         =    5.
%(process)s.FastTimerService.dqmPathTimeRange          =  100.
%(process)s.FastTimerService.dqmPathTimeResolution     =    0.5
%(process)s.FastTimerService.dqmModuleTimeRange        =   40.
%(process)s.FastTimerService.dqmModuleTimeResolution   =    0.2
# set the base DQM folder for the plots
%(process)s.FastTimerService.dqmPath                   = 'HLT/TimerService'
%(process)s.FastTimerService.enableDQMbyProcesses      = True
"""


  def instrumentDQM(self):
    if not self.config.hilton:
      # remove any reference to the hltDQMFileSaver
      if 'hltDQMFileSaver' in self.data:
        self.data = re.sub(r'\b(process\.)?hltDQMFileSaver \+ ', '', self.data)
        self.data = re.sub(r' \+ \b(process\.)?hltDQMFileSaver', '', self.data)
        self.data = re.sub(r'\b(process\.)?hltDQMFileSaver',     '', self.data)

      # instrument the HLT menu with DQMStore and DQMRootOutputModule suitable for running offline
      dqmstore  = "\n# load the DQMStore and DQMRootOutputModule\n"
      dqmstore += self.loadCffCommand('DQMServices.Core.DQMStore_cfi')
      dqmstore += "%(process)s.DQMStore.enableMultiThread = True\n"
      dqmstore += """
%(process)s.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("DQMIO.root")
)
"""

      empty_path = re.compile(r'.*\b(process\.)?DQMOutput = cms\.EndPath\( *\).*')
      other_path = re.compile(r'(.*\b(process\.)?DQMOutput = cms\.EndPath\()(.*)')
      if empty_path.search(self.data):
        # replace an empty DQMOutput path
        self.data = empty_path.sub(dqmstore + '\n%(process)s.DQMOutput = cms.EndPath( %(process)s.dqmOutput )\n', self.data)
      elif other_path.search(self.data):
        # prepend the dqmOutput to the DQMOutput path
        self.data = other_path.sub(dqmstore + r'\g<1> %(process)s.dqmOutput +\g<3>', self.data)
      else:
        # ceate a new DQMOutput path with the dqmOutput module
        self.data += dqmstore
        self.data += '\n%(process)s.DQMOutput = cms.EndPath( %(process)s.dqmOutput )\n'


  @staticmethod
  def dumppaths(paths):
    sys.stderr.write('Path selection:\n')
    for path in paths:
      sys.stderr.write('\t%s\n' % path)
    sys.stderr.write('\n\n')

  def buildPathList(self):
    self.all_paths = self.getPathList()

    if self.config.paths:
      # no path list was requested, dump the full table, minus unsupported / unwanted paths
      paths = self.config.paths.split(',')
    else:
      # dump only the requested paths, plus the eventual output endpaths
      paths = []

    if self.config.fragment or self.config.output in ('none', 'full'):
      # 'full' removes all outputs (same as 'none') and then adds a single "keep *" output (see the overrideOutput method)
      if self.config.paths:
        # paths are removed by default
        pass
      else:
        # drop all output endpaths
        paths.append( "-*Output" )
        paths.append( "-RatesMonitoring")
        paths.append( "-DQMHistograms")
    elif self.config.output == 'minimal':
      # drop all output endpaths but HLTDQMResultsOutput
      if self.config.paths:
        paths.append( "HLTDQMResultsOutput" )
      else:
        paths.append( "-*Output" )
        paths.append( "-RatesMonitoring")
        paths.append( "-DQMHistograms")
        paths.append( "HLTDQMResultsOutput" )
    else:
      # keep / add back all output endpaths
      if self.config.paths:
        paths.append( "*Output" )
      else:
        pass    # paths are kepy by default

    # drop unwanted paths for profiling (and timing studies)
    if self.config.profiling:
      paths.append( "-HLTriggerFirstPath" )
      paths.append( "-HLTAnalyzerEndpath" )

    # this should never be in any dump (nor online menu)
    paths.append( "-OfflineOutput" )

    # expand all wildcards
    paths = self.expandWildcards(paths, self.all_paths)

    if self.config.paths:
      # do an "additive" consolidation
      self.options['paths'] = self.consolidatePositiveList(paths)
      if not self.options['paths']:
        raise RuntimeError('Error: option "--paths %s" does not select any valid paths' % self.config.paths)
    else:
      # do a "subtractive" consolidation
      self.options['paths'] = self.consolidateNegativeList(paths)


  def buildOptions(self):
    # common configuration for all scenarios
    self.options['services'].append( "-DQM" )
    self.options['services'].append( "-FUShmDQMOutputService" )
    self.options['services'].append( "-MicroStateService" )
    self.options['services'].append( "-ModuleWebRegistry" )
    self.options['services'].append( "-TimeProfilerService" )

    # remove the DAQ modules and the online definition of the DQMStore and DQMFileSaver
    # unless a hilton-like configuration has been requested
    if not self.config.hilton:
      self.options['services'].append( "-EvFDaqDirector" )
      self.options['services'].append( "-FastMonitoringService" )
      self.options['services'].append( "-DQMStore" )
      self.options['modules'].append( "-hltDQMFileSaver" )

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
      self.options['esmodules'].append( "-trackerTopology" )

      self.options['esmodules'].append( "-CaloTowerGeometryFromDBEP" )
      self.options['esmodules'].append( "-CastorGeometryFromDBEP" )
      self.options['esmodules'].append( "-EcalBarrelGeometryFromDBEP" )
      self.options['esmodules'].append( "-EcalEndcapGeometryFromDBEP" )
      self.options['esmodules'].append( "-EcalPreshowerGeometryFromDBEP" )
      self.options['esmodules'].append( "-HcalGeometryFromDBEP" )
      self.options['esmodules'].append( "-ZdcGeometryFromDBEP" )
      self.options['esmodules'].append( "-XMLFromDBSource" )
      self.options['esmodules'].append( "-sistripconn" )

      self.options['services'].append( "-MessageLogger" )

      self.options['psets'].append( "-maxEvents" )
      self.options['psets'].append( "-options" )

    if self.config.fragment or (self.config.prescale and (self.config.prescale.lower() == 'none')):
      self.options['services'].append( "-PrescaleService" )

    if self.config.fragment or self.config.timing:
      self.options['services'].append( "-FastTimerService" )


  def append_filenames(self, name, filenames):
    if len(filenames) > 255:
      token_open  = "( *("
      token_close = ") )"
    else:
      token_open  = "("
      token_close = ")"

    self.data += "    %s = cms.untracked.vstring%s\n" % (name, token_open)
    for line in filenames:
      self.data += "        '%s',\n" % line
    self.data += "    %s,\n" % (token_close)


  def expand_filenames(self, input):
    # check if the input is a dataset or a list of files
    if input[0:8] == 'dataset:':
      from dasFileQuery import dasFileQuery
      # extract the dataset name, and use DAS to fine the list of LFNs
      dataset = input[8:]
      files = dasFileQuery(dataset)
    else:
      # assume a comma-separated list of input files
      files = self.config.input.split(',')
    return files

  def build_source(self):
    if self.config.input:
      # if a dataset or a list of input files was given, use it
      self.source = self.expand_filenames(self.config.input)
    elif self.config.online:
      # online we always run on data
      self.source = [ "file:/tmp/InputCollection.root" ]
    elif self.config.data:
      # offline we can run on data...
      self.source = [ "file:RelVal_Raw_%s_DATA.root" % self.config.type ]
    else:
      # ...or on mc
      self.source = [ "file:RelVal_Raw_%s_MC.root" % self.config.type ]

    if self.config.parent:
      # if a dataset or a list of input files was given for the parent data, use it
      self.parent = self.expand_filenames(self.config.parent)

    self.data += """
%(process)s.source = cms.Source( "PoolSource",
"""
    self.append_filenames("fileNames", self.source)
    if (self.parent):
      self.append_filenames("secondaryFileNames", self.parent)
    self.data += """\
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)
"""
