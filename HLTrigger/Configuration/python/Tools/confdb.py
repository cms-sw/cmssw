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
  # paths not supported by FastSim
  fastsimUnsupportedPaths = (

    # paths for which a recovery is not foreseen/possible
    "AlCa_*_v*",
    "DQM_*_v*",
    "HLT_*Calibration_v*",
    "HLT_DTErrors_v*",
    "HLT_Random_v*",
    "HLT_HcalNZS_v*",
    "HLT_HcalPhiSym_v*",
    "HLT_Activity_Ecal*_v*",
    "HLT_IsoTrackHB_v*",
    "HLT_IsoTrackHE_v*",
    "HLT_L1SingleMuOpen_AntiBPTX_v*",
    "HLT_JetE*_NoBPTX*_v*",
    "HLT_L2Mu*_NoBPTX*_v*",
    "HLT_PixelTracks_Multiplicity70_v*",
    "HLT_PixelTracks_Multiplicity80_v*",
    "HLT_PixelTracks_Multiplicity90_v*",
    "HLT_Beam*_v*",
   #"HLT_L1Tech_*_v*",
    "HLT_GlobalRunHPDNoise_v*",
    "HLT_L1TrackerCosmics_v*",
    "HLT_HcalUTCA_v*",
    
    # TODO: paths not supported by FastSim, but for which a recovery should be attempted
 
    "HLT_DoubleMu33NoFiltersNoVtx_v*",
    "HLT_DoubleMu38NoFiltersNoVtx_v*",
    "HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v*",
    "HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v*",
 
  )

  def __init__(self, configuration):
    self.config = configuration
    self.data   = None
    self.source = None
    self.parent = None

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

    if self.config.prescale and (self.config.prescale.lower() != 'none'):
      self.labels['prescale'] = self.config.prescale

    # get the configuration from ConfdB
    self.buildPathList()
    self.buildOptions()
    self.getRawConfigurationFromDB()
    self.customize()


  def getRawConfigurationFromDB(self):
    url = 'http://j2eeps.cern.ch/cms-project-confdb-hltdev/get.jsp'
    postdata = dict([ (key, ','.join(vals)) for key, vals in self.options.iteritems() if vals ])
    postdata['noedsources'] = ''
    if self.config.fragment:
      postdata['cff'] = ''
    if self.config.menu.run:
      postdata['runNumber'] = self.config.menu.run
    else:
      postdata['dbName']    = self.config.menu.db
      postdata['configName']= self.config.menu.name

    data = urllib2.urlopen(url, urllib.urlencode(postdata)).read()
    if 'Exhausted Resultset' in data or 'CONFIG_NOT_FOUND' in data:
      raise ImportError('%s is not a valid HLT menu' % self.config.menuConfig.value)
    self.data = data


  def getPathList(self):
    url = 'http://j2eeps.cern.ch/cms-project-confdb-hltdev/get.jsp'
    postdata = { 
      'noedsources': '', 
      'noes':        '',
      'noservices':  '',
      'nosequences': '',
      'nomodules' :  '',
      'cff':         '',
    }
    if self.config.menu.run:
      postdata['runNumber'] = self.config.menu.run
    else:
      postdata['dbName']    = self.config.menu.db
      postdata['configName']= self.config.menu.name

    data = urllib2.urlopen(url, urllib.urlencode(postdata)).read()
    if 'Exhausted Resultset' in data or 'CONFIG_NOT_FOUND' in data:
      raise ImportError('%s is not a valid HLT menu' % self.config.menuConfig.value)
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
    return self.data % self.labels


  # add release-specific customizations
  def releaseSpecificCustomize(self):
    # version specific customizations
    self.data += """
# CMSSW version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

# none for now
"""

# from CMSSW_7_2_0_pre6: Use Legacy Errors in "StripCPEESProducer" for HLT (PRs 5286/5151)
#if cmsswVersion >= "CMSSW_7_2":
#    if 'hltESPStripCPEfromTrackAngle' in %(dict)s:
#        %(process)shltESPStripCPEfromTrackAngle.useLegacyError = cms.bool(True)

  # customize the configuration according to the options
  def customize(self):

    # adapt the source to the current scenario
    if not self.config.fragment:
      self.build_source()

    # manual override some parameters
#    if self.config.type in ('HIon', ):
#      self.data += """
## Disable HF Noise filters in HIon menu
#if 'hltHfreco' in %(dict)s:
#    %(process)shltHfreco.setNoiseFlags = cms.bool( False )
#"""
#    else:
#      self.data += """
## Enable HF Noise filters in non-HIon menu
#if 'hltHfreco' in %(dict)s:
#    %(process)shltHfreco.setNoiseFlags = cms.bool( True )
#"""

#    self.data += """
## untracked parameters with NO default in the code
#if 'hltHcalDataIntegrityMonitor' in %(dict)s:
#    %(process)shltHcalDataIntegrityMonitor.RawDataLabel = cms.untracked.InputTag("rawDataCollector")
#if 'hltDt4DSegments' in %(dict)s:
#    %(process)shltDt4DSegments.debug = cms.untracked.bool( False )
#"""

    # if requested, override the L1 self from the GlobalTag (Xml)
    self.overrideL1MenuXml()

    # if running on MC, adapt the configuration accordingly
    self.fixForMC()

    # if requested, remove the HLT prescales
    self.fixPrescales()

    # if requested, override all ED/HLTfilters to always pass ("open" mode)
    self.instrumentOpenMode()

    # if requested, change all HLTTriggerTypeFilter EDFilters to accept only error events (SelectedTriggerType = 0)
    self.instrumentErrorEventType()

    # if requested, instrument the self with the modules and EndPath needed for timing studies
    self.instrumentTiming()

    # add version-specific customisations
    self.releaseSpecificCustomize()

    if self.config.fragment:
      
#      self.data += """
## dummyfy hltGetConditions in cff's
#if 'hltGetConditions' in %(dict)s and 'HLTriggerFirstPath' in %(dict)s :
#    %(process)shltDummyConditions = cms.EDFilter( "HLTBool",
#        result = cms.bool( True )
#    )
#    %(process)sHLTriggerFirstPath.replace(%(process)shltGetConditions,%(process)shltDummyConditions)
#"""

      # if requested, adapt the configuration for FastSim
      self.fixForFastSim()

    else:

      # override the process name and adapt the relevant filters
      self.overrideProcessName()

      # override the output modules to output root files
      self.overrideOutput()

      # add global options
      self.addGlobalOptions()

      # if requested or necessary, override the GlobalTag and connection strings (incl. L1!)
      self.overrideGlobalTag()

      # if requested, run (part of) the L1 emulator
      self.runL1Emulator()

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


  def addGlobalOptions(self):
    # add global options
    self.data += """
# limit the number of events to be processed
%%(process)smaxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( %d )
)
""" % self.config.events

    if not self.config.profiling:
      self.data += """
# enable the TrigReport and TimeReport
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
      # customise the HLT menu for running on MC
      if not self.config.fragment:
        self.data += """
# customise the HLT menu for running on MC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
process = customizeHLTforMC(process)
"""


  def fixForFastSim(self):
    if self.config.fastsim:
      # adapt the hle configuration (fragment) to run under fastsim
      self.data = re.sub( r'import FWCore.ParameterSet.Config as cms', r'\g<0>\nfrom FastSimulation.HighLevelTrigger.HLTSetup_cff import *', self.data)

      # remove the definition of streams and datasets
      self.data = re.compile( r'^streams.*\n(.*\n)*?^\)\s*\n',  re.MULTILINE ).sub( '', self.data )
      self.data = re.compile( r'^datasets.*\n(.*\n)*?^\)\s*\n', re.MULTILINE ).sub( '', self.data )

      # fix the definition of module
      # FIXME: this should be updated to take into accout the --l1-emulator option
      self._fix_parameter(                               type = 'InputTag', value = 'hltL1extraParticles',  replace = 'l1extraParticles')
      self._fix_parameter(name = 'GMTReadoutCollection', type = 'InputTag', value = 'hltGtDigis',           replace = 'simGmtDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltGtDigis',           replace = 'gtDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltL1GtObjectMap',     replace = 'gtDigis')
      self._fix_parameter(name = 'initialSeeds',         type = 'InputTag', value = 'noSeedsHere',          replace = 'globalPixelSeeds:GlobalPixel')
      self._fix_parameter(name = 'preFilteredSeeds',     type = 'bool',     value = 'True',                 replace = 'False')
      self._fix_parameter(                               type = 'InputTag', value = 'hltOfflineBeamSpot',   replace = 'offlineBeamSpot')
      self._fix_parameter(                               type = 'InputTag', value = 'hltOnlineBeamSpot',    replace = 'offlineBeamSpot')
      self._fix_parameter(                               type = 'InputTag', value = 'hltMuonCSCDigis',      replace = 'simMuonCSCDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltMuonDTDigis',       replace = 'simMuonDTDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltMuonRPCDigis',      replace = 'simMuonRPCDigis')
      self._fix_parameter(                               type = 'InputTag', value = 'hltRegionalTracksForL3MuonIsolation', replace = 'hltPixelTracks')
      self._fix_parameter(name = 'src',                  type = 'InputTag', value = 'hltHcalTowerNoiseCleaner', replace = 'hltTowerMakerForAll')
      self._fix_parameter(name = 'src',                  type = 'InputTag', value = 'hltIter4Tau3MuMerged', replace = 'hltIter4Merged')

      # MeasurementTrackerEvent
      self._fix_parameter(                               type = 'InputTag', value = 'hltSiStripClusters', replace = 'MeasurementTrackerEvent')

      # fix the definition of sequences and paths
      self.data = re.sub( r'hltMuonCSCDigis', r'cms.SequencePlaceholder( "simMuonCSCDigis" )',  self.data )
      self.data = re.sub( r'hltMuonDTDigis',  r'cms.SequencePlaceholder( "simMuonDTDigis" )',   self.data )
      self.data = re.sub( r'hltMuonRPCDigis', r'cms.SequencePlaceholder( "simMuonRPCDigis" )',  self.data )
      self.data = re.sub( r'HLTEndSequence',  r'cms.SequencePlaceholder( "HLTEndSequence" )',   self.data )
      self.data = re.sub( r'hltGtDigis',      r'HLTBeginSequence',                              self.data )


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
    %(process)sPrescaleService.forceDefault     = True
    %(process)sPrescaleService.lvl1DefaultLabel = '%(prescale)s'
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
      self.config.l1.record = 'L1GtTriggerMenuRcd'
      self.config.l1.label  = ''
      self.config.l1.tag    = self.config.l1.override
      if not self.config.l1.connect:
        self.config.l1.connect = '%(connect)s/CMS_CONDITIONS'
      self.config.l1cond = '%(tag)s,%(record)s,%(connect)s' % self.config.l1.__dict__
    else:
      self.config.l1cond = None

    if self.config.globaltag or self.config.l1cond:
      text += "    from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag as customiseGlobalTag\n"
      text += "    %(process)sGlobalTag = customiseGlobalTag(%(process)sGlobalTag"
      if self.config.globaltag:
        text += ", globaltag = %s"  % repr(self.config.globaltag)
      if self.config.l1cond:
        text += ", conditions = %s" % repr(self.config.l1cond)
      text += ")\n"

    text += """    %(process)sGlobalTag.connect   = '%(connect)s/CMS_CONDITIONS'
    %(process)sGlobalTag.pfnPrefix = cms.untracked.string('%(connect)s/')
    for pset in process.GlobalTag.toGet.value():
        pset.connect = pset.connect.value().replace('frontier://FrontierProd/', '%(connect)s/')
    # fix for multi-run processing
    %(process)sGlobalTag.RefreshEachRun = cms.untracked.bool( False )
    %(process)sGlobalTag.ReconnectEachRun = cms.untracked.bool( False )
"""
    self.data += text

  def overrideL1MenuXml(self):
    # if requested, override the L1 menu from the GlobalTag (Xml file)
    if self.config.l1Xml.XmlFile:
      text = """
# override the L1 menu from an Xml file
%%(process)sl1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",
  TriggerMenuLuminosity = cms.string('%(LumiDir)s'),
  DefXmlFile = cms.string('%(XmlFile)s'),
  VmeXmlFile = cms.string('')
)
%%(process)sL1GtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
  recordName = cms.string('L1GtTriggerMenuRcd'),
  iovIsRunNotTime = cms.bool(True),
  firstValid = cms.vuint32(1)
)
%%(process)ses_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml') 
"""
      self.data += text % self.config.l1Xml.__dict__

  def runL1EmulatorGT(self):
    # if requested, run (part of) the L1 emulator, then repack the data into a new RAW collection, to be used by the HLT
    if not self.config.emulator:
      return

    if self.config.emulator != 'gt':
      # only the GT emulator is currently supported
      return

    # run the L1 GT emulator, then repack the data into a new RAW collection, to be used by the HLT
    text = """
# run the L1 GT emulator, then repack the data into a new RAW collection, to be used by the HLT
"""
    if self.config.fragment:
      # FIXME in a cff, should also update the HLTSchedule
      text += "import Configuration.StandardSequences.SimL1EmulatorRepack_GT_cff\n"
    else:
      text += "process.load( 'Configuration.StandardSequences.SimL1EmulatorRepack_GT_cff' )\n"

    if not 'hltBoolFalse' in self.data:
      # add hltBoolFalse
      text += """
%(process)shltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
"""
    text += "process.L1Emulator = cms.Path( process.SimL1Emulator + process.hltBoolFalse )\n\n"

    self.data = re.sub(r'.*cms\.(End)?Path.*', text + r'\g<0>', self.data, 1)


  def runL1Emulator(self):
    # if requested, run (part of) the L1 emulator
    if self.config.emulator:
      # FIXME this fragment used "process" explicitly
      emulator = {
        'RawToDigi': '',
        'CustomL1T': '',
        'CustomHLT': ''
      }

      if self.config.data:
        emulator['RawToDigi'] = 'RawToDigi_Data_cff'
      else:
        emulator['RawToDigi'] = 'RawToDigi_cff'

      if self.config.emulator == 'gt':
        emulator['CustomL1T'] = 'customiseL1GtEmulatorFromRaw'
        emulator['CustomHLT'] = 'switchToSimGtDigis'
      elif self.config.emulator == 'gct,gt':
        emulator['CustomL1T'] = 'customiseL1CaloAndGtEmulatorsFromRaw'
        emulator['CustomHLT'] = 'switchToSimGctGtDigis'
      elif self.config.emulator == 'gmt,gt':
        # XXX currently unsupported
        emulator['CustomL1T'] = 'customiseL1MuonAndGtEmulatorsFromRaw'
        emulator['CustomHLT'] = 'switchToSimGmtGtDigis'
      elif self.config.emulator in ('gmt,gct,gt', 'gct,gmt,gt', 'all'):
        emulator['CustomL1T'] = 'customiseL1EmulatorFromRaw'
        emulator['CustomHLT'] = 'switchToSimGmtGctGtDigis'
      else:
        # unsupported argument, default to running the whole emulator
        emulator['CustomL1T'] = 'customiseL1EmulatorFromRaw'
        emulator['CustomHLT'] = 'switchToSimGmtGctGtDigis'

      self.data += """
# customize the L1 emulator to run %(CustomL1T)s with HLT to %(CustomHLT)s
process.load( 'Configuration.StandardSequences.%(RawToDigi)s' )
process.load( 'Configuration.StandardSequences.SimL1Emulator_cff' )
import L1Trigger.Configuration.L1Trigger_custom
process = L1Trigger.Configuration.L1Trigger_custom.%(CustomL1T)s( process )
process = L1Trigger.Configuration.L1Trigger_custom.customiseResetPrescalesAndMasks( process )

# customize the HLT to use the emulated results
import HLTrigger.Configuration.customizeHLTforL1Emulator
process = HLTrigger.Configuration.customizeHLTforL1Emulator.switchToL1Emulator( process )
process = HLTrigger.Configuration.customizeHLTforL1Emulator.%(CustomHLT)s( process )
""" % emulator


  def overrideOutput(self):
    # override the "online" ShmStreamConsumer output modules with "offline" PoolOutputModule's
    self.data = re.sub(
      r'\b(process\.)?hltOutput(\w+) *= *cms\.OutputModule\( *"ShmStreamConsumer" *,',
      r'%(process)shltOutput\2 = cms.OutputModule( "PoolOutputModule",\n    fileName = cms.untracked.string( "output\2.root" ),\n    fastCloning = cms.untracked.bool( False ),\n    dataset = cms.untracked.PSet(\n        filterName = cms.untracked.string( "" ),\n        dataTier = cms.untracked.string( "RAW" )\n    ),',
      self.data
    )

    if not self.config.fragment and self.config.output == 'full':
      # add a single "keep *" output
      self.data += """
# add a single "keep *" output
%(process)shltOutputFULL = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputFULL.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string( 'RECO' ),
        filterName = cms.untracked.string( '' )
    ),
    outputCommands = cms.untracked.vstring( 'keep *' )
)
%(process)sFULLOutput = cms.EndPath( %(process)shltOutputFULL )
"""


  # override the process name and adapt the relevant filters
  def overrideProcessName(self):
    if self.config.name is None:
      return

    # override the process name
    quote = '[\'\"]'
    self.data = re.compile(r'^(process\s*=\s*cms\.Process\(\s*' + quote + r')\w+(' + quote + r'\s*\).*)$', re.MULTILINE).sub(r'\1%s\2' % self.config.name, self.data, 1)

    # the following was stolen and adapted from HLTrigger.Configuration.customL1THLT_Options
    self.data += """
# adapt HLT modules to the correct process name
if 'hltTrigReport' in %%(dict)s:
    %%(process)shltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressCosmicsOutputSmart' in %%(dict)s:
    %%(process)shltPreExpressCosmicsOutputSmart.hltResults = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressOutputSmart' in %%(dict)s:
    %%(process)shltPreExpressOutputSmart.hltResults        = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMForHIOutputSmart' in %%(dict)s:
    %%(process)shltPreDQMForHIOutputSmart.hltResults       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMForPPOutputSmart' in %%(dict)s:
    %%(process)shltPreDQMForPPOutputSmart.hltResults       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTDQMResultsOutputSmart' in %%(dict)s:
    %%(process)shltPreHLTDQMResultsOutputSmart.hltResults  = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTDQMOutputSmart' in %%(dict)s:
    %%(process)shltPreHLTDQMOutputSmart.hltResults         = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTMONOutputSmart' in %%(dict)s:
    %%(process)shltPreHLTMONOutputSmart.hltResults         = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltDQMHLTScalers' in %%(dict)s:
    %%(process)shltDQMHLTScalers.triggerResults                   = cms.InputTag( 'TriggerResults', '', '%(name)s' )
    %%(process)shltDQMHLTScalers.processname                      = '%(name)s'

if 'hltDQML1SeedLogicScalers' in %%(dict)s:
    %%(process)shltDQML1SeedLogicScalers.processname              = '%(name)s'
""" % self.config.__dict__


  def updateMessageLogger(self):
    # request summary informations from the MessageLogger
    self.data += """
if 'MessageLogger' in %(dict)s:
    %(process)sMessageLogger.categories.append('TriggerSummaryProducerAOD')
    %(process)sMessageLogger.categories.append('L1GtTrigReport')
    %(process)sMessageLogger.categories.append('HLTrigReport')
    %(process)sMessageLogger.categories.append('FastReport')
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
      self.data += "    %%(process)s%s.%s = %s\n" % (module, parameter, value)
    self.data += "\n"


  def instrumentTiming(self):
    if self.config.profiling:
      # instrument the menu for profiling: remove the HLTAnalyzerEndpath, add/override the HLTriggerFirstPath, with hltGetRaw and hltGetConditions
      text = ''

      if not 'hltGetRaw' in self.data:
        # add hltGetRaw
        text += """
%(process)shltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
"""

      if not 'hltGetConditions' in self.data:
        # add hltGetConditions
        text += """
%(process)shltGetConditions = cms.EDAnalyzer( 'EventSetupRecordDataGetter',
    verbose = cms.untracked.bool( False ),
    toGet = cms.VPSet( )
)
"""

      if not 'hltBoolFalse' in self.data:
        # add hltBoolFalse
        text += """
%(process)shltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
"""

      # add the definition of HLTriggerFirstPath
      # FIXME in a cff, should also update the HLTSchedule
      text += """
%(process)sHLTriggerFirstPath = cms.Path( %(process)shltGetRaw + %(process)shltGetConditions + %(process)shltBoolFalse )
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
%(process)sFastTimerService.useRealTimeClock          = True
# enable specific features
%(process)sFastTimerService.enableTimingPaths         = True
%(process)sFastTimerService.enableTimingModules       = True
%(process)sFastTimerService.enableTimingExclusive     = True
# print a text summary at the end of the job
%(process)sFastTimerService.enableTimingSummary       = True
# skip the first path (disregard the time spent loading event and conditions data)
%(process)sFastTimerService.skipFirstPath             = True
# enable DQM plots
%(process)sFastTimerService.enableDQM                 = True
# enable most per-path DQM plots
%(process)sFastTimerService.enableDQMbyPathActive     = True
%(process)sFastTimerService.enableDQMbyPathTotal      = True
%(process)sFastTimerService.enableDQMbyPathOverhead   = False
%(process)sFastTimerService.enableDQMbyPathDetails    = True
%(process)sFastTimerService.enableDQMbyPathCounters   = True
%(process)sFastTimerService.enableDQMbyPathExclusive  = True
# disable per-module DQM plots
%(process)sFastTimerService.enableDQMbyModule         = False
%(process)sFastTimerService.enableDQMbyModuleType     = False
# enable per-event DQM sumary plots
%(process)sFastTimerService.enableDQMSummary          = True
# enable per-event DQM plots by lumisection
%(process)sFastTimerService.enableDQMbyLumiSection    = True
%(process)sFastTimerService.dqmLumiSectionsRange      = 2500
# set the time resolution of the DQM plots
%(process)sFastTimerService.dqmTimeRange              = 1000.
%(process)sFastTimerService.dqmTimeResolution         =    5.
%(process)sFastTimerService.dqmPathTimeRange          =  100.
%(process)sFastTimerService.dqmPathTimeResolution     =    0.5
%(process)sFastTimerService.dqmModuleTimeRange        =   40.
%(process)sFastTimerService.dqmModuleTimeResolution   =    0.2
# set the base DQM folder for the plots
%(process)sFastTimerService.dqmPath                   = 'HLT/TimerService'
%(process)sFastTimerService.enableDQMbyProcesses      = True
"""


  def instrumentDQM(self):
    # remove any reference to the hltDQMFileSaver
    if 'hltDQMFileSaver' in self.data:
      self.data = re.sub(r'\b(process\.)?hltDQMFileSaver \+ ', '', self.data)
      self.data = re.sub(r' \+ \b(process\.)?hltDQMFileSaver', '', self.data)
      self.data = re.sub(r'\b(process\.)?hltDQMFileSaver',     '', self.data)

    # instrument the HLT menu with DQMStore and DQMRootOutputModule suitable for running offline
    dqmstore  = "\n# load the DQMStore and DQMRootOutputModule\n"
    dqmstore += self.loadCffCommand('DQMServices.Core.DQMStore_cfi')
    dqmstore += "%(process)sDQMStore.enableMultiThread = True\n"
    dqmstore += """
%(process)sdqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("DQMIO.root")
)
"""

    empty_path = re.compile(r'.*\b(process\.)?DQMOutput = cms\.EndPath\( *\).*')
    other_path = re.compile(r'(.*\b(process\.)?DQMOutput = cms\.EndPath\()(.*)')
    if empty_path.search(self.data):
      # replace an empty DQMOutput path
      self.data = empty_path.sub(dqmstore + '\n%(process)sDQMOutput = cms.EndPath( %(process)sdqmOutput )\n', self.data)
    elif other_path.search(self.data):
      # prepend the dqmOutput to the DQMOutput path
      self.data = other_path.sub(dqmstore + r'\g<1> %(process)sdqmOutput +\g<3>', self.data)
    else:
      # ceate a new DQMOutput path with the dqmOutput module
      self.data += dqmstore
      self.data += '\n%(process)sDQMOutput = cms.EndPath( %(process)sdqmOutput )\n'


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
    elif self.config.output == 'minimal':
      # drop all output endpaths but HLTDQMResultsOutput
      if self.config.paths:
        paths.append( "HLTDQMResultsOutput" )
      else:
        paths.append( "-*Output" )
        paths.append( "HLTDQMResultsOutput" )
    else:
      # keep / add back all output endpaths
      if self.config.paths:
        paths.append( "*Output" )
      else:
        pass    # paths are kepy by default

    # drop paths unsupported by fastsim
    if self.config.fastsim:
      paths.extend( "-%s" % path for path in self.fastsimUnsupportedPaths )

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
    self.options['services'].append( "-EvFDaqDirector" )
    self.options['services'].append( "-FastMonitoringService" )
    self.options['services'].append( "-FUShmDQMOutputService" )
    self.options['services'].append( "-MicroStateService" )
    self.options['services'].append( "-ModuleWebRegistry" )
    self.options['services'].append( "-TimeProfilerService" )

    # drop the online definition of the DQMStore and DQMFileSaver
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

      self.options['services'].append( "-MessageLogger" )

      self.options['psets'].append( "-maxEvents" )
      self.options['psets'].append( "-options" )

    if self.config.fragment or (self.config.prescale and (self.config.prescale.lower() == 'none')):
      self.options['services'].append( "-PrescaleService" )

    if self.config.fragment or self.config.timing:
      self.options['services'].append( "-FastTimerService" )

    if self.config.fastsim:
      # remove components not supported or needed by fastsim
      self.options['esmodules'].append( "-navigationSchoolESProducer" )
      self.options['esmodules'].append( "-TransientTrackBuilderESProducer" )
      self.options['esmodules'].append( "-SteppingHelixPropagatorAny" )
      self.options['esmodules'].append( "-OppositeMaterialPropagator" )
      self.options['esmodules'].append( "-MaterialPropagator" )
      self.options['esmodules'].append( "-CaloTowerConstituentsMapBuilder" )
      self.options['esmodules'].append( "-CaloTopologyBuilder" )

      self.options['modules'].append( "hltL3MuonIsolations" )
      self.options['modules'].append( "hltPixelVertices" )
      self.options['modules'].append( "-hltCkfL1SeededTrackCandidates" )
      self.options['modules'].append( "-hltCtfL1SeededithMaterialTracks" )
      self.options['modules'].append( "-hltCkf3HitL1SeededTrackCandidates" )
      self.options['modules'].append( "-hltCtf3HitL1SeededWithMaterialTracks" )
      self.options['modules'].append( "-hltCkf3HitActivityTrackCandidates" )
      self.options['modules'].append( "-hltCtf3HitActivityWithMaterialTracks" )
      self.options['modules'].append( "-hltActivityCkfTrackCandidatesForGSF" )
      self.options['modules'].append( "-hltL1SeededCkfTrackCandidatesForGSF" )
      self.options['modules'].append( "-hltMuCkfTrackCandidates" )
      self.options['modules'].append( "-hltMuCtfTracks" )
      self.options['modules'].append( "-hltTau3MuCkfTrackCandidates" )
      self.options['modules'].append( "-hltTau3MuCtfWithMaterialTracks" )
      self.options['modules'].append( "-hltMuTrackJpsiCkfTrackCandidates" )
      self.options['modules'].append( "-hltMuTrackJpsiCtfTracks" )
      self.options['modules'].append( "-hltMuTrackJpsiEffCkfTrackCandidates" )
      self.options['modules'].append( "-hltMuTrackJpsiEffCtfTracks" )
      self.options['modules'].append( "-hltJpsiTkPixelSeedFromL3Candidate" )
      self.options['modules'].append( "-hltCkfTrackCandidatesJpsiTk" )
      self.options['modules'].append( "-hltCtfWithMaterialTracksJpsiTk" )
      self.options['modules'].append( "-hltMuTrackCkfTrackCandidatesOnia" )
      self.options['modules'].append( "-hltMuTrackCtfTracksOnia" )
      
      self.options['modules'].append( "-hltFEDSelector" )
      self.options['modules'].append( "-hltL3TrajSeedOIHit" )
      self.options['modules'].append( "-hltL3TrajSeedIOHit" )
      self.options['modules'].append( "-hltL3NoFiltersTrajSeedOIHit" )
      self.options['modules'].append( "-hltL3NoFiltersTrajSeedIOHit" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2OIState" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2OIHit" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2IOHit" )
      self.options['modules'].append( "-hltL3TrackCandidateFromL2NoVtx" )
      self.options['modules'].append( "-hltHcalDigis" )
      self.options['modules'].append( "-hltHoreco" )
      self.options['modules'].append( "-hltHfreco" )
      self.options['modules'].append( "-hltHbhereco" )
      self.options['modules'].append( "-hltESRawToRecHitFacility" )
      self.options['modules'].append( "-hltEcalRecHitAll" )
      self.options['modules'].append( "-hltESRecHitAll" )
      # === hltPF
      self.options['modules'].append( "-hltPFJetCkfTrackCandidates" )
      self.options['modules'].append( "-hltPFJetCtfWithMaterialTracks" )
      self.options['modules'].append( "-hltPFlowTrackSelectionHighPurity" )
      # === hltFastJet
      self.options['modules'].append( "-hltDisplacedHT250L1FastJetRegionalPixelSeedGenerator" )
      self.options['modules'].append( "-hltDisplacedHT250L1FastJetRegionalCkfTrackCandidates" )
      self.options['modules'].append( "-hltDisplacedHT250L1FastJetRegionalCtfWithMaterialTracks" )     
      self.options['modules'].append( "-hltDisplacedHT300L1FastJetRegionalPixelSeedGenerator" )
      self.options['modules'].append( "-hltDisplacedHT300L1FastJetRegionalCkfTrackCandidates" )
      self.options['modules'].append( "-hltDisplacedHT300L1FastJetRegionalCtfWithMaterialTracks" )     
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorbbPhiL1FastJet" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesbbPhiL1FastJet" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJet" )     
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorHbbVBF" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesHbbVBF" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksHbbVBF" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet" )
      # === hltBLifetimeRegional
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorHbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesHbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksHbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorbbPhi" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesbbPhi" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksbbPhi" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeFastRegionalPixelSeedGeneratorHbbVBF" )
      self.options['modules'].append( "-hltBLifetimeFastRegionalCkfTrackCandidatesHbbVBF" )
      self.options['modules'].append( "-hltBLifetimeFastRegionalCtfWithMaterialTracksHbbVBF" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorbbPhiL1FastJetFastPV" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesbbPhiL1FastJetFastPV" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJetFastPV" )
      self.options['modules'].append( "-hltFastPixelBLifetimeRegionalPixelSeedGeneratorHbb" )
      self.options['modules'].append( "-hltFastPixelBLifetimeRegionalCkfTrackCandidatesHbb" )
      self.options['modules'].append( "-hltFastPixelBLifetimeRegionalCtfWithMaterialTracksHbb" )
     
      self.options['modules'].append( "-hltPixelTracksForMinBias" )
      self.options['modules'].append( "-hltPixelTracksForHighMult" )
      self.options['modules'].append( "-hltRegionalPixelTracks" )
      self.options['modules'].append( "-hltPixelTracksReg" )
      self.options['modules'].append( "-hltPixelTracksL3Muon" )
      self.options['modules'].append( "-hltPixelTracksGlbTrkMuon" )
      self.options['modules'].append( "-hltPixelTracksHighPtTkMuIso" )
      self.options['modules'].append( "-hltPixelTracksHybrid" )
      self.options['modules'].append( "-hltPixelTracksForPhotons" )
      self.options['modules'].append( "-hltPixelTracksForEgamma" )
      self.options['modules'].append( "-hltPixelTracksElectrons" )
      self.options['modules'].append( "-hltPixelTracksForNoPU" )

      self.options['modules'].append( "-hltFastPixelHitsVertex" )
      self.options['modules'].append( "-hltFastPixelTracks")
      self.options['modules'].append( "-hltFastPixelTracksRecover")

      self.options['modules'].append( "-hltPixelLayerPairs" )
      self.options['modules'].append( "-hltPixelLayerTriplets" )
      self.options['modules'].append( "-hltPixelLayerTripletsReg" )
      self.options['modules'].append( "-hltPixelLayerTripletsHITHB" )
      self.options['modules'].append( "-hltPixelLayerTripletsHITHE" )
      self.options['modules'].append( "-hltMixedLayerPairs" )
      
      self.options['modules'].append( "-hltFastPrimaryVertexbbPhi")
      self.options['modules'].append( "-hltPixelTracksFastPVbbPhi")
      self.options['modules'].append( "-hltPixelTracksRecoverbbPhi" )
      self.options['modules'].append( "-hltFastPixelHitsVertexVHbb" )
      self.options['modules'].append( "-hltFastPixelTracksVHbb" )
      self.options['modules'].append( "-hltFastPixelTracksRecoverVHbb" )

      self.options['modules'].append( "-hltFastPrimaryVertex")
      self.options['modules'].append( "-hltFastPVPixelVertexFilter")
      self.options['modules'].append( "-hltFastPVPixelTracks")
      self.options['modules'].append( "-hltFastPVPixelTracksRecover" )

      self.options['modules'].append( "hltPixelMatchElectronsActivity" )

      self.options['modules'].append( "-hltMuonCSCDigis" )
      self.options['modules'].append( "-hltMuonDTDigis" )
      self.options['modules'].append( "-hltMuonRPCDigis" )
      self.options['modules'].append( "-hltGtDigis" )
      self.options['modules'].append( "-hltL1GtTrigReport" )
      self.options['modules'].append( "hltCsc2DRecHits" )
      self.options['modules'].append( "hltDt1DRecHits" )
      self.options['modules'].append( "hltRpcRecHits" )
      self.options['modules'].append( "-hltScalersRawToDigi" )

      self.options['sequences'].append( "-HLTL1SeededEgammaRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTEcalActivityEgammaRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTPixelMatchElectronActivityTrackingSequence" )
      self.options['sequences'].append( "-HLTDoLocalStripSequence" )
      self.options['sequences'].append( "-HLTDoLocalPixelSequence" )
      self.options['sequences'].append( "-HLTDoLocalPixelSequenceRegL2Tau" )
      self.options['sequences'].append( "-HLTDoLocalStripSequenceReg" )
      self.options['sequences'].append( "-HLTDoLocalPixelSequenceReg" )
      self.options['sequences'].append( "-HLTDoLocalStripSequenceRegForBTag" )
      self.options['sequences'].append( "-HLTDoLocalPixelSequenceRegForBTag" )
      self.options['sequences'].append( "-HLTDoLocalPixelSequenceRegForNoPU" )
      self.options['sequences'].append( "-hltSiPixelDigis" )
      self.options['sequences'].append( "-hltSiPixelClusters" )
      self.options['sequences'].append( "-hltSiPixelRecHits" )
      self.options['sequences'].append( "-HLTRecopixelvertexingSequence" )
      self.options['sequences'].append( "-HLTEndSequence" )
      self.options['sequences'].append( "-HLTBeginSequence" )
      self.options['sequences'].append( "-HLTBeginSequenceNZS" )
      self.options['sequences'].append( "-HLTBeginSequenceBPTX" )
      self.options['sequences'].append( "-HLTBeginSequenceAntiBPTX" )
      self.options['sequences'].append( "-HLTHBHENoiseSequence" )
      self.options['sequences'].append( "-HLTIterativeTrackingIter04" )
      self.options['sequences'].append( "-HLTIterativeTrackingIter02" )
      self.options['sequences'].append( "-HLTIterativeTracking" )
      self.options['sequences'].append( "-HLTIterativeTrackingTau3Mu" )
      self.options['sequences'].append( "-HLTIterativeTrackingReg" )
      self.options['sequences'].append( "-HLTIterativeTrackingForElectronIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingForPhotonsIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingL3MuonIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingGlbTrkMuonIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingL3MuonRegIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingHighPtTkMu" )
      self.options['sequences'].append( "-HLTIterativeTrackingHighPtTkMuIsoIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingForBTagIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingForTauIter04" )
      self.options['sequences'].append( "-HLTIterativeTrackingForTauIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingDisplacedJpsiIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingDisplacedPsiPrimeIter02" )
      self.options['sequences'].append( "-HLTIterativeTrackingDisplacedNRMuMuIter02" )
      self.options['sequences'].append( "-HLTRegionalCKFTracksForL3Isolation" )
      self.options['sequences'].append( "-HLTHBHENoiseCleanerSequence" )

      # remove HLTAnalyzerEndpath from fastsim cff's
      if self.config.fragment:
        self.options['paths'].append( "-HLTAnalyzerEndpath" )


  def build_source(self):
    if self.config.input:
      # if a dataset or a list of input files was given, use it
      if self.config.input[0:8] == 'dataset:':
        from dbsFileQuery import dbsFileQuery
        # extract the dataset name, and use DBS to fine the list of LFNs
        dataset = self.config.input[8:]
        query   = 'find file where dataset=' + dataset
        files   = dbsFileQuery(query)
        self.source = files
      else:
        # assume a list of input files
        self.source = self.config.input.split(',')
    elif self.config.online:
      # online we always run on data
      self.source = [ "file:/tmp/InputCollection.root" ]
    elif self.config.data:
      # offline we can run on data...
      self.source = [ "file:RelVal_Raw_%s_DATA.root" % self.config.type ]
    else:
      # ...or on mc
      self.source = [ "file:RelVal_Raw_%s_STARTUP.root" % self.config.type ]

    self.data += """
%(process)ssource = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
"""
    if self.source: 
      for line in self.source:
        self.data += "        '%s',\n" % line
    self.data += """    ),
    secondaryFileNames = cms.untracked.vstring(
"""
    if self.parent: 
      for line in self.parent:
        self.data += "        '%s',\n" % line
    self.data += """    ),
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)
"""
