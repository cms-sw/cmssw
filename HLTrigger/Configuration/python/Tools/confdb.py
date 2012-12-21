#!/usr/bin/env python

import sys
import re
import os
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
    "AlCa_EcalEta_v*",
    "AlCa_EcalPhiSym_v*",
    "AlCa_EcalPi0_v*",
    "AlCa_EcalPi0EBonly_v*",
    "AlCa_EcalPi0EEonly_v*",
    "AlCa_EcalEtaEBonly_v*",
    "AlCa_EcalEtaEEonly_v*",
    "AlCa_RPCMuonNoHits_v*",
    "AlCa_RPCMuonNoTriggers_v*",
    "AlCa_RPCMuonNormalisation_v*",
    "AlCa_LumiPixels_v*",
    "AlCa_LumiPixels_Random_v*",
    "AlCa_LumiPixels_ZeroBias_v*",
    "DQM_FEDIntegrity_v*",
    "DQM_HcalEmptyEvents_v*",
    "HLT_Calibration_v*",
    "HLT_EcalCalibration_v*",
    "HLT_HcalCalibration_v*",
    "HLT_TrackerCalibration_v*",
    "HLT_DTErrors_v*",
    "HLT_DTCalibration_v*",
    "HLT_Random_v*",
    "HLT_HcalNZS_v*",
    "HLT_HcalPhiSym_v*",
    "HLT_IsoTrackHB_v*",
    "HLT_IsoTrackHE_v*",
    "HLT_L1SingleMuOpen_AntiBPTX_v*",
    "HLT_JetE30_NoBPTX*_v*",
    "HLT_JetE50_NoBPTX*_v*",
    "HLT_JetE50_NoBPTX3BX_NoHalo_v*",
    "HLT_JetE70_NoBPTX3BX_NoHalo_v*",
    "HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v*",
    "HLT_L2Mu20_NoVertex_NoBPTX3BX_NoHalo_v*",
    "HLT_L2Mu30_NoVertex_NoBPTX3BX_NoHalo_v*",
    "HLT_JetE30_NoBPTX3BX_v*",
    "HLT_JetE50_NoBPTX3BX_v*",
    "HLT_JetE70_NoBPTX3BX_v*",
    "HLT_L2Mu10_NoVertex_NoBPTX3BX_v*",
    "HLT_L2Mu10_NoVertex_NoBPTX3BX_v*",
    "HLT_L2Mu10_NoVertex_NoBPTX3BX_v*",
    "HLT_L2Mu20_NoVertex_NoBPTX3BX_v*",
    "HLT_L2Mu30_NoVertex_NoBPTX3BX_v*",
    "HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v*",
    "HLT_L2Mu30_NoVertex_2Cha_NoBPTX3BX_NoHalo_v*",
    "HLT_PixelTracks_Multiplicity70_v*",
    "HLT_PixelTracks_Multiplicity80_v*",
    "HLT_PixelTracks_Multiplicity90_v*",
    "HLT_BeamGas_HF_Beam1_v*",
    "HLT_BeamGas_HF_Beam2_v*",
    "HLT_BeamHalo_v*",
    "HLT_L1Tech_CASTOR_HaloMuon_v*",
    "HLT_L1Tech_DT_GlobalOR_v*",
    "HLT_GlobalRunHPDNoise_v*",
    "HLT_L1Tech_HBHEHO_totalOR_v*",
    "HLT_L1Tech_HCAL_HF_single_channel_v*",
    "HLT_L1TrackerCosmics_v*",
    "HLT_HcalUTCA_v*",
    
# TODO: paths not supported by FastSim, but for which a recovery should be attempted
    
    "HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Reg_Jet30_v*", 
    "HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Reg_v*",
    "HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_Reg_v*",
    "HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v*",
    "HLT_IsoMu18_eta2p1_MediumIsoPFTau25_Trk1_eta2p1_Reg_v*",
# (not really needed for the five above, because the corresponding paths without regional
#  tracking are already in the HLT menu)
  
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

    # get the configuration from ConfdB
    self.buildPathList()
    self.buildOptions()
    self.getRawConfigurationFromDB()
    self.customize()


  def _build_query(self):
    if self.config.menu.run:
      return '--runNumber %s' % self.config.menu.run
    else:
      return '--%s --configName %s' % (self.config.menu.db, self.config.menu.name)

  def _build_options(self):
    return ' '.join(['--%s %s' % (key, ','.join(vals)) for key, vals in self.options.iteritems() if vals])

  def _build_cmdline(self):
    if not self.config.fragment:
      return 'edmConfigFromDB       %s --noedsources %s' % (self._build_query(), self._build_options())
    else:
      return 'edmConfigFromDB --cff %s --noedsources %s' % (self._build_query(), self._build_options())


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

# customization for CMSSW_5_2_X
if cmsswVersion.startswith('CMSSW_5_2_'):

    # force the use of the correct calo jet energy corrections
    if 'hltESPL1FastJetCorrectionESProducer' in %(dict)s:
        %(process)shltESPL1FastJetCorrectionESProducer.algorithm  = "AK5CaloHLT"

    if 'hltESPL2RelativeCorrectionESProducer' in %(dict)s:
        %(process)shltESPL2RelativeCorrectionESProducer.algorithm = "AK5CaloHLT"

    if 'hltESPL3AbsoluteCorrectionESProducer' in %(dict)s:
        %(process)shltESPL3AbsoluteCorrectionESProducer.algorithm = "AK5CaloHLT"


# customization for CMSSW_5_3_X
if cmsswVersion.startswith('CMSSW_5_3_'):

    # do not override the calo jet energy corrections in 5.3.x for consistency with the current MC samples
    pass


# customization for CMSSW_6_1_X / 6_2_X
if cmsswVersion.startswith('CMSSW_6_1_') or cmsswVersion.startswith('CMSSW_6_2_'):

    # force the use of the correct calo jet energy corrections
    if 'hltESPL1FastJetCorrectionESProducer' in %(dict)s:
        %(process)shltESPL1FastJetCorrectionESProducer.algorithm  = "AK5CaloHLT"

    if 'hltESPL2RelativeCorrectionESProducer' in %(dict)s:
        %(process)shltESPL2RelativeCorrectionESProducer.algorithm = "AK5CaloHLT"

    if 'hltESPL3AbsoluteCorrectionESProducer' in %(dict)s:
        %(process)shltESPL3AbsoluteCorrectionESProducer.algorithm = "AK5CaloHLT"

    # adapt the HLT menu to the "prototype for Event Interpretation" development
    if 'hltPFPileUp' in %(dict)s:
        # define new PFCandidateFwdPtrProducer module
        %(process)shltParticleFlowPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
            src = cms.InputTag('hltParticleFlow')
        )
        # add the new module before the hltPFPileUp module
        _sequence = None
        for _sequence in [ _sequence for _sequence in %(dict)s.itervalues() if isinstance(_sequence, cms._ModuleSequenceType)]:
            try:
                _sequence.insert( _sequence.index(%(process)shltPFPileUp), %(process)shltParticleFlowPtrs )
            except ValueError:
                pass
        # reconfigure hltPFPileUp and hltPFNoPileUp to use the new module
        %(process)shltPFPileUp.PFCandidates       = cms.InputTag( "hltParticleFlowPtrs" )
        %(process)shltPFNoPileUp.bottomCollection = cms.InputTag( "hltParticleFlowPtrs" )
"""

  # customize the configuration according to the options
  def customize(self):

    # adapt the source to the current scenario
    if not self.config.fragment:
      self.build_source()

    # manual override some parameters
    if self.config.type in ('GRun', ):
      self.data += """
# Enable HF Noise filters in GRun menu
if 'hltHfreco' in %(dict)s:
    %(process)shltHfreco.setNoiseFlags = cms.bool( True )
"""
    if self.config.type in ('HIon', ):
      self.data += """
# Disable HF Noise filters in HIon menu
if 'hltHfreco' in %(dict)s:
    %(process)shltHfreco.setNoiseFlags = cms.bool( False )
"""

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

      # load 5.2.x JECs, until they are in the GlobalTag
#      self.loadAdditionalConditions('load 5.2.x JECs',
#        {
#          'record'  : 'JetCorrectionsRecord',
#          'tag'     : 'JetCorrectorParametersCollection_AK5Calo_2012_V8_hlt_mc',
#          'label'   : 'AK5CaloHLT',
#          'connect' : '%(connect)s/CMS_COND_31X_PHYSICSTOOLS'
#        }, {
#          'record'  : 'JetCorrectionsRecord',
#          'tag'     : 'JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc',
#          'label'   : 'AK5PFHLT',
#          'connect' : '%(connect)s/CMS_COND_31X_PHYSICSTOOLS'
#        }, {
#          'record'  : 'JetCorrectionsRecord',
#          'tag'     : 'JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc',
#          'label'   : 'AK5PFchsHLT',
#          'connect' : '%(connect)s/CMS_COND_31X_PHYSICSTOOLS'
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
      self._fix_parameter(name = 'GMTReadoutCollection', type = 'InputTag', value = 'hltGtDigis',           replace = 'gmtDigis')
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

    if self.config.unprescale:
      self.data += """
# remove the HLT prescales
if 'PrescaleService' in %(dict)s:
    %(process)sPrescaleService.lvl1DefaultLabel = cms.string( '0' )
    %(process)sPrescaleService.lvl1Labels       = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    %(process)sPrescaleService.prescaleTable    = cms.VPSet( )
"""


  def instrumentOpenMode(self):
    if self.config.open:
      # find all EDfilters
      filters = [ match[1] for match in re.findall(r'(process\.)?\b(\w+) = cms.EDFilter', self.data) ]
      re_sequence = re.compile( r'cms\.(Path|Sequence)\((.*)\)' )
      # remove existing 'cms.ingore' and '~' modifiers
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
    %(process)sGlobalTag.connect   = '%(connect)s/CMS_COND_31X_GLOBALTAG'
    %(process)sGlobalTag.pfnPrefix = cms.untracked.string('%(connect)s/')
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
        self.config.l1.connect = '%(connect)s/CMS_COND_31X_L1T'
      self.config.l1cond = '%(tag)s,%(record)s,%(connect)s' % self.config.l1.__dict__
    else:
      self.config.l1cond = None

    if self.config.globaltag or self.config.l1cond:
      text += "    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag\n"
      text += "    %(process)sGlobalTag = customiseGlobalTag(%(process)sGlobalTag"
      if self.config.globaltag:
        text += ", globaltag = %s"  % repr(self.config.globaltag)
      if self.config.l1cond:
        text += ", conditions = %s" % repr(self.config.l1cond)
      text += ")\n"

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

# the following was stolen and adapted from HLTrigger.Configuration.customL1THLT_Options
    self.data += """
# override the process name
%%(process)ssetName_('%(name)s')

# adapt HLT modules to the correct process name
if 'hltTrigReport' in %%(dict)s:
    %%(process)shltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressCosmicsOutputSmart' in %%(dict)s:
    %%(process)shltPreExpressCosmicsOutputSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressOutputSmart' in %%(dict)s:
    %%(process)shltPreExpressOutputSmart.TriggerResultsTag        = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMForHIOutputSmart' in %%(dict)s:
    %%(process)shltPreDQMForHIOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMForPPOutputSmart' in %%(dict)s:
    %%(process)shltPreDQMForPPOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTDQMResultsOutputSmart' in %%(dict)s:
    %%(process)shltPreHLTDQMResultsOutputSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTDQMOutputSmart' in %%(dict)s:
    %%(process)shltPreHLTDQMOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTMONOutputSmart' in %%(dict)s:
    %%(process)shltPreHLTMONOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', '%(name)s' )

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


  def loadCff(self, module):
    # load a cfi or cff module
    if self.config.fragment:
      self.data += 'from %s import *\n' % module
    else:
      self.data += 'process.load( "%s" )\n' % module


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

      hasFST = False
      if 'FastTimerService' in self.data:
        hasFST = True

      self.data += '\n# configure the FastTimerService\n'
      if not hasFST:
        self.loadCff('HLTrigger.Timer.FastTimerService_cfi')
      self.data += """%(process)sFastTimerService.useRealTimeClock          = False
%(process)sFastTimerService.enableTimingPaths         = True
%(process)sFastTimerService.enableTimingModules       = True
%(process)sFastTimerService.enableTimingExclusive     = True
%(process)sFastTimerService.enableTimingSummary       = True
%(process)sFastTimerService.skipFirstPath             = True
%(process)sFastTimerService.enableDQM                 = True
%(process)sFastTimerService.enableDQMbyPathActive     = True
%(process)sFastTimerService.enableDQMbyPathTotal      = True
%(process)sFastTimerService.enableDQMbyPathOverhead   = True
%(process)sFastTimerService.enableDQMbyPathDetails    = True
%(process)sFastTimerService.enableDQMbyPathCounters   = True
%(process)sFastTimerService.enableDQMbyPathExclusive  = True
%(process)sFastTimerService.enableDQMbyModule         = True
%(process)sFastTimerService.enableDQMSummary          = True
%(process)sFastTimerService.enableDQMbyLuminosity     = True
%(process)sFastTimerService.enableDQMbyLumiSection    = True
%(process)sFastTimerService.enableDQMbyProcesses      = False
%(process)sFastTimerService.dqmTimeRange              =  1000. 
%(process)sFastTimerService.dqmTimeResolution         =     5. 
%(process)sFastTimerService.dqmPathTimeRange          =   100. 
%(process)sFastTimerService.dqmPathTimeResolution     =     0.5
%(process)sFastTimerService.dqmModuleTimeRange        =    40. 
%(process)sFastTimerService.dqmModuleTimeResolution   =     0.2
%(process)sFastTimerService.dqmLuminosityRange        = 1e+34
%(process)sFastTimerService.dqmLuminosityResolution   = 1e+31
%(process)sFastTimerService.dqmLumiSectionsRange      =  2500
%(process)sFastTimerService.dqmPath                   = 'HLT/TimerService'
%(process)sFastTimerService.luminosityProduct         = cms.untracked.InputTag( 'hltScalersRawToDigi' )
%(process)sFastTimerService.supportedProcesses        = cms.untracked.vuint32( )
"""

      self.data += """
# FastTimerServiceClient
%(process)sfastTimerServiceClient = cms.EDAnalyzer( "FastTimerServiceClient",
    dqmPath = cms.untracked.string( "HLT/TimerService" )
)

# DQM file saver
%(process)sdqmFileSaver = cms.EDAnalyzer( "DQMFileSaver",
    convention        = cms.untracked.string( "Offline" ),
    workflow          = cms.untracked.string( "/HLT/FastTimerService/All" ),
    dirName           = cms.untracked.string( "." ),
    saveByRun         = cms.untracked.int32(1),
    saveByLumiSection = cms.untracked.int32(-1),
    saveByEvent       = cms.untracked.int32(-1),
    saveByTime        = cms.untracked.int32(-1),
    saveByMinute      = cms.untracked.int32(-1),
    saveAtJobEnd      = cms.untracked.bool(False),
    forceRunNumber    = cms.untracked.int32(-1),
)

%(process)sTimingOutput = cms.EndPath( %(process)sfastTimerServiceClient + %(process)sdqmFileSaver )
"""

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
    self.options['services'].append( "-FUShmDQMOutputService" )

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
      self.options['services'].append( "-DQMStore" )
      self.options['services'].append( "-MicroStateService" )
      self.options['services'].append( "-ModuleWebRegistry" )
      self.options['services'].append( "-TimeProfilerService" )
      self.options['services'].append( "-FastTimerService" )

      self.options['psets'].append( "-maxEvents" )
      self.options['psets'].append( "-options" )

    if self.config.fastsim:
      # remove components not supported or needed by fastsim
      self.options['esmodules'].append( "-navigationSchoolESProducer" )
      self.options['esmodules'].append( "-TransientTrackBuilderESProducer" )
      self.options['esmodules'].append( "-SteppingHelixPropagatorAny" )
      self.options['esmodules'].append( "-OppositeMaterialPropagator" )
      self.options['esmodules'].append( "-MaterialPropagator" )
      self.options['esmodules'].append( "-CaloTowerConstituentsMapBuilder" )
      self.options['esmodules'].append( "-CaloTopologyBuilder" )

      self.options['services'].append( "-UpdaterService" )

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
      
      self.options['modules'].append( "-hltESRegionalEgammaRecHit" )
      self.options['modules'].append( "-hltEcalRegionalJetsFEDs" )
      self.options['modules'].append( "-hltEcalRegionalMuonsFEDs" )
      self.options['modules'].append( "-hltEcalRegionalEgammaFEDs" )
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
      self.options['modules'].append( "-hltIter4Merged" )
      self.options['modules'].append( "-hltFastPixelHitsVertex" )
      self.options['modules'].append( "-hltFastPixelTracks")
      self.options['modules'].append( "-hltFastPixelTracksRecover")
      
      self.options['modules'].append( "-hltFastPrimaryVertexbbPhi")
      self.options['modules'].append( "-hltPixelTracksFastPVbbPhi")
      self.options['modules'].append( "-hltPixelTracksRecoverbbPhi" )
      self.options['modules'].append( "-hltFastPixelHitsVertexVHbb" )
      self.options['modules'].append( "-hltFastPixelTracksVHbb" )
      self.options['modules'].append( "-hltFastPixelTracksRecoverVHbb" )

      self.options['modules'].append( "-hltFastPrimaryVertex")
      self.options['modules'].append( "-hltFastPVPixelTracks")
      self.options['modules'].append( "-hltFastPVPixelTracksRecover" )

      self.options['modules'].append( "-hltIter4Tau3MuMerged" )
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
      self.options['sequences'].append( "-HLTIterativeTracking" )
      self.options['sequences'].append( "-HLTIterativeTrackingTau3Mu" )
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

