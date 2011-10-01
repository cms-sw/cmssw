#!/usr/bin/env python

import sys
import re
import os
from pipe import pipe as _pipe
from options import globalTag


class HLTProcess(object):
  # paths not supported by FastSim
  fastsimUnsupportedPaths = (

  # paths for which a recovery is not foreseen/possible
    "AlCa_EcalEta_v*",
    "AlCa_EcalPhiSym_v*",
    "AlCa_EcalPi0_v*",
    "AlCa_RPCMuonNoHits_v*",
    "AlCa_RPCMuonNoTriggers_v*",
    "AlCa_RPCMuonNormalisation_v*",
    "DQM_FEDIntegrity_v*",
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

  # TODO: paths not supported by FastSim, but for which a recovery should be attempted
    "HLT_Mu3_Track3_Jpsi_v*",
    "HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v*",
    "HLT_Mu5_Track0_Jpsi_B5Q7_v*",
    "HLT_Mu5_Track2_Jpsi_v*",
    "HLT_Mu5_Track5_Jpsi_v*",
    "HLT_Mu7_Track5_Jpsi_v*",
    "HLT_Mu7_Track7_Jpsi_v*",

    "HLT_HT250_DoubleDisplacedJet60_v*",
    "HLT_HT250_DoubleDisplacedJet60_PromptTrack_v*",
  )

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
    self.buildPathList()
    self.buildOptions()
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
# version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']
"""

    # from CMSSW_4_4_0_pre8: update HF configuration for V00-09-18 RecoLocalCalo/HcalRecProducers
    if not self.config.fastsim:
      self.data += """
# from CMSSW_4_4_0_pre8: update HF configuration for V00-09-18 RecoLocalCalo/HcalRecProducers
if cmsswVersion > "CMSSW_4_4":
    if 'hltHfreco' in %(dict)s:
        %(process)shltHfreco.digiTimeFromDB = cms.bool( False )
        %(process)shltHfreco.digistat.HFdigiflagCoef = cms.vdouble(
            %(process)shltHfreco.digistat.HFdigiflagCoef0.value(),
            %(process)shltHfreco.digistat.HFdigiflagCoef1.value(),
            %(process)shltHfreco.digistat.HFdigiflagCoef2.value()
        )
        del %(process)shltHfreco.digistat.HFdigiflagCoef0
        del %(process)shltHfreco.digistat.HFdigiflagCoef1
        del %(process)shltHfreco.digistat.HFdigiflagCoef2
"""

    # from CMSSW_4_4_0_pre6: updated configuration for the HybridClusterProducer's and EgammaHLTHybridClusterProducer's
    self.data += """
# from CMSSW_4_4_0_pre6: updated configuration for the HybridClusterProducer's and EgammaHLTHybridClusterProducer's
if cmsswVersion > "CMSSW_4_4":
    if 'hltHybridSuperClustersActivity' in %(dict)s:
        %(process)shltHybridSuperClustersActivity.xi               = cms.double( 0.0 )
        %(process)shltHybridSuperClustersActivity.useEtForXi       = cms.bool( False )
    if 'hltHybridSuperClustersL1Isolated' in %(dict)s:
        %(process)shltHybridSuperClustersL1Isolated.xi             = cms.double( 0.0 )
        %(process)shltHybridSuperClustersL1Isolated.useEtForXi     = cms.bool( False )
    if 'hltHybridSuperClustersL1NonIsolated' in %(dict)s:
        %(process)shltHybridSuperClustersL1NonIsolated.xi          = cms.double( 0.0 )
        %(process)shltHybridSuperClustersL1NonIsolated.useEtForXi  = cms.bool( False )
"""

    # from CMSSW_4_4_0_pre5: updated configuration for the PFRecoTauDiscriminationByIsolation producers
    self.data += """
# from CMSSW_4_4_0_pre5: updated configuration for the PFRecoTauDiscriminationByIsolation producers
if cmsswVersion > "CMSSW_4_4":
    if 'hltPFTauTightIsoIsolationDiscriminator' in %(dict)s:
        %(process)shltPFTauTightIsoIsolationDiscriminator.qualityCuts.primaryVertexSrc = %(process)shltPFTauTightIsoIsolationDiscriminator.PVProducer
        %(process)shltPFTauTightIsoIsolationDiscriminator.qualityCuts.pvFindingAlgo    = cms.string('highestPtInEvent')
        del %(process)shltPFTauTightIsoIsolationDiscriminator.PVProducer
    if 'hltPFTauLooseIsolationDiscriminator' in %(dict)s:
        %(process)shltPFTauLooseIsolationDiscriminator.qualityCuts.primaryVertexSrc = %(process)shltPFTauLooseIsolationDiscriminator.PVProducer
        %(process)shltPFTauLooseIsolationDiscriminator.qualityCuts.pvFindingAlgo    = cms.string('highestPtInEvent')
        del %(process)shltPFTauLooseIsolationDiscriminator.PVProducer
"""

    # from CMSSW_4_4_0_pre5: updated configuration for the EcalSeverityLevelESProducer
    self.data += """
# from CMSSW_4_4_0_pre5: updated configuration for the EcalSeverityLevelESProducer
if cmsswVersion > "CMSSW_4_4":
    %(process)secalSeverityLevel = cms.ESProducer("EcalSeverityLevelESProducer",
        appendToDataLabel = cms.string(''),
        dbstatusMask=cms.PSet(
            kGood        = cms.vuint32(0),
            kProblematic = cms.vuint32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            kRecovered   = cms.vuint32(),
            kTime        = cms.vuint32(),
            kWeird       = cms.vuint32(),
            kBad         = cms.vuint32(11, 12, 13, 14, 15, 16)
        ),
        flagMask = cms.PSet (
            kGood        = cms.vstring('kGood'),
            kProblematic = cms.vstring('kPoorReco', 'kPoorCalib', 'kNoisy', 'kSaturated'),
            kRecovered   = cms.vstring('kLeadingEdgeRecovered', 'kTowerRecovered'),
            kTime        = cms.vstring('kOutOfTime'),
            kWeird       = cms.vstring('kWeird', 'kDiWeird'),
            kBad         = cms.vstring('kFaultyHardware', 'kDead', 'kKilled')
        ),
        timeThresh = cms.double(2.0)
    )
"""

    # from CMSSW_4_4_0_pre3: additional ESProducer in cfg files
    if not self.config.fragment:
      self.data += """
# from CMSSW_4_4_0_pre3: additional ESProducer in cfg files
if cmsswVersion > "CMSSW_4_4":
    %(process)shltSiPixelQualityESProducer = cms.ESProducer("SiPixelQualityESProducer",
        ListOfRecordToMerge = cms.VPSet(
            cms.PSet( record = cms.string("SiPixelQualityFromDbRcd"),
                      tag    = cms.string("")
                    ),
            cms.PSet( record = cms.string("SiPixelDetVOffRcd"),
                      tag    = cms.string("")
                    )
        )
    )
"""

    # from CMSSW_4_3_0_pre6: additional ESProducer in cfg files
    if not self.config.fragment:
      self.data += """
# from CMSSW_4_3_0_pre6: additional ESProducer in cfg files
if cmsswVersion > "CMSSW_4_3":
    %(process)shltESPStripLorentzAngleDep = cms.ESProducer("SiStripLorentzAngleDepESProducer",
        LatencyRecord = cms.PSet(
            record = cms.string('SiStripLatencyRcd'),
            label = cms.untracked.string('')
        ),
        LorentzAngleDeconvMode = cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd'),
            label = cms.untracked.string('deconvolution')
        ),
        LorentzAnglePeakMode = cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd'),
            label = cms.untracked.string('peak')
        )
    )
"""

    # from CMSSW_4_3_0_pre6: ECAL severity flags migration
    self.data += """
# from CMSSW_4_3_0_pre6: ECAL severity flags migration
if cmsswVersion > "CMSSW_4_3":
  import HLTrigger.Configuration.Tools.updateEcalSeverityFlags
  HLTrigger.Configuration.Tools.updateEcalSeverityFlags.update( %(dict)s )
"""


  # customize the configuration according to the options
  def customize(self):

    if self.config.fragment:
      # if running on MC, adapt the configuration accordingly
      self.fixForMC()

      # if requested, adapt the configuration for FastSim
      self.fixForFastSim()

      # if requested, remove the HLT prescales
      self.fixPrescales()

      # if requested, override all ED/HLTfilters to always pass ("open" mode)
      self.instrumentOpenMode()

      # if requested, instrument the self with the modules and EndPath needed for timing studies
      self.instrumentTiming()

      # add version-specific customisations
      self.releaseSpecificCustomize()

    else:
      # if running on MC, adapt the configuration accordingly
      self.fixForMC()

      # override the process name and adapt the relevant filters
      self.overrideProcessName()

      # if required, remove the HLT prescales
      self.fixPrescales()

      # if requested, override all ED/HLTfilters to always pass ("open" mode)
      self.instrumentOpenMode()

      # manual override some Heavy Ion parameters
      if self.config.type in ('HIon', ):
        self.data += """
# HIon paths in smart prescalers
if 'hltPreDQMOutputSmart' in %(dict)s:
    %(process)shltPreDQMOutputSmart.throw     = cms.bool( False )
if 'hltPreExpressOutputSmart' in %(dict)s:
    %(process)shltPreExpressOutputSmart.throw = cms.bool( False )
if 'hltPreHLTDQMOutputSmart' in %(dict)s:
    %(process)shltPreHLTDQMOutputSmart.throw  = cms.bool( False )
if 'hltPreHLTMONOutputSmart' in %(dict)s:
    %(process)shltPreHLTMONOutputSmart.throw  = cms.bool( False )
"""

      # override the output modules to output root files
      self.overrideOutput()

      # add global options
      self.addGlobalOptions()

      # if requested or necessary, override the GlobalTag and connection strings
      self.overrideGlobalTag()

      # if requested, override the L1 self from the GlobalTag (using the same connect as the GlobalTag itself)
      self.overrideL1Menu()

      # if requested, run (part of) the L1 emulator
      self.runL1Emulator()

      # request summary informations from the MessageLogger
      self.updateMessageLogger()

      # if requested, instrument the self with the modules and EndPath needed for timing studies
      self.instrumentTiming()

      # add version-specific customisations
      self.releaseSpecificCustomize()


#    # load 4.2.x JECs
#    self.loadAdditionalConditions('load 4.2.x JECs',
#      {
#        'record'  : 'JetCorrectionsRecord',
#        'tag'     : 'JetCorrectorParametersCollection_Jec11_V1_AK5Calo',
#        'label'   : 'AK5Calo',
#        'connect' : 'frontier://PromptProd/CMS_COND_31X_PHYSICSTOOLS'
#      }
#    )

  def addGlobalOptions(self):
    # add global options
    self.data += """
# limit the number of events to be processed
%(process)smaxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)
"""
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
      ## TO BE REMOVED as soon the configuration in confDB gets fixed:
      #self._fix_parameter(                               type = 'InputTag', value = 'hltPFJetCtfWithMaterialTracks', replace = 'hltIter4Merged')

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
    %(process)sPrescaleService.lvl1DefaultLabel = cms.untracked.string( '0' )
    %(process)sPrescaleService.lvl1Labels       = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    %(process)sPrescaleService.prescaleTable    = cms.VPSet( )
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
    #      - if the GT is "auto:...", insert the code to read it from Configuration.AlCa.autoCond
    text = ''
    if self.config.online:
      if self.config.globaltag:
        # override the GlobalTag connection string and pfnPrefix
        text += """
# override the GlobalTag
if 'GlobalTag' in %%(dict)s:
    %%(process)sGlobalTag.connect   = '%%(connect)s/CMS_COND_31X_GLOBALTAG'
    %%(process)sGlobalTag.pfnPrefix = cms.untracked.string('%%(connect)s/')
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
        text += "    from Configuration.AlCa.autoCond import autoCond\n"
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
# customize the L1 emulator to run only the GT, and take the GCT and GMT from data
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
      r'%(process)shltOutput\2 = cms.OutputModule( "PoolOutputModule",\n    fileName = cms.untracked.string( "output\2.root" ),\n    fastCloning = cms.untracked.bool( False ),',
      self.data
    )

    if not self.config.fragment and self.config.output == 'full':
      # add a single "keep *" output
      self.data += """
# add a single "keep *" output
%(process)shltOutputFULL = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputFULL.root" ),
    fastCloning = cms.untracked.bool( False ),
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
    %%(process)shltTrigReport.HLTriggerResults       = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreExpressSmart' in %%(dict)s:
    %%(process)shltPreExpressSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreHLTMONSmart' in %%(dict)s:
    %%(process)shltPreHLTMONSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltPreDQMSmart' in %%(dict)s:
    %%(process)shltPreDQMSmart.TriggerResultsTag     = cms.InputTag( 'TriggerResults', '', '%(name)s' )

if 'hltDQMHLTScalers' in %%(dict)s:
    %%(process)shltDQMHLTScalers.triggerResults      = cms.InputTag( 'TriggerResults', '', '%(name)s' )
    %%(process)shltDQMHLTScalers.processname         = '%(name)s'

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

      # load additional conditions needed by hltGetConditions
      self.loadAdditionalConditions('add XML geometry to keep hltGetConditions happy',
        {
          'record'  : 'GeometryFileRcd',
          'tag'     : 'XMLFILE_Geometry_311YV1_Ideal_mc',
          'label'   : 'Ideal',
          'connect' : '%(connect)s/CMS_COND_34X_GEOMETRY'
        }, {
          'record'  : 'GeometryFileRcd',
          'tag'     : 'XMLFILE_Geometry_311YV1_Extended_mc',
          'label'   : 'Extended',
          'connect' : '%(connect)s/CMS_COND_34X_GEOMETRY'
        }
      )

    # instrument the menu with the Service, EDProducer and EndPath needed for timing studies
    # FIXME in a cff, should also update the HLTSchedule
    if self.config.timing:
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
      self.options['modules'].append( "-hltCkfL1IsoTrackCandidates" )
      self.options['modules'].append( "-hltCtfL1IsoWithMaterialTracks" )
      self.options['modules'].append( "-hltCkfL1NonIsoTrackCandidates" )
      self.options['modules'].append( "-hltCtfL1NonIsoWithMaterialTracks" )
      self.options['modules'].append( "-hltCkf3HitL1IsoTrackCandidates" )
      self.options['modules'].append( "-hltCtf3HitL1IsoWithMaterialTracks" )
      self.options['modules'].append( "-hltCkf3HitL1NonIsoTrackCandidates" )
      self.options['modules'].append( "-hltCtf3HitL1NonIsoWithMaterialTracks" )
      self.options['modules'].append( "-hltCkf3HitActivityTrackCandidates" )
      self.options['modules'].append( "-hltCtf3HitActivityWithMaterialTracks" )
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
      self.options['modules'].append( "-hltPFJetPixelSeeds" )
      self.options['modules'].append( "-hltPFJetCkfTrackCandidates" )
      self.options['modules'].append( "-hltPFJetCtfWithMaterialTracks" )
      self.options['modules'].append( "-hltPFlowTrackSelectionHighPurity" )
      # === hltBLifetimeRegional
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorEleJetSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesEleJetSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksEleJetSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorIsoEleJetSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesIsoEleJetSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksIsoEleJetSingleTop" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorRA2b" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesRA2b" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksRA2b" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorRAzr" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesRAzr" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksRAzr" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorHbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesHbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksHbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixel3DSeedGeneratorJet30Hbb" )
      self.options['modules'].append( "-hltBLifetimeRegional3DCkfTrackCandidatesJet30Hbb" )
      self.options['modules'].append( "-hltBLifetimeRegional3DCtfWithMaterialTracksJet30Hbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorbbPhi" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesbbPhi" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksbbPhi" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb" )
      self.options['modules'].append( "-hltBLifetimeRegionalPixelSeedGeneratorGammaB" )
      self.options['modules'].append( "-hltBLifetimeRegionalCkfTrackCandidatesGammaB" )
      self.options['modules'].append( "-hltBLifetimeRegionalCtfWithMaterialTracksGammaB" )

      self.options['modules'].append( "-hltPixelTracksForMinBias" )
      self.options['modules'].append( "-hltPixelTracksForHighMult" )
      self.options['modules'].append( "-hltIter4Merged" )
      self.options['modules'].append( "-hltPFJetCtfWithMaterialTracks" )
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

      self.options['sequences'].append( "-HLTL1IsoEgammaRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTL1NonIsoEgammaRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTEcalActivityEgammaRegionalRecoTrackerSequence" )
      self.options['sequences'].append( "-HLTPixelMatchElectronActivityTrackingSequence" )
      self.options['sequences'].append( "-HLTDoLocalStripSequence" )
      self.options['sequences'].append( "-HLTDoLocalPixelSequence" )
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

      # remove HLTAnalyzerEndpath from fastsim cff's
      if self.config.fragment:
        self.options['paths'].append( "-HLTAnalyzerEndpath" )


  def build_source(self):
    if self.config.input:
      # if an explicit input file was given, use it
      self.source = self.config.input
    elif self.config.online:
      # online we always run on data
      self.source = "file:/tmp/InputCollection.root"
    elif self.config.data:
      # offline we can run on data...
      self.source = "/store/data/Run2011A/MinimumBias/RAW/v1/000/165/205/6C8BA6D0-F680-E011-B467-003048F118AC.root"
    else:
      # ...or on mc
      self.source = "file:RelVal_DigiL1Raw_%s.root" % self.config.type

