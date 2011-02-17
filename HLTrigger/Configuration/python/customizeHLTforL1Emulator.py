from FWCore.ParameterSet import Config, Mixins 

def switchToCustomL1Digis(process, customGmt, customGct, customGt):
  """patch the process to use custom GMT, GCT and GT results"""

  # explicit replacements to use "simGtDigis", "simGmtDigis" and "simGctDigis" instead of "hltGtDigis" or "hltGctDigis"
  if 'hltL1GtObjectMap' in process.__dict__:
    process.hltL1GtObjectMap.GmtInputTag = Config.InputTag( customGmt )
    process.hltL1GtObjectMap.GctInputTag = Config.InputTag( customGct )
  if 'hltL1extraParticles' in process.__dict__:
    process.hltL1extraParticles.muonSource          = Config.InputTag( customGmt )
    process.hltL1extraParticles.isolatedEmSource    = Config.InputTag( customGct, 'isoEm' )
    process.hltL1extraParticles.nonIsolatedEmSource = Config.InputTag( customGct, 'nonIsoEm' )
    process.hltL1extraParticles.centralJetSource    = Config.InputTag( customGct, 'cenJets' )
    process.hltL1extraParticles.forwardJetSource    = Config.InputTag( customGct, 'forJets' )
    process.hltL1extraParticles.tauJetSource        = Config.InputTag( customGct, 'tauJets' )
    process.hltL1extraParticles.etTotalSource       = Config.InputTag( customGct )
    process.hltL1extraParticles.etHadSource         = Config.InputTag( customGct )
    process.hltL1extraParticles.etMissSource        = Config.InputTag( customGct )
  if 'hltL2MuonSeeds' in process.__dict__:
    process.hltL2MuonSeeds.GMTReadoutCollection = Config.InputTag( customGmt )

  # automatic replacements to use "simGtDigis" and "simGctDigis" instead of "hltGtDigis" or "hltGctDigis"
  for module in process.__dict__.itervalues():
    if isinstance(module, Mixins._Parameterizable):
      for parameter in module.__dict__.itervalues():
        if isinstance(parameter, Config.InputTag):
          if parameter.moduleLabel == 'hltGtDigis':
            parameter.moduleLabel = customGt
          elif parameter.moduleLabel == 'hltGctDigis':
            parameter.moduleLabel = customGct

  # check if "hltGtDigis" and "hltGctDigis" are defined
  hasGtDigis  = 'hltGtDigis'  in process.producers
  hasGctDigis = 'hltGctDigis' in process.producers

  # remove "hltGtDigis" and "hltGctDigis" from all paths, endpaths and sequences
  for iterable in process.sequences.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  for iterable in process.paths.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  for iterable in process.endpaths.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  return process


def switchToSimGtDigis(process):
  """patch the process to use newly emulated GMT, GCT and GT results"""
  return switchToCustomL1Digis(process, 'simGmtDigis', 'simGctDigis', 'simGtDigis')


def switchToSimGctGtDigis(process):
  """patch the process to use gtDigis for GMT results, and newly emulated GCT and GT results"""
  return switchToCustomL1Digis(process, 'gtDigis', 'simGctDigis', 'simGtDigis')


def switchToSimGmtGtDigis(process):
  """patch the process to use gctDigis for GCT results, and newly emulated GMT and GT results"""
  return switchToCustomL1Digis(process, 'simGmtDigis', 'gctDigis', 'simGtDigis')
