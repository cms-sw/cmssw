from FWCore.ParameterSet import Config, Mixins 

def switchToSimGtDigis(process):
  """patch the process to use 'sim*Digis' from the L1 emulator instead of 'hlt*Digis' from the RAW data"""

  # explicit replacements to use "simGtDigis", "simGmtDigis" and "simGctDigis" instead of "hltGtDigis", "hltGmtDigis" or "hltGctDigis"
  if 'hltL1GtObjectMap' in process.__dict__:
    process.hltL1GtObjectMap.GmtInputTag = Config.InputTag( 'simGmtDigis' )
    process.hltL1GtObjectMap.GctInputTag = Config.InputTag( 'simGctDigis' )
  if 'hltL1extraParticles' in process.__dict__:
    process.hltL1extraParticles.muonSource = Config.InputTag( 'simGmtDigis' )
    process.hltL1extraParticles.isolatedEmSource = Config.InputTag( 'simGctDigis','isoEm' )
    process.hltL1extraParticles.nonIsolatedEmSource = Config.InputTag( 'simGctDigis','nonIsoEm' )
    process.hltL1extraParticles.centralJetSource = Config.InputTag( 'simGctDigis','cenJets' )
    process.hltL1extraParticles.forwardJetSource = Config.InputTag( 'simGctDigis','forJets' )
    process.hltL1extraParticles.tauJetSource = Config.InputTag( 'simGctDigis','tauJets' )
    process.hltL1extraParticles.etTotalSource = Config.InputTag( 'simGctDigis' )
    process.hltL1extraParticles.etHadSource = Config.InputTag( 'simGctDigis' )
    process.hltL1extraParticles.etMissSource = Config.InputTag( 'simGctDigis' )
  if 'hltL2MuonSeeds' in process.__dict__:
    process.hltL2MuonSeeds.GMTReadoutCollection = Config.InputTag( 'simGmtDigis' )

  # automatic replacements to use "simGtDigis", "simGmtDigis" and "simGctDigis" instead of "hltGtDigis", "hltGmtDigis" or "hltGctDigis"
  for module in process.__dict__.itervalues():
    if isinstance(module, Mixins._Parameterizable):
      for parameter in module.__dict__.itervalues():
        if isinstance(parameter, Config.InputTag):
          if parameter.moduleLabel == 'hltGtDigis':
            parameter.moduleLabel = 'simGtDigis'
          elif parameter.moduleLabel == 'hltGmtDigis':
            parameter.moduleLabel = 'simGmtDigis'
          elif parameter.moduleLabel == 'hltGctDigis':
            parameter.moduleLabel = 'simGctDigis'

  # check if "hltGtDigis", "hltGmtDigis" and "hltGctDigis" are defined
  hasGtDigis  = 'hltGtDigis'  in process.producers
  hasGmtDigis = 'hltGmtDigis' in process.producers
  hasGctDigis = 'hltGctDigis' in process.producers

  # remove "hltGtDigis", "hltGmtDigis" and "hltGctDigis" from all paths, endpaths and sequences
  for iterable in process.sequences.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGmtDigis: iterable.remove( process.hltGmtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  for iterable in process.paths.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGmtDigis: iterable.remove( process.hltGmtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

  for iterable in process.endpaths.itervalues():
    if hasGtDigis:  iterable.remove( process.hltGtDigis )
    if hasGmtDigis: iterable.remove( process.hltGmtDigis )
    if hasGctDigis: iterable.remove( process.hltGctDigis )

