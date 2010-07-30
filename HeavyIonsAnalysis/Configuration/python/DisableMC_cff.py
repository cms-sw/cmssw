## this should probably be moved to PhysicsTools.PatAlgos.tools.heavyIonTools

# Turn off MC dependence in HI PAT sequence
def removePatMCMatch(process):
  process.makeHeavyIonJets.remove(process.genPartons)
  process.makeHeavyIonJets.remove(process.heavyIonCleanedGenJets)
  process.makeHeavyIonJets.remove(process.hiPartons)
  process.makeHeavyIonJets.remove(process.patJetGenJetMatch)
  process.makeHeavyIonJets.remove(process.patJetPartonMatch)

  process.patJets.addGenPartonMatch   = False
  process.patJets.embedGenPartonMatch = False
  process.patJets.genPartonMatch      = ''
  process.patJets.addGenJetMatch      = False
  process.patJets.genJetMatch	      = ''
  process.patJets.getJetMCFlavour     = False
  process.patJets.JetPartonMapSource  = ''

  process.makeHeavyIonMuons.remove(process.muonMatch)

  process.patMuons.addGenMatch        = False
  process.patMuons.embedGenMatch      = False
  
  return process

# Top Config to turn off all MC dependence
def disableMC(process):
  process.heavyIon.doMC = False
  removePatMCMatch(process)
  return process
