import FWCore.ParameterSet.Config as cms


# require the EXISTANCE of a track - not necessarily above any pt cut (above the basic 0.5 GeV filter)
leadTrackFinding = cms.PSet(
      Producer = cms.InputTag('pfRecoTauDiscriminationByLeadingTrackFinding'),
      cut = cms.double(0.5)
)

# Require no prediscriminants
noPrediscriminants = cms.PSet(
      BooleanOperator = cms.string("and"),
      )

# Require the existence of a lead track
requireLeadTrack = cms.PSet(
      BooleanOperator = cms.string("and"),
      leadTrack = leadTrackFinding,
      )

# Require a existence of a lead track in a CaloTau.
requireLeadTrackCalo = cms.PSet(
      BooleanOperator = cms.string("and"),
      leadTrack = cms.PSet(
         Producer = cms.InputTag('caloRecoTauDiscriminationByLeadingTrackFinding'),
         cut = cms.double(0.5)
         )
      )

# This is equivalent to the lead track case, and shoudl be deprecated.  
#  Preserved for backwards compatibility
requireLeadPion = cms.PSet(
      BooleanOperator = cms.string("and"),
      leadPion = leadTrackFinding,
      )

import re
tauTypeRegex = re.compile("(\w*)Producer")

def subParameterSets(pSet):
   ''' Generator to return all sub-PSets in a PSet '''
   for name, value in pSet.parameters_().iteritems():
      if isinstance(value, cms.PSet):
         yield getattr(pSet, name)

def setTauSource(discriminatorModule, tauSource):
   ''' Set the PFTau producer to tauSource, and update the list of prediscriminants correspondingly '''
   # get the old tau type
   oldTauType = tauTypeRegex.match(discriminatorModule.PFTauProducer.value()).group(1)
   #print oldTauType
   # get the new tau type
   newTauType = tauTypeRegex.match(tauSource).group(1)
   #print newTauType

   discriminatorModule.PFTauProducer = cms.InputTag(tauSource)

   tauTypeReplacement = re.compile(oldTauType)

   # change the tau type of the prediscriminants
   for aPrediscriminant in subParameterSets(discriminatorModule.Prediscriminants):
      aPrediscriminant.Producer = cms.InputTag( tauTypeReplacement.sub(newTauType, aPrediscriminant.Producer.value()) )

