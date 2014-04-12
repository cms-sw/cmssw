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

def subParameterSets(pSet):
   ''' Generator to return all sub-PSets in a PSet '''
   for name, value in pSet.parameters_().iteritems():
      if isinstance(value, cms.PSet):
         yield getattr(pSet, name)

# For RECO type taus, where the tau producer is [tauType]Producer 
import re
recoTauTypeMapperRegex = re.compile("(\w*)Producer")
def recoTauTypeMapper(tauProducer):
   return recoTauTypeMapperRegex.match(tauProducer).group(1)

# For taus where the producer name is the type, like "allLayer1Taus", etc
producerIsTauTypeMapper = lambda tauProducer: tauProducer

def adaptTauDiscriminator(discriminator, newTauProducer='shrinkingConePFTauProducer',
      oldTauTypeMapper=recoTauTypeMapper, newTauTypeMapper=recoTauTypeMapper,
      preservePFTauProducer = False):
   ''' Change a TauDiscriminator to use a different tau/prediscriminant sources

   Tau discriminators use the following convention: 
        [tauType]DiscriminationByXXX

   i.e. fixedConePFTauDiscriminationByIsolation,
   allLayer1TausDiscriminationByIsolation, etc

   However, the mapping of tauType to tau producer name is not constant.  In
   RECO, the form is [tauType]Producer.  In PAT, the producer is just named
   [tauType].  To manage this oldTauTypeMapper and newTauTypeMapper are
   functions with signature f(str) that translate a TauProducer name (like
   shrinkingConePFTauProducer) to its type (shrinkingConePFTau).  Two types of
   mapping are provided, 
        * recoTauTypeMapper
              shrinkingConePFTauProducer->shrinkingConePFTau
        * producerIsTauTypeMapper
              allLayer1Taus->allLayer1Taus

   '''

   oldTauProducer = discriminator.PFTauProducer
   if isinstance(newTauProducer, str):
      newTauProducer = cms.InputTag(newTauProducer)

   # This is useful for the PF2PAT case where you wish to set the producer name
   # seperately
   if not preservePFTauProducer: 
      discriminator.PFTauProducer = newTauProducer

   oldTauType = oldTauTypeMapper(oldTauProducer.value())
   newTauType = newTauTypeMapper(newTauProducer.value())

   replacementRegex = re.compile(oldTauType)

   # Adapt all the prediscriminants
   for prediscriminant in subParameterSets(discriminator.Prediscriminants):
      oldProducer = prediscriminant.Producer.value() 
      # Replace the old tau type by the new tau type in the prediscrimant
      # producer
      prediscriminant.Producer = cms.InputTag(replacementRegex.sub(newTauType,
         oldProducer))

def adaptTauDiscriminatorSequence(sequence, **kwargs):
   def fixer(discriminator):
      if hasattr(discriminator, "Prediscriminants"):
         adaptTauDiscriminator(discriminator, **kwargs)
   sequence.visit(fixer)

def setTauSource(discriminator, newTauProducer):
   ''' Same as adaptTauDiscriminator, kept for backwards compatibility'''
   adaptTauDiscriminator(discriminator, newTauProducer)

