import FWCore.ParameterSet.Config as cms

pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('BvsAll'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probb'),
            ),
         denominator=cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probb'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probg'),
         ),
      ),
      cms.PSet(
         name = cms.string('CvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probg'),
            ),
         ),
      cms.PSet(
         name = cms.string('CvsB'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probb'),
            ),
         ),
      cms.PSet(
         name = cms.string('QvsG'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probuds'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probg'),
            ),
         ),
      cms.PSet(
         name = cms.string('TauVsJet'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h1p'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probb'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probg'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h1p'),
            ),
         ),
      cms.PSet(
         name = cms.string('TauVsEle'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h1p'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probele'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h1p'),
            ),
         ),
      
      cms.PSet(
         name = cms.string('TauVsMu'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h1p'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probmu'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiCentralJetTags', 'probtaum3h1p'),
            ),
         ),

      )
   )

pfParticleNetFromMiniAODAK4PuppiForwardDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('QvsG'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiForwardJetTags', 'probq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiForwardJetTags', 'probq'),
            cms.InputTag('pfParticleNetFromMiniAODAK4PuppiForwardJetTags', 'probg'),
            ),
         ),

      )
   )

pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('BvsAll'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probb'),
            ),
         denominator=cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probb'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probg'),
         ),
      ),
      cms.PSet(
         name = cms.string('CvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probg'),
            ),
         ),
      cms.PSet(
         name = cms.string('CvsB'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probb'),
            ),
         ),
      cms.PSet(
         name = cms.string('QvsG'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probuds'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probg'),
            ),
         ),
      cms.PSet(
         name = cms.string('TauVsJet'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h1p'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probb'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probc'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probuds'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probg'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h1p'),
            ),
         ),
      cms.PSet(
         name = cms.string('TauVsEle'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h1p'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probele'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h1p'),
            ),
         ),
      
      cms.PSet(
         name = cms.string('TauVsMu'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h1p'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probmu'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaup3h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h1p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum1h2p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h0p'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSCentralJetTags', 'probtaum3h1p'),
            ),
         ),

      )
   )

pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('QvsG'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSForwardJetTags', 'probq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSForwardJetTags', 'probq'),
            cms.InputTag('pfParticleNetFromMiniAODAK4CHSForwardJetTags', 'probg'),
            ),
         ),

      )
   )
