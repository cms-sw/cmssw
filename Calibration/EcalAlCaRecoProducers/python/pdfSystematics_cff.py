import FWCore.ParameterSet.Config as cms

# Produce PDF weights (maximum is 3)
pdfWeights = cms.EDProducer("PdfWeightProducer",
                            # Fix POWHEG if buggy (this PDF set will also appear on output,
                            # so only two more PDF sets can be added in PdfSetNames if not "")
                            #FixPOWHEG = cms.untracked.string("cteq66.LHgrid"),
                            #GenTag = cms.untracked.InputTag("genParticles"),
                            PdfInfoTag = cms.untracked.InputTag("generator"),
                            PdfSetNames = cms.untracked.vstring(
    "cteq66.LHgrid"
    , "MRST2006nnlo.LHgrid"
    , "NNPDF10_100.LHgrid"
    )
                            )

# Produce event weights to estimate missing O(alpha) terms + NLO QED terms
fsrWeight = cms.EDProducer("FSRWeightProducer",
                           GenTag = cms.untracked.InputTag("genParticles"),
                           )


# Produce event weights to estimate missing weak terms (=> include missing rho factor for Z diagrams)
weakWeight = cms.EDProducer("WeakEffectsWeightProducer",
                            GenParticlesTag = cms.untracked.InputTag("genParticles"),
                            RhoParameter = cms.untracked.double(1.004)
                            )



    
