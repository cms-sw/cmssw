import FWCore.ParameterSet.Config as cms

def enablePileUpCorrectionInPF2PAT( process, postfix, sequence='PF2PAT' ):
    """
    Modifies the PF2PAT sequence according to the recipe of JetMET:
    """

    # pile up subtraction
    getattr(process,"pfNoPileUp"+postfix).enable = True 
    getattr(process,"pfPileUp"+postfix).checkClosestZVertex = False 
    getattr(process,"pfJets"+postfix).Vertices = cms.InputTag('goodOfflinePrimaryVertices')

    getattr(process,"pfJets"+postfix).doAreaFastjet = True
    getattr(process,"pfJets"+postfix).doRhoFastjet = False

    from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets

    setattr( process, 'kt6PFJets'+postfix, kt4PFJets.clone( rParam = cms.double(0.6),
                                                            src = cms.InputTag('pfNoElectron'+postfix),
                                                            doAreaFastjet = cms.bool(True),
                                                            doRhoFastjet = cms.bool(True),
                                                            voronoiRfact = cms.double(0.9)
                                                            )
             )
    getattr(process,sequence+postfix).replace( getattr(process,"pfNoElectron"+postfix), getattr(process,"pfNoElectron"+postfix)*getattr(process,"kt6PFJets"+postfix) )

def enablePileUpCorrectionInPAT( process, postfix, sequence ):
    # PAT specific stuff:
    jetCorrFactors = getattr(process,"patJetCorrFactors"+postfix)
    jetCorrFactors.rho = cms.InputTag("kt6PFJets"+postfix, "rho")


def enablePileUpCorrection( process, postfix, sequence='patPF2PATSequence'):
    """
    Enables the pile-up correction for jets in a PF2PAT+PAT sequence
    to be called after the usePF2PAT function.
    """

    enablePileUpCorrectionInPF2PAT( process, postfix, sequence )
    enablePileUpCorrectionInPAT( process, postfix, sequence )


