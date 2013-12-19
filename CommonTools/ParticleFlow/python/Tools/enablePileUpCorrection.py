import FWCore.ParameterSet.Config as cms

def enablePileUpCorrectionInPF2PAT( process, postfix, sequence='PF2PAT'):
    """
    Modifies the PF2PAT sequence according to the recipe of JetMET:
    """

    # pile up subtraction
    getattr(process,"pfNoPileUp"+postfix).enable = True 
    getattr(process,"pfPileUp"+postfix).Enable = True 
    getattr(process,"pfPileUp"+postfix).checkClosestZVertex = False 
    getattr(process,"pfPileUp"+postfix).Vertices = 'goodOfflinePrimaryVertices'

    getattr(process,"pfJets"+postfix).doAreaFastjet = True
    getattr(process,"pfJets"+postfix).doRhoFastjet = False
        
    # adding goodOfflinePrimaryVertices before pfPileUp
    process.load('CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi')   
    getattr(process, 'pfNoPileUpSequence'+postfix).replace( getattr(process,"pfPileUp"+postfix),
                                                            process.goodOfflinePrimaryVertices +
                                                            getattr(process,"pfPileUp"+postfix) )
    
def enablePileUpCorrectionInPAT( process, postfix, sequence ):
    # PAT specific stuff:

    jetCorrFactors = getattr(process,"patJetCorrFactors"+postfix)
    # using non-pileup-charged-hadron-substracted kt6PFJets consistently with JetMET recommendation
    jetCorrFactors.rho = cms.InputTag("fixedGridRhoFastjetAll")


def enablePileUpCorrection( process, postfix, sequence='patPF2PATSequence'):
    """
    Enables the pile-up correction for jets in a PF2PAT+PAT sequence
    to be called after the usePF2PAT function.
    """

    enablePileUpCorrectionInPF2PAT( process, postfix, sequence)
    enablePileUpCorrectionInPAT( process, postfix, sequence)


