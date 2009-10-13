import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.jetTools import *


def run33xOn31xMC(process,
                  jetSrc = cms.InputTag("antikt5CaloJets"),
                  jetIdTag = "antikt5" ):
    """
    ------------------------------------------------------------------
    switch appropriate jet collections to run 33x on 31x MC

    process : process
    jetSrc  : jet source to use
    jetID   : jet ID to make
    ------------------------------------------------------------------    
    """
    addJetID( process, jetSrc, jetIdTag )
    # in PAT (iterativeCone5) to ak5 (anti-kt cone = 0.5)
    switchJetCollection(process, 
                        cms.InputTag('antikt5CaloJets'),   
                        doJTA            = True,            
                        doBTagging       = True,            
                        jetCorrLabel     = ('AK5','Calo'),  
                        doType1MET       = True,
                        genJetCollection = cms.InputTag("antikt5GenJets"),
                        doJetID          = True,
                        jetIdLabel       = "antikt5"
                        )
    
