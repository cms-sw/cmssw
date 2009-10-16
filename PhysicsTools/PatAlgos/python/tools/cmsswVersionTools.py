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
    print "*********************************************************************"
    print "NOTE TO USER: when running on 31X samples with this CMSSW version    "
    print "              of PAT the collection label for anti-kt has to be      "
    print "              switched from \'ak*\' to \'antikt*\'. This is going    "
    print "              to be done now. Also note that the *JetId collections  "
    print "              are not stored on these input files in contrary to     "
    print "              input files in 33X. Please use the _addJetId_ tool     "
    print "              as described on SWGuidePATTools, when adding new jet   "
    print "              collections! Such a line could look like this:         "
    print ""
    print "  addJetID( process, \"sisCone5CaloJets\", \"sc5\")"
    print "  from PhysicsTools.PatAlgos.tools.jetTools import *"
    print "  addJetCollection(process,cms.InputTag('sisCone5CaloJets'),"
    print "  ..."
    print "  )"
    print "*********************************************************************"
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
    
