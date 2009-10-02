import FWCore.ParameterSet.Config as cms


def addTcMET(process,
             postfixLabel = 'TC'):
    """
    ------------------------------------------------------------------
    add track corrected MET collection to patEventContent:

    process : process
    ------------------------------------------------------------------
    """
    ## add module as process to the default sequence
    def addAlso (label,value):
        existing = getattr(process, label)
        setattr( process, label+postfixLabel, value)
        process.patDefaultSequence.replace( existing, existing*value )        

    ## clone and add a module as process to the
    ## default sequence
    def addClone(label,**replaceStatements):
        new = getattr(process, label).clone(**replaceStatements)
        addAlso(label, new)

    ## addClone('corMetType1Icone5Muons', uncorMETInputTag = cms.InputTag("tcMet"))
    addClone('layer1METs', metSource = cms.InputTag("tcMet"))

    ## add new met collections output to the pat summary
    process.allLayer1Summary.candidates += [ cms.InputTag('layer1METs'+postfixLabel) ]

def addPfMET(process,
             postfixLabel = 'PF'):
    """
    ------------------------------------------------------------------
    add pflow MET collection to patEventContent:

    process : process
    ------------------------------------------------------------------
    """
    ## add module as process to the default sequence
    def addAlso (label,value):
        existing = getattr(process, label)
        setattr( process, label+postfixLabel, value)
        process.patDefaultSequence.replace( existing, existing*value )        

    ## clone and add a module as process to the
    ## default sequence
    def addClone(label,**replaceStatements):
        new = getattr(process, label).clone(**replaceStatements)
        addAlso(label, new)

    ## addClone('corMetType1Icone5Muons', uncorMETInputTag = cms.InputTag("tcMet"))
    addClone('layer1METs', metSource = cms.InputTag("pfMet"), addMuonCorrections = False)

    ## add new met collections output to the pat summary
    process.allLayer1Summary.candidates += [ cms.InputTag('layer1METs'+postfixLabel) ]
