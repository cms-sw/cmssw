import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import *

def restrictInputToAOD(process,
                   names=['All']
                   ):
    """
    ------------------------------------------------------------------
    remove pat object production steps which rely on RECO event
    content:

    process : process
    name    : list of collection names; supported are 'Photons', 
              'Electrons',, 'Muons', 'Taus', 'Jets', 'METs', 'All'
    ------------------------------------------------------------------    
    """
    for obj in range(len(names)):
        print "---------------------------------------------------------------------"
        print "WARNING: the following additional information can only be used on "
        print "         RECO format:"
        if( names[obj] == 'Photons' or names[obj] == 'All' ):
            print "          * nothing needs to be done for Photons"
        if( names[obj] == 'Electrons' or names[obj] == 'All' ):
            print "          * nothing needs to be done for Electrons"            
        if( names[obj] == 'Muons' or names[obj] == 'All' ):
            print "          * nothing needs to be done for Muons"            
        if( names[obj] == 'Taus' or names[obj] == 'All' ):
            print "          * nothing needs to be done for Taus"            
        if( names[obj] == 'Jets' or names[obj] == 'All' ):
            print "          * nothing needs to be done for Jets"            
        if( names[obj] == 'METs' or names[obj] == 'All' ):
            print "          * nothing needs to be done for METs"            
    print "---------------------------------------------------------------------"
    

def removeMCMatching(process,
                     name
                     ):
    """
    ------------------------------------------------------------------
    remove monte carlo matching from a given collection or all PAT
    candidate collections:

    process : process
    name    : collection name; supported are 'Photons', 'Electrons',
              'Muons', 'Taus', 'Jets', 'METs', 'All'
    ------------------------------------------------------------------    
    """
    if( name == 'Photons'   or name == 'All' ):
        _removeMCMatchingForPATObject(process, 'photonMatch', 'allLayer1Photons') 
    if( name == 'Electrons' or name == 'All' ):
        _removeMCMatchingForPATObject(process, 'electronMatch', 'allLayer1Electrons') 
    if( name == 'Muons'     or name == 'All' ):
        _removeMCMatchingForPATObject(process, 'muonMatch', 'allLayer1Muons') 
    if( name == 'Taus'      or name == 'All' ):
        _removeMCMatchingForPATObject(process, 'tauMatch', 'allLayer1Taus')
        ## remove mc extra modules for taus
        process.patDefaultSequence.remove(process.tauGenJets)
        process.patDefaultSequence.remove(process.tauGenJetMatch)
        ## remove mc extra configs for taus
        tauProducer = getattr(process, 'allLayer1Taus')
        tauProducer.addGenJetMatch      = False
        tauProducer.embedGenJetMatch    = False
        tauProducer.genJetMatch         = ''         
    if( name == 'Jets'      or name == 'All' ):
        ## remove mc extra modules for jets
        process.patDefaultSequence.remove(process.jetPartonMatch)
        process.patDefaultSequence.remove(process.jetGenJetMatch)
        process.patDefaultSequence.remove(process.jetFlavourId)
        ## remove mc extra configs for jets
        jetProducer = getattr(process, 'allLayer1Jets')
        jetProducer.addGenPartonMatch   = False
        jetProducer.embedGenPartonMatch = False
        jetProducer.genPartonMatch      = ''
        jetProducer.addGenJetMatch      = False
        jetProducer.genJetMatch         = ''
        jetProducer.getJetMCFlavour     = False
        jetProducer.JetPartonMapSource  = ''       
    if( name == 'METs'      or name == 'All' ):
        ## remove mc extra configs for jets
        metProducer = getattr(process, 'layer1METs')        
        metProducer.addGenMET           = False
        metProducer.genMETSource        = ''       


def _removeMCMatchingForPATObject(process, matcherName, producerName):
    ## remove mcMatcher from the default sequence
    objectMatcher = getattr(process, matcherName)
    process.patDefaultSequence.remove(objectMatcher)
    ## straighten photonProducer
    objectProducer = getattr(process, producerName)
    objectProducer.addGenMatch      = False
    objectProducer.embedGenMatch    = False
    objectProducer.genParticleMatch = ''    


def removeAllPATObjectsBut(process,
                           names,
                           outputInProcess=True
                           ):
    """
    ------------------------------------------------------------------
    remove all PAT objects from the default sequence but a specific
    one:

    process         : process
    name            : list of collection names; supported are
                      'Photons', 'Electrons', 'Muons', 'Taus',
                      'Jets', 'METs'
    outputInProcess : indicate whether there is an output module
                      specified for the process (default is True)            
    ------------------------------------------------------------------    
    """
    removeTheseObjectCollections = ['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs']
    for obj in range(len(names)):
        removeTheseObjectCollections.remove(names[obj])
    removeSpecificPATObjects(process, removeTheseObjectCollections, outputInProcess)


def removeSpecificPATObjects(process,
                             names,
                             outputInProcess=True
                            ):
    """
    ------------------------------------------------------------------
    remove a specific PAT object from the default sequence:

    process         : process
    names           : listr of collection names; supported are
                      'Photons', 'Electrons', 'Muons', 'Taus', 'Jets',
                      'METs'
    outputInProcess : indicate whether there is an output module
                      specified for the process (default is True)
    ------------------------------------------------------------------    
    """
    ## remove pre object production steps from the default sequence
    for obj in range(len(names)):
        if( names[obj] == 'Photons' ):
            process.patDefaultSequence.remove(getattr(process, 'patPhotonIsolation'))
            process.patDefaultSequence.remove(getattr(process, 'photonMatch'))            
        if( names[obj] == 'Electrons' ):
            process.patDefaultSequence.remove(getattr(process, 'patElectronId'))
            process.patDefaultSequence.remove(getattr(process, 'patElectronIsolation'))
            process.patDefaultSequence.remove(getattr(process, 'electronMatch'))        
        if( names[obj] == 'Muons' ):
            process.patDefaultSequence.remove(getattr(process, 'muonMatch'))
        if( names[obj] == 'Taus' ):
            process.patDefaultSequence.remove(getattr(process, 'patPFCandidateIsoDepositSelection'))
            process.patDefaultSequence.remove(getattr(process, 'patPFTauIsolation'))
            process.patDefaultSequence.remove(getattr(process, 'tauMatch'))
            process.patDefaultSequence.remove(getattr(process, 'tauGenJets'))
            process.patDefaultSequence.remove(getattr(process, 'tauGenJetMatch'))
        if( names[obj] == 'Jets' ):
            process.patDefaultSequence.remove(getattr(process, 'patJetCharge'))
            process.patDefaultSequence.remove(getattr(process, 'patJetCorrections'))
            process.patDefaultSequence.remove(getattr(process, 'jetPartonMatch'))
            process.patDefaultSequence.remove(getattr(process, 'jetGenJetMatch'))
            process.patDefaultSequence.remove(getattr(process, 'jetFlavourId'))                
        if( names[obj] == 'METs' ):
            process.patDefaultSequence.remove(getattr(process, 'patMETCorrections'))                
        ## remove cleaning for the moment; in principle only the removed object
        ## could be taken out of the checkOverlaps PSet
        removeCleaning(process, outputInProcess)
        
        ## remove object production steps from the default sequence    
        if( names[obj] == 'METs' ):
            process.allLayer1Objects.remove( getattr(process, 'layer1'+names[obj]) )
        else:
            process.allLayer1Objects.remove( getattr(process, 'allLayer1'+names[obj]) )
            process.selectedLayer1Objects.remove( getattr(process, 'selectedLayer1'+names[obj]) )
            process.countLayer1Objects.remove( getattr(process, 'countLayer1'+names[obj]) )
        ## in the case of leptons, the lepton counter must be modified as well
        if( names[obj] == 'Electrons' ):
            print 'removed from lepton counter: electrons'
            process.countLayer1Leptons.countElectrons = False
        elif( names[obj] == 'Muons' ):
            print 'removed from lepton counter: muons'
            process.countLayer1Leptons.countMuons = False
        elif( names[obj] == 'Taus' ):
            print 'removed from lepton counter: taus'
            process.countLayer1Leptons.countTaus = False
        ## remove from summary
        if( names[obj] == 'METs' ):
            process.allLayer1Summary.candidates.remove( cms.InputTag('layer1'+names[obj]) )
        else:
            process.allLayer1Summary.candidates.remove( cms.InputTag('allLayer1'+names[obj]) )
            process.selectedLayer1Summary.candidates.remove( cms.InputTag('selectedLayer1'+names[obj]) )
            process.cleanLayer1Summary.candidates.remove( cms.InputTag('cleanLayer1'+names[obj]) )
    

def removeCleaning(process, outputInProcess=True):
    """
    ------------------------------------------------------------------
    remove PAT cleaning from the default sequence:

    process         : process
    outputInOricess : indicate whether there is an output module
                      specified for the process (default is True)
    ------------------------------------------------------------------    
    """
    ## adapt single object counters
    for m in listModules(process.countLayer1Objects):
        if hasattr(m, 'src'): m.src = m.src.value().replace('cleanLayer1','selectedLayer1')
    ## adapt lepton counter
    countLept = process.countLayer1Leptons
    countLept.electronSource = countLept.electronSource.value().replace('cleanLayer1','selectedLayer1')
    countLept.muonSource = countLept.muonSource.value().replace('cleanLayer1','selectedLayer1')
    countLept.tauSource = countLept.tauSource.value().replace('cleanLayer1','selectedLayer1')
    process.patDefaultSequence.remove(process.cleanLayer1Objects)
    if ( outputInProcess ):
        ## add selected layer1 objects to the pat output
        from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoLayer1Cleaning
        process.out.outputCommands = patEventContentNoLayer1Cleaning


def addCleaning(process):
    """
    ------------------------------------------------------------------
    add PAT cleaning from the default sequence:

    process : process
    ------------------------------------------------------------------    
    """
    ## adapt single object counters
    process.patDefaultSequence.replace(process.countLayer1Objects, process.cleanLayer1Objects * process.countLayer1Objects)
    for m in listModules(process.countLayer1Objects):
        if hasattr(m, 'src'): m.src = m.src.value().replace('selectedLayer1','cleanLayer1')
    ## adapt lepton counter
    countLept = process.countLayer1Leptons
    countLept.electronSource = countLept.electronSource.value().replace('selectedLayer1','cleanLayer1')
    countLept.muonSource = countLept.muonSource.value().replace('selectedLayer1','cleanLayer1')
    countLept.tauSource = countLept.tauSource.value().replace('selectedLayer1','cleanLayer1')
    ## add clean layer1 objects to the pat output
    from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
    process.out.outputCommands = patEventContent               
