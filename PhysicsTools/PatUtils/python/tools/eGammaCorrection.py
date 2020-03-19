from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.ConfigToolBase import *
#import PhysicsTools.PatAlgos.tools.helpers as configtools

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def eGammaCorrection(process,
                     electronCollection,
                     photonCollection,
                     corElectronCollection,
                     corPhotonCollection,
                     metCollections,
                     pfCandMatching=False,
                     pfCandidateCollection="",
                     #correctAlreadyExistingMET,
                     corMetName="corEGSlimmedMET",
                     postfix=""
                     ):

    task = getPatAlgosToolsTask(process)

    process.load("PhysicsTools.PatAlgos.cleaningLayer1.photonCleaner_cfi")
    task.add(process.cleanPatPhotons)
    #cleaning the bad collections
    cleanedPhotonCollection="cleanedPhotons"+postfix
    cleanPhotonProducer = getattr(process, "cleanPatPhotons").clone( 
                    src = photonCollection,
                    
                    )
    cleanPhotonProducer.checkOverlaps.electrons.src = electronCollection
    cleanPhotonProducer.checkOverlaps.electrons.requireNoOverlaps=cms.bool(True)
    
    #cleaning the good collections
    cleanedCorPhotonCollection="cleanedCorPhotons"+postfix
    cleanCorPhotonProducer = getattr(process, "cleanPatPhotons").clone( 
                    src = corPhotonCollection
                    )
    cleanCorPhotonProducer.checkOverlaps.electrons.src = corElectronCollection
    cleanCorPhotonProducer.checkOverlaps.electrons.requireNoOverlaps=cms.bool(True)


    #matching between objects
    matchPhotonCollection="matchedPhotons"+postfix
    matchPhotonProducer=cms.EDProducer("PFMatchedCandidateRefExtractor",
                                       col1=cms.InputTag(cleanedPhotonCollection),
                                       col2=cms.InputTag(cleanedCorPhotonCollection),
                                       pfCandCollection=cms.InputTag(pfCandidateCollection),
                                       extractPFCandidates=cms.bool(pfCandMatching) )
    
    matchElectronCollection="matchedElectrons"+postfix
    matchElectronProducer=cms.EDProducer("PFMatchedCandidateRefExtractor",
                                         col1=cms.InputTag(electronCollection),
                                         col2=cms.InputTag(corElectronCollection),
                                         pfCandCollection=cms.InputTag(pfCandidateCollection),
                                         extractPFCandidates=cms.bool(pfCandMatching) )
    

    #removal of old objects, and replace by the new 
    tag1= "pfCandCol1" if pfCandMatching else "col1"
    tag2= "pfCandCol2" if pfCandMatching else "col2"
    correctionPhoton="corMETPhoton"+postfix
    corMETPhoton = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                                  srcOriginal = cms.InputTag(matchPhotonCollection,tag1),
                                  srcShifted = cms.InputTag(matchPhotonCollection,tag2),
                                  )
    correctionElectron="corMETElectron"+postfix
    corMETElectron=cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                                  srcOriginal=cms.InputTag(matchElectronCollection,tag1),
                                  srcShifted=cms.InputTag(matchElectronCollection,tag2),
                                  )


    addToProcessAndTask(cleanedPhotonCollection,cleanPhotonProducer, process, task)
    addToProcessAndTask(cleanedCorPhotonCollection,cleanCorPhotonProducer, process, task)
    addToProcessAndTask(matchPhotonCollection,matchPhotonProducer, process, task)
    addToProcessAndTask(matchElectronCollection,matchElectronProducer, process, task)
    addToProcessAndTask(correctionPhoton,corMETPhoton, process, task)
    addToProcessAndTask(correctionElectron,corMETElectron, process, task)

    sequence=cms.Sequence()
    sequence+=getattr(process,cleanedPhotonCollection)
    sequence+=getattr(process,cleanedCorPhotonCollection)
    sequence+=getattr(process,correctionPhoton)
    sequence+=getattr(process,correctionElectron)


    
    #MET corrector
    for metCollection in metCollections:
        #print "---------->>>> ",metCollection, postfix
        if not hasattr(process, metCollection+postfix):
            #print " ==>> aqui"
            #raw met is the only one that does not have already a valid input collection
            inputMetCollection=metCollection.replace("Raw","",1) 
            corMETModuleName=corMetName+postfix
            corMETModule = cms.EDProducer("CorrectedPATMETProducer",
                 src = cms.InputTag( inputMetCollection ),
                 #"patPFMet" if ("Raw" in metCollection )else metCollection
                 srcCorrections = cms.VInputTag( cms.InputTag(correctionPhoton),
                                                 cms.InputTag(correctionElectron),
                                                 )
                                          )
            addToProcessAndTask(metCollection+postfix, corMETModule, process, task) #corMETModuleName
            sequence+=getattr(process,metCollection+postfix) #corMETModuleName
        else:
            print(metCollection)
            getattr(process,metCollection).srcCorrections.append(cms.InputTag(correctionPhoton))
            getattr(process,metCollection).srcCorrections.append(cms.InputTag(correctionElectron))
            
    return sequence
