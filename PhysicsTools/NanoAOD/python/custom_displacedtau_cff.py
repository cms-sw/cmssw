import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.muons_cff import *
from PhysicsTools.NanoAOD.jetsAK4_CHS_cff import *
from PhysicsTools.NanoAOD.jetsAK4_Puppi_cff import *
import os

def add_displacedtauCHSTables(process, isMC):

    process.linkedObjectsCHS = cms.EDProducer("PATObjectCrossLinker",
                                      jets=cms.InputTag("finalJets"),
                                      muons=cms.InputTag("finalMuons"),
                                      electrons=cms.InputTag("finalElectrons"),
                                      lowPtElectrons=cms.InputTag("finalLowPtElectrons"),
                                      taus=cms.InputTag("finalTaus"),
                                      boostedTaus=cms.InputTag("finalBoostedTaus"),
                                      photons=cms.InputTag("finalPhotons"),
                                      vertices=cms.InputTag("slimmedSecondaryVertices")
                                      )
    
    del process.updatedJetsWithUserData.userFloats.leadTrackPt
    del process.updatedJetsWithUserData.userFloats.leptonPtRelv0
    del process.updatedJetsWithUserData.userFloats.leptonPtRelInvv0
    del process.updatedJetsWithUserData.userFloats.leptonDeltaR
    del process.updatedJetsWithUserData.userFloats.vtxPt
    del process.updatedJetsWithUserData.userFloats.vtxMass
    del process.updatedJetsWithUserData.userFloats.vtx3dL
    del process.updatedJetsWithUserData.userFloats.vtx3deL
    del process.updatedJetsWithUserData.userFloats.ptD
    del process.updatedJetsWithUserData.userFloats.qgl
    del process.updatedJetsWithUserData.userFloats.puIdNanoDisc
    del process.updatedJetsWithUserData.userFloats.muonSubtrRawPt
    del process.updatedJetsWithUserData.userFloats.muonSubtrRawEta
    del process.updatedJetsWithUserData.userFloats.muonSubtrRawPhi

    del process.updatedJetsWithUserData.userInts.vtxNtrk
    del process.updatedJetsWithUserData.userInts.leptonPdgId
    del process.updatedJetsWithUserData.userInts.puIdNanoId

    print(process.updatedJetsWithUserData.dumpPython())

    #
    # Customize jetTable
    #
    process.jetTable.src = cms.InputTag("linkedObjectsCHS","jets")
    process.jetTable.name = "JetCHS" # Change collection name from "Jet" ->" JetCHS"

    #
    # Remove these tagger branches since for CHS, we just want to store ParticleNet.
    # Remove also branches related to object linking. It is only done for AK4 Puppi.
    #
    for varName in process.jetTable.variables.parameterNames_():
        if "btagDeepFlav" in varName or "btagRobustParT" in varName or "btagUParT" in varName:
          delattr(process.jetTable.variables, varName)
        if "UParTAK4Reg" in varName:
          delattr(process.jetTable.variables, varName)
        if "svIdx" in varName or "muonIdx" in varName or "electronIdx" in varName:
          delattr(process.jetTable.variables, varName)
        if "nSVs" in varName or "nElectrons" in varName or "nMuons" in varName:
          delattr(process.jetTable.variables, varName)

    del process.jetTable.variables.muonSubtrFactor
    del process.jetTable.variables.muonSubtrDeltaEta
    del process.jetTable.variables.muonSubtrDeltaPhi
    del process.jetTable.variables.qgl
    del process.jetTable.variables.puIdDisc
    del process.jetTable.variables.puId

    del process.jetTable.externalVariables.bRegCorr
    del process.jetTable.externalVariables.bRegRes
    del process.jetTable.externalVariables.cRegCorr
    del process.jetTable.externalVariables.cRegRes

    process.jetUserDataTask = cms.Task(
        process.jercVars,
    )
    process.nanoTableTaskCommon.add(process.jetUserDataTask)




    ## displaced tau part
    file = "RecoTauTag/TrainingFiles/DisplacedTauId/particlenet_v1_a27159734e304ea4b7f9e0042baa9e22.pb"
    process.options = cms.untracked.PSet(
        numberOfThreads = cms.untracked.uint32(4),  # Global thread count
        numberOfStreams = cms.untracked.uint32(4),   # Should match threads
    )
     
    process.disTauTag = cms.EDProducer(
            "DisTauTag",
        graphPath = cms.FileInPath(file),
        jets = cms.InputTag("linkedObjectsCHS","jets"),
        pfCandidates = cms.InputTag('packedPFCandidates'),
        save_inputs  = cms.bool(False),
        batchSize = cms.uint32(8),
        #numThreads = cms.untracked.uint32(4)
        allowUnscheduled = cms.untracked.bool(True)
    )
    
    process.jetImpactParameters = cms.EDProducer(
        "JetImpactParameters",
        jets = cms.InputTag("linkedObjectsCHS","jets"),
        pfCandidates = cms.InputTag('packedPFCandidates'),
        deltaRMax = cms.double(0.4)
    )
    
    
    
    d_disTauTagVars = {
        "disTauTag_score0":     ExtVar("disTauTag:score0"       , float, doc = "Score 0"),
        "disTauTag_score1":     ExtVar("disTauTag:score1"       , float, doc = "Score 1"),
        "dxy": ExtVar("jetImpactParameters:jetDxy", float, doc = "leadingPtPFCand_dxy which is within dR=0.4 and charged/hasTrackDetails"),
        "dz": ExtVar("jetImpactParameters:jetDz", float, doc = "leadingPtPFCand_dz which is within dR=0.4 and charged/hasTrackDetails"),
        "dxyerror": ExtVar("jetImpactParameters:jetDxyError", float, doc = "leadingPtPFCand_dxyerror which is within dR=0.4 and charged/hasTrackDetails"),
        "dzerror": ExtVar("jetImpactParameters:jetDzError", float, doc = "leadingPtPFCand_dzerror which is within dR=0.4 and charged/hasTrackDetails"),
        "charge": ExtVar("jetImpactParameters:jetCharge", float, doc = "leadingPtPFCand_charge which is within dR=0.4 and charged/hasTrackDetails"), 
    }

    #if useCHSJets:
    process.jetTable.externalVariables = process.jetTable.externalVariables.clone(**d_disTauTagVars)
        ## for puppi jets, use this!
    #else:
    #    process.jetPuppiTable.externalVariables = process.jetPuppiTable.externalVariables.clone(**d_disTauTagVars)
      
    process.jetTask = cms.Task(
        process.jetCorrFactorsNano,
        process.updatedJets,
        process.linkedObjectsCHS,
        process.updatedJetsWithUserData,
        process.finalJets,
        process.disTauTag,
        process.jetImpactParameters
        
    )

    process.nanoTableTaskCommon.add(process.jetTask)

    process.jetTablesTask = cms.Task(
        process.jetTable
    )
    process.nanoTableTaskCommon.add(process.jetTablesTask)

    #
    # Only for MC
    #
    if isMC:
        process.jetCHSMCTable = process.jetMCTable.clone(
            src = process.jetTable.src,
            name = process.jetTable.name
        )
        process.jetMCTask.add(process.jetCHSMCTable)

    return process


