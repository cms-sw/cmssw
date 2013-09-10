#!/usr/bin/env python

import os

version = 'tauId_v1_14'

inputFilePath  = "/data2/veelken/CMSSW_5_3_x/Ntuples/tauIdMVATraining/%s/" % version
inputFilePath += "user/veelken/CMSSW_5_3_x/Ntuples/tauIdMVATraining/%s/" % version

outputFilePath = "/data1/veelken/tmp/tauIdMVATraining/%s_2/" % version

preselection_oldDMs = \
    'decayModeFindingOldDMs > 0.5' \
  + ' && numSelectedOfflinePrimaryVertices >= 1 && TMath::Abs(recTauVtxZ - selectedOfflinePrimaryVertexZ) < 0.4 && recJetLooseId > 0.5' \
  + ' && leadPFChargedHadrCandPt > 1.'
preselection_newDMs = \
    'decayModeFindingNewDMs > 0.5' \
  + ' && numSelectedOfflinePrimaryVertices >= 1 && TMath::Abs(recTauVtxZ - selectedOfflinePrimaryVertexZ) < 0.4 && recJetLooseId > 0.5' \
  + ' && leadPFChargedHadrCandPt > 1.'  

mvaDiscriminators = {
##     'mvaIsolation3HitsDeltaR05opt1bI' : {
##         'preselection'        : preselection_newDMs,
##         'applyPtReweighting'  : True,
##         'applyEtaReweighting' : True,
##         'reweight'            : 'min:KILL',
##         'applyEventPruning'   : 0,
##         'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
##         'inputVariables'      : [
##             'TMath::Log(TMath::Max(1., recTauPt))/F',
##             'TMath::Abs(recTauEta)/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.125*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##         ],
##         'spectatorVariables'  : [
##             ##'recTauPt/F',
##             'recTauDecayMode/I',
##             'leadPFChargedHadrCandPt/F',
##             'numOfflinePrimaryVertices/I',
##             'genVisTauPt/F',
##             'genTauPt/F'
##         ],
##         'legendEntry'         : "MVA opt1bI",
##         'color'               : 1,
##     },
##     'mvaIsolation3HitsDeltaR05opt1bII' : {
##        'preselection'        : preselection_newDMs,
##        'applyPtReweighting'  : True,
##        'applyEtaReweighting' : True,
##        'reweight'            : 'min:KILL',
##        'applyEventPruning'   : 0,
##        'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
##        'inputVariables'      : [
##            'TMath::Log(TMath::Max(1., recTauPt))/F',
##            'TMath::Abs(recTauEta)/F',
##            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
##            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))/F',
##            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##        ],
##        'spectatorVariables'  : [
##            ##'recTauPt/F',
##            'recTauDecayMode/I',
##            'leadPFChargedHadrCandPt/F',
##            'numOfflinePrimaryVertices/I',
##            'genVisTauPt/F',
##            'genTauPt/F'
##        ],
##        'legendEntry'         : "MVA opt1bII",
##        'color'               : 2,
##     },
##     'mvaIsolation3HitsDeltaR05opt1c' : {
##        'preselection'        : preselection_oldDMs,
##        'applyPtReweighting'  : True,
##        'applyEtaReweighting' : True,
##        'reweight'            : 'min:KILL',
##        'applyEventPruning'   : 0,
##        'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
##        'inputVariables'      : [
##            'TMath::Abs(recTauEta)/F',
##            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
##            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.125*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##        ],
##        'spectatorVariables'  : [
##            'recTauPt/F',
##            'recTauDecayMode/I',
##            'leadPFChargedHadrCandPt/F',
##            'numOfflinePrimaryVertices/I',
##            'genVisTauPt/F',
##            'genTauPt/F'
##        ],
##        'legendEntry'         : "MVA opt1c",
##        'color'               : 3,
##     },    
    'mvaIsolation3HitsDeltaR05opt2a' : {
        'preselection'        : preselection_oldDMs,
        'applyPtReweighting'  : True,
        'applyEtaReweighting' : True,
        'reweight'            : 'min:KILL',
        'applyEventPruning'   : 2,
        'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
        'inputVariables'      : [
            'TMath::Log(TMath::Max(1., recTauPt))/F',
            'TMath::Abs(recTauEta)/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
            'recTauDecayMode/I'
        ],
        'spectatorVariables'  : [
            ##'recTauPt/F',
            'leadPFChargedHadrCandPt/F',
            'numOfflinePrimaryVertices/I',
            'genVisTauPt/F',
            'genTauPt/F'
        ],
        'legendEntry'         : "MVA opt2a",
        'color'               : 1
    },
    'mvaIsolation3HitsDeltaR05opt2aLT' : {
        'preselection'        : preselection_oldDMs,
        'applyPtReweighting'  : True,
        'applyEtaReweighting' : True,
        'reweight'            : 'min:KILL',
        'applyEventPruning'   : 2,
        'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
        'inputVariables'      : [
            'TMath::Log(TMath::Max(1., recTauPt))/F',
            'TMath::Abs(recTauEta)/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
            'recTauDecayMode/I',
            'TMath::Sign(+1., recImpactParam)/F',
            'TMath::Sqrt(TMath::Abs(TMath::Min(1., recImpactParam)))/F',
            'TMath::Min(10., TMath::Abs(recImpactParamSign))/F',
            'hasRecDecayVertex/I',
            'TMath::Sqrt(recDecayDistMag)/F',
            'TMath::Min(10., recDecayDistSign)/F'
        ],
        'spectatorVariables'  : [
            ##'recTauPt/F',
            'leadPFChargedHadrCandPt/F',
            'numOfflinePrimaryVertices/I',
            'genVisTauPt/F',
            'genTauPt/F'
        ],
        'legendEntry'         : "MVA opt2aLT",
        'color'               : 2
    },
    'mvaIsolation3HitsDeltaR05opt2b' : {
        'preselection'        : preselection_newDMs,
        'applyPtReweighting'  : True,
        'applyEtaReweighting' : True,
        'reweight'            : 'min:KILL',
        'applyEventPruning'   : 2,
        'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
        'inputVariables'      : [
            'TMath::Log(TMath::Max(1., recTauPt))/F',
            'TMath::Abs(recTauEta)/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
            'recTauDecayMode/I'
        ],
        'spectatorVariables'  : [
            ##'recTauPt/F',
            'leadPFChargedHadrCandPt/F',
            'numOfflinePrimaryVertices/I',
            'genVisTauPt/F',
            'genTauPt/F'
        ],
        'legendEntry'         : "MVA opt2b",
        'color'               : 6
    },
    'mvaIsolation3HitsDeltaR05opt2bLT' : {
        'preselection'        : preselection_newDMs,
        'applyPtReweighting'  : True,
        'applyEtaReweighting' : True,
        'reweight'            : 'min:KILL',
        'applyEventPruning'   : 2,
        'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
        'inputVariables'      : [
            'TMath::Log(TMath::Max(1., recTauPt))/F',
            'TMath::Abs(recTauEta)/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))/F',
            'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
            'recTauDecayMode/I',
            'TMath::Sign(+1., recImpactParam)/F',
            'TMath::Sqrt(TMath::Abs(TMath::Min(1., recImpactParam)))/F',
            'TMath::Min(10., TMath::Abs(recImpactParamSign))/F',
            'hasRecDecayVertex/I',
            'TMath::Sqrt(recDecayDistMag)/F',
            'TMath::Min(10., recDecayDistSign)/F'
        ],
        'spectatorVariables'  : [
            ##'recTauPt/F',
            'leadPFChargedHadrCandPt/F',
            'numOfflinePrimaryVertices/I',
            'genVisTauPt/F',
            'genTauPt/F'
        ],
        'legendEntry'         : "MVA opt2bLT",
        'color'               : 4
    },
##     'mvaIsolation3HitsDeltaR05opt5a' : {
##         'preselection'        : preselection_oldDMs,
##         'applyPtReweighting'  : True,
##         'applyEtaReweighting' : True,
##         'reweight'            : 'min:KILL',
##         'applyEventPruning'   : 0,
##         'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
##         'inputVariables'      : [
##             'TMath::Log(TMath::Max(1., recTauPt))/F',
##             'TMath::Abs(recTauEta)/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.125*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'recTauDecayMode/I',
##             'recTauM/F',
##             'chargedHadron1Pt/recTauPt/F',
##             'chargedHadron2Pt/recTauPt/F',
##             'chargedHadron3Pt/recTauPt/F',
##             'piZero1Pt/recTauPt/F',
##             'TMath::Min(0.2, piZero1M)/F'
##         ],
##         'spectatorVariables'  : [
##             ##'recTauPt/F',
##             'leadPFChargedHadrCandPt/F',
##             'numOfflinePrimaryVertices/I',
##             'genVisTauPt/F',
##             'genTauPt/F'
##         ],
##         'legendEntry'         : "MVA opt5a",
##         'color'               : 1
##     },
##     'mvaIsolation3HitsDeltaR05opt5aLT' : {
##         'preselection'        : preselection_oldDMs,
##         'applyPtReweighting'  : True,
##         'applyEtaReweighting' : True,
##         'reweight'            : 'min:KILL',
##         'applyEventPruning'   : 0,
##         'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
##         'inputVariables'      : [
##             'TMath::Log(TMath::Max(1., recTauPt))/F',
##             'TMath::Abs(recTauEta)/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.125*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'recTauDecayMode/I',
##             'recTauM/F',
##             'chargedHadron1Pt/recTauPt/F',
##             'chargedHadron2Pt/recTauPt/F',
##             'chargedHadron3Pt/recTauPt/F',
##             'piZero1Pt/recTauPt/F',
##             'TMath::Min(0.2, piZero1M)/F',
##             'TMath::Sign(+1., recImpactParam)/F',
##             'TMath::Sqrt(TMath::Abs(TMath::Min(1., recImpactParam)))/F',
##             'TMath::Min(10., TMath::Abs(recImpactParamSign))/F',
##             'hasRecDecayVertex/I',
##             'TMath::Sqrt(recDecayDistMag)/F',
##             'TMath::Min(10., recDecayDistSign)/F'
##         ],
##         'spectatorVariables'  : [
##             ##'recTauPt/F',
##             'leadPFChargedHadrCandPt/F',
##             'numOfflinePrimaryVertices/I',
##             'genVisTauPt/F',
##             'genTauPt/F'
##         ],
##         'legendEntry'         : "MVA opt5aLT",
##         'color'               : 2
##     },
##     'mvaIsolation3HitsDeltaR05opt5b' : {
##         'preselection'        : preselection_newDMs,
##         'applyPtReweighting'  : True,
##         'applyEtaReweighting' : True,
##         'reweight'            : 'min:KILL',
##         'applyEventPruning'   : 0,
##         'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
##         'inputVariables'      : [
##             'TMath::Log(TMath::Max(1., recTauPt))/F',
##             'TMath::Abs(recTauEta)/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.125*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'recTauDecayMode/I',
##             'recTauM/F',
##             'chargedHadron1Pt/recTauPt/F',
##             'chargedHadron2Pt/recTauPt/F',
##             'chargedHadron3Pt/recTauPt/F',
##             'piZero1Pt/recTauPt/F',
##             'TMath::Min(0.2, piZero1M)/F'
##         ],
##         'spectatorVariables'  : [
##             ##'recTauPt/F',            
##             'leadPFChargedHadrCandPt/F',
##             'numOfflinePrimaryVertices/I',
##             'genVisTauPt/F',
##             'genTauPt/F'
##         ],
##         'legendEntry'         : "MVA opt5b",
##         'color'               : 6
##     },
##     'mvaIsolation3HitsDeltaR05opt5bLT' : {
##         'preselection'        : preselection_newDMs,
##         'applyPtReweighting'  : True,
##         'applyEtaReweighting' : True,
##         'reweight'            : 'min:KILL',
##         'applyEventPruning'   : 0,
##         'mvaTrainingOptions'  : "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
##         'inputVariables'      : [
##             'TMath::Log(TMath::Max(1., recTauPt))/F',
##             'TMath::Abs(recTauEta)/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.125*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))/F',
##             'recTauDecayMode/I',
##             'recTauM/F',
##             'chargedHadron1Pt/recTauPt/F',
##             'chargedHadron2Pt/recTauPt/F',
##             'chargedHadron3Pt/recTauPt/F',
##             'piZero1Pt/recTauPt/F',
##             'TMath::Min(0.2, piZero1M)/F',
##             'TMath::Sign(+1., recImpactParam)/F',
##             'TMath::Sqrt(TMath::Abs(TMath::Min(1., recImpactParam)))/F',
##             'TMath::Min(10., TMath::Abs(recImpactParamSign))/F',
##             'hasRecDecayVertex/I',
##             'TMath::Sqrt(recDecayDistMag)/F',
##             'TMath::Min(10., recDecayDistSign)/F'
##         ],
##         'spectatorVariables'  : [
##             ##'recTauPt/F',            
##             'leadPFChargedHadrCandPt/F',
##             'numOfflinePrimaryVertices/I',
##             'genVisTauPt/F',
##             'genTauPt/F'
##         ],
##         'legendEntry'         : "MVA opt5bLT",
##         'color'               : 7
##     }
}

cutDiscriminators = {
##     'hpsCombinedIsolation3HitsDeltaR04DB015' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsChargedIsoPtSum'
##                                 '                            + TMath::Max(0., tauIsoDeltaR04PtThresholdsLoose3HitsNeutralIsoPtSum - 0.15*tauIsoDeltaR04PtThresholdsLoose3HitsPUcorrPtSum)))',
##         'numBins'             : 120,
##         'min'                 : -5.,
##         'max'                 : +7.,
##         'legendEntry'         : "HPS 3hit: #DeltaR = 0.4, #Delta#beta = 0.15",
##         'color'               : 6
##     },
##     'hpsCombinedIsolation3HitsDeltaR04DB030' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsChargedIsoPtSum'
##                                 '                            + TMath::Max(0., tauIsoDeltaR04PtThresholdsLoose3HitsNeutralIsoPtSum - 0.30*tauIsoDeltaR04PtThresholdsLoose3HitsPUcorrPtSum)))',
##         'numBins'             : 120,
##         'min'                 : -5.,
##         'max'                 : +7.,
##         'legendEntry'         : "HPS 3hit: #DeltaR = 0.4, #Delta#beta = 0.30",
##         'color'               : 8
##     },
##     'hpsCombinedIsolation3HitsDeltaR04DB045' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsChargedIsoPtSum'
##                                 '                            + TMath::Max(0., tauIsoDeltaR04PtThresholdsLoose3HitsNeutralIsoPtSum - 0.4576*tauIsoDeltaR04PtThresholdsLoose3HitsPUcorrPtSum)))',
##         'numBins'             : 120,
##         'min'                 : -5.,
##         'max'                 : +7.,
##         'legendEntry'         : "HPS 3hit: #DeltaR = 0.4, #Delta#beta = 0.45",
##         'color'               : 1
##     },
##     'hpsCombinedIsolation3HitsDeltaR05DB015' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum'
##                                 '                            + TMath::Max(0., tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.15*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum)))',
##         'numBins'             : 120,
##         'min'                 : -5.,
##         'max'                 : +7.,
##         'legendEntry'         : "HPS 3hit: #DeltaR = 0.5, #Delta#beta = 0.15",
##         'color'               : 2
##     },
##     'hpsCombinedIsolation3HitsDeltaR05DB030' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum'
##                                 '                            + TMath::Max(0., tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.30*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum)))',
##         'numBins'             : 120,
##         'min'                 : -5.,
##         'max'                 : +7.,
##         'legendEntry'         : "HPS 3hit: #DeltaR = 0.5, #Delta#beta = 0.30",
##         'color'               : 4
##     },
    'hpsCombinedIsolation3HitsDeltaR05DB045' : {
        'preselection'        : preselection_oldDMs,
        'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum'
                                '                            + TMath::Max(0., tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum - 0.4576*tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum)))',
        'numBins'             : 120,
        'min'                 : -5.,
        'max'                 : +7.,
        'legendEntry'         : "HPS 3hit: #DeltaR = 0.5, #Delta#beta = 0.45",
        'color'               : 8
    },
    'hpsCombinedIsolation3HitsLoose' : {
        'preselection'        : preselection_oldDMs,
        'discriminator'       : 'byLooseCombinedIsolationDeltaBetaCorr3Hits',
        'numBins'             : 2,
        'min'                 : -0.5,
        'max'                 : +1.5,
        'legendEntry'         : "",
        'color'               : 8,
        'markerStyle'         : 20
    },
    'hpsCombinedIsolation3HitsMedium' : {
        'preselection'        : preselection_oldDMs,
        'discriminator'       : 'byMediumCombinedIsolationDeltaBetaCorr3Hits',
        'numBins'             : 2,
        'min'                 : -0.5,
        'max'                 : +1.5,
        'legendEntry'         : "",
        'color'               : 8,
        'markerStyle'         : 21
    },
    'hpsCombinedIsolation3HitsTight' : {
        'preselection'        : preselection_oldDMs,
        'discriminator'       : 'byTightCombinedIsolationDeltaBetaCorr3Hits',
        'numBins'             : 2,
        'min'                 : -0.5,
        'max'                 : +1.5,
        'legendEntry'         : "",
        'color'               : 8,
        'markerStyle'         : 33,
        'markerSize'          : 2
    },        
##     'hpsCombinedIsolation8HitsDeltaR05DB045' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose8HitsChargedIsoPtSum'
##                                 '                            + TMath::Max(0., tauIsoDeltaR05PtThresholdsLoose8HitsNeutralIsoPtSum - 0.4576*tauIsoDeltaR05PtThresholdsLoose8HitsPUcorrPtSum)))',
##         'numBins'             : 120,
##         'min'                 : -5.,
##         'max'                 : +7.,
##         'legendEntry'         : "HPS 8hit: #DeltaR = 0.5, #Delta#beta = 0.45",
##         'color'               : 6
##     },
##     'hpsCombinedIsolation8HitsLoose' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byLooseCombinedIsolationDeltaBetaCorr8Hits',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 6,
##         'markerStyle'         : 24
##     },
##     'hpsCombinedIsolation8HitsMedium' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byMediumCombinedIsolationDeltaBetaCorr8Hits',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 6,
##         'markerStyle'         : 25
##     },
##     'hpsCombinedIsolation8HitsTight' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byTightCombinedIsolationDeltaBetaCorr8Hits',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 6,
##         'markerStyle'         : 27
##     },
##     'hpsChargedIsolation3HitsDeltaR05' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))',
##         'numBins'             : 120,
##         'min'                 : -5.,
##         'max'                 : +7.,
##         'legendEntry'         : "Charged 3hit: #DeltaR = 0.5",
##         'color'               : 3
##     },
##     'rawMVA3oldDMwoLT'  : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byIsolationMVA3oldDMwoLTraw',
##         'numBins'             : 2020,
##         'min'                 : -1.01,
##         'max'                 : +1.01,
##         'legendEntry'         : "MVA3newDMwoLT",
##         'color'               : 1
##     },
##     'wpVLooseMVA3oldDMwoLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVLooseIsolationMVA3oldDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 20
##     },
##     'wpLooseMVA3oldDMwoLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byLooseIsolationMVA3oldDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 21
##     },
##     'wpMediumMVA3oldDMwoLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byMediumIsolationMVA3oldDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },
##     'wpTightMVA3oldDMwoLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byTightIsolationMVA3oldDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 22
##     },
##     'wpVTightMVA3oldDMwoLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVTightIsolationMVA3oldDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 23
##     },
##     'wpVVTightMVA3oldDMwoLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVVTightIsolationMVA3oldDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },    
##     'rawMVA3oldDMwLT'  : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byIsolationMVA3oldDMwLTraw',
##         'numBins'             : 2020,
##         'min'                 : -1.01,
##         'max'                 : +1.01,
##         'legendEntry'         : "MVA3newDMwoLT",
##         'color'               : 2
##     },
##     'wpVLooseMVA3oldDMwLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVLooseIsolationMVA3oldDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 20
##     },
##     'wpLooseMVA3oldDMwLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byLooseIsolationMVA3oldDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 21
##     },
##     'wpMediumMVA3oldDMwLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byMediumIsolationMVA3oldDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },
##     'wpTightMVA3oldDMwLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byTightIsolationMVA3oldDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 22
##     },
##     'wpVTightMVA3oldDMwLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVTightIsolationMVA3oldDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 23
##     },
##     'wpVVTightMVA3oldDMwLT' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVVTightIsolationMVA3oldDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },    
##     'rawMVA3newDMwoLT'  : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byIsolationMVA3newDMwoLTraw',
##         'numBins'             : 2020,
##         'min'                 : -1.01,
##         'max'                 : +1.01,
##         'legendEntry'         : "MVA3newDMwoLT",
##         'color'               : 3
##     },
##     'wpVLooseMVA3newDMwoLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byVLooseIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 3,
##         'markerStyle'         : 20
##     },
##     'wpLooseMVA3newDMwoLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byLooseIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 3,
##         'markerStyle'         : 21
##     },
##     'wpMediumMVA3newDMwoLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byMediumIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 3,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },
##     'wpTightMVA3newDMwoLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byTightIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 3,
##         'markerStyle'         : 22
##     },
##     'wpVTightMVA3newDMwoLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byVTightIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 3,
##         'markerStyle'         : 23
##     },
##     'wpVVTightMVA3newDMwoLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byVVTightIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 3,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },    
##     'rawMVA3newDMwLT'  : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byIsolationMVA3newDMwLTraw',
##         'numBins'             : 2020,
##         'min'                 : -1.01,
##         'max'                 : +1.01,
##         'legendEntry'         : "MVA3newDMwoLT",
##         'color'               : 4
##     },
##     'wpVLooseMVA3newDMwLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byVLooseIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 4,
##         'markerStyle'         : 20
##     },
##     'wpLooseMVA3newDMwLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byLooseIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 4,
##         'markerStyle'         : 21
##     },
##     'wpMediumMVA3newDMwLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byMediumIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 4,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },
##     'wpTightMVA3newDMwLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byTightIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 4,
##         'markerStyle'         : 22
##     },
##     'wpVTightMVA3newDMwLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byVTightIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 4,
##         'markerStyle'         : 23
##     },
##     'wpVVTightMVA3newDMwLT' : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byVVTightIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 4,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },    
##     'rawMVA3newDMwoLToldDMcutAdded'  : {
##         'preselection'        : preselection_newDMs,
##         'discriminator'       : 'byIsolationMVA3newDMwoLTraw',
##         'numBins'             : 2020,
##         'min'                 : -1.01,
##         'max'                 : +1.01,
##         'legendEntry'         : "MVA3newDMwoLT",
##         'color'               : 1
##     },
##     'wpVLooseMVA3newDMwoLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVLooseIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 20
##     },
##     'wpLooseMVA3newDMwoLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byLooseIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 21
##     },
##     'wpMediumMVA3newDMwoLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byMediumIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },
##     'wpTightMVA3newDMwoLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byTightIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 22
##     },
##     'wpVTightMVA3newDMwoLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVTightIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 23
##     },
##     'wpVVTightMVA3newDMwoLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVVTightIsolationMVA3newDMwoLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 1,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },    
##     'rawMVA3newDMwLToldDMcutAdded'  : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byIsolationMVA3newDMwLTraw',
##         'numBins'             : 2020,
##         'min'                 : -1.01,
##         'max'                 : +1.01,
##         'legendEntry'         : "MVA3newDMwoLT",
##         'color'               : 2
##     },
##     'wpVLooseMVA3newDMwLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVLooseIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 20
##     },
##     'wpLooseMVA3newDMwLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byLooseIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 21
##     },
##     'wpMediumMVA3newDMwLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byMediumIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     },
##     'wpTightMVA3newDMwLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byTightIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 22
##     },
##     'wpVTightMVA3newDMwLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVTightIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 23
##     },
##     'wpVVTightMVA3newDMwLToldDMcutAdded' : {
##         'preselection'        : preselection_oldDMs,
##         'discriminator'       : 'byVVTightIsolationMVA3newDMwLT',
##         'numBins'             : 2,
##         'min'                 : -0.5,
##         'max'                 : +1.5,
##         'legendEntry'         : "",
##         'color'               : 2,
##         'markerStyle'         : 33,
##         'markerSize'          : 2
##     }
}

plots = {
##     'hpsCombinedIsolation' : {
##         'graphs' : [            
##             'hpsCombinedIsolation3HitsDeltaR04DB045',
##             'hpsCombinedIsolation3HitsDeltaR05DB015',
##             'hpsCombinedIsolation3HitsDeltaR05DB030',
##             'hpsCombinedIsolation3HitsDeltaR05DB045',
##             'hpsCombinedIsolation3HitsLoose',
##             'hpsCombinedIsolation3HitsMedium',
##             'hpsCombinedIsolation3HitsTight',
##             'hpsCombinedIsolation8HitsDeltaR05DB045',
##             'hpsCombinedIsolation8HitsLoose',
##             'hpsCombinedIsolation8HitsMedium',
##             'hpsCombinedIsolation8HitsTight',
##             'hpsChargedIsolation3HitsDeltaR05'
##         ]
##     },
##     'mvaIsolation' : {
##         'graphs' : [
##             'mvaIsolation3HitsDeltaR05opt2a',
##             'mvaIsolation3HitsDeltaR05opt2aLT',
##             'mvaIsolation3HitsDeltaR05opt2b',
##             'mvaIsolation3HitsDeltaR05opt2bLT',
##             'mvaIsolation3HitsDeltaR05opt5b',
##             'mvaIsolation3HitsDeltaR05opt5bLT',
##             'hpsCombinedIsolation3HitsDeltaR05DB045',
##             'hpsCombinedIsolation3HitsLoose',
##             'hpsCombinedIsolation3HitsMedium',
##             'hpsCombinedIsolation3HitsTight'
##         ]
##     },
##     'mvaIsolation_opt1' : {
##         'graphs' : [
##             'mvaIsolation3HitsDeltaR05opt1bI',
##             'mvaIsolation3HitsDeltaR05opt1bII',
##             'mvaIsolation3HitsDeltaR05opt1c',
##             'hpsCombinedIsolation3HitsDeltaR05DB045',
##             'hpsCombinedIsolation3HitsLoose',
##             'hpsCombinedIsolation3HitsMedium',
##             'hpsCombinedIsolation3HitsTight'
##         ]
##     },
    'mvaIsolation_opt2' : {
        'graphs' : [
            'mvaIsolation3HitsDeltaR05opt2a',
            'mvaIsolation3HitsDeltaR05opt2aLT',
            'mvaIsolation3HitsDeltaR05opt2b',
            'mvaIsolation3HitsDeltaR05opt2bLT',
            'hpsCombinedIsolation3HitsDeltaR05DB045',
            'hpsCombinedIsolation3HitsLoose',
            'hpsCombinedIsolation3HitsMedium',
            'hpsCombinedIsolation3HitsTight'
        ]
    },
##     'mvaIsolation_opt5' : {
##         'graphs' : [
##             'mvaIsolation3HitsDeltaR05opt5a',
##             'mvaIsolation3HitsDeltaR05opt5aLT',
##             'mvaIsolation3HitsDeltaR05opt5b',
##             'mvaIsolation3HitsDeltaR05opt5bLT',
##             'hpsCombinedIsolation3HitsDeltaR05DB045',
##             'hpsCombinedIsolation3HitsLoose',
##             'hpsCombinedIsolation3HitsMedium',
##             'hpsCombinedIsolation3HitsTight'
##         ]
##     }
##     'mvaIsolation_cmssw' : {
##         'graphs' : [
##             'rawMVA3oldDMwoLT',
##             'wpVLooseMVA3oldDMwoLT',
##             'wpLooseMVA3oldDMwoLT',
##             'wpMediumMVA3oldDMwoLT',
##             'wpTightMVA3oldDMwoLT',
##             'wpVTightMVA3oldDMwoLT',
##             'wpVVTightMVA3oldDMwoLT',            
##             'rawMVA3oldDMwLT',
##             'wpVLooseMVA3oldDMwLT',
##             'wpLooseMVA3oldDMwLT',
##             'wpTightMVA3oldDMwLT',
##             'wpVTightMVA3oldDMwLT',
##             'wpVVTightMVA3oldDMwLT',
##             'rawMVA3newDMwoLT',
##             'wpVLooseMVA3newDMwoLT',
##             'wpLooseMVA3newDMwoLT',
##             'wpMediumMVA3newDMwoLT',
##             'wpTightMVA3newDMwoLT',
##             'wpVTightMVA3newDMwoLT',
##             'wpVVTightMVA3newDMwoLT',
##             'rawMVA3newDMwLT',
##             'wpVLooseMVA3newDMwLT',
##             'wpLooseMVA3newDMwLT',
##             'wpMediumMVA3newDMwLT',
##             'wpTightMVA3newDMwLT',
##             'wpVTightMVA3newDMwLT',
##             'wpVVTightMVA3newDMwLT',
##             'hpsCombinedIsolation3HitsDeltaR05DB045',
##             'hpsCombinedIsolation3HitsLoose',
##             'hpsCombinedIsolation3HitsMedium',
##             'hpsCombinedIsolation3HitsTight'
##         ]
##     },
##     'mvaIsolation_cmssw_newDMcutAdded' : {
##         'graphs' : [
##             'rawMVA3oldDMwoLT',
##             'wpVLooseMVA3oldDMwoLT',
##             'wpLooseMVA3oldDMwoLT',
##             'wpMediumMVA3oldDMwoLT',
##             'wpTightMVA3oldDMwoLT',
##             'wpVTightMVA3oldDMwoLT',
##             'wpVVTightMVA3oldDMwoLT',
##             'rawMVA3oldDMwLT',
##             'wpVLooseMVA3oldDMwLT',
##             'wpLooseMVA3oldDMwLT',
##             'wpTightMVA3oldDMwLT',
##             'wpVTightMVA3oldDMwLT',
##             'wpVVTightMVA3oldDMwLT',
##             'rawMVA3newDMwoLToldDMcutAdded',
##             'wpVLooseMVA3newDMwoLToldDMcutAdded',
##             'wpLooseMVA3newDMwoLToldDMcutAdded',
##             'wpMediumMVA3newDMwoLToldDMcutAdded',
##             'wpTightMVA3newDMwoLToldDMcutAdded',
##             'wpVTightMVA3newDMwoLToldDMcutAdded',
##             'wpVVTightMVA3newDMwoLToldDMcutAdded',
##             'rawMVA3newDMwLToldDMcutAdded',
##             'wpVLooseMVA3newDMwLToldDMcutAdded',
##             'wpLooseMVA3newDMwLToldDMcutAdded',
##             'wpMediumMVA3newDMwLToldDMcutAdded',
##             'wpTightMVA3newDMwLToldDMcutAdded',
##             'wpVTightMVA3newDMwLToldDMcutAdded',
##             'wpVVTightMVA3newDMwLToldDMcutAdded',
##             'hpsCombinedIsolation3HitsDeltaR05DB045',
##             'hpsCombinedIsolation3HitsLoose',
##             'hpsCombinedIsolation3HitsMedium',
##             'hpsCombinedIsolation3HitsTight'
##         ]
##     }
}

allDiscriminators = {}
allDiscriminators.update(mvaDiscriminators)
allDiscriminators.update(cutDiscriminators)

signalSamples = [
    "ZplusJets_madgraph"
]
smHiggsMassPoints = [ 80, 90, 100, 110, 120, 130, 140 ]
for massPoint in smHiggsMassPoints:
    ggSampleName = "ggHiggs%1.0ftoTauTau" % massPoint
    signalSamples.append(ggSampleName)
    vbfSampleName = "vbfHiggs%1.0ftoTauTau" % massPoint
    signalSamples.append(vbfSampleName)
mssmHiggsMassPoints = [ 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000 ]
for massPoint in mssmHiggsMassPoints:
    ggSampleName = "ggA%1.0ftoTauTau" % massPoint
    signalSamples.append(ggSampleName)
    bbSampleName = "bbA%1.0ftoTauTau" % massPoint
    signalSamples.append(bbSampleName)
ZprimeMassPoints = [ 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500 ]
for massPoint in ZprimeMassPoints:
    sampleName = "Zprime%1.0ftoTauTau" % massPoint
    signalSamples.append(sampleName)
WprimeMassPoints = [ 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3500, 4000 ]
for massPoint in WprimeMassPoints:
    sampleName = "Wprime%1.0ftoTauNu" % massPoint
    signalSamples.append(sampleName)

backgroundSamples = [
    "PPmuXptGt20Mu15",
    "QCDmuEnrichedPt50to80",
    "QCDmuEnrichedPt80to120",
    "QCDmuEnrichedPt120to170",
    "QCDmuEnrichedPt170to300",
    "QCDmuEnrichedPt300to470",
    "QCDmuEnrichedPt470to600",
    "QCDmuEnrichedPt600to800",
    "QCDmuEnrichedPt800to1000",
    "QCDmuEnrichedPtGt1000",
    "WplusJets_madgraph",    
    "QCDjetsFlatPt15to3000",
    "QCDjetsPt50to80",
    "QCDjetsPt80to120",
    "QCDjetsPt120to170",
    "QCDjetsPt170to300",
    "QCDjetsPt300to470",
    "QCDjetsPt470to600",
    "QCDjetsPt600to800",
    "QCDjetsPt800to1000",
    "QCDjetsPt1000to1400",
    "QCDjetsPt1400to1800",
    "QCDmuEnrichedPtGt1800"
]

execDir = "%s/bin/%s/" % (os.environ['CMSSW_BASE'], os.environ['SCRAM_ARCH'])

executable_preselectTreeTauIdMVA = execDir + 'preselectTreeTauIdMVA'
executable_reweightTreeTauIdMVA  = execDir + 'reweightTreeTauIdMVA'
executable_trainTauIdMVA         = execDir + 'trainTauIdMVA'
executable_makeROCcurveTauIdMVA  = execDir + 'makeROCcurveTauIdMVA'
executable_showROCcurvesTauIdMVA = execDir + 'showROCcurvesTauIdMVA'
executable_hadd                  = 'hadd'
executable_rm                    = 'rm -f'

nice = 'nice '

configFile_preselectTreeTauIdMVA = 'preselectTreeTauIdMVA_cfg.py'
configFile_reweightTreeTauIdMVA  = 'reweightTreeTauIdMVA_cfg.py'
configFile_trainTauIdMVA         = 'trainTauIdMVA_cfg.py'
configFile_makeROCcurveTauIdMVA  = 'makeROCcurveTauIdMVA_cfg.py'
configFile_showROCcurvesTauIdMVA = 'showROCcurvesTauIdMVA_cfg.py'

def getInputFileNames(inputFilePath, samples):
    inputFileNames = []
    for sample in samples:
        try:
            inputFileNames_sample = [ os.path.join(inputFilePath, sample, file) for file in os.listdir(os.path.join(inputFilePath, sample)) ]
            print "sample = %s: #inputFiles = %i" % (sample, len(inputFileNames_sample))
            inputFileNames.extend(inputFileNames_sample)
        except OSError:
            print "inputFilePath = %s does not exist --> skipping !!" % os.path.join(inputFilePath, sample)
            continue
    return inputFileNames

inputFileNames_signal     = getInputFileNames(inputFilePath, signalSamples)
if not len(inputFileNames_signal) > 0:
    raise ValueError("Failed to find signal samples !!")
inputFileNames_background = getInputFileNames(inputFilePath, backgroundSamples)
if not len(inputFileNames_background) > 0:
    raise ValueError("Failed to find background samples !!")

inputFileNames = []
inputFileNames.extend(inputFileNames_signal)
inputFileNames.extend(inputFileNames_background)

# create outputFilePath in case it does not yet exist
def createFilePath_recursively(filePath):
    filePath_items = filePath.split('/')
    currentFilePath = "/"
    for filePath_item in filePath_items:
        currentFilePath = os.path.join(currentFilePath, filePath_item)
        if len(currentFilePath) <= 1:
            continue
        if not os.path.exists(currentFilePath):
            os.mkdir(currentFilePath)

if not os.path.isdir(outputFilePath):
    print "outputFilePath does not yet exist, creating it."
    createFilePath_recursively(outputFilePath)

def getStringRep_bool(flag):
    retVal = None
    if flag:
        retVal = "True"
    else:
        retVal = "False"
    return retVal

print "Info: building config files for MVA training"
preselectTreeTauIdMVA_configFileNames     = {} # key = discriminator, "signal" or "background"
preselectTreeTauIdMVA_outputFileNames     = {} # key = discriminator, "signal" or "background"
preselectTreeTauIdMVA_logFileNames        = {} # key = discriminator, "signal" or "background"
reweightTreeTauIdMVA_configFileNames      = {} # key = discriminator, "signal" or "background"
reweightTreeTauIdMVA_outputFileNames      = {} # key = discriminator, "signal" or "background"
reweightTreeTauIdMVA_logFileNames         = {} # key = discriminator, "signal" or "background"
trainTauIdMVA_configFileNames             = {} # key = discriminator
trainTauIdMVA_outputFileNames             = {} # key = discriminator
trainTauIdMVA_logFileNames                = {} # key = discriminator
for discriminator in mvaDiscriminators.keys():

    print "processing discriminator = %s" % discriminator

    #----------------------------------------------------------------------------
    # build config file for preselecting training trees for signal and background
    preselectTreeTauIdMVA_configFileNames[discriminator] = {}
    preselectTreeTauIdMVA_outputFileNames[discriminator] = {}
    preselectTreeTauIdMVA_logFileNames[discriminator]    = {}
    for sample in [ "signal", "background" ]:
        outputFileName = os.path.join(outputFilePath, "preselectTreeTauIdMVA_%s_%s.root" % (discriminator, sample))
        print " outputFileName = '%s'" % outputFileName
        preselectTreeTauIdMVA_outputFileNames[discriminator][sample] = outputFileName

        cfgFileName_original = configFile_preselectTreeTauIdMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n"
        for inputFileName in inputFileNames:
            cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % inputFileName
        cfg_modified += "\n"
        eventPruningLevel = None
        if sample == 'signal':
            cfg_modified += "process.preselectTreeTauIdMVA.samples = cms.vstring(%s)\n" % signalSamples
            eventPruningLevel = 0
        else:
            cfg_modified += "process.preselectTreeTauIdMVA.samples = cms.vstring(%s)\n" % backgroundSamples
            eventPruningLevel = mvaDiscriminators[discriminator]['applyEventPruning']
        cfg_modified += "process.preselectTreeTauIdMVA.preselection = cms.string('%s')\n" % mvaDiscriminators[discriminator]['preselection']
        cfg_modified += "process.preselectTreeTauIdMVA.applyEventPruning = cms.int32(%i)\n" % eventPruningLevel
        cfg_modified += "process.preselectTreeTauIdMVA.inputVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['inputVariables']
        cfg_modified += "process.preselectTreeTauIdMVA.spectatorVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['spectatorVariables']
        cfg_modified += "process.preselectTreeTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, sample)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        preselectTreeTauIdMVA_configFileNames[discriminator][sample] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        preselectTreeTauIdMVA_logFileNames[discriminator][sample] = logFileName
    #----------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------
    # CV: build config file for Pt, eta reweighting
    reweightTreeTauIdMVA_configFileNames[discriminator] = {}
    reweightTreeTauIdMVA_outputFileNames[discriminator] = {}
    reweightTreeTauIdMVA_logFileNames[discriminator]    = {}
    for sample in [ "signal", "background" ]:
        outputFileName = os.path.join(outputFilePath, "reweightTreeTauIdMVA_%s_%s.root" % (discriminator, sample))
        print " outputFileName = '%s'" % outputFileName
        reweightTreeTauIdMVA_outputFileNames[discriminator][sample] = outputFileName

        cfgFileName_original = configFile_reweightTreeTauIdMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n" 
        cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % preselectTreeTauIdMVA_outputFileNames[discriminator]['signal']
        cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % preselectTreeTauIdMVA_outputFileNames[discriminator]['background']
        cfg_modified += "\n"
        cfg_modified += "process.reweightTreeTauIdMVA.signalSamples = cms.vstring('signal')\n"
        cfg_modified += "process.reweightTreeTauIdMVA.backgroundSamples = cms.vstring('background')\n"
        cfg_modified += "process.reweightTreeTauIdMVA.applyPtReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyPtReweighting'])
        cfg_modified += "process.reweightTreeTauIdMVA.applyEtaReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyEtaReweighting'])
        cfg_modified += "process.reweightTreeTauIdMVA.reweight = cms.string('%s')\n" % mvaDiscriminators[discriminator]['reweight']
        cfg_modified += "process.reweightTreeTauIdMVA.inputVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['inputVariables']
        cfg_modified += "process.reweightTreeTauIdMVA.spectatorVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['spectatorVariables']
        cfg_modified += "process.reweightTreeTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfg_modified += "process.reweightTreeTauIdMVA.save = cms.string('%s')\n" % sample
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, sample)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        reweightTreeTauIdMVA_configFileNames[discriminator][sample] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        reweightTreeTauIdMVA_logFileNames[discriminator][sample] = logFileName
    #----------------------------------------------------------------------------
        
    #----------------------------------------------------------------------------    
    # CV: build config file for actual MVA training
        
    outputFileName = os.path.join(outputFilePath, "trainTauIdMVA_%s.root" % discriminator)
    print " outputFileName = '%s'" % outputFileName
    trainTauIdMVA_outputFileNames[discriminator] = outputFileName

    cfgFileName_original = configFile_trainTauIdMVA
    cfgFile_original = open(cfgFileName_original, "r")
    cfg_original = cfgFile_original.read()
    cfgFile_original.close()
    cfg_modified  = cfg_original
    cfg_modified += "\n"
    cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n"
    cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % reweightTreeTauIdMVA_outputFileNames[discriminator]['signal']
    cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % reweightTreeTauIdMVA_outputFileNames[discriminator]['background']
    cfg_modified += "\n"
    cfg_modified += "process.trainTauIdMVA.signalSamples = cms.vstring('signal')\n"
    cfg_modified += "process.trainTauIdMVA.backgroundSamples = cms.vstring('background')\n"
    cfg_modified += "process.trainTauIdMVA.applyPtReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyPtReweighting'])
    cfg_modified += "process.trainTauIdMVA.applyEtaReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyEtaReweighting'])
    cfg_modified += "process.trainTauIdMVA.reweight = cms.string('%s')\n" % mvaDiscriminators[discriminator]['reweight']
    cfg_modified += "process.trainTauIdMVA.mvaName = cms.string('%s')\n" % discriminator
    cfg_modified += "process.trainTauIdMVA.mvaTrainingOptions = cms.string('%s')\n" % mvaDiscriminators[discriminator]['mvaTrainingOptions']
    cfg_modified += "process.trainTauIdMVA.inputVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['inputVariables']
    cfg_modified += "process.trainTauIdMVA.spectatorVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['spectatorVariables']
    cfg_modified += "process.trainTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
    cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_cfg.py" % discriminator))
    print " cfgFileName_modified = '%s'" % cfgFileName_modified
    cfgFile_modified = open(cfgFileName_modified, "w")
    cfgFile_modified.write(cfg_modified)
    cfgFile_modified.close()
    trainTauIdMVA_configFileNames[discriminator] = cfgFileName_modified

    logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
    trainTauIdMVA_logFileNames[discriminator] = logFileName

print "Info: building config files for evaluating MVA performance"
makeROCcurveTauIdMVA_configFileNames = {} # key = discriminator, "TestTree" or "TrainTree"
makeROCcurveTauIdMVA_outputFileNames = {} # key = discriminator, "TestTree" or "TrainTree"
makeROCcurveTauIdMVA_logFileNames    = {} # key = discriminator, "TestTree" or "TrainTree"
for discriminator in mvaDiscriminators.keys():

    print "processing discriminator = %s" % discriminator

    makeROCcurveTauIdMVA_configFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_outputFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_logFileNames[discriminator]    = {}
        
    for tree in [ "TestTree", "TrainTree" ]:
        outputFileName = os.path.join(outputFilePath, "makeROCcurveTauIdMVA_%s_%s.root" % (discriminator, tree))
        print " outputFileName = '%s'" % outputFileName
        makeROCcurveTauIdMVA_outputFileNames[discriminator][tree] = outputFileName

        cfgFileName_original = configFile_makeROCcurveTauIdMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring('%s')\n" % trainTauIdMVA_outputFileNames[discriminator]
        cfg_modified += "\n"    
        cfg_modified += "delattr(process.makeROCcurveTauIdMVA, 'signalSamples')\n"
        cfg_modified += "delattr(process.makeROCcurveTauIdMVA, 'backgroundSamples')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.treeName = cms.string('%s')\n" % tree
        ##cfg_modified += "process.makeROCcurveTauIdMVA.preselection = cms.string('%s')\n" % mvaDiscriminators[discriminator]['preselection']
        cfg_modified += "process.makeROCcurveTauIdMVA.preselection = cms.string('')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.classId_signal = cms.int32(0)\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.classId_background = cms.int32(1)\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.branchNameClassId = cms.string('classID')\n"
        if 'recTauPt/F' in mvaDiscriminators[discriminator]['spectatorVariables']:
            cfg_modified += "process.makeROCcurveTauIdMVA.branchNameLogTauPt = cms.string('')\n"
            cfg_modified += "process.makeROCcurveTauIdMVA.branchNameTauPt = cms.string('recTauPt')\n"
        else:
            cfg_modified += "process.makeROCcurveTauIdMVA.branchNameLogTauPt = cms.string('TMath_Log_TMath_Max_1.,recTauPt__')\n"
            cfg_modified += "process.makeROCcurveTauIdMVA.branchNameTauPt = cms.string('')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.discriminator = cms.string('BDTG')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.branchNameEvtWeight = cms.string('weight')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.graphName = cms.string('%s_%s')\n" % (discriminator, tree)
        cfg_modified += "process.makeROCcurveTauIdMVA.binning.numBins = cms.int32(%i)\n" % 30000
        cfg_modified += "process.makeROCcurveTauIdMVA.binning.min = cms.double(%1.2f)\n" % -1.5
        cfg_modified += "process.makeROCcurveTauIdMVA.binning.max = cms.double(%1.2f)\n" % +1.5
        cfg_modified += "process.makeROCcurveTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, tree)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        makeROCcurveTauIdMVA_configFileNames[discriminator][tree] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        makeROCcurveTauIdMVA_logFileNames[discriminator][tree] = logFileName

    plotName = "mvaIsolation_%s_overtraining" % discriminator
    plots[plotName] = {
        'graphs' : [
            '%s:TestTree' % discriminator,
            '%s:TrainTree' % discriminator
        ]
    }

for discriminator in cutDiscriminators.keys():

    print "processing discriminator = %s" % discriminator

    makeROCcurveTauIdMVA_configFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_outputFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_logFileNames[discriminator]    = {}

    outputFileName = os.path.join(outputFilePath, "makeROCcurveTauIdMVA_%s.root" % discriminator)
    print " outputFileName = '%s'" % outputFileName
    makeROCcurveTauIdMVA_outputFileNames[discriminator]['TestTree'] = outputFileName

    cfgFileName_original = configFile_makeROCcurveTauIdMVA
    cfgFile_original = open(cfgFileName_original, "r")
    cfg_original = cfgFile_original.read()
    cfgFile_original.close()
    cfg_modified  = cfg_original
    cfg_modified += "\n"
    cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n"
    for inputFileName in inputFileNames:
        cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % inputFileName
    cfg_modified += "\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.signalSamples = cms.vstring(%s)\n" % signalSamples
    cfg_modified += "process.makeROCcurveTauIdMVA.backgroundSamples = cms.vstring(%s)\n" % backgroundSamples
    cfg_modified += "process.makeROCcurveTauIdMVA.treeName = cms.string('tauIdMVATrainingNtupleProducer/tauIdMVATrainingNtuple')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.preselection = cms.string('%s')\n" % cutDiscriminators[discriminator]['preselection']
    cfg_modified += "process.makeROCcurveTauIdMVA.branchNameLogTauPt = cms.string('')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.branchNameTauPt = cms.string('recTauPt')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.discriminator = cms.string('%s')\n" % cutDiscriminators[discriminator]['discriminator']
    cfg_modified += "process.makeROCcurveTauIdMVA.graphName = cms.string('%s_%s')\n" % (discriminator, "TestTree")
    cfg_modified += "process.makeROCcurveTauIdMVA.binning.numBins = cms.int32(%i)\n" % cutDiscriminators[discriminator]['numBins']
    cfg_modified += "process.makeROCcurveTauIdMVA.binning.min = cms.double(%1.2f)\n" % cutDiscriminators[discriminator]['min']
    cfg_modified += "process.makeROCcurveTauIdMVA.binning.max = cms.double(%1.2f)\n" % cutDiscriminators[discriminator]['max']
    cfg_modified += "process.makeROCcurveTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
    cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_cfg.py" % discriminator))
    print " cfgFileName_modified = '%s'" % cfgFileName_modified
    cfgFile_modified = open(cfgFileName_modified, "w")
    cfgFile_modified.write(cfg_modified)
    cfgFile_modified.close()
    makeROCcurveTauIdMVA_configFileNames[discriminator]['TestTree'] = cfgFileName_modified

    logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
    makeROCcurveTauIdMVA_logFileNames[discriminator]['TestTree'] = logFileName

hadd_inputFileNames = []
for discriminator in makeROCcurveTauIdMVA_outputFileNames.keys():
    for tree in [ "TestTree", "TrainTree" ]:
        if tree in makeROCcurveTauIdMVA_outputFileNames[discriminator].keys():
            hadd_inputFileNames.append(makeROCcurveTauIdMVA_outputFileNames[discriminator][tree])
hadd_outputFileName = os.path.join(outputFilePath, "makeROCcurveTauIdMVA_all.root")

print "Info: building config files for displaying results"
showROCcurvesTauIdMVA_configFileNames = {} # key = plot
showROCcurvesTauIdMVA_outputFileNames = {} # key = plot
showROCcurvesTauIdMVA_logFileNames    = {} # key = plot
for plot in plots.keys():

    print "processing plot = %s" % plot
    
    outputFileName = os.path.join(outputFilePath, "showROCcurvesTauIdMVA_%s.png" % plot)
    print " outputFileName = '%s'" % outputFileName
    showROCcurvesTauIdMVA_outputFileNames[plot] = outputFileName

    cfgFileName_original = configFile_showROCcurvesTauIdMVA
    cfgFile_original = open(cfgFileName_original, "r")
    cfg_original = cfgFile_original.read()
    cfgFile_original.close()
    cfg_modified  = cfg_original
    cfg_modified += "\n"
    cfg_modified += "process.fwliteInput.fileNames = cms.vstring('%s')\n" % hadd_outputFileName
    cfg_modified += "\n"
    cfg_modified += "process.showROCcurvesTauIdMVA.graphs = cms.VPSet(\n"
    for graph in plots[plot]['graphs']:
        discriminator = None
        tree = None
        legendEntry = None
        markerStyle = None
        markerSize  = None
        markerColor = None
        if graph.find(":") != -1:
            discriminator = graph[:graph.find(":")]
            tree = graph[graph.find(":") + 1:]
            legendEntry = tree
            if tree == "TestTree":
                markerStyle = 20
                markerColor = 1
                markerSize  = 1
            elif tree == "TrainTree":
                markerStyle = 24
                markerColor = 2
                markerSize  = 1
            else:
                raise ValueError("Invalid Parameter 'tree' = %s !!" % tree)
        else:
            discriminator = graph
            tree = "TestTree"
            legendEntry = allDiscriminators[graph]['legendEntry']
            if 'markerStyle' in allDiscriminators[graph].keys():
                markerStyle = allDiscriminators[graph]['markerStyle']
            if 'markerSize' in allDiscriminators[graph].keys():
                markerSize = allDiscriminators[graph]['markerSize']
            markerColor = allDiscriminators[graph]['color']
        cfg_modified += "    cms.PSet(\n"
        cfg_modified += "        graphName = cms.string('%s_%s'),\n" % (discriminator, tree)
        cfg_modified += "        legendEntry = cms.string('%s'),\n" % legendEntry
        if markerStyle:
            cfg_modified += "        markerStyle = cms.int32(%i),\n" % markerStyle
        if markerSize:
            cfg_modified += "        markerSize = cms.int32(%i),\n" % markerSize    
        cfg_modified += "        color = cms.int32(%i)\n" % markerColor
        cfg_modified += "    ),\n"
    cfg_modified += ")\n"
    cfg_modified += "process.showROCcurvesTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
    cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_cfg.py" % plot))
    print " cfgFileName_modified = '%s'" % cfgFileName_modified
    cfgFile_modified = open(cfgFileName_modified, "w")
    cfgFile_modified.write(cfg_modified)
    cfgFile_modified.close()
    showROCcurvesTauIdMVA_configFileNames[plot] = cfgFileName_modified

    logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
    showROCcurvesTauIdMVA_logFileNames[plot] = logFileName
    
def make_MakeFile_vstring(list_of_strings):
    retVal = ""
    for i, string_i in enumerate(list_of_strings):
        if i > 0:
            retVal += " "
        retVal += string_i
    return retVal

# done building config files, now build Makefile...
makeFileName = os.path.join(outputFilePath, "Makefile_runTauIdMVATraining_%s" % version)
makeFile = open(makeFileName, "w")
makeFile.write("\n")
outputFileNames = []
for discriminator in trainTauIdMVA_outputFileNames.keys():
    for sample in [ "signal", "background" ]:
        outputFileNames.append(preselectTreeTauIdMVA_outputFileNames[discriminator][sample])
        outputFileNames.append(reweightTreeTauIdMVA_outputFileNames[discriminator][sample])
    outputFileNames.append(trainTauIdMVA_outputFileNames[discriminator])
for discriminator in makeROCcurveTauIdMVA_outputFileNames.keys():
    for tree in makeROCcurveTauIdMVA_outputFileNames[discriminator]:
        outputFileNames.append(makeROCcurveTauIdMVA_outputFileNames[discriminator][tree])
outputFileNames.append(hadd_outputFileName)    
for plot in showROCcurvesTauIdMVA_outputFileNames.keys():
    outputFileNames.append(showROCcurvesTauIdMVA_outputFileNames[plot])
makeFile.write("all: %s\n" % make_MakeFile_vstring(outputFileNames))
makeFile.write("\techo 'Finished tau ID MVA training.'\n")
makeFile.write("\n")
for discriminator in trainTauIdMVA_outputFileNames.keys():
    for sample in [ "signal", "background" ]:
        makeFile.write("%s:\n" %
          (preselectTreeTauIdMVA_outputFileNames[discriminator][sample]))
        makeFile.write("\t%s%s %s &> %s\n" %
          (nice, executable_preselectTreeTauIdMVA,
           preselectTreeTauIdMVA_configFileNames[discriminator][sample],
           preselectTreeTauIdMVA_logFileNames[discriminator][sample]))
        makeFile.write("%s: %s\n" %
          (reweightTreeTauIdMVA_outputFileNames[discriminator][sample],
           make_MakeFile_vstring([ preselectTreeTauIdMVA_outputFileNames[discriminator]['signal'], preselectTreeTauIdMVA_outputFileNames[discriminator]['background'] ])))
        makeFile.write("\t%s%s %s &> %s\n" %
          (nice, executable_reweightTreeTauIdMVA,
           reweightTreeTauIdMVA_configFileNames[discriminator][sample],
           reweightTreeTauIdMVA_logFileNames[discriminator][sample]))
    makeFile.write("%s: %s\n" %
      (trainTauIdMVA_outputFileNames[discriminator],
       make_MakeFile_vstring([ reweightTreeTauIdMVA_outputFileNames[discriminator]['signal'], reweightTreeTauIdMVA_outputFileNames[discriminator]['background'] ])))
    makeFile.write("\t%s%s %s &> %s\n" %
      (nice, executable_trainTauIdMVA,
       trainTauIdMVA_configFileNames[discriminator],
       trainTauIdMVA_logFileNames[discriminator]))
makeFile.write("\n")    
for discriminator in makeROCcurveTauIdMVA_outputFileNames.keys():
    for tree in [ "TestTree", "TrainTree" ]:
        if tree in makeROCcurveTauIdMVA_outputFileNames[discriminator].keys():
            if discriminator in trainTauIdMVA_outputFileNames.keys():
                makeFile.write("%s: %s %s\n" %
                  (makeROCcurveTauIdMVA_outputFileNames[discriminator][tree],
                   trainTauIdMVA_outputFileNames[discriminator],
                   #executable_makeROCcurveTauIdMVA,
                   ""))
            else:
                makeFile.write("%s:\n" %
                  (makeROCcurveTauIdMVA_outputFileNames[discriminator][tree]))
            makeFile.write("\t%s%s %s &> %s\n" %
              (nice, executable_makeROCcurveTauIdMVA,
               makeROCcurveTauIdMVA_configFileNames[discriminator][tree],
               makeROCcurveTauIdMVA_logFileNames[discriminator][tree]))
makeFile.write("\n")
makeFile.write("%s: %s\n" %
  (hadd_outputFileName,
   make_MakeFile_vstring(hadd_inputFileNames)))
makeFile.write("\t%s%s %s\n" %
  (nice, executable_rm,
   hadd_outputFileName))
makeFile.write("\t%s%s %s %s\n" %
  (nice, executable_hadd,
   hadd_outputFileName, make_MakeFile_vstring(hadd_inputFileNames)))
makeFile.write("\n")
for plot in showROCcurvesTauIdMVA_outputFileNames.keys():
    makeFile.write("%s: %s %s\n" %
      (showROCcurvesTauIdMVA_outputFileNames[plot],
       hadd_outputFileName,
       executable_showROCcurvesTauIdMVA))
    makeFile.write("\t%s%s %s &> %s\n" %
      (nice, executable_showROCcurvesTauIdMVA,
       showROCcurvesTauIdMVA_configFileNames[plot],
       showROCcurvesTauIdMVA_logFileNames[plot]))
makeFile.write("\n")
makeFile.write(".PHONY: clean\n")
makeFile.write("clean:\n")
makeFile.write("\t%s %s\n" % (executable_rm, make_MakeFile_vstring(outputFileNames)))
makeFile.write("\techo 'Finished deleting old files.'\n")
makeFile.write("\n")
makeFile.close()

print("Finished building Makefile. Now execute 'make -f %s'." % makeFileName)
