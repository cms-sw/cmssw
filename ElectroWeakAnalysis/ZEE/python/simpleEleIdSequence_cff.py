import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZEE.simpleCutBasedElectronIDSpring10_cfi import *


simpleEleId95relIso = simpleCutBasedElectronID.clone()
simpleEleId95relIso.electronQuality = '95relIso'

simpleEleId90relIso = simpleCutBasedElectronID.clone()
simpleEleId90relIso.electronQuality = '90relIso'

simpleEleId85relIso = simpleCutBasedElectronID.clone()
simpleEleId85relIso.electronQuality = '85relIso'

simpleEleId80relIso = simpleCutBasedElectronID.clone()
simpleEleId80relIso.electronQuality = '80relIso'

simpleEleId70relIso = simpleCutBasedElectronID.clone()
simpleEleId70relIso.electronQuality = '70relIso'

simpleEleId60relIso = simpleCutBasedElectronID.clone()
simpleEleId60relIso.electronQuality = '60relIso'


simpleEleId95cIso = simpleCutBasedElectronID.clone()
simpleEleId95cIso.electronQuality = '95cIso'

simpleEleId90cIso = simpleCutBasedElectronID.clone()
simpleEleId90cIso.electronQuality = '90cIso'

simpleEleId85cIso = simpleCutBasedElectronID.clone()
simpleEleId85cIso.electronQuality = '85cIso'

simpleEleId80cIso = simpleCutBasedElectronID.clone()
simpleEleId80cIso.electronQuality = '80cIso'

simpleEleId70cIso = simpleCutBasedElectronID.clone()
simpleEleId70cIso.electronQuality = '70cIso'

simpleEleId60cIso = simpleCutBasedElectronID.clone()
simpleEleId60cIso.electronQuality = '60cIso'



simpleEleIdSequence = cms.Sequence(simpleEleId95relIso+
                                   simpleEleId90relIso+
                                   simpleEleId85relIso+
                                   simpleEleId80relIso+
                                   simpleEleId70relIso+
                                   simpleEleId60relIso+
                                   simpleEleId95cIso+
                                   simpleEleId90cIso+
                                   simpleEleId85cIso+
                                   simpleEleId80cIso+
                                   simpleEleId70cIso+
                                   simpleEleId60cIso
                                   )

