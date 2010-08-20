import FWCore.ParameterSet.Config as cms


#from DPGAnalysis.Skims.MinBiasPDSkim_cfg import SkimCfg
from Configuration.EventContent.EventContent_cff import FEVTEventContent
skimContent = FEVTEventContent.clone()
skimContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimContent.outputCommands.append("drop *_*_*_SKIM")

#############
from  DPGAnalysis.Skims.logErrorSkim_cff import *
pathlogerror =cms.Path(logerrorseq)

SKIMStreamLogerror = cms.FilteredStream(
    responsible = 'reco convener',
    name = 'Logerror',
    paths = (pathlogerror),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


##############
from  DPGAnalysis.Skims.BeamBkgSkim_cff import *
pathpfgskim3noncross = cms.Path(pfgskim3noncrossseq)

SKIMStreamBEAMBKGV3 = cms.FilteredStream(
    responsible = 'PFG',
    name = 'BEAMBKGV3',
    paths = (pathpfgskim3noncross),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

###########
    
from DPGAnalysis.Skims.cscSkim_cff import *
pathCSCSkim =cms.Path(cscHaloSkimseq)  

SKIMStreamCSC = cms.FilteredStream(
    responsible = 'DPG',
    name = 'CSC',
    paths = (pathCSCSkim),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.dtActivitySkim_cff import *
pathdtSkim =cms.Path(dtSkimseq)  
pathHLTdtSkim =cms.Path(dtHLTSkimseq)
    
SKIMStreamDT = cms.FilteredStream(
    responsible = 'DPG',
    name = 'DT',
    paths = (pathdtSkim,pathHLTdtSkim),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


#####################

from DPGAnalysis.Skims.L1MuonBitSkim_cff import *
pathL1MuBitSkim =cms.Path(l1MuBitsSkimseq)  

SKIMStreamL1MuBit = cms.FilteredStream(
    responsible = 'DPG',
    name = 'L1MuBit',
    paths = (pathL1MuBitSkim),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.RPCSkim_cff import *
pathrpcTecSkim =cms.Path(rpcTecSkimseq)  

SKIMStreamRPC = cms.FilteredStream(
    responsible = 'DPG',
    name = 'RPC',
    paths = (pathrpcTecSkim),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.singleMuonSkim_cff import *
from DPGAnalysis.Skims.singleElectronSkim_cff import *
from DPGAnalysis.Skims.muonTagProbeFilters_cff import *
from DPGAnalysis.Skims.electronTagProbeFilters_cff import *
from DPGAnalysis.Skims.singlePhotonSkim_cff import *
from DPGAnalysis.Skims.jetSkim_cff import *
from DPGAnalysis.Skims.METSkim_cff import *
from DPGAnalysis.Skims.singlePfTauSkim_cff import *

singleMuPt5SkimPath=cms.Path(singleMuPt5RecoQualitySeq)
singleElectronPt5SkimPath=cms.Path(singleElectronPt5RecoQualitySeq)
singlePhotonPt5SkimPath=cms.Path(singlePhotonPt5QualitySeq)
muonJPsiMMSkimPath=cms.Path(muonJPsiMMRecoQualitySeq)
jetSkimPath=cms.Path(jetRecoQualitySeq)
singlePfTauPt15SkimPath=cms.Path(singlePfTauPt15QualitySeq)
SKIMStreamTPG = cms.FilteredStream(
    responsible = 'TPG',
    name = 'TPG',
    paths = (singleMuPt5SkimPath,singleElectronPt5SkimPath,singlePhotonPt5SkimPath,muonJPsiMMSkimPath,jetSkimPath,singlePfTauPt15SkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )
    
