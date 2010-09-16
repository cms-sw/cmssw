import FWCore.ParameterSet.Config as cms

def documentSkims():
    import Configuration.StandardSequences.Skims_cff as Skims

    listOfOptions=[]
    for skim in Skims.__dict__:
        skimstream = getattr(Skims,skim)
        if (not isinstance(skimstream,cms.FilteredStream)):
            continue
        
        shortname = skim.replace('SKIMStream','')
        print shortname
        if shortname!=skimstream['name']:
            print '#### ERROR ####'
            print 'skim name and stream name should be the same for consistency',shortname,'!=',skimstream['name']
            
        for token in ['name','responsible','dataTier']:
            print token,":",skimstream[token]
            
        listOfOptions.append(skimstream['name'])

    print 'possible cmsDriver options for skimming:'
    print 'SKIM:'+'+'.join(listOfOptions)
            
    


from Configuration.EventContent.EventContent_cff import FEVTEventContent
skimContent = FEVTEventContent.clone()
skimContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimContent.outputCommands.append("drop *_*_*_SKIM")

#############
from  DPGAnalysis.Skims.logErrorSkim_cff import *
pathlogerror =cms.Path(logerrorseq)

SKIMStreamLogError = cms.FilteredStream(
    responsible = 'reco convener',
    name = 'LogError',
    paths = (pathlogerror),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


##############
from  DPGAnalysis.Skims.BeamBkgSkim_cff import *
pathpfgskim3noncross = cms.Path(pfgskim3noncrossseq)

SKIMStreamBeamBkg = cms.FilteredStream(
    responsible = 'PFG',
    name = 'BeamBkg',
    paths = (pathpfgskim3noncross),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

###########
    
from DPGAnalysis.Skims.cscSkim_cff import *
pathCSCSkim =cms.Path(cscSkimseq)  
pathCSCHLTSkim = cms.Path(cscHLTSkimSeq)
pathCSCAloneSkim = cms.Path(cscSkimAloneSeq)

SKIMStreamCSC = cms.FilteredStream(
    responsible = 'DPG',
    name = 'CSC',
    paths = (pathCSCSkim),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )
SKIMStreamCSCHLT = cms.FilteredStream(
    responsible = 'DPG',
    name = 'CSCHLT',
    paths = (pathCSCHLTSkim),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )
SKIMStreamCSCAlone = cms.FilteredStream(
    responsible = 'DPG',
    name = 'CSCAlone',
    paths = (pathCSCAloneSkim),
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
    
#####################

from DPGAnalysis.Skims.HSCPSkim_cff import *

HSCPSkimPath = cms.Path( HSCPSkim )
SKIMStreamHSCP = cms.FilteredStream(
    responsible = '',
    name = 'HSCP',
    paths = (HSCPSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.cosmicSPSkim_cff import *
cosmicSPSkimPath = cms.Path( cosmicSPSkim )
SKIMStreamCosmicSP = cms.FilteredStream(
        responsible = '',
        name = 'CosmicSP',
        paths = (cosmicSPSkimPath),
        content = skimContent.outputCommands,
        selectEvents = cms.untracked.PSet(),
        dataTier = cms.untracked.string('RAW-RECO')
        )

#####################

from DPGAnalysis.Skims.ecalrechitsSkim_cff import *
ecalrechitSkimPath = cms.Path(ecalrechitSkim)
SKIMStreamEcalRH = cms.FilteredStream(
    responsible = 'Ecal DPG',
    name = 'EcalRH',
    paths = (ecalrechitSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.goodvertexSkim_cff import *
goodvertexSkimPath = cms.Path(goodvertexSkim)
SKIMStreamGoodVtx = cms.FilteredStream(
    responsible = 'Tracking POG',
    name = 'GoodVtx',
    paths = (goodvertexSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.muonTracksSkim_cff import *
muonTracksSkimPath = cms.Path(muonTracksSkim)
SKIMStreamMuonTrack = cms.FilteredStream(
        responsible = 'Muon POG',
        name = 'MuonTrack',
        paths = (muonTracksSkimPath),
        content = skimContent.outputCommands,
        selectEvents = cms.untracked.PSet(),
        dataTier = cms.untracked.string('RAW-RECO')
        )

#####################

from DPGAnalysis.Skims.valSkim_cff import *
relvaltrackSkimPath = cms.Path( relvaltrackSkim )
relvalmuonSkimPath = cms.Path( relvalmuonSkim )
SKIMStreamValSkim = cms.FilteredStream(
    responsible = 'RECO',
    name = 'ValSkim',
    paths = (relvaltrackSkimPath,relvalmuonSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.WZEGSkim_cff import *
WZEGSkimPath = cms.Path ( WZfilterSkim )
SKIMStreamWZ = cms.FilteredStream(
    responsible = 'ECAL DPG',
    name = 'WZ',
    paths = ( WZEGSkimPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from Configuration.Skimming.PDWG_DiJetAODSkim_cff import *
diJetAveSkimPath = cms.Path(DiJetAveSkim_Trigger)
SKIMStreamDiJet = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'DiJet',
    paths = (diJetAveSkimPath),
    content = DiJetAveSkim_EventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

#####################

from Configuration.Skimming.PDWG_TauSkim_cff import *
tauSkimPath = cms.Path( tauSkimSequence )
SKIMStreamTau = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'Tau',
    paths = (tauSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


#####################

from Configuration.Skimming.PDWG_OniaSkim_cff import *
oniaSkimPath = cms.Path(oniaSkimSequence)
SKIMStreamOnia = cms.FilteredStream(
    responsible = 'PDWG',
    name = 'Onia',
    paths = (oniaSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

