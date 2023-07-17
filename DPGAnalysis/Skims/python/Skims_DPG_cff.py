import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import FEVTEventContent
skimContent = FEVTEventContent.clone()
skimContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimContent.outputCommands.append("drop *_*_*_SKIM")

from Configuration.EventContent.EventContent_cff import RAWMINIAODEventContent 
skimMiniAODContent = RAWMINIAODEventContent.clone()
skimMiniAODContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimMiniAODContent.outputCommands.append("drop *_*_*_SKIM")

###### HCAL DPG Skims #########

from DPGAnalysis.Skims.SinglePhotonJetPlusHOFilter_cff import *
SinglePhotonJetPlusHOFilterPath = cms.Path( SinglePhotonJetPlusHOFilterSequence )
SKIMStreamSinglePhotonJetPlusHOFilter = cms.FilteredStream(
    responsible = 'HCAL DPG',
    name = 'SinglePhotonJetPlusHOFilter',
    paths = ( SinglePhotonJetPlusHOFilterPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
)
from DPGAnalysis.Skims.JetHTJetPlusHOFilter_cff import *
JetHTJetPlusHOFilterPath = cms.Path( JetHTJetPlusHOFilterSequence )
SKIMStreamJetHTJetPlusHOFilter = cms.FilteredStream(
    responsible = 'HCAL DPG',
    name = 'JetHTJetPlusHOFilter',
    paths = ( JetHTJetPlusHOFilterPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
)

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

#############
pathlogerrormonitor =cms.Path(logerrormonitorseq)

SKIMStreamLogErrorMonitor = cms.FilteredStream(
    responsible = 'reco convener',
    name = 'LogErrorMonitor',
    paths = (pathlogerrormonitor),
    content = cms.untracked.vstring('drop *_*_*_*', 'keep edmErrorSummaryEntrys_*_*_*'),
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

##############
#from  DPGAnalysis.Skims.BeamBkgSkim_cff import *
#pathpfgskim3noncross = cms.Path(pfgskim3noncrossseq)

#SKIMStreamBeamBkg = cms.FilteredStream(
#    responsible = 'PFG',
#    name = 'BeamBkg',
#    paths = (pathpfgskim3noncross),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

###########
    
#from DPGAnalysis.Skims.cscSkim_cff import *
#pathCSCSkim =cms.Path(cscSkimseq)  
#pathCSCHLTSkim = cms.Path(cscHLTSkimSeq)
#pathCSCAloneSkim = cms.Path(cscSkimAloneSeq)

#SKIMStreamCSC = cms.FilteredStream(
#    responsible = 'DPG',
#    name = 'CSC',
#    paths = (pathCSCSkim),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )
#SKIMStreamCSCHLT = cms.FilteredStream(
#    responsible = 'DPG',
#    name = 'CSCHLT',
#    paths = (pathCSCHLTSkim),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )
#SKIMStreamCSCAlone = cms.FilteredStream(
#    responsible = 'DPG',
#    name = 'CSCAlone',
#    paths = (pathCSCAloneSkim),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#   )

#####################

#from DPGAnalysis.Skims.dtActivitySkim_cff import *
#pathdtSkim =cms.Path(dtSkimseq)  
#pathHLTdtSkim =cms.Path(dtHLTSkimseq)
    
#SKIMStreamDT = cms.FilteredStream(
#    responsible = 'DPG',
#    name = 'DT',
#    paths = (pathHLTdtSkim),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

from DPGAnalysis.Skims.EcalActivitySkim_cff import *
EcalActivityPath = cms.Path( ecalActivitySeq )
SKIMStreamEcalActivity = cms.FilteredStream(
    responsible = 'ECAL DPG',
    name = 'EcalActivity',
    paths = ( EcalActivityPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


#####################

#from DPGAnalysis.Skims.L1MuonBitSkim_cff import *
#pathL1MuBitSkim =cms.Path(l1MuBitsSkimseq)  

#SKIMStreamL1MuBit = cms.FilteredStream(
#    responsible = 'DPG',
#    name = 'L1MuBit',
#    paths = (pathL1MuBitSkim),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

#from DPGAnalysis.Skims.RPCSkim_cff import *
#pathrpcTecSkim =cms.Path(rpcTecSkimseq)  

#SKIMStreamRPC = cms.FilteredStream(
#    responsible = 'DPG',
#    name = 'RPC',
#    paths = (pathrpcTecSkim),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

#from DPGAnalysis.Skims.singleMuonSkim_cff import *
#from DPGAnalysis.Skims.singleElectronSkim_cff import *
#from DPGAnalysis.Skims.muonTagProbeFilters_cff import *
#from DPGAnalysis.Skims.electronTagProbeFilters_cff import *
#from DPGAnalysis.Skims.singlePhotonSkim_cff import *
#from DPGAnalysis.Skims.jetSkim_cff import *
#from DPGAnalysis.Skims.METSkim_cff import *
#from DPGAnalysis.Skims.singlePfTauSkim_cff import *

#singleMuPt5SkimPath=cms.Path(singleMuPt5RecoQualitySeq)
#singleElectronPt5SkimPath=cms.Path(singleElectronPt5RecoQualitySeq)
#singlePhotonPt5SkimPath=cms.Path(singlePhotonPt5QualitySeq)
#muonJPsiMMSkimPath=cms.Path(muonJPsiMMRecoQualitySeq)
#jetSkimPath=cms.Path(jetRecoQualitySeq)
#singlePfTauPt15SkimPath=cms.Path(singlePfTauPt15QualitySeq)
#SKIMStreamTPG = cms.FilteredStream(
#    responsible = 'TPG',
#    name = 'TPG',
#    paths = (singleMuPt5SkimPath,singleElectronPt5SkimPath,singlePhotonPt5SkimPath,muonJPsiMMSkimPath,jetSkimPath,singlePfTauPt15SkimPath),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('USER')
#    )
    
#####################

#from DPGAnalysis.Skims.ecalrechitsSkim_cff import *
#ecalrechitSkimPath = cms.Path(ecalrechitSkim)
#SKIMStreamEcalRH = cms.FilteredStream(
#    responsible = 'Ecal DPG',
#    name = 'EcalRH',
#    paths = (ecalrechitSkimPath),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

#from DPGAnalysis.Skims.goodvertexSkim_cff import *
#goodvertexSkimPath = cms.Path(goodvertexSkim)
#SKIMStreamGoodVtx = cms.FilteredStream(
#    responsible = 'Tracking POG',
#    name = 'GoodVtx',
#    paths = (goodvertexSkimPath),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#############

#from DPGAnalysis.Skims.goodcollSkim_cff import *
#pathgoodcoll1 = cms.Path(goodcollL1requirement)
#pathgoodcolhf = cms.Path(goodcollHFrequirement)

#SKIMStreamGoodCol = cms.FilteredStream(
#    responsible = 'PVT',
#    name = 'GoodCol',
#    paths = (goodvertexSkimPath,pathgoodcoll1,pathgoodcolhf),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

#from DPGAnalysis.Skims.muonTracksSkim_cff import *
#muonTracksSkimPath = cms.Path(muonTracksSkim)
#SKIMStreamMuonTrack = cms.FilteredStream(
#        responsible = 'Muon POG',
#        name = 'MuonTrack',
#        paths = (muonTracksSkimPath),
#        content = skimContent.outputCommands,
#        selectEvents = cms.untracked.PSet(),
#        dataTier = cms.untracked.string('RAW-RECO')
#        )

#####################

#SKIMStreamMuonDPG = cms.FilteredStream(
#        responsible = 'Muon DPG',
#        name = 'MuonDPG',
#        paths = (muonTracksSkimPath,pathdtSkim,pathHLTdtSkim,pathCSCHLTSkim,pathCSCAloneSkim,pathrpcTecSkim,),
#        content = skimContent.outputCommands,
#        selectEvents = cms.untracked.PSet(),
#        dataTier = cms.untracked.string('RAW-RECO')
#        )

#####################

#from DPGAnalysis.Skims.valSkim_cff import *
#relvaltrackSkimPath = cms.Path( relvaltrackSkim )
#relvalmuonSkimPath = cms.Path( relvalmuonSkim )
#SKIMStreamValSkim = cms.FilteredStream(
#    responsible = 'RECO',
#    name = 'ValSkim',
#    paths = (relvaltrackSkimPath,relvalmuonSkimPath),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

#from DPGAnalysis.Skims.WZEGSkim_cff import *
#WZEGSkimPath = cms.Path ( WZfilterSkim )
#SKIMStreamWZEG = cms.FilteredStream(
#    responsible = 'ECAL DPG & EGM POG',
#    name = 'WZEG',
#    paths = ( WZEGSkimPath ),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )


#####################


from DPGAnalysis.Skims.WMuSkim_cff import *
WMuPath = cms.Path( pfMetWMuNuSeq )
SKIMStreamWMu = cms.FilteredStream(
    responsible = 'DDT Test',
    name = 'WMu',
    paths = ( WMuPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )


from DPGAnalysis.Skims.WElectronSkim_cff import *
WElectronPath = cms.Path( elecMetSeq )
SKIMStreamWElectron = cms.FilteredStream(
    responsible = 'Egamma POG',
    name = 'WElectron',
    paths = ( WElectronPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )



from DPGAnalysis.Skims.TopMuEGSkim_cff import *
TopMuEGPath = cms.Path( TopMuEGsequence )
SKIMStreamTopMuEG = cms.FilteredStream(
    responsible = 'BTV',
    name = 'TopMuEG',
    paths = ( TopMuEGPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )



from DPGAnalysis.Skims.ZMuSkim_cff import *
from DPGAnalysis.Skims.ZmmgSkim_cff import *

ZMuSkimPath = cms.Path(diMuonSelSeq)
ZmmgSkimPath = cms.Path(ZmmgSkimSeq)
WtcMetSkimPath = cms.Path(tcMetWMuNuSeq)
WpfMetSkimPath = cms.Path(pfMetWMuNuSeq)
#SKIMStreamWZMu   = cms.FilteredStream(
#    responsible = 'Muon DPG-POG-MuEW',
#    name = 'WZMu',
#    paths = (ZMuSkimPath, WtcMetSkimPath, WpfMetSkimPath),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )
SKIMStreamZMu   = cms.FilteredStream(
    responsible = 'Muon DPG-POG-MuEW',
    name = 'ZMu',
    paths = (ZMuSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )
SKIMStreamZmmg   = cms.FilteredStream(
    responsible = 'ECAL DPG',
    name = 'Zmmg',
    paths = (ZmmgSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )
#SKIMStreamWMu   = cms.FilteredStream(
#    responsible = 'Muon DPG-POG-MuEW',
#    name = 'WMu',
#    paths = (WtcMetSkimPath, WpfMetSkimPath),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

#from DPGAnalysis.Skims.TkSDSkim_cff import *
#TkSDSkimPath = cms.Path (TkSD_Seq)
#SKIMStreamTkSD = cms.FilteredStream(
#    responsible = 'Tracker DPG & Tracking POG',
#    name = 'TkSD',
#    paths = ( TkSDSkimPath ),
#    content = skimContent.outputCommands,
#    selectEvents = cms.untracked.PSet(),
#    dataTier = cms.untracked.string('RAW-RECO')
#    )

#####################

from DPGAnalysis.Skims.WElectronSkim_cff import *
WElectronPath = cms.Path( elecMetSeq )
SKIMStreamWElectron = cms.FilteredStream(
    responsible = 'ECAL DPG',
    name = 'WElectron',
    paths = ( WElectronPath ),
    content = WElectronSkimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

#####################

from DPGAnalysis.Skims.ZElectronSkim_cff import *
ZElectronPathGsf = cms.Path( zdiElectronSequence )

SKIMStreamZElectron = cms.FilteredStream(
    responsible = 'ECAL DPG',
    name = 'ZElectron',
    paths = ( ZElectronPathGsf ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.IsoPhotonEBSkim_cff import *
isoPhotonEBPath = cms.Path( isoPhotonEBSequence )

SKIMStreamIsoPhotonEB = cms.FilteredStream(
    responsible = 'L1 DPG',
    name = 'IsoPhotonEB',
    paths = ( isoPhotonEBPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from DPGAnalysis.Skims.HighMETSkim_cff import *
condPath = cms.Path(CondMETSelSeq)
#pfPath = cms.Path(pfMETSelSeq)
#caloPath = cms.Path(caloMETSelSeq)
#miniPath = cms.Path(miniMETSelSeq)
SKIMStreamHighMET   = cms.FilteredStream(
            responsible = 'MET Group',
            name = 'HighMET',
            paths = (condPath),
#           paths = (pfPath),
#           paths = (caloPath),
#           paths = (miniPath),
            content = skimContent.outputCommands,
            selectEvents = cms.untracked.PSet(),
            dataTier = cms.untracked.string('RAW-RECO')
            )

#####################

###### MuTau Skim #########
from DPGAnalysis.Skims.MuTauSkim_cff import *
MuTauPath = cms.Path( mutauSkimSequence )
SKIMStreamMuTau = cms.FilteredStream(
    responsible = 'Tau POG',
    name = 'MuTau',
    paths = ( MuTauPath ),
    content = skimMiniAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

##########################


from DPGAnalysis.Skims.ZHbbSkim_cff import *
ZHbbSkimPath = cms.Path(ZHbbSkim)
SKIMStreamZHbb = cms.FilteredStream(
    responsible = 'BTV',
    name = 'ZHbb',
    paths = (ZHbbSkimPath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )




