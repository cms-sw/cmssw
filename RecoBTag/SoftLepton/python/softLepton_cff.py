import FWCore.ParameterSet.Config as cms

from RecoBTag.SoftLepton.softLepton_EventSetup_cff import *

from RecoBTag.SoftLepton.softPFElectronTagInfos_cfi import *
from RecoBTag.SoftLepton.softPFMuonTagInfos_cfi import *
from RecoBTag.SoftLepton.SoftLeptonByMVA_cff import *
from RecoBTag.SoftLepton.SoftLeptonByPt_cff import *
from RecoBTag.SoftLepton.SoftLeptonByIP3d_cff import *
from RecoBTag.SoftLepton.SoftLeptonByIP2d_cff import *

softLeptonTask = cms.Task(
    softPFElectronsTagInfos,
    softPFMuonsTagInfos,
    softPFElectronBJetTags,
    negativeSoftPFElectronBJetTags,
    positiveSoftPFElectronBJetTags,
    softPFMuonBJetTags,
    negativeSoftPFMuonBJetTags,
    positiveSoftPFMuonBJetTags,
    softPFElectronByPtBJetTags,
    negativeSoftPFElectronByPtBJetTags,
    positiveSoftPFElectronByPtBJetTags,
    softPFMuonByPtBJetTags,
    negativeSoftPFMuonByPtBJetTags,
    positiveSoftPFMuonByPtBJetTags,
    softPFElectronByIP3dBJetTags,
    negativeSoftPFElectronByIP3dBJetTags,
    positiveSoftPFElectronByIP3dBJetTags,
    softPFMuonByIP3dBJetTags,
    negativeSoftPFMuonByIP3dBJetTags,
    positiveSoftPFMuonByIP3dBJetTags,
    softPFElectronByIP2dBJetTags,
    negativeSoftPFElectronByIP2dBJetTags,
    positiveSoftPFElectronByIP2dBJetTags,
    softPFMuonByIP2dBJetTags,
    negativeSoftPFMuonByIP2dBJetTags,
    positiveSoftPFMuonByIP2dBJetTags
)
