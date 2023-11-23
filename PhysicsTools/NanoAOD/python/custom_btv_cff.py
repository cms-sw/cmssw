import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var
#from PhysicsTools.NanoAOD.jetsAK4_CHS_cff import jetTable, jetCorrFactorsNano, updatedJets, finalJets, qgtagger, hfJetShowerShapeforNanoAOD
from PhysicsTools.NanoAOD.jetsAK4_Puppi_cff import jetPuppiTable, jetPuppiCorrFactorsNano, updatedJetsPuppi, updatedJetsPuppiWithUserData
from PhysicsTools.NanoAOD.jetsAK8_cff import fatJetTable, subJetTable
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from PhysicsTools.PatAlgos.tools.helpers import addToProcessAndTask, getPatAlgosToolsTask
from PhysicsTools.NanoAOD.common_cff import Var, CandVars
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

## Move PFNano (https://github.com/cms-jet/PFNano/) to NanoAOD


## From: https://github.com/cms-jet/PFNano/blob/13_0_7_from124MiniAOD/python/addBTV.py
## Define customized GenParticles 
def customize_BTV_GenTable(process):
    process.finalGenParticles.select += [
        "keep (4 <= abs(pdgId) <= 5) && statusFlags().isLastCopy()", # BTV: keep b/c quarks in their last copy
        "keep (abs(pdgId) == 310 || abs(pdgId) == 3122) && statusFlags().isLastCopy()", # BTV: keep K0s and Lambdas in their last copy
    ]
    process.genParticleTable.variables = cms.PSet(
        process.genParticleTable.variables,
        vx = Var("vx", "float", doc="x coordinate of vertex position"),
        vy = Var("vy", "float", doc="y coordinate of vertex position"),
        vz = Var("vz", "float", doc="z coordinate of vertex position"),
        genPartIdxMother2 = Var("?numberOfMothers>1?motherRef(1).key():-1", "int", doc="index of the second mother particle, if valid"),
    )

def update_jets_AK4(process):
    # Based on ``nanoAOD_addDeepInfo``
    # in https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/nano_cff.py
    # DeepJet flav_names as found in
    # https://github.com/cms-sw/cmssw/blob/master/RecoBTag/ONNXRuntime/plugins/DeepFlavourONNXJetTagsProducer.cc#L86
    # and https://twiki.cern.ch/twiki/bin/view/CMS/DeepJet
    from RecoBTag.ONNXRuntime.pfParticleTransformerAK4_cff import _pfParticleTransformerAK4JetTagsAll as pfParticleTransformerAK4JetTagsAll
    from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll as pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
    from RecoBTag.ONNXRuntime.pfParticleTransformerAK4_cff import _pfNegativeParticleTransformerAK4JetTagsProbs as pfNegativeParticleTransformerAK4JetTagsProbs
    from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs as pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs

    _btagDiscriminators = [
        'pfJetProbabilityBJetTags',
        'pfJetBProbabilityBJetTags',
        'pfNegativeOnlyJetProbabilityBJetTags',
        'pfNegativeOnlyJetBProbabilityBJetTags',
        'pfDeepFlavourJetTags:probb',
        'pfDeepFlavourJetTags:probbb',
        'pfDeepFlavourJetTags:problepb',
        'pfDeepFlavourJetTags:probc',
        'pfDeepFlavourJetTags:probuds',
        'pfDeepFlavourJetTags:probg',
        'pfNegativeDeepFlavourJetTags:probb',
        'pfNegativeDeepFlavourJetTags:probbb',
        'pfNegativeDeepFlavourJetTags:problepb',
        'pfNegativeDeepFlavourJetTags:probc',
        'pfNegativeDeepFlavourJetTags:probuds',
        'pfNegativeDeepFlavourJetTags:probg',
    ] + pfParticleTransformerAK4JetTagsAll + pfNegativeParticleTransformerAK4JetTagsProbs \
        + pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll + pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs
    
    updateJetCollection(
        process,
        jetSource=cms.InputTag('slimmedJetsPuppi'),
        jetCorrections=('AK4PFPuppi',
                        cms.vstring(
                            ['L1FastJet', 'L2Relative', 'L3Absolute',
                             'L2L3Residual']), 'None'),
        btagDiscriminators=_btagDiscriminators,
        postfix='PuppiWithDeepInfo',
    )
    process.load("Configuration.StandardSequences.MagneticField_cff")
    process.jetPuppiCorrFactorsNano.src = "selectedUpdatedPatJetsPuppiWithDeepInfo"
    process.updatedJetsPuppi.jetSource = "selectedUpdatedPatJetsPuppiWithDeepInfo"
    
    
    process.updatedPatJetsTransientCorrectedPuppiWithDeepInfo.tagInfoSources.append(cms.InputTag("pfDeepFlavourTagInfosPuppiWithDeepInfo"))
    process.updatedPatJetsTransientCorrectedPuppiWithDeepInfo.addTagInfos = cms.bool(True)

    
    
    return process

def update_jets_AK8(process):
    # Based on ``nanoAOD_addDeepInfoAK8``
    # in https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/nano_cff.py
    # Care needs to be taken to make sure no discriminators from stock Nano are excluded -> would results in unfilled vars
    _btagDiscriminators = [
        'pfMassIndependentDeepDoubleBvLV2JetTags:probHbb',
        'pfMassIndependentDeepDoubleCvLV2JetTags:probHcc',
        'pfMassIndependentDeepDoubleCvBV2JetTags:probHcc',
        ]
    from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll
    _btagDiscriminators += pfParticleNetJetTagsAll
    updateJetCollection(
        process,
        jetSource=cms.InputTag('slimmedJetsAK8'),
        pvSource=cms.InputTag('offlineSlimmedPrimaryVertices'),
        svSource=cms.InputTag('slimmedSecondaryVertices'),
        rParam=0.8,
        jetCorrections=('AK8PFPuppi',
                        cms.vstring([
                            'L1FastJet', 'L2Relative', 'L3Absolute',
                            'L2L3Residual'
                        ]), 'None'),
        btagDiscriminators=_btagDiscriminators,
        postfix='AK8WithDeepInfo',
        # this should work but doesn't seem to enable the tag info with addTagInfos
        # btagInfos=['pfDeepDoubleXTagInfos'],
        printWarning=False)
    process.jetCorrFactorsAK8.src = "selectedUpdatedPatJetsAK8WithDeepInfo"
    process.updatedJetsAK8.jetSource = "selectedUpdatedPatJetsAK8WithDeepInfo"
    # add DeepDoubleX taginfos
    process.updatedPatJetsTransientCorrectedAK8WithDeepInfo.tagInfoSources.append(cms.InputTag("pfDeepDoubleXTagInfosAK8WithDeepInfo"))
    process.updatedPatJetsTransientCorrectedAK8WithDeepInfo.addTagInfos = cms.bool(True)
    return process

def update_jets_AK8_subjet(process):
    # Based on ``nanoAOD_addDeepInfoAK8``
    # in https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/nano_cff.py
    # and https://github.com/alefisico/RecoBTag-PerformanceMeasurements/blob/10_2_X_boostedCommissioning/test/runBTagAnalyzer_cfg.py
    _btagDiscriminators = [
        'pfJetProbabilityBJetTags',
        'pfDeepCSVJetTags:probb',
        'pfDeepCSVJetTags:probc',
        'pfDeepCSVJetTags:probbb',
        'pfDeepCSVJetTags:probudsg',
        ]
    updateJetCollection(
        process,
        labelName='SoftDropSubjetsPF',
        jetSource=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked", "SubJets"),
        jetCorrections=('AK4PFPuppi',
                        ['L2Relative', 'L3Absolute'], 'None'),
        btagDiscriminators=list(_btagDiscriminators),
        explicitJTA=True,  # needed for subjet b tagging
        svClustering=False,  # needed for subjet b tagging (IMPORTANT: Needs to be set to False to disable ghost-association which does not work with slimmed jets)
        fatJets=cms.InputTag('slimmedJetsAK8'),  # needed for subjet b tagging
        rParam=0.8,  # needed for subjet b tagging
        sortByPt=False, # Don't change order (would mess with subJetIdx for FatJets)
        postfix='AK8SubjetsWithDeepInfo')

    process.subJetTable.src = 'selectedUpdatedPatJetsSoftDropSubjetsPFAK8SubjetsWithDeepInfo' 
    

    return process

def get_DDX_vars():
    # retreive 27 jet-level features used in double-b and deep double-x taggers
    # defined in arXiv:1712.07158
    DDXVars = cms.PSet(
        DDX_jetNTracks = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.jetNTracks", int, doc="number of tracks associated with the jet"),
        DDX_jetNSecondaryVertices = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.jetNSecondaryVertices", int, doc="number of SVs associated with the jet"),
        DDX_tau1_trackEtaRel_0 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_trackEtaRel_0", float, doc="1st smallest track pseudorapidity, relative to the jet axis, associated to the 1st N-subjettiness axis", precision=10),
        DDX_tau1_trackEtaRel_1 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_trackEtaRel_1", float, doc="2nd smallest track pseudorapidity, relative to the jet axis, associated to the 1st N-subjettiness axis", precision=10),
        DDX_tau1_trackEtaRel_2 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_trackEtaRel_2", float, doc="3rd smallest track pseudorapidity, relative to the jet axis, associated to the 1st N-subjettiness axis", precision=10),
        DDX_tau2_trackEtaRel_0 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_trackEtaRel_0", float, doc="1st smallest track pseudorapidity, relative to the jet axis, associated to the 2nd N-subjettiness axis", precision=10),
        DDX_tau2_trackEtaRel_1 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_trackEtaRel_1", float, doc="2nd smallest track pseudorapidity, relative to the jet axis, associated to the 2nd N-subjettiness axis", precision=10),
        DDX_tau2_trackEtaRel_3 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_trackEtaRel_2", float, doc="3rd smallest track pseudorapidity, relative to the jet axis, associated to the 2nd N-subjettiness axis", precision=10),
        DDX_tau1_flightDistance2dSig = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_flightDistance2dSig", float, doc="transverse distance significance between primary and secondary vertex associated to the 1st N-subjettiness axis", precision=10),
        DDX_tau2_flightDistance2dSig = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_flightDistance2dSig", float, doc="transverse distance significance between primary and secondary vertex associated to the 2nd N-subjettiness axis", precision=10),
        DDX_tau1_vertexDeltaR = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_vertexDeltaR", float, doc="deltaR between the 1st N-subjettiness axis and secondary vertex direction", precision=10),
        DDX_tau1_vertexEnergyRatio = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_vertexEnergyRatio", float, doc="ratio of energy at secondary vertex over total energy associated to the 1st N-subjettiness axis", precision=10),
        DDX_tau2_vertexEnergyRatio = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_vertexEnergyRatio", float, doc="ratio of energy at secondary vertex over total energy associated to the 2nd N-subjettiness axis", precision=10),
        DDX_tau1_vertexMass = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_vertexMass", float, doc="mass of track sum at secondary vertex associated to the 1st N-subjettiness axis", precision=10),
        DDX_tau2_vertexMass = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_vertexMass", float, doc="mass of track sum at secondary vertex associated to the 2nd N-subjettiness axis", precision=10),
        DDX_trackSip2dSigAboveBottom_0 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.trackSip2dSigAboveBottom_0", float, doc="track 2D signed impact parameter significance of 1st track lifting mass above bottom", precision=10),
        DDX_trackSip2dSigAboveBottom_1 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.trackSip2dSigAboveBottom_1", float, doc="track 2D signed impact parameter significance of 2nd track lifting mass above bottom", precision=10),
        DDX_trackSip2dSigAboveCharm = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.trackSip2dSigAboveCharm", float, doc="track 2D signed impact parameter significance of 1st track lifting mass above charm", precision=10),
        DDX_trackSip3dSig_0 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.trackSip3dSig_0", float, doc="1st largest track 3D signed impact parameter significance", precision=10),
        DDX_tau1_trackSip3dSig_0 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_trackSip3dSig_0", float, doc="1st largest track 3D signed impact parameter significance associated to the 1st N-subjettiness axis", precision=10),
        DDX_tau1_trackSip3dSig_1 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau1_trackSip3dSig_1", float, doc="2nd largest track 3D signed impact parameter significance associated to the 1st N-subjettiness axis", precision=10),
        DDX_trackSip3dSig_1 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.trackSip3dSig_1", float, doc="2nd largest track 3D signed impact parameter significance", precision=10),
        DDX_tau2_trackSip3dSig_0 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_trackSip3dSig_0", float, doc="1st largest track 3D signed impact parameter significance associated to the 2nd N-subjettiness axis", precision=10),
        DDX_tau2_trackSip3dSig_1 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.tau2_trackSip3dSig_1", float, doc="2nd largest track 3D signed impact parameter significance associated to the 2nd N-subjettiness axis", precision=10),
        DDX_trackSip3dSig_2 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.trackSip3dSig_2", float, doc="3rd largest track 3D signed impact parameter significance", precision=10),
        DDX_trackSip3dSig_3 = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.trackSip3dSig_3", float, doc="4th largest track 3D signed impact parameter significance", precision=10),
        DDX_z_ratio = Var("tagInfo(\'pfDeepDoubleX\').features().tag_info_features.z_ratio", float, doc="z = deltaR(SV0,SV1)*pT(SV1)/m(SV0,SV1), defined in Eq. 7 of arXiv:1712.07158", precision=10)
    )
    return DDXVars

def get_DeepCSV_vars():
    DeepCSVVars = cms.PSet(
        # Tagger inputs also include jet pt and eta
        # Track based (keep only jet-based features for DeepCSV from Run 3 commissioning)
        # DeepCSV_trackPtRel_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRel\')[0]:-999", float, doc="track transverse momentum, relative to the jet axis", precision=10),
        # DeepCSV_trackPtRel_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRel\')[1]:-999", float, doc="track transverse momentum, relative to the jet axis", precision=10),
        # DeepCSV_trackPtRel_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRel\')[2]:-999", float, doc="track transverse momentum, relative to the jet axis", precision=10),
        # DeepCSV_trackPtRel_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRel\')[3]:-999", float, doc="track transverse momentum, relative to the jet axis", precision=10),
        # DeepCSV_trackPtRel_4 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRel\')[4]:-999", float, doc="track transverse momentum, relative to the jet axis", precision=10),
        # DeepCSV_trackPtRel_5 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRel\')[5]:-999", float, doc="track transverse momentum, relative to the jet axis", precision=10),
        # DeepCSV_trackJetDistVal_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackJetDistVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackJetDistVal\')[0]:-999", float, doc="minimum track approach distance to jet axis", precision=10),
        # DeepCSV_trackJetDistVal_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackJetDistVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackJetDistVal\')[1]:-999", float, doc="minimum track approach distance to jet axis", precision=10),
        # DeepCSV_trackJetDistVal_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackJetDistVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackJetDistVal\')[2]:-999", float, doc="minimum track approach distance to jet axis", precision=10),
        # DeepCSV_trackJetDistVal_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackJetDistVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackJetDistVal\')[3]:-999", float, doc="minimum track approach distance to jet axis", precision=10),
        # DeepCSV_trackJetDistVal_4 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackJetDistVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackJetDistVal\')[4]:-999", float, doc="minimum track approach distance to jet axis", precision=10),
        # DeepCSV_trackJetDistVal_5 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackJetDistVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackJetDistVal\')[5]:-999", float, doc="minimum track approach distance to jet axis", precision=10),
        # DeepCSV_trackDeltaR_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDeltaR\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDeltaR\')[0]:-999", float, doc="track pseudoangular distance from the jet axis", precision=10),
        # DeepCSV_trackDeltaR_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDeltaR\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDeltaR\')[1]:-999", float, doc="track pseudoangular distance from the jet axis", precision=10),
        # DeepCSV_trackDeltaR_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDeltaR\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDeltaR\')[2]:-999", float, doc="track pseudoangular distance from the jet axis", precision=10),
        # DeepCSV_trackDeltaR_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDeltaR\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDeltaR\')[3]:-999", float, doc="track pseudoangular distance from the jet axis", precision=10),
        # DeepCSV_trackDeltaR_4 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDeltaR\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDeltaR\')[4]:-999", float, doc="track pseudoangular distance from the jet axis", precision=10),
        # DeepCSV_trackDeltaR_5 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDeltaR\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDeltaR\')[5]:-999", float, doc="track pseudoangular distance from the jet axis", precision=10),
        # DeepCSV_trackPtRatio_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRatio\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRatio\')[0]:-999", float, doc="track transverse momentum, relative to the jet axis, normalized to its energy", precision=10),
        # DeepCSV_trackPtRatio_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRatio\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRatio\')[1]:-999", float, doc="track transverse momentum, relative to the jet axis, normalized to its energy", precision=10),
        # DeepCSV_trackPtRatio_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRatio\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRatio\')[2]:-999", float, doc="track transverse momentum, relative to the jet axis, normalized to its energy", precision=10),
        # DeepCSV_trackPtRatio_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRatio\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRatio\')[3]:-999", float, doc="track transverse momentum, relative to the jet axis, normalized to its energy", precision=10),
        # DeepCSV_trackPtRatio_4 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRatio\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRatio\')[4]:-999", float, doc="track transverse momentum, relative to the jet axis, normalized to its energy", precision=10),
        # DeepCSV_trackPtRatio_5 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackPtRatio\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackPtRatio\')[5]:-999", float, doc="track transverse momentum, relative to the jet axis, normalized to its energy", precision=10),
        # DeepCSV_trackSip3dSig_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip3dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip3dSig\')[0]:-999", float, doc="track 3D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip3dSig_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip3dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip3dSig\')[1]:-999", float, doc="track 3D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip3dSig_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip3dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip3dSig\')[2]:-999", float, doc="track 3D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip3dSig_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip3dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip3dSig\')[3]:-999", float, doc="track 3D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip3dSig_4 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip3dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip3dSig\')[4]:-999", float, doc="track 3D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip3dSig_5 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip3dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip3dSig\')[5]:-999", float, doc="track 3D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip2dSig_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip2dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip2dSig\')[0]:-999", float, doc="track 2D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip2dSig_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip2dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip2dSig\')[1]:-999", float, doc="track 2D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip2dSig_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip2dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip2dSig\')[2]:-999", float, doc="track 2D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip2dSig_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip2dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip2dSig\')[3]:-999", float, doc="track 2D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip2dSig_4 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip2dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip2dSig\')[4]:-999", float, doc="track 2D signed impact parameter significance", precision=10),
        # DeepCSV_trackSip2dSig_5 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackSip2dSig\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackSip2dSig\')[5]:-999", float, doc="track 2D signed impact parameter significance", precision=10),
        # DeepCSV_trackDecayLenVal_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDecayLenVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDecayLenVal\')[0]:-999", float, doc="track decay length", precision=10),
        # DeepCSV_trackDecayLenVal_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDecayLenVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDecayLenVal\')[1]:-999", float, doc="track decay length", precision=10),
        # DeepCSV_trackDecayLenVal_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDecayLenVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDecayLenVal\')[2]:-999", float, doc="track decay length", precision=10),
        # DeepCSV_trackDecayLenVal_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDecayLenVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDecayLenVal\')[3]:-999", float, doc="track decay length", precision=10),
        # DeepCSV_trackDecayLenVal_4 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDecayLenVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDecayLenVal\')[4]:-999", float, doc="track decay length", precision=10),
        # DeepCSV_trackDecayLenVal_5 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackDecayLenVal\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackDecayLenVal\')[5]:-999", float, doc="track decay length", precision=10),
        # DeepCSV_trackEtaRel_0 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackEtaRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackEtaRel\')[0]:-999", float, doc="track pseudorapidity, relative to the jet axis", precision=10),
        # DeepCSV_trackEtaRel_1 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackEtaRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackEtaRel\')[1]:-999", float, doc="track pseudorapidity, relative to the jet axis", precision=10),
        # DeepCSV_trackEtaRel_2 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackEtaRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackEtaRel\')[2]:-999", float, doc="track pseudorapidity, relative to the jet axis", precision=10),
        # DeepCSV_trackEtaRel_3 = Var("?tagInfo(\'pfDeepCSV\').taggingVariables.checkTag(\'trackEtaRel\')?tagInfo(\'pfDeepCSV\').taggingVariables.getList(\'trackEtaRel\')[3]:-999", float, doc="track pseudorapidity, relative to the jet axis", precision=10),
        # Jet based
        DeepCSV_trackJetPt = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'trackJetPt\', -999)", float, doc="track-based jet transverse momentum", precision=10),
        DeepCSV_vertexCategory = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'vertexCategory\', -999)", float, doc="category of secondary vertex (Reco, Pseudo, No)", precision=10),
        DeepCSV_jetNSecondaryVertices = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'jetNSecondaryVertices\', -999)", int, doc="number of reconstructed possible secondary vertices in jet"),
        DeepCSV_jetNSelectedTracks = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'jetNSelectedTracks\', -999)", int, doc="selected tracks in the jet"), 
        DeepCSV_jetNTracksEtaRel = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'jetNTracksEtaRel\', -999)", int, doc="number of tracks for which etaRel is computed"), 
        DeepCSV_trackSumJetEtRatio = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'trackSumJetEtRatio\', -999)", float, doc="ratio of track sum transverse energy over jet energy", precision=10),
        DeepCSV_trackSumJetDeltaR = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'trackSumJetDeltaR\', -999)", float, doc="pseudoangular distance between jet axis and track fourvector sum", precision=10),
        DeepCSV_trackSip2dValAboveCharm = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'trackSip2dValAboveCharm\', -999)", float, doc="track 2D signed impact parameter of first track lifting mass above charm", precision=10),
        DeepCSV_trackSip2dSigAboveCharm = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'trackSip2dSigAboveCharm\', -999)", float, doc="track 2D signed impact parameter significance of first track lifting mass above charm", precision=10),
        DeepCSV_trackSip3dValAboveCharm = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'trackSip3dValAboveCharm\', -999)", float, doc="track 3D signed impact parameter of first track lifting mass above charm", precision=10),
        DeepCSV_trackSip3dSigAboveCharm = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'trackSip3dSigAboveCharm\', -999)", float, doc="track 3D signed impact parameter significance of first track lifting mass above charm", precision=10),
        DeepCSV_vertexMass = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'vertexMass\', -999)", float, doc="mass of track sum at secondary vertex", precision=10),
        DeepCSV_vertexNTracks = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'vertexNTracks\', -999)", int, doc="number of tracks at secondary vertex"),
        DeepCSV_vertexEnergyRatio = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'vertexEnergyRatio\', -999)", float, doc="ratio of energy at secondary vertex over total energy", precision=10),
        DeepCSV_vertexJetDeltaR = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'vertexJetDeltaR\', -999)", float, doc="pseudoangular distance between jet axis and secondary vertex direction", precision=10),
        DeepCSV_flightDistance2dVal = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'flightDistance2dVal\', -999)", float, doc="transverse distance between primary and secondary vertex", precision=10),
        DeepCSV_flightDistance2dSig = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'flightDistance2dSig\', -999)", float, doc="transverse distance significance between primary and secondary vertex", precision=10),
        DeepCSV_flightDistance3dVal = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'flightDistance3dVal\', -999)", float, doc="distance between primary and secondary vertex", precision=10),
        DeepCSV_flightDistance3dSig = Var("tagInfo(\'pfDeepCSV\').taggingVariables.get(\'flightDistance3dSig\', -999)", float, doc="distance significance between primary and secondary vertex", precision=10),
    )
    return DeepCSVVars

## Store all output nodes, negative tagger for SF 
def get_DeepJet_outputs():
    DeepJetOutputVars = cms.PSet(
        btagDeepFlavB_b=Var("bDiscriminator('pfDeepFlavourJetTags:probb')",
                            float,
                            doc="DeepJet b tag probability",
                            precision=10),
        btagDeepFlavB_bb=Var("bDiscriminator('pfDeepFlavourJetTags:probbb')",
                            float,
                            doc="DeepJet bb tag probability",
                            precision=10),
        btagDeepFlavB_lepb=Var("bDiscriminator('pfDeepFlavourJetTags:problepb')",
                            float,
                            doc="DeepJet lepb tag probability",
                            precision=10),
        btagDeepFlavC=Var("bDiscriminator('pfDeepFlavourJetTags:probc')",
                            float,
                            doc="DeepJet c tag probability",
                            precision=10),
        btagDeepFlavUDS=Var("bDiscriminator('pfDeepFlavourJetTags:probuds')",
                            float,
                            doc="DeepJet uds tag probability",
                            precision=10),
        btagDeepFlavG=Var("bDiscriminator('pfDeepFlavourJetTags:probg')",
                            float,
                            doc="DeepJet gluon tag probability",
                            precision=10),
        # discriminators are already part of jets_cff.py from NanoAOD and therefore not added here     

        # negative taggers
        btagNegDeepFlavB = Var("bDiscriminator('pfNegativeDeepFlavourJetTags:probb')+bDiscriminator('pfNegativeDeepFlavourJetTags:probbb')+bDiscriminator('pfNegativeDeepFlavourJetTags:problepb')",
                            float,
                            doc="Negative DeepJet b+bb+lepb tag discriminator",
                            precision=10),
        btagNegDeepFlavCvL = Var("?(bDiscriminator('pfNegativeDeepFlavourJetTags:probc')+bDiscriminator('pfNegativeDeepFlavourJetTags:probuds')+bDiscriminator('pfNegativeDeepFlavourJetTags:probg'))>0?bDiscriminator('pfNegativeDeepFlavourJetTags:probc')/(bDiscriminator('pfNegativeDeepFlavourJetTags:probc')+bDiscriminator('pfNegativeDeepFlavourJetTags:probuds')+bDiscriminator('pfNegativeDeepFlavourJetTags:probg')):-1",
                            float,
                            doc="Negative DeepJet c vs uds+g discriminator",
                            precision=10),
        btagNegDeepFlavCvB = Var("?(bDiscriminator('pfNegativeDeepFlavourJetTags:probc')+bDiscriminator('pfNegativeDeepFlavourJetTags:probb')+bDiscriminator('pfNegativeDeepFlavourJetTags:probbb')+bDiscriminator('pfNegativeDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfNegativeDeepFlavourJetTags:probc')/(bDiscriminator('pfNegativeDeepFlavourJetTags:probc')+bDiscriminator('pfNegativeDeepFlavourJetTags:probb')+bDiscriminator('pfNegativeDeepFlavourJetTags:probbb')+bDiscriminator('pfNegativeDeepFlavourJetTags:problepb')):-1",
                            float,
                            doc="Negative DeepJet c vs b+bb+lepb discriminator",
                            precision=10),
        btagNegDeepFlavQG = Var("?(bDiscriminator('pfNegativeDeepFlavourJetTags:probg')+bDiscriminator('pfNegativeDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfNegativeDeepFlavourJetTags:probg')/(bDiscriminator('pfNegativeDeepFlavourJetTags:probg')+bDiscriminator('pfNegativeDeepFlavourJetTags:probuds')):-1",
                            float,
                            doc="Negative DeepJet g vs uds discriminator",
                            precision=10),
        btagNegDeepFlavB_b = Var("bDiscriminator('pfNegativeDeepFlavourJetTags:probb')",
                            float,
                            doc="Negative DeepJet b tag probability",
                            precision=10),
        btagNegDeepFlavB_bb = Var("bDiscriminator('pfNegativeDeepFlavourJetTags:probbb')",
                            float,
                            doc="Negative DeepJet bb tag probability",
                            precision=10),
        btagNegDeepFlavB_lepb = Var("bDiscriminator('pfNegativeDeepFlavourJetTags:problepb')",
                            float,
                            doc="Negative DeepJet lepb tag probability",
                            precision=10),
        btagNegDeepFlavC = Var("bDiscriminator('pfNegativeDeepFlavourJetTags:probc')",
                            float,
                            doc="Negative DeepJet c tag probability",
                            precision=10),
        btagNegDeepFlavUDS = Var("bDiscriminator('pfNegativeDeepFlavourJetTags:probuds')",
                            float,
                            doc="Negative DeepJet uds tag probability",
                            precision=10),
        btagNegDeepFlavG = Var("bDiscriminator('pfNegativeDeepFlavourJetTags:probg')",
                            float,
                            doc="Negative DeepJet gluon tag probability",
                            precision=10),
    )
    return DeepJetOutputVars

def get_ParticleNetAK4_outputs():
    ## default scores in jetAK4_Puppi_cff.py collections
    ParticleNetAK4OutputVars = cms.PSet(
        # raw scores
        btagPNetProbB = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb'):-1",
                            float,
                            doc="ParticleNet b tag probability",
                            precision=10),
        btagPNetProbC = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probc'):-1",
                            float,
                            doc="ParticleNet c tag probability",
                            precision=10),
        btagPNetProbUDS = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds'):-1",
                            float,
                            doc="ParticleNet uds tag probability",
                            precision=10),
        btagPNetProbG = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probg')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probg'):-1",
                            float,
                            doc="ParticleNet gluon tag probability",
                            precision=10),
        
        # negative taggers
        btagNegPNetB = Var("?(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probb')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probg'))>0?(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probb'))/(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probb')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probg')):-1",
                            float,
                            doc="Negative ParticleNet b vs. udscg",
                            precision=10),
        btagNegPNetCvL = Var("?(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probg'))>0?(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc'))/(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probg')):-1",
                            float,
                            doc="Negative ParticleNet c vs. udsg",
                            precision=10),
        btagNegPNetCvB = Var("?(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probb'))>0?(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc'))/(bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')+bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probb')):-1",
                            float,
                            doc="Negative ParticleNet c vs. b",
                            precision=10),
        btagNegPNetProbB = Var("?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probb')>0?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probb'):-1",
                            float,
                            doc="Negative ParticleNet b tag probability",
                            precision=10),
        btagNegPNetProbC = Var("?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')>0?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probc'):-1",
                            float,
                            doc="Negative ParticleNet c tag probability",
                            precision=10),
        btagNegPNetProbUDS = Var("?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds')>0?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds'):-1",
                            float,
                            doc="Negative ParticleNet uds tag probability",
                            precision=10),
        btagNegPNetProbG = Var("?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probg')>0?bDiscriminator('pfNegativeParticleNetFromMiniAODAK4PuppiCentralJetTags:probg'):-1",
                            float,
                            doc="Negative ParticleNet gluon tag probability",
                            precision=10),
    )

    return ParticleNetAK4OutputVars

def get_ParticleTransformerAK4_outputs():
    ParticleTransformerAK4OutputVars = cms.PSet(       
        btagRobustParTAK4B_b=Var("bDiscriminator('pfParticleTransformerAK4JetTags:probb')",
                            float,
                            doc="RobustParTAK4 b tag probability",
                            precision=10),
        btagRobustParTAK4B_bb=Var("bDiscriminator('pfParticleTransformerAK4JetTags:probbb')",
                            float,
                            doc="RobustParTAK4 bb tag probability",
                            precision=10),
        btagRobustParTAK4B_lepb=Var("bDiscriminator('pfParticleTransformerAK4JetTags:problepb')",
                            float,
                            doc="RobustParTAK4 lepb tag probability",
                            precision=10),
        btagRobustParTAK4C=Var("bDiscriminator('pfParticleTransformerAK4JetTags:probc')",
                            float,
                            doc="RobustParTAK4 c tag probability",
                            precision=10),
        btagRobustParTAK4UDS=Var("bDiscriminator('pfParticleTransformerAK4JetTags:probuds')",
                            float,
                            doc="RobustParTAK4 uds tag probability",
                            precision=10),
        btagRobustParTAK4G=Var("bDiscriminator('pfParticleTransformerAK4JetTags:probg')",
                            float,
                            doc="RobustParTAK4 gluon tag probability",
                            precision=10),

        # negative taggers
        btagNegRobustParTAK4B = Var("bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probb')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:problepb')",
                            float,
                            doc="Negative RobustParTAK4 b+bb+lepb tag discriminator",
                            precision=10),
        btagNegRobustParTAK4CvL = Var("?(bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probc')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probuds')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probg'))>0?bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probc')/(bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probc')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probuds')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probg')):-1",
                            float,
                            doc="Negative RobustParTAK4 c vs uds+g discriminator",
                            precision=10),
        btagNegRobustParTAK4CvB = Var("?(bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probc')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probb')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:problepb'))>0?bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probc')/(bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probc')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probb')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:problepb')):-1",
                            float,
                            doc="Negative RobustParTAK4 c vs b+bb+lepb discriminator",
                            precision=10),
        btagNegRobustParTAK4QG = Var("?(bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probg')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probuds'))>0?bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probg')/(bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probg')+bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probuds')):-1",
                            float,
                            doc="Negative RobustParTAK4 g vs uds discriminator",
                            precision=10),
        btagNegRobustParTAK4B_b = Var("bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probb')",
                            float,
                            doc="Negative RobustParTAK4 b tag probability",
                            precision=10),
        btagNegRobustParTAK4B_bb = Var("bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probbb')",
                            float,
                            doc="Negative RobustParTAK4 bb tag probability",
                            precision=10),
        btagNegRobustParTAK4B_lepb = Var("bDiscriminator('pfNegativeParticleTransformerAK4JetTags:problepb')",
                            float,
                            doc="Negative RobustParTAK4 lepb tag probability",
                            precision=10),
        btagNegRobustParTAK4C = Var("bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probc')",
                            float,
                            doc="Negative RobustParTAK4 c tag probability",
                            precision=10),
        btagNegRobustParTAK4UDS = Var("bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probuds')",
                            float,
                            doc="Negative RobustParTAK4 uds tag probability",
                            precision=10),
        btagNegRobustParTAK4G = Var("bDiscriminator('pfNegativeParticleTransformerAK4JetTags:probg')",
                            float,
                            doc="Negative RobustParTAK4 gluon tag probability",
                            precision=10),
    )

    return ParticleTransformerAK4OutputVars

def add_BTV(process, runOnMC=False, addAK4=False, addAK8=False):
    if addAK4:
        process = update_jets_AK4(process)
    if addAK8:
        process = update_jets_AK8(process)
        process = update_jets_AK8_subjet(process)

    process.customizeJetTask = cms.Task()
    process.schedule.associate(process.customizeJetTask)

    CommonVars = cms.PSet(
        Proba=Var("bDiscriminator('pfJetProbabilityBJetTags')",
                  float,
                  doc="Jet Probability (Usage:BTV)",
                  precision=10),
        ProbaN=Var("bDiscriminator('pfNegativeOnlyJetProbabilityBJetTags')",
                  float,
                  doc="Negative-only Jet Probability (Usage:BTV)",
                  precision=10),
        Bprob=Var("bDiscriminator('pfJetBProbabilityBJetTags')",
                  float,
                  doc="Jet B Probability (Usage:BTV)",
                  precision=10),
        BprobN=Var("bDiscriminator('pfNegativeOnlyJetBProbabilityBJetTags')",
                  float,
                  doc="Negative-only Jet B Probability (Usage:BTV)",
                  precision=10),
    )
    
    # decouple these from CommonVars, not relevant for data
    HadronCountingVars = cms.PSet(
        nBHadrons=Var("jetFlavourInfo().getbHadrons().size()",
                      int,
                      doc="number of b-hadrons"),
        nCHadrons=Var("jetFlavourInfo().getcHadrons().size()",
                      int,
                      doc="number of c-hadrons")
    )

    # AK4
    process.customJetExtTable = cms.EDProducer(
        "SimpleCandidateFlatTableProducer",
        src=jetPuppiTable.src,
        cut=jetPuppiTable.cut,
        name=jetPuppiTable.name,
        doc=jetPuppiTable.doc,
        singleton=cms.bool(False),  # the number of entries is variable
        extension=cms.bool(True),  # this is the extension table for Jets
        variables=cms.PSet(
            CommonVars,
            HadronCountingVars if runOnMC else cms.PSet(), # hadrons from Generator only relevant for MC
            get_DeepCSV_vars(),
            get_DeepJet_outputs(),  # outputs are added in any case, inputs only if requested
            get_ParticleNetAK4_outputs(),
            get_ParticleTransformerAK4_outputs(),
        ))

    # disable the ParT branches in default jetPuppi table
    from PhysicsTools.NanoAOD.nano_eras_cff import run3_nanoAOD_122, run3_nanoAOD_124
    (run3_nanoAOD_122 | run3_nanoAOD_124).toModify(
        process.jetPuppiTable.variables,
        btagRobustParTAK4B = None,
        btagRobustParTAK4CvL = None,
        btagRobustParTAK4CvB = None,
        btagRobustParTAK4QG = None,
    )
    
    
    # from Run3 onwards, always set storeAK4Truth to True for MC
    process.customAK4ConstituentsForDeepJetTable = cms.EDProducer("PatJetDeepJetTableProducer",
                                                                      jets = cms.InputTag("linkedObjects","jets"),
                                                                      storeAK4Truth = cms.bool(runOnMC),
                                                                      )
    
    # AK8
    process.customFatJetExtTable = cms.EDProducer(
        "SimpleCandidateFlatTableProducer",
        src=fatJetTable.src,
        cut=fatJetTable.cut,
        name=fatJetTable.name,
        doc=fatJetTable.doc,
        singleton=cms.bool(False),  # the number of entries is variable
        extension=cms.bool(True),  # this is the extension table for FatJets
        variables=cms.PSet(
            CommonVars,
            #HadronCountingVars if runOnMC else cms.PSet(), # only necessary before 106x
            get_DDX_vars() ,
        ))

    # Subjets
    process.customSubJetExtTable = cms.EDProducer(
        "SimpleCandidateFlatTableProducer",
        src=subJetTable.src,
        cut=subJetTable.cut,
        name=subJetTable.name,
        doc=subJetTable.doc,
        singleton=cms.bool(False),  # the number of entries is variable
        extension=cms.bool(True),  # this is the extension table for FatJets
        variables=cms.PSet(
            CommonVars,
            #HadronCountingVars if runOnMC else cms.PSet(), # only necessary before 106x
    ))

    process.customSubJetMCExtTable = cms.EDProducer(
    "SimpleCandidateFlatTableProducer",
    src = subJetTable.src,
    cut = subJetTable.cut,
        name = subJetTable.name,
        doc=subJetTable.doc,
        singleton = cms.bool(False),
        extension = cms.bool(True),
        variables = cms.PSet(
        subGenJetAK8Idx = Var("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().key():-1",
        int,
        doc="index of matched gen Sub jet"),
       )
    )

    if addAK4:
        process.customizeJetTask.add(process.customJetExtTable)
        process.customizeJetTask.add(process.customAK4ConstituentsForDeepJetTable)

    if addAK8:
        process.customizeJetTask.add(process.customFatJetExtTable)
        process.customizeJetTask.add(process.customSubJetExtTable)
        if runOnMC:
            process.customizeJetTask.add(process.customSubJetMCExtTable)

    ## customize BTV GenTable
    if runOnMC:
        customize_BTV_GenTable(process)

#  From https://github.com/cms-jet/PFNano/blob/13_0_7_from124MiniAOD/python/addPFCands_cff.py
def addPFCands(process, runOnMC=False, allPF = False, addAK4=False, addAK8=False):
    process.customizedPFCandsTask = cms.Task( )
    process.schedule.associate(process.customizedPFCandsTask)

    process.finalJetsAK8Constituents = cms.EDProducer("PatJetConstituentPtrSelector",
                                            src = cms.InputTag("finalJetsAK8"),
                                            cut = cms.string("")
                                            )
    process.finalJetsAK4Constituents = cms.EDProducer("PatJetConstituentPtrSelector",
                                            src = cms.InputTag("finalJetsPuppi"),
                                            cut = cms.string("")
                                            )
    if allPF:
        candInput = cms.InputTag("packedPFCandidates")
    elif not addAK8:
        candList = cms.VInputTag(cms.InputTag("finalJetsAK4Constituents", "constituents"))
        process.customizedPFCandsTask.add(process.finalJetsAK4Constituents)
        process.finalJetsConstituents = cms.EDProducer("PackedCandidatePtrMerger", src = candList, skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        candInput = cms.InputTag("finalJetsConstituents")
    elif not addAK4:
        candList = cms.VInputTag(cms.InputTag("finalJetsAK8Constituents", "constituents"))
        process.customizedPFCandsTask.add(process.finalJetsAK8Constituents)
        process.finalJetsConstituents = cms.EDProducer("PackedCandidatePtrMerger", src = candList, skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        candInput = cms.InputTag("finalJetsConstituents")
    else:
        candList = cms.VInputTag(cms.InputTag("finalJetsAK4Constituents", "constituents"), cms.InputTag("finalJetsAK8Constituents", "constituents"))
        process.customizedPFCandsTask.add(process.finalJetsAK4Constituents)
        process.customizedPFCandsTask.add(process.finalJetsAK8Constituents)
        process.finalJetsConstituents = cms.EDProducer("PackedCandidatePtrMerger", src = candList, skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        candInput = cms.InputTag("finalJetsConstituents")
    process.customConstituentsExtTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                        src = candInput,
                                                        cut = cms.string(""), #we should not filter after pruning
                                                        name = cms.string("PFCands"),
                                                        doc = cms.string("interesting particles from AK4 and AK8 jets"),
                                                        singleton = cms.bool(False), # the number of entries is variable
                                                        extension = cms.bool(False), # this is the extension table for the AK8 constituents
                                                        variables = cms.PSet(CandVars,
                                                            puppiWeight = Var("puppiWeight()", float, doc="Puppi weight",precision=10),
                                                            puppiWeightNoLep = Var("puppiWeightNoLep()", float, doc="Puppi weight removing leptons",precision=10),
                                                            vtxChi2 = Var("?hasTrackDetails()?vertexChi2():-1", float, doc="vertex chi2",precision=10),
                                                            trkChi2 = Var("?hasTrackDetails()?pseudoTrack().normalizedChi2():-1", float, doc="normalized trk chi2", precision=10),
                                                            dz = Var("?hasTrackDetails()?dz():-1", float, doc="pf dz", precision=10),
                                                            dzErr = Var("?hasTrackDetails()?dzError():-1", float, doc="pf dz err", precision=10),
                                                            d0 = Var("?hasTrackDetails()?dxy():-1", float, doc="pf d0", precision=10),
                                                            d0Err = Var("?hasTrackDetails()?dxyError():-1", float, doc="pf d0 err", precision=10),
                                                            pvAssocQuality = Var("pvAssociationQuality()", int, doc="primary vertex association quality. 0: NotReconstructedPrimary, 1: OtherDeltaZ, 4: CompatibilityBTag, 5: CompatibilityDz, 6: UsedInFitLoose, 7: UsedInFitTight"),
                                                            lostInnerHits = Var("lostInnerHits()", int, doc="lost inner hits. -1: validHitInFirstPixelBarrelLayer, 0: noLostInnerHits, 1: oneLostInnerHit, 2: moreLostInnerHits"),
                                                            lostOuterHits = Var("?hasTrackDetails()?pseudoTrack().hitPattern().numberOfLostHits('MISSING_OUTER_HITS'):0", int, doc="lost outer hits"),
                                                            numberOfHits = Var("numberOfHits()", int, doc="number of hits"),
                                                            numberOfPixelHits = Var("numberOfPixelHits()", int, doc="number of pixel hits"),
                                                            trkQuality = Var("?hasTrackDetails()?pseudoTrack().qualityMask():0", int, doc="track quality mask"),
                                                            trkHighPurity = Var("?hasTrackDetails()?pseudoTrack().quality('highPurity'):0", bool, doc="track is high purity"),
                                                            trkAlgo = Var("?hasTrackDetails()?pseudoTrack().algo():-1", int, doc="track algorithm"),
                                                            trkP = Var("?hasTrackDetails()?pseudoTrack().p():-1", float, doc="track momemtum", precision=-1),
                                                            trkPt = Var("?hasTrackDetails()?pseudoTrack().pt():-1", float, doc="track pt", precision=-1),
                                                            trkEta = Var("?hasTrackDetails()?pseudoTrack().eta():-1", float, doc="track pt", precision=12),
                                                            trkPhi = Var("?hasTrackDetails()?pseudoTrack().phi():-1", float, doc="track phi", precision=12),
                                                         )
                                    )
    process.customAK8ConstituentsTable = cms.EDProducer("PatJetConstituentTableProducer",
                                                        candidates = candInput,
                                                        jets = cms.InputTag("finalJetsAK8"),
                                                        jet_radius = cms.double(0.8),
                                                        name = cms.string("FatJetPFCands"),
                                                        idx_name = cms.string("pFCandsIdx"),
                                                        nameSV = cms.string("FatJetSVs"),
                                                        idx_nameSV = cms.string("sVIdx"),
                                                        )
    process.customAK4ConstituentsTable = cms.EDProducer("PatJetConstituentTableProducer",
                                                        candidates = candInput,
                                                        jets = cms.InputTag("finalJetsPuppi"), # was finalJets before
                                                        jet_radius = cms.double(0.4),
                                                        name = cms.string("JetPFCands"),
                                                        idx_name = cms.string("pFCandsIdx"),
                                                        nameSV = cms.string("JetSVs"),
                                                        idx_nameSV = cms.string("sVIdx"),
                                                        )
    if not allPF:
        process.customizedPFCandsTask.add(process.finalJetsConstituents)
    process.customizedPFCandsTask.add(process.customConstituentsExtTable)
    # linkedObjects are WIP for Run3
    process.customizedPFCandsTask.add(process.customAK8ConstituentsTable)
    process.customizedPFCandsTask.add(process.customAK4ConstituentsTable)
    
    if runOnMC:

        process.genJetsAK8Constituents = cms.EDProducer("GenJetPackedConstituentPtrSelector",
                                                    src = cms.InputTag("slimmedGenJetsAK8"),
                                                    cut = cms.string("pt > 100.")
                                                    )

      
        process.genJetsAK4Constituents = process.genJetsAK8Constituents.clone(
                                                    src = cms.InputTag("slimmedGenJets"),
                                                    cut = cms.string("pt > 20")
                                                    )
        if allPF:
            genCandInput = cms.InputTag("packedGenParticles")
        elif addAK4 and not addAK8:
            genCandList = cms.VInputTag(cms.InputTag("genJetsAK4Constituents", "constituents"))
            genCandInput =  cms.InputTag("genJetsConstituents")
            process.genJetsConstituents = cms.EDProducer("PackedGenParticlePtrMerger", src = genCandList, skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        elif addAK8 and not addAK4:
            genCandList = cms.VInputTag(cms.InputTag("genJetsAK8Constituents", "constituents"))
            genCandInput =  cms.InputTag("genJetsConstituents")
            process.genJetsConstituents = cms.EDProducer("PackedGenParticlePtrMerger", src = genCandList, skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        else:
            genCandList = cms.VInputTag(cms.InputTag("genJetsAK4Constituents", "constituents"), cms.InputTag("genJetsAK8Constituents", "constituents"))
            genCandInput =  cms.InputTag("genJetsConstituents")
            process.genJetsConstituents = cms.EDProducer("PackedGenParticlePtrMerger", src = genCandList, skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        process.genJetsParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                         src = genCandInput,
                                                         cut = cms.string(""), #we should not filter after pruning
                                                         name= cms.string("GenCands"),
                                                         doc = cms.string("interesting gen particles from AK4 and AK8 jets"),
                                                         singleton = cms.bool(False), # the number of entries is variable
                                                         extension = cms.bool(False), # this is the main table for the AK8 constituents
                                                         variables = cms.PSet(CandVars
                                                                          )
                                                     )
        process.genAK8ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                         candidates = genCandInput,
                                                         jets = cms.InputTag("genJetsAK8Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                         name = cms.string("GenFatJetCands"),
                                                         nameSV = cms.string("GenFatJetSVs"),
                                                         idx_name = cms.string("pFCandsIdx"),
                                                         idx_nameSV = cms.string("sVIdx"),
                                                         readBtag = cms.bool(False))
        process.genAK4ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                         candidates = genCandInput,
                                                         jets = cms.InputTag("genJetsAK4Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                         name = cms.string("GenJetCands"),
                                                         nameSV = cms.string("GenJetSVs"),
                                                         idx_name = cms.string("pFCandsIdx"),
                                                         idx_nameSV = cms.string("sVIdx"),
                                                         readBtag = cms.bool(False))
        process.customizedPFCandsTask.add(process.genJetsAK4Constituents) #Note: For gen need to add jets to the process to keep pt cuts.
        process.customizedPFCandsTask.add(process.genJetsAK8Constituents)
        if not allPF:
            process.customizedPFCandsTask.add(process.genJetsConstituents)
        process.customizedPFCandsTask.add(process.genJetsParticleTable)
        process.customizedPFCandsTask.add(process.genAK8ConstituentsTable)
        process.customizedPFCandsTask.add(process.genAK4ConstituentsTable)
        
    return process

## Switches for BTV nano
# Default(store SFs PFCands+TaggerInputs) for both AK4 & AK8 jets
# nanoAOD_addbtagAK4_switch, nanoAOD_addbtagAK8_switch True, nanoAOD_allPF_switch  False

nanoAOD_allPF_switch = False  # Add all PF candidates, use for training
nanoAOD_addbtagAK4_switch = True # AK4 SFs vars
nanoAOD_addbtagAK8_switch = False # AK8 SFs vars

def PrepBTVCustomNanoAOD_MC(process):    
    addPFCands(process, True, nanoAOD_allPF_switch,nanoAOD_addbtagAK4_switch,nanoAOD_addbtagAK8_switch)
    add_BTV(process, True,nanoAOD_addbtagAK4_switch,nanoAOD_addbtagAK8_switch)
    return process
def PrepBTVCustomNanoAOD_DATA(process):
    # only run if SFvar or allPFCands
    addPFCands(process, False, nanoAOD_allPF_switch,nanoAOD_addbtagAK4_switch,nanoAOD_addbtagAK8_switch)
    add_BTV(process, False,nanoAOD_addbtagAK4_switch,nanoAOD_addbtagAK8_switch)
    return process