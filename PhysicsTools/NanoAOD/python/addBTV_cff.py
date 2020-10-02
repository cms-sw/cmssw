import FWCore.ParameterSet.Config as cms
# from  PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.common_cff import Var
from PhysicsTools.NanoAOD.jets_cff import jetTable, fatJetTable, subJetTable
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection


def update_jets_AK4(process):
    # Based on ``nanoAOD_addDeepInfo``
    # in https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/nano_cff.py
    _btagDiscriminators = [
        'pfJetProbabilityBJetTags',
        'pfDeepCSVJetTags:probb',
        'pfDeepCSVJetTags:probc',
        'pfDeepCSVJetTags:probbb',
        'pfDeepCSVJetTags:probudsg',
    ]
    updateJetCollection(
        process,
        jetSource=cms.InputTag('slimmedJets'),
        jetCorrections=('AK4PFchs',
                        cms.vstring(
                            ['L1FastJet', 'L2Relative', 'L3Absolute',
                             'L2L3Residual']), 'None'),
        btagDiscriminators=_btagDiscriminators,
        postfix='WithDeepInfo',
    )
    process.load("Configuration.StandardSequences.MagneticField_cff")
    process.jetCorrFactorsNano.src = "selectedUpdatedPatJetsWithDeepInfo"
    process.updatedJets.jetSource = "selectedUpdatedPatJetsWithDeepInfo"
    return process


def update_jets_AK8(process):
    # Based on ``nanoAOD_addDeepInfoAK8``
    # in https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/nano_cff.py
    _btagDiscriminators = [
        'pfJetProbabilityBJetTags',
        'pfDeepCSVJetTags:probb',
        'pfDeepCSVJetTags:probc',
        'pfDeepCSVJetTags:probbb',
        'pfDeepCSVJetTags:probudsg',
        'pfMassIndependentDeepDoubleBvLV2JetTags:probHbb',
        'pfMassIndependentDeepDoubleCvLV2JetTags:probHcc',
        'pfMassIndependentDeepDoubleCvBV2JetTags:probHcc',
        ]
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
        postfix='AK8SubjetsWithDeepInfo')

    process.subJetTable.src = 'updatedPatJetsTransientCorrectedSoftDropSubjetsPFAK8SubjetsWithDeepInfo'  ### VERY LONG NAME!!! :P
    return process


def get_DDX_vars():
    # retreive 27 jet-level features used in double-b and deep double-x taggers
    # defiend in arXiv:1712.07158

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


def addBTV(process, runOnMC=False, onlyAK4=False, onlyAK8=False):
    addAK4 = not onlyAK8
    addAK8 = not onlyAK4

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
        nBHadrons=Var("jetFlavourInfo().getbHadrons().size()",
                      int,
                      doc="number of b-hadrons"),
        nCHadrons=Var("jetFlavourInfo().getcHadrons().size()",
                      int,
                      doc="number of c-hadrons"),
        btagDeepB_b=Var("bDiscriminator('pfDeepCSVJetTags:probb')",
                        float,
                        doc="DeepCSV b tag discriminator",
                        precision=10),
        btagDeepB_bb=Var("bDiscriminator('pfDeepCSVJetTags:probbb')",
                         float,
                         doc="DeepCSV bb tag discriminator",
                         precision=10),
        btagDeepL=Var("bDiscriminator('pfDeepCSVJetTags:probudsg')",
                      float,
                      doc="DeepCSV light btag discriminator",
                      precision=10),
    )

    # AK4
    process.customJetExtTable = cms.EDProducer(
        "SimpleCandidateFlatTableProducer",
        src=jetTable.src,
        cut=jetTable.cut,
        name=jetTable.name,
        doc=jetTable.doc,
        singleton=cms.bool(False),  # the number of entries is variable
        extension=cms.bool(True),  # this is the extension table for Jets
        variables=cms.PSet(
            CommonVars,
        ))

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
            cms.PSet(
                btagDDBvLV2 = Var("bDiscriminator('pfMassIndependentDeepDoubleBvLV2JetTags:probHbb')",float,doc="DeepDoubleX V2 discriminator for H(Z)->bb vs QCD",precision=10),
                btagDDCvLV2 = Var("bDiscriminator('pfMassIndependentDeepDoubleCvLV2JetTags:probHcc')",float,doc="DeepDoubleX V2 discriminator for H(Z)->cc vs QCD",precision=10),
                btagDDCvBV2 = Var("bDiscriminator('pfMassIndependentDeepDoubleCvBV2JetTags:probHcc')",float,doc="DeepDoubleX V2 discriminator for H(Z)->cc vs H(Z)->bb",precision=10),
            ),
            get_DDX_vars(),
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
            # Proba=Var("bDiscriminator('pfJetProbabilityBJetTags')",
            #           float,
            #           doc="Jet Probability (Usage:BTV)",
            #           precision=10),
             btagDeepC = Var("bDiscriminator('pfDeepCSVJetTags:probc')",
                        float,
                        doc="DeepCSV charm btag discriminator",
                        precision=10),

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
    if addAK8:
        process.customizeJetTask.add(process.customFatJetExtTable)
        process.customizeJetTask.add(process.customSubJetExtTable)
    if runOnMC and addAK8:
        process.customizeJetTask.add(process.customSubJetMCExtTable)

    return process
