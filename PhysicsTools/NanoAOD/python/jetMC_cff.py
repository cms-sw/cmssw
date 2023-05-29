import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer
from PhysicsTools.NanoAOD.jetsAK8_cff import fatJetTable as _fatJetTable
from PhysicsTools.NanoAOD.jetsAK8_cff import subJetTable as _subJetTable

jetMCTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","jets"),
    name = cms.string("Jet"),
    extension = cms.bool(True), # this is an extension  table for the jets
    variables = cms.PSet(
        partonFlavour = Var("partonFlavour()", "int16", doc="flavour from parton matching"),
        hadronFlavour = Var("hadronFlavour()", "uint8", doc="flavour from hadron ghost clustering"),
        # cut should follow genJetTable.cut
        genJetIdx = Var("?genJetFwdRef().backRef().isNonnull() && genJetFwdRef().backRef().pt() > 10.?genJetFwdRef().backRef().key():-1", "int16", doc="index of matched gen jet"),
    )
)
genJetTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("slimmedGenJets"),
    cut = cms.string("pt > 10"),
    name = cms.string("GenJet"),
    doc  = cms.string("slimmedGenJets, i.e. ak4 Jets made with visible genparticles"),
    variables = cms.PSet(P4Vars,
	#anything else?
    )
)

patJetPartonsNano = cms.EDProducer('HadronAndPartonSelector',
    src = cms.InputTag("generator"),
    particles = cms.InputTag("prunedGenParticles"),
    partonMode = cms.string("Auto"),
    fullChainPhysPartons = cms.bool(True)
)

genJetFlavourAssociation = cms.EDProducer("JetFlavourClustering",
    jets = genJetTable.src,
    bHadrons = cms.InputTag("patJetPartonsNano","bHadrons"),
    cHadrons = cms.InputTag("patJetPartonsNano","cHadrons"),
    partons = cms.InputTag("patJetPartonsNano","physicsPartons"),
    leptons = cms.InputTag("patJetPartonsNano","leptons"),
    jetAlgorithm = cms.string("AntiKt"),
    rParam = cms.double(0.4),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)

genJetFlavourTable = cms.EDProducer("GenJetFlavourTableProducer",
    name = genJetTable.name,
    src = genJetTable.src,
    cut = genJetTable.cut,
    deltaR = cms.double(0.1),
    jetFlavourInfos = cms.InputTag("slimmedGenJetsFlavourInfos"),
)

genJetAK8Table = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("slimmedGenJetsAK8"),
    cut = cms.string("pt > 100."),
    name = cms.string("GenJetAK8"),
    doc  = cms.string("slimmedGenJetsAK8, i.e. ak8 Jets made with visible genparticles"),
    variables = cms.PSet(P4Vars,
	#anything else?
    )
)

genJetAK8FlavourAssociation = cms.EDProducer("JetFlavourClustering",
    jets = genJetAK8Table.src,
    bHadrons = cms.InputTag("patJetPartonsNano","bHadrons"),
    cHadrons = cms.InputTag("patJetPartonsNano","cHadrons"),
    partons = cms.InputTag("patJetPartonsNano","physicsPartons"),
    leptons = cms.InputTag("patJetPartonsNano","leptons"),
    jetAlgorithm = cms.string("AntiKt"),
    rParam = cms.double(0.8),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)

genJetAK8FlavourTable = cms.EDProducer("GenJetFlavourTableProducer",
    name = genJetAK8Table.name,
    src = genJetAK8Table.src,
    cut = genJetAK8Table.cut,
    deltaR = cms.double(0.1),
    jetFlavourInfos = cms.InputTag("genJetAK8FlavourAssociation"),
)
fatJetMCTable = simpleCandidateFlatTableProducer.clone(
    src = _fatJetTable.src,
    cut = _fatJetTable.cut,
    name = _fatJetTable.name,
    extension = cms.bool(True),
    variables = cms.PSet(
        nBHadrons = Var("jetFlavourInfo().getbHadrons().size()", "uint8", doc="number of b-hadrons"),
        nCHadrons = Var("jetFlavourInfo().getcHadrons().size()", "uint8", doc="number of c-hadrons"),
        hadronFlavour = Var("hadronFlavour()", "uint8", doc="flavour from hadron ghost clustering"),
        # cut should follow genJetAK8Table.cut
        genJetAK8Idx = Var("?genJetFwdRef().backRef().isNonnull() && genJetFwdRef().backRef().pt() > 100.?genJetFwdRef().backRef().key():-1", "int16", doc="index of matched gen AK8 jet"),
    )
)

genSubJetAK8Table = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("slimmedGenJetsAK8SoftDropSubJets"),
    name = cms.string("SubGenJetAK8"),
    doc  = cms.string("slimmedGenJetsAK8SoftDropSubJets, i.e. subjets of ak8 Jets made with visible genparticles"),
    variables = cms.PSet(P4Vars,
	#anything else?
    )
)
subjetMCTable = simpleCandidateFlatTableProducer.clone(
    src = _subJetTable.src,
    cut = _subJetTable.cut,
    name = _subJetTable.name,
    extension = cms.bool(True),
    variables = cms.PSet(
        nBHadrons = Var("jetFlavourInfo().getbHadrons().size()", "uint8", doc="number of b-hadrons"),
        nCHadrons = Var("jetFlavourInfo().getcHadrons().size()", "uint8", doc="number of c-hadrons"),
        hadronFlavour = Var("hadronFlavour()", "uint8", doc="flavour from hadron ghost clustering"),
    )
)


jetMCTaskak4 = cms.Task(jetMCTable,genJetTable,patJetPartonsNano,genJetFlavourTable)
jetMCTaskak8 = cms.Task(genJetAK8Table,genJetAK8FlavourAssociation,genJetAK8FlavourTable,fatJetMCTable,genSubJetAK8Table,subjetMCTable)
jetMCTask = jetMCTaskak4.copyAndAdd(jetMCTaskak8)
