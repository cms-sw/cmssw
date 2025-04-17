import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer
from PhysicsTools.NanoAOD.jetsAK8_cff import fatJetTable, subJetTable
from PhysicsTools.NanoAOD.jetsAK4_Puppi_cff import jetPuppiTable
from PhysicsTools.NanoAOD.genparticles_cff import *
# when storing the flat table, always do "T"able in the naming convention

def addGenCands(process, allPF = False, addAK4=False, addAK8=False):
    process.btvGenTask = cms.Task()
    process.schedule.associate(process.btvGenTask)

    process.finalGenParticles.select +=[
            "keep (4 <= abs(pdgId) <= 5) && statusFlags().isLastCopy()", # BTV: keep b/c quarks in their last copy
            "keep (abs(pdgId) == 310 || abs(pdgId) == 3122) && statusFlags().isLastCopy()", # BTV: keep K0s and Lambdas in their last copy
            "++keep (400 < abs(pdgId) < 600) || (4000 < abs(pdgId) < 6000)", # keep all B and C hadron + their decay products
        ]

    process.btvGenTable =  cms.EDProducer(
        "SimpleGenParticleFlatTableProducer",
        src = cms.InputTag("finalGenParticles"),
        name= cms.string("GenPart"),
        doc = cms.string("interesting gen particles "),
        singleton=cms.bool(False),
        variables=
        cms.PSet(
            genParticleTable.variables,
        vx = Var("vx", "float", doc="x coordinate of vertex position"),
        vy = Var("vy", "float", doc="y coordinate of vertex position"),
        vz = Var("vz", "float", doc="z coordinate of vertex position"),
        genPartIdxMother2 = Var("?numberOfMothers>1?motherRef(1).key():-1", "int", doc="index of the second mother particle, if valid")
        )
    )
    process.btvGenTask.add(process.btvGenTable)
    process.genParticleTablesTask.replace(process.genParticleTable,process.btvGenTable)

    if addAK4:
        process.btvMCTable = cms.EDProducer("BTVMCFlavourTableProducer",name=jetPuppiTable.name,src=cms.InputTag("linkedObjects","jets"),genparticles=cms.InputTag("prunedGenParticles"))
        process.btvGenTask.add(process.btvMCTable)

        process.btvAK4JetExtTable = cms.EDProducer(
            "SimplePATJetFlatTableProducer",
            src=jetPuppiTable.src,
            cut=jetPuppiTable.cut,
            name=jetPuppiTable.name,
            doc=jetPuppiTable.doc,
            singleton=cms.bool(False),  # the number of entries is variable
            extension=cms.bool(True),  # this is the extension table for Jets
            variables=cms.PSet(
            nBHadrons=Var("jetFlavourInfo().getbHadrons().size()",
                        int, doc="number of b-hadrons"),
            nCHadrons=Var("jetFlavourInfo().getcHadrons().size()",
                        int, doc="number of c-hadrons"),
            )
        )
        process.btvGenTask.add(process.btvAK4JetExtTable)

    if addAK8:
        process.btvSubJetMCExtTable = cms.EDProducer(
            "SimplePATJetFlatTableProducer",
            src = subJetTable.src,
            cut = subJetTable.cut,
            name = subJetTable.name,
            doc=subJetTable.doc,
            singleton = cms.bool(False),
            extension = cms.bool(True),
            variables = cms.PSet(
                subGenJetAK8Idx = Var("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().key():-1",
                int, doc="index of matched gen Sub jet"),
            )
        )
        process.btvGenTask.add(process.btvSubJetMCExtTable)

    if addAK4:
        process.genJetsAK4Constituents = cms.EDProducer("GenJetPackedConstituentPtrSelector",
            src = cms.InputTag("slimmedGenJets"),
            cut = cms.string("pt > 20")
        )
        process.btvGenTask.add(process.genJetsAK4Constituents)
    if addAK8:
        process.genJetsAK8Constituents = cms.EDProducer("GenJetPackedConstituentPtrSelector",
            src = cms.InputTag("slimmedGenJetsAK8"),
            cut = cms.string("pt > 100.")
        )
        process.btvGenTask.add(process.genJetsAK8Constituents)

    if allPF:
        genCandInput = cms.InputTag("packedGenParticles")
    elif not addAK8:
        process.genJetsConstituentsTable = cms.EDProducer("PackedGenParticlePtrMerger", src = cms.VInputTag(cms.InputTag("genJetsAK4Constituents", "constituents")), skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        genCandInput = cms.InputTag("genJetsConstituentsTable")
    elif not addAK4:
        process.genJetsConstituentsTable = cms.EDProducer("PackedGenParticlePtrMerger", src = cms.VInputTag(cms.InputTag("genJetsAK8Constituents", "constituents")), skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        genCandInput = cms.InputTag("genJetsConstituentsTable")
    else:
        process.genJetsConstituentsTable = cms.EDProducer("PackedGenParticlePtrMerger", src = cms.VInputTag(cms.InputTag("genJetsAK4Constituents", "constituents"), cms.InputTag("genJetsAK8Constituents", "constituents")), skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
        genCandInput = cms.InputTag("genJetsConstituentsTable")
    if not allPF:
        process.btvGenTask.add(process.genJetsConstituentsTable)

    GenCandVars = CandVars.clone()
    GenCandVars.pdgId.doc = cms.string("PDG id")

    process.genCandsTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
        src = genCandInput,
        cut = cms.string(""), #we should not filter after pruning
        name= cms.string("GenCands"),
        doc = cms.string("Final-state gen particles"),
        singleton = cms.bool(False), # the number of entries is variable
        extension = cms.bool(False), # this is the main table for the GEN constituents
        variables = cms.PSet(GenCandVars)
    )
    process.btvGenTask.add(process.genCandsTable)

    kwargs = { }
    import os
    sv_sort = os.getenv('CMSSW_NANOAOD_SV_SORT')
    if sv_sort is not None: kwargs['sv_sort'] = cms.untracked.string(sv_sort)
    pf_sort = os.getenv('CMSSW_NANOAOD_PF_SORT')
    if pf_sort is not None: kwargs['pf_sort'] = cms.untracked.string(pf_sort)

    if addAK4:
        process.genAK4ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
            candidates = genCandInput,
            jets = cms.InputTag("genJetsAK4Constituents"), # Note: The name has "Constituents" in it, but these are the jets
            name = cms.string("GenJetCands"),
            nameSV = cms.string("GenJetSVs"),
            idx_name = cms.string("genCandsIdx"),
            idx_nameSV = cms.string("sVIdx"),
            readBtag = cms.bool(False),
            **kwargs,
        )
        process.btvGenTask.add(process.genAK4ConstituentsTable)
    if addAK8:
        process.genAK8ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
            candidates = genCandInput,
            jets = cms.InputTag("genJetsAK8Constituents"), # Note: The name has "Constituents" in it, but these are the jets
            name = cms.string("GenFatJetCands"),
            nameSV = cms.string("GenFatJetSVs"),
            idx_name = cms.string("genCandsIdx"),
            idx_nameSV = cms.string("sVIdx"),
            readBtag = cms.bool(False),
            **kwargs,
        )
        process.btvGenTask.add(process.genAK8ConstituentsTable)

    process.genCandMotherTable = cms.EDProducer("GenCandMotherTableProducer",
        src     = genCandInput,
        mcMap   = cms.InputTag("finalGenParticles"),
        objName = cms.string("GenCands"),
        branchName = cms.string("genPart"),
    )
    process.btvGenTask.add(process.genCandMotherTable)

    if allPF:
        pfCandInput = cms.InputTag("packedPFCandidates")
    else:
        pfCandInput = cms.InputTag("finalJetsConstituentsTable")

    process.genCandMCMatchTable = cms.EDProducer("PackedCandMCMatchTableProducer",
        src = pfCandInput,
        genparticles = genCandInput,
        objName = cms.string("PFCands"),
        branchName = cms.string("genCand"),
        docString = cms.string("MC matching to status==1 genCands"),
    )
    process.btvGenTask.add(process.genCandMCMatchTable)

    return process