import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer
from PhysicsTools.NanoAOD.jetsAK8_cff import fatJetTable, subJetTable
from PhysicsTools.NanoAOD.jetsAK4_Puppi_cff import jetPuppiTable
from PhysicsTools.NanoAOD.genparticles_cff import *
# when storing the flat table, always do "T"able in the naming convention


finalGenParticles.select +=[
        "keep (4 <= abs(pdgId) <= 5) && statusFlags().isLastCopy()", # BTV: keep b/c quarks in their last copy
        "keep (abs(pdgId) == 310 || abs(pdgId) == 3122) && statusFlags().isLastCopy()", # BTV: keep K0s and Lambdas in their last copy
        "++keep (400 < abs(pdgId) < 600) || (4000 < abs(pdgId) < 6000)", # keep all B and C hadron + their decay products
    ]

btvGenTable =  cms.EDProducer(
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
        ))
genParticleTablesTask.replace(genParticleTable,btvGenTable)
btvMCTable = cms.EDProducer("BTVMCFlavourTableProducer",name=jetPuppiTable.name,src=cms.InputTag("linkedObjects","jets"),genparticles=cms.InputTag("prunedGenParticles"))

btvAK4JetExtTable = cms.EDProducer(
        "SimplePATJetFlatTableProducer",
        src=jetPuppiTable.src,
        cut=jetPuppiTable.cut,
        name=jetPuppiTable.name,
        doc=jetPuppiTable.doc,
        singleton=cms.bool(False),  # the number of entries is variable
        extension=cms.bool(True),  # this is the extension table for Jets
        variables=cms.PSet(
        nBHadrons=Var("jetFlavourInfo().getbHadrons().size()",
                      int,
                      doc="number of b-hadrons"),
        nCHadrons=Var("jetFlavourInfo().getcHadrons().size()",
                      int,
                      doc="number of c-hadrons"),
        ))

btvSubJetMCExtTable = cms.EDProducer(
    "SimplePATJetFlatTableProducer",
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
genJetsAK8Constituents = cms.EDProducer("GenJetPackedConstituentPtrSelector",
                                                src = cms.InputTag("slimmedGenJetsAK8"),
                                                cut = cms.string("pt > 100.")
                                                )


genJetsAK4Constituents = cms.EDProducer("GenJetPackedConstituentPtrSelector",
                                                src = cms.InputTag("slimmedGenJets"),
                                                cut = cms.string("pt > 20")
                                                )





ak4onlygenJetsConstituents = cms.EDProducer("PackedGenParticlePtrMerger", src = cms.VInputTag(cms.InputTag("genJetsAK4Constituents", "constituents")), skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
ak8onlygenJetsConstituents = cms.EDProducer("PackedGenParticlePtrMerger", src = cms.VInputTag(cms.InputTag("genJetsAK8Constituents", "constituents")), skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))
ak4ak8genJetsConstituents = cms.EDProducer("PackedGenParticlePtrMerger", src = cms.VInputTag(cms.InputTag("genJetsAK4Constituents", "constituents"), cms.InputTag("genJetsAK8Constituents", "constituents")), skipNulls = cms.bool(True), warnOnSkip = cms.bool(True))

GenCandVars = CandVars.clone()
GenCandVars.pdgId.doc = cms.string("PDG id")

allGENParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("packedGenParticles"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("Stable gen particles from the whole event"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the GEN constituents
                                                     variables = cms.PSet(GenCandVars
                                                                      )
                                                 )
ak4onlygenJetsParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("ak4onlygenJetsConstituents"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("Stable gen particles from AK4 jets"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the AK4 constituents
                                                     variables = cms.PSet(GenCandVars
                                                                      )
                                                 )
ak8onlygenJetsParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("ak8onlygenJetsConstituents"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("Stable gen particles from AK8 jets"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the AK8 constituents
                                                     variables = cms.PSet(GenCandVars
                                                                      )
                                                 )
ak4ak8genJetsParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("ak4ak8genJetsConstituents"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("Stable gen particles from AK4, AK8 jets"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the AK4,AK8 constituents
                                                     variables = cms.PSet(GenCandVars
                                                                      )
                                                 )

ak4onlygenAK4ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak4onlygenJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK4Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenJetCands"),
                                                     nameSV = cms.string("GenJetSVs"),
                                                     idx_name = cms.string("genCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))
ak8onlygenAK8ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak8onlygenJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK8Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenFatJetCands"),
                                                     nameSV = cms.string("GenFatJetSVs"),
                                                     idx_name = cms.string("genCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))
ak4ak8genAK4ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak4ak8genJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK4Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenJetCands"),
                                                     nameSV = cms.string("GenJetSVs"),
                                                     idx_name = cms.string("genCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))
ak4ak8genAK8ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak4ak8genJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK8Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenFatJetCands"),
                                                     nameSV = cms.string("GenFatJetSVs"),
                                                     idx_name = cms.string("genCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))

allGenCandMotherTable = cms.EDProducer("GenCandMotherTableProducer",
    src     = cms.InputTag("packedGenParticles"), # FIXME: needs a ptr collection as input
    mcMap   = cms.InputTag("finalGenParticles"),
    objName = cms.string("GenCands"),
    branchName = cms.string("genPart"),
)
ak4GenCandMotherTable = cms.EDProducer("GenCandMotherTableProducer",
    src     = cms.InputTag("ak4onlygenJetsConstituents"),
    mcMap   = cms.InputTag("finalGenParticles"),
    objName = cms.string("GenCands"),
    branchName = cms.string("genPart"),
)
ak8GenCandMotherTable = cms.EDProducer("GenCandMotherTableProducer",
    src     = cms.InputTag("ak8onlygenJetsConstituents"),
    mcMap   = cms.InputTag("finalGenParticles"),
    objName = cms.string("GenCands"),
    branchName = cms.string("genPart"),
)
ak4ak8GenCandMotherTable = cms.EDProducer("GenCandMotherTableProducer",
    src     = cms.InputTag("ak4ak8genJetsConstituents"),
    mcMap   = cms.InputTag("finalGenParticles"),
    objName = cms.string("GenCands"),
    branchName = cms.string("genPart"),
)

allCandMCMatchTable = cms.EDProducer("PackedCandMCMatchTableProducer",
    src = cms.InputTag("packedPFCandidates"),
    genparticles = cms.InputTag("packedGenParticles"), # FIXME: needs a ptr collection as input
    objName = cms.string("PFCands"),
    branchName = cms.string("genCand"),
    docString = cms.string("MC matching to status==1 genCands"),
)
ak4CandMCMatchTable = cms.EDProducer("PackedCandMCMatchTableProducer",
    src = cms.InputTag("finalJetsConstituentsTable"),
    genparticles = cms.InputTag("ak4onlygenJetsConstituents"), # final mc-truth particle collection
    objName = cms.string("PFCands"),
    branchName = cms.string("genCand"),
    docString = cms.string("MC matching to status==1 genCands"),
)
ak8CandMCMatchTable = cms.EDProducer("PackedCandMCMatchTableProducer",
    src = cms.InputTag("finalJetsConstituentsTable"),
    genparticles = cms.InputTag("ak8onlygenJetsConstituents"), # final mc-truth particle collection
    objName = cms.string("PFCands"),
    branchName = cms.string("genCand"),
    docString = cms.string("MC matching to status==1 genCands"),
)
ak4ak8CandMCMatchTable = cms.EDProducer("PackedCandMCMatchTableProducer",
    src = cms.InputTag("finalJetsConstituentsTable")
    genparticles = cms.InputTag("ak4ak8genJetsConstituents"), # final mc-truth particle collection
    objName = cms.string("PFCands"),
    branchName = cms.string("genCand"),
    docString = cms.string("MC matching to status==1 genCands"),
)

# process.genWeightsTable.keepAllPSWeights = True

btvAK4MCSequence = cms.Sequence(btvGenTable+btvAK4JetExtTable+btvMCTable)
btvAK8MCSequence = cms.Sequence(btvGenTable+btvSubJetMCExtTable)
#PF Cands of AK4 only , with cross linking to AK4 jets
ak4onlyPFCandsMCSequence=cms.Sequence(genJetsAK4Constituents+ak4onlygenJetsConstituents+ak4onlygenJetsParticleTable+ak4GenCandMotherTable+ak4CandMCMatchTable+ak4onlygenAK4ConstituentsTable)+btvAK4MCSequence
#PF Cands of AK8 only , with cross linking to AK8 jets
ak8onlyPFCandsMCSequence=cms.Sequence(genJetsAK8Constituents+ak8onlygenJetsConstituents+ak8onlygenJetsParticleTable+ak8GenCandMotherTable+ak8CandMCMatchTable+ak8onlygenAK8ConstituentsTable)+btvAK8MCSequence
#PF Cands of AK4+AK8, with cross linking to AK4,AK8 jets
ak4ak8PFCandsMCSequence=cms.Sequence(genJetsAK4Constituents+genJetsAK8Constituents+ak4ak8genJetsConstituents+ak4ak8genJetsParticleTable+ak4ak8GenCandMotherTable+ak4ak8genAK4ConstituentsTable+ak4ak8genAK8ConstituentsTable)+btvAK4MCSequence+btvAK8MCSequence
#All PFCands, with cross linking to AK4,AK8 jets
allPFCandsMCSequence=cms.Sequence(genJetsAK4Constituents+genJetsAK8Constituents+ak4ak8genJetsConstituents+allGENParticleTable+ak4ak8CandMCMatchTable+ak4ak8genAK4ConstituentsTable+ak4ak8genAK8ConstituentsTable)+btvAK4MCSequence+btvAK8MCSequence




