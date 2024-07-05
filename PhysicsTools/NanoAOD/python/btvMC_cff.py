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
    ]

btvGenTable =  cms.EDProducer(
        "SimpleGenParticleFlatTableProducer",
        src=finalGenParticles.src,
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
    
allPFParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("packedGenParticles"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("interesting gen particles from PF candidates"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the AK8 constituents
                                                     variables = cms.PSet(CandVars
                                                                      )
                                                 )
ak4onlygenJetsParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("ak4onlygenJetsConstituents"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("interesting gen particles from AK4 jets"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the AK8 constituents
                                                     variables = cms.PSet(CandVars
                                                                      )
                                                 )
ak8onlygenJetsParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("ak8onlygenJetsConstituents"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("interesting gen particles from AK8 jets"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the AK8 constituents
                                                     variables = cms.PSet(CandVars
                                                                      )
                                                 )
ak4ak8genJetsParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
                                                     src = cms.InputTag("ak4ak8genJetsConstituents"),
                                                     cut = cms.string(""), #we should not filter after pruning
                                                     name= cms.string("GenCands"),
                                                     doc = cms.string("interesting gen particles from AK4, AK8 jets"),
                                                     singleton = cms.bool(False), # the number of entries is variable
                                                     extension = cms.bool(False), # this is the main table for the AK8 constituents
                                                     variables = cms.PSet(CandVars
                                                                      )
                                                 )
ak8onlygenAK8ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak8onlygenJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK8Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenFatJetCands"),
                                                     nameSV = cms.string("GenFatJetSVs"),
                                                     idx_name = cms.string("pFCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))
ak4onlygenAK4ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak4onlygenJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK4Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenJetCands"),
                                                     nameSV = cms.string("GenJetSVs"),
                                                     idx_name = cms.string("pFCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))
ak4ak8genAK4ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak4ak8genJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK4Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenJetCands"),
                                                     nameSV = cms.string("GenJetSVs"),
                                                     idx_name = cms.string("pFCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))

ak4ak8genAK8ConstituentsTable = cms.EDProducer("GenJetConstituentTableProducer",
                                                     candidates = cms.InputTag("ak4ak8genJetsConstituents"),
                                                     jets = cms.InputTag("genJetsAK8Constituents"), # Note: The name has "Constituents" in it, but these are the jets
                                                     name = cms.string("GenFatJetCands"),
                                                     nameSV = cms.string("GenFatJetSVs"),
                                                     idx_name = cms.string("pFCandsIdx"),
                                                     idx_nameSV = cms.string("sVIdx"),
                                                     readBtag = cms.bool(False))
btvAK4MCSequence = cms.Sequence(btvGenTable+btvAK4JetExtTable+btvMCTable)
btvAK8MCSequence = cms.Sequence(btvGenTable+btvSubJetMCExtTable)
#PF Cands of AK4 only , with cross linking to AK4 jets
ak4onlyPFCandsMCSequence=cms.Sequence(genJetsAK4Constituents+ak4onlygenJetsConstituents+ak4onlygenJetsParticleTable+ak4onlygenAK4ConstituentsTable)+btvAK4MCSequence
#PF Cands of AK8 only , with cross linking to AK8 jets
ak8onlyPFCandsMCSequence=cms.Sequence(genJetsAK8Constituents+ak8onlygenJetsConstituents+ak8onlygenJetsParticleTable+ak8onlygenAK8ConstituentsTable)+btvAK8MCSequence
#PF Cands of AK4+AK8, with cross linking to AK4,AK8 jets
ak4ak8PFCandsMCSequence=cms.Sequence(genJetsAK4Constituents+genJetsAK8Constituents+ak4ak8genJetsConstituents+ak4ak8genJetsParticleTable+ak4ak8genAK4ConstituentsTable+ak4ak8genAK8ConstituentsTable)+btvAK4MCSequence+btvAK8MCSequence
#All PFCands, with cross linking to AK4,AK8 jets
allPFPFCandsMCSequence=cms.Sequence(genJetsAK4Constituents+genJetsAK8Constituents+ak4ak8genJetsConstituents+allPFParticleTable+ak4ak8genAK4ConstituentsTable+ak4ak8genAK8ConstituentsTable)+btvAK4MCSequence+btvAK8MCSequence




