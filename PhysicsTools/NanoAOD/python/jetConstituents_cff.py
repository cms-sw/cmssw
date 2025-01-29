import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.jetsAK8_cff import fatJetTable as _fatJetTable

##############################################################
# Take AK8 jets and collect their PF constituents
###############################################################
finalJetsAK8PFConstituents = cms.EDProducer("PatJetConstituentPtrSelector",
    src = _fatJetTable.src,
    cut = cms.string("abs(eta) <= 2.5")
)

selectedFinalJetsAK8PFConstituents = cms.EDFilter("PATPackedCandidatePtrSelector",
    src = cms.InputTag("finalJetsAK8PFConstituents", "constituents"),
    cut = cms.string("")
)

##############################################################
# Setup PF candidates table
##############################################################
finalPFCandidates = cms.EDProducer("PackedCandidatePtrMerger",
    src = cms.VInputTag(cms.InputTag("selectedFinalJetsAK8PFConstituents")),
    skipNulls = cms.bool(True),
    warnOnSkip = cms.bool(True)
)

pfCandidatesTable = cms.EDProducer("SimplePATCandidateFlatTableProducer",
    src = cms.InputTag("finalPFCandidates"),
    cut = cms.string(""),
    name = cms.string("PFCand"),
    doc = cms.string("PF candidate constituents of AK8 puppi jets (FatJet) with |eta| <= 2.5"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt * puppiWeight()", float, doc="Puppi-weighted pt", precision=10),
        mass = Var("mass * puppiWeight()", float, doc="Puppi-weighted mass", precision=10),
        eta = Var("eta", float, precision=12),
        phi = Var("phi", float, precision=12),
        pdgId = Var("pdgId", int, doc="PF candidate type (+/-211 = ChgHad, 130 = NeuHad, 22 = Photon, +/-11 = Electron, +/-13 = Muon, 1 = HFHad, 2 = HFEM)")
  )
)

##############################################################
# Setup AK8 jet constituents table
##############################################################
finalJetsAK8ConstituentsTable = cms.EDProducer("SimplePatJetConstituentTableProducer",
  name = cms.string(_fatJetTable.name.value()+"PFCand"),
  candIdxName = cms.string("pfCandIdx"),
  candIdxDoc = cms.string("Index in the PFCand table"),
  candidates = pfCandidatesTable.src,
  jets = _fatJetTable.src,
  jetCut = _fatJetTable.cut,
  jetConstCut = selectedFinalJetsAK8PFConstituents.cut
)

jetConstituentsTask = cms.Task(finalJetsAK8PFConstituents,selectedFinalJetsAK8PFConstituents)
jetConstituentsTablesTask = cms.Task(finalPFCandidates,pfCandidatesTable,finalJetsAK8ConstituentsTable)


def SaveAK4JetConstituents(process, jetCut="", jetConstCut=""):
    """
    This function can be used as a cmsDriver customization
    function to add AK4 jet constituents, on top of the AK8
    jet constituents.
    """
    process.finalJetsPuppiPFConstituents = process.finalJetsAK8PFConstituents.clone(
        src = process.jetPuppiTable.src,
        cut = jetCut
    )
    process.jetConstituentsTask.add(process.finalJetsPuppiPFConstituents)

    process.selectedFinalJetsPuppiPFConstituents = process.selectedFinalJetsAK8PFConstituents.clone(
        src = cms.InputTag("finalJetsPuppiPFConstituents", "constituents"),
        cut = jetConstCut
    )
    process.jetConstituentsTask.add(process.selectedFinalJetsPuppiPFConstituents)

    process.finalPFCandidates.src += ["selectedFinalJetsPuppiPFConstituents"]
    process.pfCandidatesTable.doc = pfCandidatesTable.doc.value()+" and AK4 puppi jets (Jet)"

    process.finalJetsPuppiConstituentsTable = process.finalJetsAK8ConstituentsTable.clone(
        name = process.jetPuppiTable.name.value()+"PFCand",
        jets = process.jetPuppiTable.src,
        jetCut = process.jetPuppiTable.cut,
        jetConstCut = process.selectedFinalJetsPuppiPFConstituents.cut
    )
    process.jetConstituentsTablesTask.add(process.finalJetsPuppiConstituentsTable)

    return process

def SaveGenJetConstituents(process, addGenJetConst, addGenJetAK8Const, genJetConstCut="",genJetAK8ConstCut=""):
    """
    This function can be used as a cmsDriver
    customization function to add gen jet
    constituents.
    """
    process.genjetConstituentsTask = cms.Task()
    process.genjetConstituentsTableTask = cms.Task()

    if addGenJetConst:
        process.genJetConstituents = cms.EDProducer("GenJetPackedConstituentPtrSelector",
            src = process.genJetTable.src,
            cut = process.genJetTable.cut,
        )
        process.genjetConstituentsTask.add(process.genJetConstituents)

        process.selectedGenJetConstituents = cms.EDFilter("PATPackedGenParticlePtrSelector",
            src = cms.InputTag("genJetConstituents", "constituents"),
            cut = cms.string(genJetConstCut)
        )
        process.genjetConstituentsTask.add(process.selectedGenJetConstituents)

    if addGenJetAK8Const:
        process.genJetAK8Constituents = cms.EDProducer("GenJetPackedConstituentPtrSelector",
            src = process.genJetAK8Table.src,
            cut = process.genJetAK8Table.cut,
        )
        process.genjetConstituentsTask.add(process.genJetAK8Constituents)

        process.selectedGenJetAK8Constituents = cms.EDFilter("PATPackedGenParticlePtrSelector",
            src = cms.InputTag("genJetAK8Constituents", "constituents"),
            cut = cms.string(genJetConstCut)
        )
        process.genjetConstituentsTask.add(process.selectedGenJetAK8Constituents)

    if addGenJetConst or addGenJetConst:
        process.finalGenPartCandidates = cms.EDProducer("PackedGenParticlePtrMerger",
            src = cms.VInputTag(),
            skipNulls = cms.bool(True),
            warnOnSkip = cms.bool(True)
        )
        process.genjetConstituentsTableTask.add(process.finalGenPartCandidates)

        process.genPartCandidatesTable = cms.EDProducer("SimplePATGenParticleFlatTableProducer",
            src = cms.InputTag("finalGenPartCandidates"),
            cut = cms.string(""),
            name = cms.string("GenPartCand"),
            doc = cms.string("Gen particle constituents:"),
            singleton = cms.bool(False),
            extension = cms.bool(False),
            variables = cms.PSet(P4Vars,
                pdgId = Var("pdgId", int, doc="pdgId")
          )
        )
        process.genjetConstituentsTableTask.add(process.genPartCandidatesTable)
        process.genPartCandidatesTable.variables.pt.precision=10
        process.genPartCandidatesTable.variables.mass.precision=10

        if addGenJetConst:
            process.finalGenPartCandidates.src += ["selectedGenJetConstituents"]
            process.genPartCandidatesTable.doc = process.genPartCandidatesTable.doc.value()+" AK4 Gen jets (GenJet) "

            process.genJetConstituentsTable = cms.EDProducer("SimpleGenJetConstituentTableProducer",
              name = cms.string(process.genJetTable.name.value()+"GenPartCand"),
              candIdxName = cms.string("genPartCandIdx"),
              candIdxDoc = cms.string("Index in the GenPartCand table"),
              candidates = pfCandidatesTable.src,
              jets = process.genJetTable.src,
              jetCut = process.genJetTable.cut,
              jetConstCut = process.selectedGenJetConstituents.cut
            )
            process.genjetConstituentsTableTask.add(process.genJetConstituentsTable)

        if addGenJetAK8Const:
            process.finalGenPartCandidates.src += ["selectedGenJetAK8Constituents"]
            process.genPartCandidatesTable.doc = process.genPartCandidatesTable.doc.value()+" AK8 Gen jets (GenJetAK8)"

            process.genJetAK8ConstituentsTable = cms.EDProducer("SimpleGenJetConstituentTableProducer",
              name = cms.string(process.genJetAK8Table.name.value()+"GenPartCand"),
              candIdxName = cms.string("genPartCandIdx"),
              candIdxDoc = cms.string("Index in the GenPartCand table"),
              candidates = pfCandidatesTable.src,
              jets = process.genJetAK8Table.src,
              jetCut = process.genJetAK8Table.cut,
              jetConstCut = process.selectedGenJetConstituents.cut
            )
            process.genjetConstituentsTableTask.add(process.genJetAK8ConstituentsTable)

        process.nanoTableTaskFS.add(process.genjetConstituentsTask)
        process.nanoTableTaskFS.add(process.genjetConstituentsTableTask)

    return process

def SaveGenJetAK4Constituents(process):
    process = SaveGenJetConstituents(process,True,False)
    return process
def SaveGenJetAK8Constituents(process):
    process = SaveGenJetConstituents(process,True,False)
    return process
def SaveGenJetAK4AK8Constituents(process):
    process = SaveGenJetConstituents(process,True,True)
    return process

