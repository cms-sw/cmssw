#!/usr/bin/env python3
"""Member parity check for pat::GenJet.

Uses ROOT/cling reflection of the built dictionaries (no text parsing).
pat::GenJet keeps the pat::Jet interface except for the members that have no generator meaning:
  (a) every member in REMOVED (b tagging, calo tower / PF / JPT specifics, flavour, jet ID, self link and the reco-jet ->
      pat::GenJet link, associated tracks,
      HF/HO quantities incl. HF multiplicities, calo-tower quantities, ...) must not be declared by pat::io_v1::GenJet;
  (b) every other public member name of pat::io_v1::Jet and of PATObject<reco::io_v1::Jet> must exist
      as a public member (declared or inherited) of pat::io_v1::GenJet (the "kept" set).
      Constructor/destructor names are mapped Jet -> GenJet. The only allowed exception
      is the scouting constructor Jet(const Run3ScoutingPFJet&), which is checked to be absent;
  (c) the generator constituent members setGenCandidates, genCandidates, genCandidatesFwdPtr and
      updateFwdGenCandidateFwdPtr (the counterparts of the pat::Jet PF candidate members) must exist.
Member templates (userData<T>, addUserData<T>) and the free operator<< are invisible to reflection;
they are checked by compiling a snippet.

Informational: argument lists of kept pat::Jet members that pat::GenJet has under the same
name but with different argument types (expected: GenJet types instead of Jet ones).

Run under cmsenv:  python3 patGenJet_check_member_parity.py   (exit code 0 = pass)
"""
import sys
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
for lib in ("libFWCoreFWLite.so", "libDataFormatsPatCandidates.so"):
    if ROOT.gSystem.Load(lib) < 0:
        print(f"FAIL: cannot load {lib}")
        sys.exit(1)
ROOT.FWLiteEnabler.enable()


def public_methods(cname, inherited):
    cl = ROOT.TClass.GetClass(cname)
    if not cl or not cl.HasDictionary():
        print(f"FAIL: no dictionary for {cname}")
        sys.exit(1)
    lst = cl.GetListOfAllPublicMethods() if inherited else cl.GetListOfMethods()
    out = {}
    for m in lst:
        if not inherited and not (m.Property() & ROOT.kIsPublic):
            continue
        args = ",".join(a.GetFullTypeName() for a in m.GetListOfMethodArgs())
        const = " const" if (m.Property() & ROOT.kIsConstMethod) else ""
        out.setdefault(m.GetName(), set()).add(f"({args}){const}")
    return out


jet = public_methods("pat::io_v1::Jet", inherited=False)
patobj = public_methods("pat::PATObject<reco::io_v1::Jet>", inherited=False)
genjet = public_methods("pat::io_v1::GenJet", inherited=True)


def mapped(name, origin):
    if origin == "Jet":
        return {"Jet": "GenJet", "~Jet": "~GenJet"}.get(name, name)
    return {"PATObject<reco::io_v1::Jet>": "GenJet", "PATObject": "GenJet",
            "~PATObject<reco::io_v1::Jet>": "~GenJet", "~PATObject": "~GenJet"}.get(name, name)


# Members removed from pat::GenJet (no generator meaning). They must not be declared by pat::GenJet.
REMOVED = {
    # b tagging
    "bDiscriminator", "getPairDiscri", "tagInfoLabels", "hasTagInfo", "tagInfo", "tagInfoCandIP", "tagInfoTrackIP",
    "tagInfoCandSoftLepton", "tagInfoSoftLepton", "tagInfoCandSecondaryVertex", "tagInfoSecondaryVertex",
    "tagInfoBoostedDoubleSV", "tagInfoPixelCluster", "addBDiscriminatorPair", "addTagInfo", "tagInfosFwdPtr",
    "updateFwdTagInfoFwdPtr",
    # calo tower / PF candidate constituents
    "setCaloTowers", "setPFCandidates", "getCaloConstituent", "getCaloConstituents", "getPFConstituent",
    "getPFConstituents", "caloTowersFwdPtr", "pfCandidatesFwdPtr", "updateFwdCaloTowerFwdPtr",
    "updateFwdPFCandidateFwdPtr",
    # self link
    "setGenJetRef",
    # link from a reco jet to its matched pat::GenJet (pat::Jet class version 4): a gen jet is its own match
    "patGenJetRef", "patGenJet", "setPatGenJetRef", "matchedGenJetPtr",
    # flavour
    "partonFlavour", "hadronFlavour", "jetFlavourInfo", "setPartonFlavour", "setHadronFlavour", "setJetFlavourInfo",
    # jet ID
    "jetID", "setJetID",
    # jet type and the Calo/JPT/PF specifics
    "isCaloJet", "isJPTJet", "isPFJet", "isBasicJet", "hasPFSpecific", "caloSpecific", "jptSpecific", "pfSpecific",
    "setCaloSpecific", "setJPTSpecific", "setPFSpecific",
    # calo-tower quantities
    "maxEInEmTowers", "maxEInHadTowers", "energyFractionHadronic", "emEnergyFraction", "hadEnergyInHB",
    "hadEnergyInHO", "hadEnergyInHE", "hadEnergyInHF", "emEnergyInEB", "emEnergyInEE", "emEnergyInHF", "towersArea",
    "n90", "n60",
    # JPT track lists
    "pionsInVertexInCalo", "pionsInVertexOutCalo", "pionsOutVertexInCalo", "muonsInVertexInCalo",
    "muonsInVertexOutCalo", "muonsOutVertexInCalo", "elecsInVertexInCalo", "elecsInVertexOutCalo",
    "elecsOutVertexInCalo", "elecMultiplicity",
    # HF and HO energies
    "HFHadronEnergy", "HFHadronEnergyFraction", "HFEMEnergy", "HFEMEnergyFraction", "hoEnergy", "hoEnergyFraction",
    "HFHadronMultiplicity", "HFEMMultiplicity", "associatedTracks", "setAssociatedTracks",
}
# New on pat::GenJet: the generator constituents, as pat::Jet has the PF candidates.
ADDED = {"setGenCandidates", "genCandidates", "genCandidatesFwdPtr", "updateFwdGenCandidateFwdPtr"}

failures = []
infos = []

# (a) every REMOVED member is not declared by pat::GenJet itself (or PATObject<reco::GenJet>).
# A removed name that a reco base class still provides would not be a pat::GenJet member.
genjet_own = public_methods("pat::io_v1::GenJet", inherited=False)
genjet_patobj = public_methods("pat::PATObject<reco::GenJet>", inherited=False)
for name in sorted(REMOVED):
    if name in genjet_own or name in genjet_patobj:
        failures.append(f"removed member {name} is still declared on pat::GenJet")
    elif name in genjet:
        infos.append(f"removed member {name} is not declared by pat::GenJet but inherited from a reco base class")

# (c) the new generator constituent members
for name in sorted(ADDED):
    if name not in genjet_own:
        failures.append(f"new member {name} missing on pat::GenJet")

# (b) every other pat::Jet / PATObject<reco::Jet> member is kept
for origin, names in (("Jet", jet), ("PATObject<reco::Jet>", patobj)):
    for name, sigs in sorted(names.items()):
        if name in REMOVED:
            continue
        target = mapped(name, origin)
        if target not in genjet:
            failures.append(f"{origin}::{name} missing on pat::GenJet")
            continue
        if origin == "Jet":
            for sig in sorted(sigs):
                if "Run3ScoutingPFJet" in sig:
                    continue
                if sig not in genjet[target]:
                    infos.append(f"{name}{sig}")

# the scouting constructor exists on pat::Jet and must not exist on pat::GenJet
scouting_on_jet = any("Run3ScoutingPFJet" in s for s in jet.get("Jet", set()))
scouting_on_genjet = any("Run3ScoutingPFJet" in s for s in genjet.get("GenJet", set()))
if not scouting_on_jet:
    failures.append("reference pat::Jet has no scouting constructor (reflection problem?)")
if scouting_on_genjet:
    failures.append("pat::GenJet has a Run3ScoutingPFJet constructor")

# member templates and the free operator<<: compile a snippet
ok = ROOT.gInterpreter.Declare('''
#include "DataFormats/PatCandidates/interface/GenJet.h"
#include <sstream>
namespace parity_check {
  inline bool templatesAndStream() {
    pat::GenJet j;
    j.addUserData<reco::Candidate::LorentzVector>("x", reco::Candidate::LorentzVector(1., 2., 3., 4.));
    const auto* p = j.userData<reco::Candidate::LorentzVector>("x");
    std::ostringstream os;
    reco::operator<<(os, j);
    os << j;
    return p != nullptr && p->E() == 4. && !os.str().empty();
  }
}
''')
if not ok or not ROOT.parity_check.templatesAndStream():
    failures.append("userData<T>/addUserData<T>/operator<< check failed")

print(f"pat::io_v1::Jet: {len(jet)} public names; PATObject<reco::io_v1::Jet>: {len(patobj)} public names")
print(f"pat::io_v1::GenJet (with inherited): {len(genjet)} public names")
if infos:
    print(f"\nINFO: {len(infos)} pat::Jet argument lists with no identical overload on pat::GenJet (name exists; expected where pat::GenJet swaps Jet types for GenJet types):")
    for i in infos:
        print("  " + i)
# The intended type swaps: exactly these pat::Jet argument lists may differ.
EXPECTED_SWAPS = {
    "Jet(const edm::Ptr<pat::Jet>&)", "Jet(const edm::Ptr<reco::Jet>&)", "Jet(const edm::RefToBase<pat::Jet>&)",
    "Jet(const edm::RefToBase<reco::Jet>&)", "Jet(const pat::io_v1::Jet&)", "Jet(const reco::Jet&)",
    "addSubjets(const pat::JetPtrCollection&,const string&)", "operator=(const pat::io_v1::Jet&)",
    "updateFwdGenJetFwdRef(edm::Ref<reco::GenJetCollection>)",
}
for i in sorted(set(infos) - EXPECTED_SWAPS):
    failures.append("unexpected signature difference: " + i)
for i in sorted(EXPECTED_SWAPS - set(infos)):
    failures.append("expected type swap not seen (pat::GenJet now has the pat::Jet signature?): " + i)
if failures:
    print(f"\nFAIL: {len(failures)} problem(s)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nPASS: removed members absent, every other public name of pat::Jet and PATObject<reco::Jet> "
      "exists on pat::GenJet, genCandidates members present (scouting constructor absent)")
