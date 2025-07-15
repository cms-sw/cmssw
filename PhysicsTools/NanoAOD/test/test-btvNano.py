#!/bin/env python3

import ROOT

f = ROOT.TFile.Open("btvNanoMC_NANO.root")
t = f.Get("Events")

eps = 1e-10

def checkHist(maxMean, maxRMS):
    h = ROOT.gPad.GetPrimitive("htemp")
    assert h.GetMean() < maxMean
    assert h.GetRMS() < maxRMS

# Check JetCand matching, must be exact
t.Draw("JetPFCands_pt - PFCands_pt[JetPFCands_pFCandsIdx]")
checkHist(eps, eps)
t.Draw("GenJetCands_pt - GenCands_pt[GenJetCands_genCandsIdx]")
checkHist(eps, eps)

# Check matching of candidates to jets, RMS should be smaller than jet radius / 2
t.Draw("PFCands_eta[JetPFCands_pFCandsIdx] - Jet_eta[JetPFCands_jetIdx]")
checkHist(1e-2, 0.2)
t.Draw("PFCands_eta[FatJetPFCands_pFCandsIdx] - FatJet_eta[FatJetPFCands_jetIdx]")
checkHist(1e-2, 0.4)

# Check matching of JetSVs to SVs, must be exact
t.Draw("JetSVs_mass - SV_mass[JetSVs_sVIdx]", "JetSVs_sVIdx>-1")
checkHist(eps, eps)

# Check matching of SVs to leading jet, RMS should be smaller than jet radius / 2
t.Draw("Jet_eta[0] - SV_eta[JetSVs_sVIdx]", "JetSVs_sVIdx>-1 && JetSVs_jetIdx==0")
checkHist(0.1, 0.2)

# Check matching of PFCands to SVs
t.Draw("PFCands_eta[JetPFCands_pFCandsIdx] - SV_eta[JetSVs_sVIdx[JetPFCands_jetSVIdx]]", "JetPFCands_jetSVIdx>-1 && JetSVs_sVIdx[JetPFCands_jetSVIdx]>-1")
checkHist(1e-2, 0.2)

# Check matching of PFCands to GenCands
t.Draw("PFCands_eta - GenCands_eta[PFCands_genCandIdx]", "PFCands_genCandIdx>-1")
checkHist(1e-3, 1e-2)

# Check matching of Muons to GenParts
t.Draw("Muon_eta - GenPart_eta[Muon_genPartIdx]", "Muon_genPartIdx>-1")
checkHist(1e-3, 1e-2)

# Check matching of GenCands from B hadrons to GenPart mothers
t.Draw("GenCands_eta - GenPart_eta[GenCands_genPartMotherIdx]", "GenCands_genPartMotherIdx>1 && GenCands_isFromB==2")
checkHist(1e-3, 0.2)
