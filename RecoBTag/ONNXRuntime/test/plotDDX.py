from __future__ import print_function
import ROOT
from DataFormats.FWLite import Handle, Events

events_c = Events('output_test_DDX.root')

handleJ  = Handle ("std::vector<pat::Jet>")
labelJ = ("selectedUpdatedPatJets","","PATtest")

h_probQ_ddb = ROOT.TH1F('h_probQ_ddb', ';prob Q;', 40, 0., 1.)
h_probH_ddb = ROOT.TH1F('h_probH_ddb', ';prob H;', 40, 0., 1.)

h_probQ_ddc = ROOT.TH1F('h_probQ_ddc', ';prob Q;', 40, 0., 1.)
h_probH_ddc = ROOT.TH1F('h_probH_ddc', ';prob H;', 40, 0., 1.)

for iev,event in enumerate(events_c):
    event.getByLabel (labelJ, handleJ)
    jets = handleJ.product()
    for jet in jets  :
        if jet.pt() < 300 or jet.pt() > 2000: continue
        if jet.mass() < 40 or jet.mass() > 200: continue

        print(jet.pt(), jet.mass())
        print("DDB", jet.bDiscriminator("pfDeepDoubleBvLJetTags:probQCD"), jet.bDiscriminator("pfDeepDoubleBvLJetTags:probHbb"))
        print("DDB", jet.bDiscriminator("pfMassIndependentDeepDoubleBvLJetTags:probQCD"), jet.bDiscriminator("pfMassIndependentDeepDoubleBvLJetTags:probHbb"))
        print("DDCvL", jet.bDiscriminator("pfDeepDoubleCvLJetTags:probQCD"), jet.bDiscriminator("pfDeepDoubleCvLJetTags:probHcc"))
        print("DDCvL", jet.bDiscriminator("pfMassIndependentDeepDoubleCvLJetTags:probQCD"), jet.bDiscriminator("pfMassIndependentDeepDoubleCvLJetTags:probHcc"))
        print("DDCvB", jet.bDiscriminator("pfDeepDoubleCvBJetTags:probHbb"), jet.bDiscriminator("pfDeepDoubleCvBJetTags:probHcc") )
        print("DDCvB", jet.bDiscriminator("pfMassIndependentDeepDoubleCvBJetTags:probHbb"), jet.bDiscriminator("pfMassIndependentDeepDoubleCvBJetTags:probHcc"))
        h_probQ_ddb.Fill(jet.bDiscriminator("pfDeepDoubleBvLJetTags:probQCD"))
        h_probH_ddb.Fill(jet.bDiscriminator("pfDeepDoubleBvLJetTags:probHbb"))
        h_probQ_ddc.Fill(jet.bDiscriminator("pfDeepDoubleCvLJetTags:probQCD"))
        h_probH_ddc.Fill(jet.bDiscriminator("pfDeepDoubleCvLJetTags:probHcc"))

c1a = ROOT.TCanvas()
h_probH_ddb.Draw("HISTO")
h_probH_ddb.SetLineColor(632)
h_probH_ddb.SetLineStyle(10)
h_probQ_ddb.Draw("SAME")
c1a.Draw()
c1a.SaveAs("ProbQ_vc_vb.png")

c1b = ROOT.TCanvas()
h_probH_ddc.Draw("HISTO")
h_probH_ddc.SetLineColor(632)
h_probH_ddc.SetLineStyle(10)
h_probQ_ddc.Draw("SAME")
c1b.Draw()
c1b.SaveAs("ProbH_vc_vb.png")
