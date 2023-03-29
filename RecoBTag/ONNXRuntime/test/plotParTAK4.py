from __future__ import print_function
import ROOT
from DataFormats.FWLite import Handle, Events
import numpy as np
import pickle

events_c = Events('miniFromAOD_mcRun3_withParT.root')

handleJ  = Handle("std::vector<pat::Jet>")
pu_mitigation_technique = 'PUPPI' # PUPPI
labelJ = ("slimmedJetsPuppi","","PAT") if pu_mitigation_technique == 'PUPPI' else ("slimmedJets","","PAT")

h_disc_bvsall = ROOT.TH1F('robustpart_bvsall', ';BvsAll;', 100, -1., 1.)
h_disc_bvsl = ROOT.TH1F('robustpart_bvsl', ';BvsL;', 100, -1., 1.)
h_disc_cvsl = ROOT.TH1F('robustpart_cvsl', ';CvsL;', 100, -1., 1.)
h_disc_cvsb = ROOT.TH1F('robustpart_cvsb', ';CvsB;', 100, -1., 1.)

h_probb = ROOT.TH1F('robustpart_probb', ';probb;', 100, -1., 1.)
h_probbb = ROOT.TH1F('robustpart_probbb', ';probbb;', 100, -1., 1.)
h_problepb = ROOT.TH1F('robustpart_problepb', ';problepb;', 100, -1., 1.)
h_probc = ROOT.TH1F('robustpart_probc', ';probc;', 100, -1., 1.)
h_probuds = ROOT.TH1F('robustpart_probuds', ';probuds;', 100, -1., 1.)
h_probuds_L = ROOT.TH1F('robustpart_probuds_L', ';probuds_L;', 100, -1., 1.)
h_probg = ROOT.TH1F('robustpart_probg', ';probg;', 100, -1., 1.)

info = {}
info['pt'] = []
info['eta'] = []
info['mass'] = []
info['probb'] = []
info['probbb'] = []
info['problepb'] = []
info['probc'] = []
info['probuds'] = []
info['probg'] = []
info['BvsAll'] = []
info['BvsL'] = []
info['CvsL'] = []
info['CvsB'] = []
a = 0
for iev, event in enumerate(events_c):
    a += 1
    
print(a, ' events')
for iev, event in enumerate(events_c):
    event.getByLabel(labelJ, handleJ)
    jets = handleJ.product()
    #print(iev)
    for jet in jets:
        if jet.pt() < 15 or abs(jet.eta()) > 2.5: continue
        #if jet.pt() < 300 or jet.pt() > 2000: continue
        #if jet.mass() < 40 or jet.mass() > 200: continue

        jet_pt, jet_eta, jet_mass = jet.pt(), jet.eta(), jet.mass()
        # print(jet_pt, jet_eta, jet_mass)
        jet_probb = jet.bDiscriminator("pfParticleTransformerAK4JetTags:probb")
        jet_probbb = jet.bDiscriminator("pfParticleTransformerAK4JetTags:probbb")
        jet_problepb = jet.bDiscriminator("pfParticleTransformerAK4JetTags:problepb")
        jet_probc = jet.bDiscriminator("pfParticleTransformerAK4JetTags:probc")
        jet_probuds = jet.bDiscriminator("pfParticleTransformerAK4JetTags:probuds")
        jet_probg = jet.bDiscriminator("pfParticleTransformerAK4JetTags:probg")
        
        h_probb.Fill(jet_probb)
        h_probbb.Fill(jet_probbb)
        h_problepb.Fill(jet_problepb)
        h_probc.Fill(jet_probc)
        h_probuds.Fill(jet_probuds)
        if jet.hadronFlavour() < 4:
            h_probuds_L.Fill(jet_probuds)
        h_probg.Fill(jet_probg)
        
        #print("probb", jet_probb)
        #print("probbb", jet_probbb)
        #print("problepb", jet_problepb)
        #print("probc", jet_probc)
        #print("probuds", jet_probuds)
        #print("probg", jet_probg)
        
        jet_BvsAll = jet.bDiscriminator("pfParticleTransformerAK4DiscriminatorsJetTags:BvsAll")
        jet_BvsL = jet.bDiscriminator("pfParticleTransformerAK4DiscriminatorsJetTags:BvsL")
        jet_CvsL = jet.bDiscriminator("pfParticleTransformerAK4DiscriminatorsJetTags:CvsL")
        jet_CvsB = jet.bDiscriminator("pfParticleTransformerAK4DiscriminatorsJetTags:CvsB")
        
        #print("BvsAll", jet_BvsAll)
        #print("BvsL", jet_BvsL)
        #print("CvsL", jet_CvsL) 
        #print("CvsB", jet_CvsB)
        
        h_disc_bvsall.Fill(jet_BvsAll)
        h_disc_bvsl.Fill(jet_BvsL)
        h_disc_cvsl.Fill(jet_CvsL)
        h_disc_cvsb.Fill(jet_CvsB)

        info['mass'].append(jet_mass)
        info['pt'].append(jet_pt)
        info['eta'].append(jet_eta)
        
        info['probb'].append(jet_probb)
        info['probbb'].append(jet_probbb)
        info['problepb'].append(jet_problepb)
        info['probc'].append(jet_probc)
        info['probuds'].append(jet_probuds)
        info['probg'].append(jet_probg)
        
        info['BvsAll'].append(jet_BvsAll)
        info['BvsL'].append(jet_BvsL)
        info['CvsL'].append(jet_CvsL)
        info['CvsB'].append(jet_CvsB)
    break

with open('outputs.pkl', 'wb') as handle:
    pickle.dump(info, handle)

canv = ROOT.TCanvas()
h_disc_bvsall.Draw("HISTO")
h_disc_bvsall.SetLineColor(632)
h_disc_bvsall.SetLineStyle(10)
canv.Draw()
canv.SaveAs(f"RobustParTAK4_BvsAll{pu_mitigation_technique}.png")

canv2 = ROOT.TCanvas()
h_disc_bvsl.Draw("HISTO")
h_disc_bvsl.SetLineColor(632)
h_disc_bvsl.SetLineStyle(10)
canv2.Draw()
canv2.SaveAs(f"RobustParTAK4_BvsL{pu_mitigation_technique}.png")

canv3 = ROOT.TCanvas()
h_disc_cvsl.Draw("HISTO")
h_disc_cvsl.SetLineColor(632)
h_disc_cvsl.SetLineStyle(10)
canv3.Draw()
canv3.SaveAs(f"RobustParTAK4_CvsL{pu_mitigation_technique}.png")

canv4 = ROOT.TCanvas()
h_disc_cvsb.Draw("HISTO")
h_disc_cvsb.SetLineColor(632)
h_disc_cvsb.SetLineStyle(10)
canv4.Draw()
canv4.SaveAs(f"RobustParTAK4_CvsB{pu_mitigation_technique}.png")


pcanv = ROOT.TCanvas()
h_probb.Draw("HISTO")
h_probb.SetLineColor(632)
h_probb.SetLineStyle(10)
pcanv.Draw()
pcanv.SaveAs(f"RobustParTAK4_probb{pu_mitigation_technique}.png")

pcanv2 = ROOT.TCanvas()
h_probbb.Draw("HISTO")
h_probbb.SetLineColor(632)
h_probbb.SetLineStyle(10)
pcanv2.Draw()
pcanv2.SaveAs(f"RobustParTAK4_probbb{pu_mitigation_technique}.png")

pcanv3 = ROOT.TCanvas()
h_problepb.Draw("HISTO")
h_problepb.SetLineColor(632)
h_problepb.SetLineStyle(10)
pcanv3.Draw()
pcanv3.SaveAs(f"RobustParTAK4_problepb{pu_mitigation_technique}.png")

pcanv4 = ROOT.TCanvas()
h_probc.Draw("HISTO")
h_probc.SetLineColor(632)
h_probc.SetLineStyle(10)
pcanv4.Draw()
pcanv4.SaveAs(f"RobustParTAK4_probc{pu_mitigation_technique}.png")

pcanv5 = ROOT.TCanvas()
h_probuds.Draw("HISTO")
h_probuds.SetLineColor(632)
h_probuds.SetLineStyle(10)
pcanv5.Draw()
pcanv5.SaveAs(f"RobustParTAK4_probuds{pu_mitigation_technique}.png")

pcanv5L = ROOT.TCanvas()
h_probuds_L.Draw("HISTO")
h_probuds_L.SetLineColor(632)
h_probuds_L.SetLineStyle(10)
pcanv5L.Draw()
pcanv5L.SaveAs(f"RobustParTAK4_probuds_L{pu_mitigation_technique}.png")

pcanv6 = ROOT.TCanvas()
h_probg.Draw("HISTO")
h_probg.SetLineColor(632)
h_probg.SetLineStyle(10)
pcanv6.Draw()
pcanv6.SaveAs(f"RobustParTAK4_probg{pu_mitigation_technique}.png")