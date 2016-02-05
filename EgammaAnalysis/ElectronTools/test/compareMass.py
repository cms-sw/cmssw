import ROOT

masseb = []
massee = []

f1 = ROOT.TFile("plots_data.root")
t1 = f1.Get("zeeUncalibTree/probe_tree")
e = t1.GetEntries()

masseb.append(ROOT.TH1F("masseb1", "", 80, 70, 110))
massee.append(ROOT.TH1F("massee1", "", 80, 70, 110))
for z in xrange(e):
    t1.GetEntry(z)
    if(abs(t1.l1eta)<1.479  and abs(t1.l2eta)<1.479):
        masseb[-1].Fill(t1.mass)
    else:
        massee[-1].Fill(t1.mass)

f2 = ROOT.TFile("plots_data.root")
t2 = f2.Get("zeeCalibTree/probe_tree")
e = t2.GetEntries()

masseb.append(ROOT.TH1F("masseb2", "", 80, 70, 110))
massee.append(ROOT.TH1F("massee2", "", 80, 70, 110))

for z in xrange(e):
    t2.GetEntry(z)
    if(abs(t2.l1eta)<1.479  and abs(t2.l2eta)<1.479):
        masseb[-1].Fill(t2.mass)
    else:
        massee[-1].Fill(t2.mass)

f3 = ROOT.TFile("plots_mc.root")
t3 = f3.Get("zeeUncalibTree/probe_tree")
e = t3.GetEntries()

masseb.append(ROOT.TH1F("masseb3", "", 80, 70, 110))
massee.append(ROOT.TH1F("massee3", "", 80, 70, 110))

for z in xrange(e):
    t3.GetEntry(z)
    if(abs(t3.l1eta)<1.479  and abs(t3.l2eta)<1.479):
        masseb[-1].Fill(t3.mass)
    else:
        massee[-1].Fill(t3.mass)

f4 = ROOT.TFile("plots_mc.root")
t4 = f4.Get("zeeCalibTree/probe_tree")
e = t4.GetEntries()

masseb.append(ROOT.TH1F("masseb4", "", 80, 70, 110))
massee.append(ROOT.TH1F("massee4", "", 80, 70, 110))

for z in xrange(e):
    t4.GetEntry(z)
    if(abs(t4.l1eta)<1.479  and abs(t4.l2eta)<1.479):
        masseb[-1].Fill(t4.mass)
    else:
        massee[-1].Fill(t4.mass)


c = []
ratio = []
for i in xrange(2):
    c.append(ROOT.TCanvas("c"+str(i), "c"))
    c[-1].Divide(2,2)
    c[-1].cd(1)
    masseb[i+2].Scale(masseb[i].Integral()/masseb[i+2].Integral())
    masseb[i+2].Draw()
    masseb[i+2].SetFillColor(ROOT.kRed)
    masseb[i].Draw("PESAME")
    masseb[i].SetMarkerStyle(20)
    c[-1].cd(3)
    ratio.append(masseb[i+2].Clone())
    ratio[-1].Sumw2()
    ratio[-1].Divide(masseb[i])
    ratio[-1].Draw("PE")
    
    c[-1].cd(2)
    massee[i+2].Scale(massee[i].Integral()/massee[i+2].Integral())
    massee[i+2].Draw()
    massee[i+2].SetFillColor(ROOT.kRed)
    massee[i].Draw("PESAME")
    massee[i].SetMarkerStyle(20)
raw_input()
