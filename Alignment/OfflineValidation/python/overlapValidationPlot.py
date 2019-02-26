import math
import ROOT
from TkAlStyle import TkAlStyle

def hist(tree_file_name, hist_name):
    f = ROOT.TFile(tree_file_name)
    t = f.Get("analysis/Overlaps")

    h = ROOT.TH1F(hist_name, hist_name, 100, -300, 300)

    h.SetDirectory(0)

    for entry in t:
        if not ((t.subdetID == 1) and (abs(t.moduleZ[0] - t.moduleZ[1]) < 0.5)):

            continue

        modulePhi0 = math.atan2(t.moduleY[0], t.moduleX[0]) 
        modulePhi1 = math.atan2(t.moduleY[1], t.moduleX[1])

        if modulePhi0 > modulePhi1:
            hitXA = t.hitX[1]
            hitXB = t.hitX[0]
            predXA = t.predX[1]
            predXB = t.predX[0]
            deltaPhiA = t.deltaphi[1]
            deltaPhiB = t.deltaphi[0]
        else:
            hitXA = t.hitX[0]
            hitXB = t.hitX[1]
            predXA = t.predX[0]
            predXB = t.predX[1]
            deltaPhiA = t.deltaphi[0]
            deltaPhiB = t.deltaphi[1]

        residualA = hitXA - predXA
        residualB = hitXB - predXB
        if deltaPhiA < 0:
            residualA *= -1
        if deltaPhiB < 0:
            residualB *= -1
        
        A = 10000*(residualA - residualB)
        h.Fill(A)
    return h

def plot(file_name, *filesTitlesColorsStyles):
    hstack = ROOT.THStack("hstack","")
    legend = TkAlStyle.legend(len(filesTitlesColorsStyles), 0.3)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    hs = []
    
    for files, title, color, style in filesTitlesColorsStyles:
        h = hist(files, files.replace("/",""))
        h.SetLineColor(color)
        h.SetLineStyle(style)
        legend.AddEntry(h, title, "l")
        hstack.Add(h)
        hs.append(h)
    
    c = ROOT.TCanvas()
    hstack.Draw("nostack")
    legend.Draw()
    
    hstack.GetXaxis().SetTitle("hit_{A} - pred_{A} - (hit_{B} - pred_{B}) (#mum)")
    hstack.GetYaxis().SetTitle("number of events")
    hstack.GetXaxis().SetNdivisions(404)

    TkAlStyle.drawStandardTitle()
    
    save_as_file_name = file_name

    for ext in "png", "eps", "root", "pdf":
        c.SaveAs(save_as_file_name+"." +ext)

