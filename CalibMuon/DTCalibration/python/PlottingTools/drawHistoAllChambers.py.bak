import ROOT

def drawHisto(histo,title,ymin,ymax,option="HISTOP",draw=True):
    histo.SetStats(0)
    histo.SetLineWidth(3)
    histo.SetMarkerStyle(20)
    histo.SetMarkerSize(0.9)
    histo.GetYaxis().SetRangeUser(ymin,ymax)
    histo.GetYaxis().SetTitle(title)
    histo.GetXaxis().SetLabelSize(0.04)
    histo.GetXaxis().SetTickLength(0.)
    histo.LabelsOption("d","X")
 
    fillColor = 0
    canvas = None 
    if draw:
        canvas = ROOT.TCanvas("c_" + histo.GetName())
        canvas.SetGridy()
        canvas.SetFillColor(fillColor)
    if draw: histo.Draw(option)

    linesWh = {}
    linesSt = {}
    labels = {}
    for idx_st in range(1,5):
        nSectors = 12
        if idx_st == 4: nSectors = 14
        for idx_wh in range(-1,3):
            xline = (idx_st - 1)*60 + (idx_wh + 2)*nSectors
            if xline >= histo.GetNbinsX(): continue 

            linesWh[(idx_st,idx_wh)] = ROOT.TLine(xline,ymin,xline,ymax)
            linesWh[(idx_st,idx_wh)].SetLineStyle(2)
            if draw: linesWh[(idx_st,idx_wh)].Draw("SAME")

    for idx in range(1,4):
        xline = idx*60
        if xline >= histo.GetNbinsX(): continue

        linesSt[idx] = ROOT.TLine(xline,ymin,xline,ymax)
        linesSt[idx].SetLineStyle(2)
        linesSt[idx].SetLineWidth(2)
        if draw: linesSt[idx].Draw("SAME")

    for idx in range(1,5):
        xlabel = (idx - 1)*60 + 20
        ylabel = ymin + 0.75*(ymax -ymin)
        if xlabel >= histo.GetNbinsX(): continue

        strSt = "MB%d" % idx
        labels[idx] = ROOT.TPaveLabel(xlabel,ylabel,(xlabel+20),(ylabel + 0.10*(ymax -ymin)),strSt)
        labels[idx].SetTextSize(0.5)
        labels[idx].SetFillColor(fillColor)
        if draw: labels[idx].Draw("SAME")

    objects = []
    objects.append(linesWh)
    objects.append(linesSt)
    objects.append(labels)

    return (canvas,histo,objects)
