from __future__ import print_function
import ROOT
from drawHistoAllChambers import drawHisto

def binNumber(station,sector):
    start = (station - 1)*12
    return start + sector

def plot(fileName,sl,ymin=300,ymax=360,option="HISTOP",draw=True):

    slType = sl
    slStr = "SL%d" % slType
    verbose = False

    ROOT.TH1.AddDirectory(False)

    file = ROOT.TFile(fileName,'read')

    wheels = (-2,-1,0,1,2)
    stations = (1,2,3,4)

    histosWheel = {}
    for wh in wheels:
        histoName = 'Wheel%d_%s_TTrig' % (wh,slStr)
        print("Accessing",histoName)
        histosWheel[wh] = file.Get(histoName)

    # (Wh-2 MB1 Sec1 ... Wh-2 MB1 Sec12 ... Wh-1 MB1 Sec1 ... Wh-1 MB1 Sec12 ...)
    # (Wh-2 MB2 Sec1 ... Wh-2 MB2 Sec12 ... Wh-1 MB2 Sec1 ... Wh-1 MB1 Sec12 ...) ...  
    nBins = 250
    if slType == 2: nBins = 180
    histo = ROOT.TH1F("h_TTrigAll","TTrig",nBins,0,nBins)
    for st in stations:
        nSectors = 12
        if st == 4: nSectors = 14 
        if st == 4 and slType == 2: continue
        if verbose: print("Station",st)
        for wh in wheels:
            if verbose: print("Wheel",wh) 
            for sec in range(1,nSectors+1):
                if verbose: print("Sector",sec)
                binHisto = binNumber(st,sec)
                if verbose: print("Bin from histos:",binHisto) 
                value = histosWheel[wh].GetBinContent(binHisto)

                binHistoNew = (st - 1)*60 + (wh + 2)*nSectors + sec
                if verbose: print("Bin final",binHistoNew)
                histo.SetBinContent(binHistoNew,value) 

                if sec == 1:
                    label = "Wheel %d" % wh
                    if wh == -2: label += " MB%d" % st  
                    histo.GetXaxis().SetBinLabel(binHistoNew,label) 

    objects = drawHisto(histo,
                        title="t_{Trig} (ns)",
                        ymin=ymin,ymax=ymax,option=option,draw=draw)

    return objects

def compare(fileNames,sl,ymin=300,ymax=360,labels=[]):
    option = "HISTOP"
    colors = (2,4,12,44,55,38,27,46)
    markers = (24,25,26,27,28,30,32,5)

    idx = 0
    canvas = None
    objects = None
    histos = []
    for fileName in fileNames:
        draw = False
        if not idx: draw = True

        objs = plot(fileName,sl,ymin,ymax,option,draw)
        histos.append(objs[1])
        histos[-1].SetName( "%s_%d" % (histos[-1].GetName(),idx) )
        if not idx:
            canvas = objs[0]
            objects = objs[2]

        canvas.cd()
        if idx:
            histos[-1].SetLineColor(colors[ (idx - 1) % len(colors) ])
            histos[-1].SetMarkerColor(colors[ (idx - 1) % len(colors) ])
            histos[-1].SetMarkerStyle(markers[ (idx - 1) % len(markers) ])

            histos[-1].Draw(option + "SAME")

        idx += 1

    legend = ROOT.TLegend(0.4,0.7,0.95,0.8)
    for idx in range( len(histos) ):
        histo = histos[idx]
        label = histo.GetName()
        if len(labels): label = labels[idx]
        legend.AddEntry(histo,label,"LP")

        idx += 1

    canvas.cd()
    legend.SetFillColor( canvas.GetFillColor() )
    legend.Draw("SAME")

    objects.append(legend)

    return (canvas,histos,objects)

def compareDiff(fileNames,sl,ymin=-15.,ymax=15.):
    option = "HISTOP"
    colors = (2,4,9,12,38,44,46,55)
    markers = (24,25,26,27,28,30,32,5)

    idx = 0
    canvases = [None,None]
    objects = None
    histoRef = None
    histos = []
    histosDist = []
    for fileName in fileNames:
        objs = plot(fileName,sl,300,360,'',False)
        histos.append( objs[1].Clone(objs[1].GetName() + "_diff") )
        histos[-1].SetName( "%s_%d" % (histos[-1].GetName(),idx) )
        if not idx:
            histoRef = objs[1]
            histos[-1].Reset()
        else:
            histos[-1].Add(histoRef,-1.) 

        draw = False
        if not idx: draw = True

        objs = drawHisto(histos[-1],
                         title="t_{Trig} difference (ns)",
                         ymin=ymin,ymax=ymax,option=option,draw=draw)

        if not idx: 
            canvases[0] = objs[0]
            objects = objs[2]

        if idx:
            canvases[0].cd()
            histos[-1].SetLineColor(colors[ (idx - 1) % len(colors) ])
            histos[-1].SetMarkerColor(colors[ (idx - 1) % len(colors) ])
            histos[-1].SetMarkerStyle(markers[ (idx - 1) % len(markers) ])

            histos[-1].Draw(option + "SAME")

            histosDist.append( ROOT.TH1F(histos[-1].GetName() + "_dist","tTrig distribution",200,ymin,ymax) )
            for ibin in range(1,histos[-1].GetNbinsX()+1):
                histosDist[-1].Fill( histos[-1].GetBinContent(ibin) )

            histosDist[-1].SetLineColor(colors[ (idx - 1) % len(colors) ])
            histosDist[-1].SetMarkerColor(colors[ (idx - 1) % len(colors) ])
            histosDist[-1].SetMarkerStyle(markers[ (idx - 1) % len(markers) ])

        idx += 1


    canvases[1] = ROOT.TCanvas("c_tTrigDist")
    canvases[1].SetGridy()
    canvases[1].SetFillColor(0)
    canvases[1].cd()
    option = "HISTO"
    idx = 0
    for histo in histosDist:
        if not idx:
            histo.GetXaxis().SetTitle("t_{Trig} difference (ns)")
            histo.GetYaxis().SetTitle("Number of chambers")
            histo.Draw(option)
        else:
            histo.Draw(option + "SAME") 
        idx += 1

    return (canvases,histos,histosDist,objects)
