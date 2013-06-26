import ROOT
from drawHistoAllChambers import drawHisto

def binNumber(station,sector):
    start = (station - 1)*12
    return start + sector
 
def plot(fileName,sl,option="HISTOP",draw=True):

    slType = sl
    slStr = "SL%d" % slType
    verbose = False

    ROOT.TH1.AddDirectory(False)

    file = ROOT.TFile(fileName,'read')

    wheels = (-2,-1,0,1,2)
    stations = (1,2,3,4)

    histosWheel = {}
    for wh in wheels:
        histoName = 'Wheel%d_%s_VDrift' % (wh,slStr)
        print "Accessing",histoName
        histosWheel[wh] = file.Get(histoName)

    # (Wh-2 MB1 Sec1 ... Wh-2 MB1 Sec12 ... Wh-1 MB1 Sec1 ... Wh-1 MB1 Sec12 ...)
    # (Wh-2 MB2 Sec1 ... Wh-2 MB2 Sec12 ... Wh-1 MB2 Sec1 ... Wh-1 MB1 Sec12 ...) ...  
    nBins = 250
    if slType == 2: nBins = 180
    histo = ROOT.TH1F("h_VDriftAll","VDrift",nBins,0,nBins)
    for st in stations:
        nSectors = 12
        if st == 4: nSectors = 14 
        if st == 4 and slType == 2: continue 
        if verbose: print "Station",st
        for wh in wheels:
            if verbose: print "Wheel",wh 
            for sec in range(1,nSectors+1):
                if verbose: print "Sector",sec
                binHisto = binNumber(st,sec)
                if verbose: print "Bin from histos:",binHisto 
                value = histosWheel[wh].GetBinContent(binHisto)
                # From cm/ns to micron/ns
                value *= 10000.
 
                binHistoNew = (st - 1)*60 + (wh + 2)*nSectors + sec
                if verbose: print "Bin final",binHistoNew
                histo.SetBinContent(binHistoNew,value) 
  
                if sec == 1:
                    label = "Wheel %d" % wh
                    if wh == -2: label += " MB%d" % st  
                    histo.GetXaxis().SetBinLabel(binHistoNew,label) 

    objects = drawHisto(histo,
                        title="v_{drift} (#mum/ns)",
                        ymin=53,ymax=57,option=option,draw=draw)

    return objects

def compare(fileNames,sl):
    option = "HISTOP"
    colors = (2,4,12,44,55)
    markers = (24,25,26,27)

    idx = 0
    canvas = None
    objects = None
    histos = []
    for fileName in fileNames:
        draw = False
        if not idx: draw = True

        objs = plot(fileName,sl,option,draw)
        if not idx:
            canvas = objs[0]
            objects = objs[2]
        histos.append(objs[1])
        
        canvas.cd()
        if idx:
            histos[-1].SetLineColor(colors[ (idx - 1) % len(colors) ])
            histos[-1].SetMarkerColor(colors[ (idx - 1) % len(colors) ])
            histos[-1].SetMarkerStyle(markers[ (idx - 1) % len(markers) ])

            histos[-1].Draw(option + "SAME")

        idx += 1

    return (canvas,histos,objects)
