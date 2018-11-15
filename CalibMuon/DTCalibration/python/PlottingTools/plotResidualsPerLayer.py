from __future__ import print_function
import ROOT
from fitResidual import fitResidual
from drawHistoAllChambers import drawHisto

layerCorrectionFactors = {'SL1':(1.17,1.16,1.15,1.14),
                          'SL2':(1.83,1.20,1.20,1.83),
                          'SL3':(1.14,1.15,1.16,1.17)}

def plotResLayer(fileName,sl,layer,
                          dir='DQMData/Run 1/DT/Run summary/DTCalibValidation',
                          option="HISTOPE1",draw=True):

    mean_ymin = -0.02
    mean_ymax =  0.02
    sig_ymin = 0.
    sig_ymax = 0.1

    slType = sl
    slStr = "SL%d" % slType
    layerType = layer
    layerStr = "Layer%d" % layerType
    verbose = False
    nSigmas = 2

    ROOT.TH1.AddDirectory(False)

    file = ROOT.TFile(fileName,'read')

    wheels = (-2,-1,0,1,2)
    stations = (1,2,3,4)

    # (Wh-2 MB1 Sec1 ... Wh-2 MB1 Sec12 ... Wh-1 MB1 Sec1 ... Wh-1 MB1 Sec12 ...)
    # (Wh-2 MB2 Sec1 ... Wh-2 MB2 Sec12 ... Wh-1 MB2 Sec1 ... Wh-1 MB1 Sec12 ...) ...  
    nBins = 250
    if slType == 2: nBins = 180
    histoMean = ROOT.TH1F("h_ResMeanAll_%s_%s" % (slStr,layerStr),"Mean of residuals",nBins,0,nBins)
    histoSigma = ROOT.TH1F("h_ResSigmaAll_%s_%s" % (slStr,layerStr),"Sigma of residuals",nBins,0,nBins)
    for st in stations:
        nSectors = 12
        if st == 4: nSectors = 14
        if st == 4 and slType == 2: continue 
        if verbose: print("Station",st)
        for wh in wheels:
            if verbose: print("Wheel",wh) 
            for sec in range(1,nSectors+1):
                if verbose: print("Sector",sec)
                # Get histogram
                histoName = "%s/Wheel%d/Station%d/Sector%d/%s/hResDist_STEP3_W%d_St%d_Sec%d_%s_%s" % (dir,wh,st,sec,slStr,wh,st,sec,slStr,layerStr) 
                print("Accessing",histoName)
                histo = file.Get(histoName)
                (histo,fitFunc) = fitResidual(histo,nSigmas,verbose)
                fitMean = fitFunc.GetParameter(1)
                fitMeanErr = fitFunc.GetParError(1)
                fitSigma = fitFunc.GetParameter(2)
                fitSigmaErr = fitFunc.GetParError(2)

                layerIdx = (layer - 1)
                corrFactor = layerCorrectionFactors[slStr][layerIdx]

                binHistoNew = (st - 1)*60 + (wh + 2)*nSectors + sec
                if verbose: print("Bin in summary histo",binHistoNew)
                histoMean.SetBinContent(binHistoNew,fitMean)
                histoMean.SetBinError(binHistoNew,fitMeanErr)
                histoSigma.SetBinContent(binHistoNew,fitSigma*corrFactor)
                histoSigma.SetBinError(binHistoNew,fitSigmaErr*corrFactor)

                if sec == 1:
                    label = "Wheel %d" % wh
                    if wh == -2: label += " MB%d" % st  
                    histoMean.GetXaxis().SetBinLabel(binHistoNew,label) 
                    histoSigma.GetXaxis().SetBinLabel(binHistoNew,label)

    objectsMean = drawHisto(histoMean,title="Mean of residuals (cm)",
                                      ymin=mean_ymin,ymax=mean_ymax,option=option,draw=draw)
    objectsSigma = drawHisto(histoSigma,title="Resolution (cm)",
                                        ymin=sig_ymin,ymax=sig_ymax,option=option,draw=draw)

    return (objectsMean,objectsSigma)

def plot(fileName,sl,
                  dir='DQMData/Run 1/DT/Run summary/DTCalibValidation',type='mean',option='HISTOPE1'):
    colors = (2,4,12,44,55,38,27,46)
    markers = (24,25,26,27,28,30,32,5)
    labels=['Layer 1','Layer 2','Layer 3','Layer 4']
    idx_type = None
    if type == 'mean': idx_type = 0
    elif type == 'sigma': idx_type = 1
    else: raise RuntimeError("Wrong option: %s" % type)

    idx = 0
    canvas = None
    objects = None
    histos = []
    for layer in range(1,5):
        draw = False
        if not idx: draw = True

        objs = plotResLayer(fileName,sl,layer,dir,option,draw)
        histos.append(objs[idx_type][1])
        histos[-1].SetName( "%s_%d" % (histos[-1].GetName(),idx) )
        if not idx:
            canvas = objs[idx_type][0]
            objects = objs[idx_type][2]

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

    # Compute averages
    # (Wh-2 MB1 Sec1 ... Wh-2 MB1 Sec12 ... Wh-1 MB1 Sec1 ... Wh-1 MB1 Sec12 ...)
    # (Wh-2 MB2 Sec1 ... Wh-2 MB2 Sec12 ... Wh-1 MB2 Sec1 ... Wh-1 MB1 Sec12 ...) ...  
    import math
    wheels = (-2,-1,0,1,2)
    stations = (1,2,3,4)
    slType = sl
    slStr = "SL%d" % slType

    nBinsAve = len(stations)*len(wheels)
    histoAverage = ROOT.TH1F("h_AverageAll_" + slStr,"",nBinsAve,0,nBinsAve)
    averages = {}
    averagesErr = {}
    averagesSumw = {}
    print("Averages:")
    for st in stations:
        nSectors = 12
        if st == 4: nSectors = 14
        if st == 4 and slType == 2: continue 
        for wh in wheels:
            binHistoAve = (st - 1)*5 + (wh + 2) + 1
            label = "Wheel %d" % wh
            if wh == -2: label += " MB%d" % st  
            histoAverage.GetXaxis().SetBinLabel(binHistoAve,label) 

            averages[(st,wh)] = 0.
            averagesSumw[(st,wh)] = 0.
            for sec in range(1,nSectors+1):
                binHisto = (st - 1)*60 + (wh + 2)*nSectors + sec
                for idx in range( len(histos) ):
                    histo = histos[idx]
                    value = histo.GetBinContent( binHisto ) 
                    error = histo.GetBinError( binHisto ) 
                    averages[(st,wh)]     += value/( error*error ) 
                    averagesSumw[(st,wh)] += 1./( error*error )
            # Average per (st,wh)
            averages[(st,wh)] = averages[(st,wh)]/averagesSumw[(st,wh)]
            averagesErr[(st,wh)] = math.sqrt( 1./averagesSumw[(st,wh)] )
            histoAverage.SetBinContent(binHistoAve,averages[(st,wh)])
            histoAverage.SetBinError(binHistoAve,averagesErr[(st,wh)])
            print("Station %d, Wheel %d: %.4f +/- %.6f" % (st,wh,averages[(st,wh)],averagesErr[(st,wh)]))

    canvasAverage = ROOT.TCanvas("c_" + histoAverage.GetName())
    canvasAverage.SetGridx()
    canvasAverage.SetGridy()
    canvasAverage.SetFillColor( 0 )
    canvasAverage.cd()
    mean_ymin = -0.02
    mean_ymax =  0.02
    sig_ymin = 0.
    sig_ymax = 0.1
    if type == 'mean':
        histoAverage.GetYaxis().SetTitle("Mean of residuals (cm)")
        histoAverage.GetYaxis().SetRangeUser(mean_ymin,mean_ymax)
    elif type == 'sigma':
        histoAverage.GetYaxis().SetTitle("Resolution (cm)")
        histoAverage.GetYaxis().SetRangeUser(sig_ymin,sig_ymax)

    histoAverage.SetStats(0)
    histoAverage.SetLineWidth(2)
    histoAverage.SetMarkerStyle( 27 )
    histoAverage.SetMarkerSize( 1.5 )
    histoAverage.LabelsOption("d","X")
    histoAverage.Draw("E2")           

    return ( (canvas,canvasAverage),(histos,histoAverage),objects )

def plotMean(fileName,sl,dir='DQMData/Run 1/DT/Run summary/DTCalibValidation',option='HISTOPE1'):
    type = 'mean'
    objs = plot(fileName,sl,dir,type,option)
    return objs

def plotSigma(fileName,sl,dir='DQMData/Run 1/DT/Run summary/DTCalibValidation',option='HISTOPE1'):
    type = 'sigma'
    objs = plot(fileName,sl,dir,type,option)
    return objs

def plotSigmaAll(fileName,dir='DQMData/Run 1/DT/Run summary/DTCalibValidation',option='HISTOPE1',outputFileName=''):
    colors = (2,4,12,44,55,38,27,46)
    markers = (24,25,26,27,28,30,32,5)

    slList = (1,2,3)
    labels = ('R-#phi SL1','R-z SL2','R-#phi SL3') 
    canvas = None
    objects = None
    histos = []
    idx = 0
    for sl in slList:
        draw = False
        if not idx: draw = True

        objs = plotSigma(fileName,sl,dir,option)
        histos.append(objs[1][1])
        histos[-1].SetName( "%s_%d" % (histos[-1].GetName(),idx) )
        if not idx:
            canvas = objs[0][1]
            #objects = objs[2][1]

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
    if not objects: objects = [legend]
    else:           objects.append(legend)

    if outputFileName:
        outputFile = ROOT.TFile(outputFileName,'recreate')
        outputFile.cd()
        for histo in histos: histo.Write()
        outputFile.Close()
        return 0
    else:       
        return (canvas,histos,objects)

#def plotDataVsMCFromFile(fileNameData,fileNameMC,labels=[]):
def plotFromFile(fileNames,labels=[]):

    AddDirectoryStatus_ = ROOT.TH1.AddDirectoryStatus()
    ROOT.TH1.AddDirectory(False)

    #fileData = ROOT.TFile(fileNameData,'read')
    #fileMC = ROOT.TFile(fileNameMC,'read')
    rootFiles = []
    for file in fileNames: rootFiles.append( ROOT.TFile(file,'read') ) 

    variables = ['h_AverageAll_SL1_0',
                 'h_AverageAll_SL2_1',
                 'h_AverageAll_SL3_2']

    colors = (1,2,4,12,44,55,38,27,46)
    markers = (20,24,25,26,27,28,30,32,5)
    objects = None
    canvases = []
    legends = []
    histos = []
    idx_var = 0
    for var in variables:
        print("Accessing",var) 
        #histoData = fileData.Get(var)
        #histoData.SetName(histoData.GetName() + "_data")
        #histoMC = fileMC.Get(var)
        #histoMC.SetName(histoMC.GetName() + "_mc")
        #histoData.SetLineColor(1)
        #histoData.SetMarkerStyle(20)
        #histoData.SetMarkerSize(1.4)
        #histoData.SetMarkerColor(1)

        #histoMC.SetLineColor(2)
        #histoMC.SetMarkerStyle(24)
        #histoMC.SetMarkerSize(1.4)
        #histoMC.SetMarkerColor(2)

        histos_tmp = []
        idx = 0
        for file in rootFiles:
            histos_tmp.append( file.Get(var) )
            histos_tmp[-1].SetName( "%s_%d" % (histos_tmp[-1].GetName(),idx) )
            print("Created",histos_tmp[-1].GetName())
            histos_tmp[-1].SetLineColor(colors[ idx % len(colors) ]) 
            histos_tmp[-1].SetMarkerColor(colors[ idx % len(colors) ]) 
            histos_tmp[-1].SetMarkerStyle(markers[ idx % len(markers) ]) 
            histos_tmp[-1].SetMarkerSize(1.4)
            idx += 1
        histos.append( histos_tmp )

        canvases.append( ROOT.TCanvas("c_" + var,var) ) 
        canvases[-1].SetGridx()
        canvases[-1].SetGridy()
        canvases[-1].SetFillColor(0) 
        canvases[-1].cd()
        #histoData.Draw()
        #histoMC.Draw("SAME")
        #histos.append( (histoData,histoMC) )
        histos[-1][0].Draw()
        for histo in histos[-1][1:]: histo.Draw("SAME")

        if len(labels):
            #labelData = labels[0]
            #labelMC = labels[1]
            legends.append( ROOT.TLegend(0.4,0.7,0.95,0.8) )
            idx = 0
            for histo in histos[-1]:
                legends[-1].AddEntry(histo,labels[idx],"LP")
                idx += 1

            legends[-1].SetFillColor( canvases[-1].GetFillColor() )
            legends[-1].Draw("SAME")

        idx_var += 1

    if not objects: objects = [legends]
    else:           objects.append(legends)

    ROOT.TH1.AddDirectory(AddDirectoryStatus_)

    return (canvases,histos,objects)
