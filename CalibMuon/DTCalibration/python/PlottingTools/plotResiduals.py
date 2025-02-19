import ROOT
from fitResidual import fitResidual
from drawHistoAllChambers import drawHisto

def plot(fileName,sl,dir='DQMData/Run 1/DT/Run summary/DTCalibValidation',option="HISTOPE1",draw=True):

    mean_ymin = -0.02
    mean_ymax =  0.02
    sig_ymin = 0.
    sig_ymax = 0.07

    slType = sl
    slStr = "SL%d" % slType
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
    histoMean = ROOT.TH1F("h_ResMeanAll","Mean of residuals",nBins,0,nBins)
    histoSigma = ROOT.TH1F("h_ResSigmaAll","Sigma of residuals",nBins,0,nBins)
    for st in stations:
        nSectors = 12
        if st == 4: nSectors = 14
        if st == 4 and slType == 2: continue 
        if verbose: print "Station",st
        for wh in wheels:
            if verbose: print "Wheel",wh 
            for sec in range(1,nSectors+1):
                if verbose: print "Sector",sec
                # Get histogram
                histoName = "%s/Wheel%d/Station%d/Sector%d/hResDist_STEP3_W%d_St%d_Sec%d_%s" % (dir,wh,st,sec,wh,st,sec,slStr) 
                print "Accessing",histoName
                histo = file.Get(histoName)
                (histo,fitFunc) = fitResidual(histo,nSigmas,verbose)
                fitMean = fitFunc.GetParameter(1)
                fitMeanErr = fitFunc.GetParError(1)
                fitSigma = fitFunc.GetParameter(2)
                fitSigmaErr = fitFunc.GetParError(2)

                binHistoNew = (st - 1)*60 + (wh + 2)*nSectors + sec
                if verbose: print "Bin in summary histo",binHistoNew
                histoMean.SetBinContent(binHistoNew,fitMean)
                histoMean.SetBinError(binHistoNew,fitMeanErr)
                histoSigma.SetBinContent(binHistoNew,fitSigma)
                histoSigma.SetBinError(binHistoNew,fitSigmaErr)
  
                if sec == 1:
                    label = "Wheel %d" % wh
                    if wh == -2: label += " MB%d" % st  
                    histoMean.GetXaxis().SetBinLabel(binHistoNew,label) 
                    histoSigma.GetXaxis().SetBinLabel(binHistoNew,label)

    objectsMean = drawHisto(histoMean,title="Mean of residuals (cm)",
                                      ymin=mean_ymin,ymax=mean_ymax,option=option,draw=draw)
    objectsSigma = drawHisto(histoSigma,title="Sigma of residuals (cm)",
                                        ymin=sig_ymin,ymax=sig_ymax,option=option,draw=draw)

    return (objectsMean,objectsSigma)
