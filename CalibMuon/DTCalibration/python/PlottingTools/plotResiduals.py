import ROOT
from fitResidual import fitResidual
from drawHistoAllChambers import drawHisto

def plot(fileName,sl,dir='DQMData/Run 1/DT/Run summary/DTCalibValidation',option="HISTOPE1",mode="None",draw=True):

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
        if verbose: print("Station",st)
        for wh in wheels:
            if verbose: print("Wheel",wh) 
            for sec in range(1,nSectors+1):
                if verbose: print("Sector",sec)
                # Get histogram
                histoName = "%s/Wheel%d/Station%d/Sector%d/hResDist_STEP3_W%d_St%d_Sec%d_%s" % (dir,wh,st,sec,wh,st,sec,slStr)
    
                print("Accessing",histoName)
                histo = file.Get(histoName)
                (histo,fitFunc) = fitResidual(histo,nSigmas,verbose)
                fitMean = fitFunc.GetParameter(1)
                fitMeanErr = fitFunc.GetParError(1)
                fitSigma = fitFunc.GetParameter(2)
                fitSigmaErr = fitFunc.GetParError(2)

                binHistoNew = (st - 1)*60 + (wh + 2)*nSectors + sec
                if verbose: print("Bin in summary histo",binHistoNew)
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
    objectsSigma = drawHisto(histoSigma,title="Sigma of residuals (cm)",ymin=sig_ymin,ymax=sig_ymax,option=option,draw=draw)


    objectsMean[0].cd()
    objectsMean[0].SaveAs("plot_"+mode+"_Mean.png")


    objectsSigma[0].cd()
    objectsSigma[0].SaveAs("plot_"+mode+"_Sigma.png")
    
    return (objectsMean,objectsSigma)



if __name__ == "__main__":

    fileName = "./Run379617-PPtest_v1/Residuals/results/residuals.root"
    directory = "DTResiduals"
    mode = "Test"

    #fileName = "./Run379617-ValTest_v1/TtrigValidation/results/DQM_V0001_R000000001__ExpressPhysics__Run2024C-Express-v1__FEVT.root"
    #directory = "DQMData/Run 1/DT/Run summary/DTCalibValidation"
    #mode = "Val"
    sl = 2
    re = plot(fileName, sl, dir=directory, mode=mode)[0]

    print(re)
