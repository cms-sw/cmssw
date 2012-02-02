import ROOT

def fitResidual(histo,nSigmas=2,verbose=False):
    option = "R0"
    if not verbose: option += "Q"

    minFit = histo.GetMean() - histo.GetRMS()
    maxFit = histo.GetMean() + histo.GetRMS()

    funcName = histo.GetName() + "_gaus"
    fitFunc = ROOT.TF1(funcName,"gaus",minFit,maxFit)
    histo.Fit(fitFunc,option)

    minFit = fitFunc.GetParameter(1) - nSigmas*fitFunc.GetParameter(2)
    maxFit = fitFunc.GetParameter(1) + nSigmas*fitFunc.GetParameter(2)
    fitFunc.SetRange(minFit,maxFit)
    histo.Fit(fitFunc,option)

    return (histo,fitFunc)
