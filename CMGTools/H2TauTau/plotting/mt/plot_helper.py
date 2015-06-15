import math
import copy
import ROOT

from CMGTools.H2TauTau.proto.plotter.rootutils import *
from CMGTools.H2TauTau.proto.plotter.blind import blind
from CMGTools.H2TauTau.proto.plotter.plotmod import *
from plot_H2TauTauDataMC_TauMu_All import *

def qcdIsoPlots(hist, NBINS, XMIN, XMAX, parameters):

    parameters['subtractBGForQCDShape'] = True
    parameters['antiMuIsoForQCD'] = False
    ssignAM, osignAM, ssQCDAM, osQCDAM = makePlot( hist, NBINS, XMIN, XMAX, **parameters)

    parameters['subtractBGForQCDShape'] = False
    parameters['antiMuIsoForQCD'] = True
    ssignATM, osignATM, ssQCDATM, osQCDATM = makePlot( hist, NBINS, XMIN, XMAX, **parameters)

    # can = buildCanvasOfficial()
    can, pad, padr = buildCanvas()
    pad.cd()
    #
    # hQCD = osQCD.Hist('QCD').weighted.Clone()
    # hQCD.GetXaxis().SetTitle('m_{#tau #tau} [GeV]')
    # hQCD.GetYaxis().SetTitle('Events')
    # hQCD.SetFillColor(0)
    # hQCD.SetLineColor(hQCD.GetMarkerColor())
    # hQCD.SetTitle('SS')
    # hQCDAM = osQCDAM.Hist('QCD').weighted
    # hQCDAM.SetTitle('QCD SS #mu')
    #
    hQCDAM = osQCDAM.Hist('QCD').weighted
    hQCDAM.SetTitle('SS BG-sub')
    hQCDAM.GetXaxis().SetTitle('m_{#tau #tau} [GeV]')
    hQCDAM.GetYaxis().SetTitle('Events')
    #
    hQCDAM.SetMarkerColor(1)
    hQCDAM.SetLineColor(1)
    hQCDAM.SetFillColor(0)
    #
    hQCDATM = osQCDATM.Hist('QCD').weighted
    # hQCDATM.SetTitle('QCD SS #mu #tau')
    hQCDATM.SetTitle('SS anti-#mu iso')
    # hQCDATM.SetTitle('QCD SS')
    hQCDATM.SetMarkerColor(2)
    hQCDATM.SetLineColor(2)
    hQCDATM.SetFillColor(0)
    leg = ROOT.TLegend(0.5,0.46,0.88,0.89)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetLineColor(0)
    # leg.AddEntry(hQCD, hQCD.GetTitle())
    leg.AddEntry(hQCDAM, hQCDAM.GetTitle())
    leg.AddEntry(hQCDATM, hQCDATM.GetTitle())
    # ks1 = hQCD.KolmogorovTest(hQCDAM)
    # ks2 = hQCD.KolmogorovTest(hQCDATM)
    ks3 = hQCDAM.KolmogorovTest(hQCDATM)
    # chi2_1 = hQCD.Chi2Test(hQCDAM, 'WW')
    # chi2_2 = hQCD.Chi2Test(hQCDATM, 'WW')
    chi2_3 = hQCDAM.Chi2Test(hQCDATM, 'WW')
    oldTitle = hQCDAM.GetTitle()
    hQCDAM.SetTitle('')
    hQCDAM.Draw()
    maxVal = max(hQCDATM.GetMaximum() + hQCDATM.GetBinError(hQCDATM.GetMaximumBin()), hQCDAM.GetMaximum() + hQCDAM.GetBinError(hQCDAM.GetMaximumBin()))
    hQCDAM.GetYaxis().SetRangeUser(0., maxVal*1.3)
    # hQCD.Draw("SAME")
    hQCDATM.Draw("SAME")
    dummyl = ROOT.TLine()
    dummyl.SetLineColor(0)
    # leg.AddEntry(dummyl, 'KS: ' + str(round(ks1, 2)) + ' #chi^{2}: ' + str(round(chi2_1, 2)), 'l')
    # leg.AddEntry(dummyl, 'KS: ' + str(round(ks2, 2)) + ' #chi^{2}: ' + str(round(chi2_2, 2)), 'l')
    leg.AddEntry(dummyl, 'KS: ' + str(round(ks3, 2)) + ' #chi^{2}: ' + str(round(chi2_3, 2)), 'l')
    leg.Draw()
    padr.cd()
    hr = copy.deepcopy(hQCDAM)
    hr.Divide(hQCDATM)
    hr.GetYaxis().SetNdivisions(4)
    # hr.GetYaxis().SetTitle(oldTitle+'/'+hQCDATM.GetTitle())
    hr.GetYaxis().SetTitle('BG-sub/anti-#mu-iso')
    hr.GetYaxis().SetTitleSize(0.1)
    hr.GetYaxis().SetTitleOffset(0.5)
    hr.GetXaxis().SetTitle('{xtitle}'.format(xtitle='m_{#tau #tau} [GeV]'))
    hr.GetXaxis().SetTitleSize(0.13)
    hr.GetXaxis().SetTitleOffset(0.9)
    rls = 0.075
    hr.GetYaxis().SetLabelSize(rls)
    hr.GetXaxis().SetLabelSize(rls)
    hr.GetYaxis().SetRangeUser(0.5, 1.5)

    hrup = hr.Clone()
    hrup.SetTitle(hr.GetTitle()+'up')
    hrup.SetMarkerStyle(22)
    hrdown = hr.Clone()
    hrdown.SetTitle(hr.GetTitle()+'down')
    hrdown.SetMarkerStyle(23)
    hrdown.SetMarkerStyle(23)

    for iBin in range(hr.GetNbinsX()):
        if hQCDATM.GetBinContent(iBin+1) == 0. and hQCDAM.GetBinContent(iBin+1) != 0.:
            hr.SetBinContent(iBin+1, 9999.)
        elif hQCDATM.GetBinContent(iBin+1) == 0. and hQCDAM.GetBinContent(iBin+1) == 0.:
            hr.SetBinContent(iBin+1, 1.)

        if hr.GetBinContent(iBin + 1) > 1.5:
            hrup.SetBinContent(iBin+1, 1.4)
        else:
            hrup.SetBinContent(iBin+1, -9999.)

        if hr.GetBinContent(iBin + 1) < 0.5:
            hrdown.SetBinContent(iBin+1, 0.6)
        else:
            hrdown.SetBinContent(iBin+1, -9999.)
        hrdown.SetBinError(iBin+1, 0.)
        hrup.SetBinError(iBin+1, 0.)

    hr.Draw()
    hrup.Draw('same p')
    hrdown.Draw('same p')
    line = ROOT.TLine()
    if XMIN or XMAX:
        line.DrawLine(float(XMIN), 1, float(XMAX), 1)
    else:
        line.DrawLine(NBINS[0], 1, NBINS[-1], 1)
    padr.Update()
    can.Print('QCD_comparison'+parameters['cutName']+'.pdf')

    return osQCDAM, osQCDATM

def printDataVsQCDInfo(osQCD, ssQCD):
    print "Fit 0-60"
    osQCD.ratioTotalHist.weighted.Fit('pol0', '', '', 0., 60.)
    print "Fit Inclusive"
    osQCD.ratioTotalHist.weighted.Fit('pol0')
        
    dataHist = osQCD.Hist('Data')
    totalMC = osQCD.stack.totalHist
    qcdHist = osQCD.Hist('QCD')

    ssDataHist = ssQCD.Hist('Data')

    print ' --------- KS Test Probability', dataHist.weighted.KolmogorovTest(totalMC.weighted) 
    print ' ------- Chi2 Test Probability', dataHist.weighted.Chi2Test(totalMC.weighted, 'UW') 

    print 'Range: Inclusive'
    print '--- Yield data SS', ssDataHist.Yield()
    print '--- Yield data', dataHist.Yield()
    print '--- Yield stack', totalMC.Yield()
    yieldDataMinusMC = dataHist.Yield() - osQCD.Hist('Ztt').Yield() - osQCD.Hist('TTJets').Yield()- osQCD.Hist('electroweak').Yield()
    print '--- Yield data - MC / Yield QCD', yieldDataMinusMC/qcdHist.Yield()
    print '--- Rough error estimate ----', math.sqrt(yieldDataMinusMC+qcdHist.Yield())/yieldDataMinusMC

    rangePairs = [(0., 30.), (0., 40.), (0., 50.), (0., 60.), (0., 70.), (0., 80.), (40., 80.), (40., 100.),  (60., 100.), (0., 100.), (0., 150.)]
    for pair in rangePairs:
        xmin, xmax = pair
        print 'Range:', xmin, xmax
        print '--- Yield data SS', ssDataHist.Integral(xmin=xmin, xmax=xmax)
        print '--- Yield data', dataHist.Integral(xmin=xmin, xmax=xmax)
        print '--- Yield stack', totalMC.Integral(xmin=xmin, xmax=xmax)
        yieldDataMinusMC = dataHist.Integral(xmin=xmin, xmax=xmax) - osQCD.Hist('Ztt').Integral(xmin=xmin, xmax=xmax) - osQCD.Hist('TTJets').Integral(xmin=xmin, xmax=xmax)- osQCD.Hist('electroweak').Integral(xmin=xmin, xmax=xmax)
        print '--- Yield data - MC / Yield QCD', yieldDataMinusMC/qcdHist.Integral(xmin=xmin, xmax=xmax)
        print '--- Rough error estimate ----', math.sqrt(yieldDataMinusMC+qcdHist.Integral(xmin=xmin, xmax=xmax))/yieldDataMinusMC



        