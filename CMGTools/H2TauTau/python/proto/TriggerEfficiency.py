import copy
from CMGTools.RootTools.RootTools import *

from ROOT import gSystem

# loadLibs()

gSystem.Load("libCMGToolsH2TauTau")

from ROOT import TriggerEfficiency
    

if __name__ == '__main__':

    from ROOT import TGraph, TCanvas
    
    triggerEfficiency = TriggerEfficiency()

    npoints = 1000
    
    tauCurves = dict( p2011A = (triggerEfficiency.effTau2011A,
                                TGraph(npoints)),
                      p2011B = (triggerEfficiency.effTau2011B,
                                TGraph(npoints)),
                      p2011AB = (triggerEfficiency.effTau2011AB,
                                 TGraph(npoints))
                      )

    tauCurves2012barrel = dict( p2012A = (triggerEfficiency.effTau2012A,
                                    TGraph(npoints)),
                          p2012B = (triggerEfficiency.effTau2012B,
                                    TGraph(npoints)),
                          p2012AB = (triggerEfficiency.effTau2012AB,
                                     TGraph(npoints)),
                          p2012MC = (triggerEfficiency.effTau2012MC,
                                     TGraph(npoints))
                          )

    tauCurves2012endcaps = dict( p2012A = (triggerEfficiency.effTau2012A,
                                    TGraph(npoints)),
                          p2012B = (triggerEfficiency.effTau2012B,
                                    TGraph(npoints)),
                          p2012AB = (triggerEfficiency.effTau2012AB,
                                     TGraph(npoints)),
                          p2012MC = (triggerEfficiency.effTau2012MC,
                                     TGraph(npoints))
                          )

    muCurves2012barrel = dict( p2012A = (triggerEfficiency.effMu2012A,
                                    TGraph(npoints)),
                          p2012B = (triggerEfficiency.effMu2012B,
                                    TGraph(npoints)),
                          p2012AB = (triggerEfficiency.effMu2012AB,
                                     TGraph(npoints)),
                          p2012MC = (triggerEfficiency.effMu2012MC,
                                     TGraph(npoints))
                          )

    muCurves2012endcaps = dict( p2012A = (triggerEfficiency.effMu2012A,
                                    TGraph(npoints)),
                          p2012B = (triggerEfficiency.effMu2012B,
                                    TGraph(npoints)),
                          p2012AB = (triggerEfficiency.effMu2012AB,
                                     TGraph(npoints)),
                          p2012MC = (triggerEfficiency.effMu2012MC,
                                     TGraph(npoints))
                          )
    
    tauCurves2 = dict( tau20 = (triggerEfficiency.effIsoTau20,
                                 TGraph(npoints)),
                       tau25 = (triggerEfficiency.effIsoTau25,
                                TGraph(npoints)),
                       tau35 = (triggerEfficiency.effIsoTau35,
                                TGraph(npoints)),
                       tau45 = (triggerEfficiency.effIsoTau45,
                                TGraph(npoints)),
                      )
    
    muCurves = dict( p2011A = (triggerEfficiency.effMu2011A,
                               TGraph(npoints)),
                     p2011B = (triggerEfficiency.effMu2011B,
                               TGraph(npoints)),
                     p2011AB = (triggerEfficiency.effMu2011AB,
                                TGraph(npoints))
                     )

    eleCurves = dict( p2011A = (triggerEfficiency.effEle2011A,
                               TGraph(npoints)),
                     p2011B = (triggerEfficiency.effEle2011B,
                               TGraph(npoints)),
                     p2011AB = (triggerEfficiency.effEle2011AB,
                                TGraph(npoints))
                     )

    def fillGraphs( curves, region=None):
        for np in range(0, npoints):
            pt = np / 10.
            for period, struct in curves.iteritems():
                (fun, gr) = struct
                if region == 'Barrel':
                    gr.SetPoint( np, pt, fun( pt, 0 ) )
                elif region == 'Endcaps':
                    gr.SetPoint( np, pt, fun( pt, 2 ) )
                    

    fillGraphs( tauCurves )
    fillGraphs( tauCurves2 )
    # tauCurves2012barrel = copy.deepcopy(tauCurves2012)
    # tauCurves2012endcaps = copy.deepcopy(tauCurves2012)
    fillGraphs( tauCurves2012barrel, region='Barrel' )
    fillGraphs( tauCurves2012endcaps, region='Endcaps' )
    fillGraphs( muCurves2012barrel, region='Barrel' )
    fillGraphs( muCurves2012endcaps, region='Endcaps' )
    fillGraphs( muCurves, region='Barrel' )
    fillGraphs( eleCurves, region='Endcaps' )


    keeper = []

    def drawCurves( curves, name):
        first = True

        tauC = TCanvas(name, name)
        color = 0
        legend = TLegend(0.5,0.15,0.8, 0.45)
        keeper.append( legend )
        for name, (dummy, gr) in curves.iteritems():
            color += 1
            gr.SetLineColor(color)
            gr.SetLineWidth(3)
            legend.AddEntry( gr,name,'l' )
            if first is True: 
                gr.Draw('AL')
                gr.GetYaxis().SetRangeUser(0,1)
                first = False
            else:
                gr.Draw('Lsame')
        legend.Draw('Lsame')
        return tauC

    
    # can1 = drawCurves( tauCurves, 'tau')
    # can1b = drawCurves( tauCurves2, 'tau2')
    #can1b = drawCurves( tauCurves2012barrel, 'tau2012, barrel')
    # can1ec = drawCurves( tauCurves2012endcaps, 'tau2012, endcaps')
    can1b = drawCurves( muCurves2012barrel, 'mu2012, barrel')
    can1ec = drawCurves( muCurves2012endcaps, 'mu2012, endcaps')
    # can2 = drawCurves( muCurves, 'mu')
    # can3 = drawCurves( eleCurves, 'ele')


    def graphEta( curves, pt = 20):

        
        etaMax = 3.
        etaBin = 2*etaMax / npoints
        
        for np in range(0, npoints):
            eta = -etaMax + np * etaBin  
            for period, struct in curves.iteritems():
                (fun, gr) = struct
                gr.SetPoint( np, eta, fun( pt, eta) )

