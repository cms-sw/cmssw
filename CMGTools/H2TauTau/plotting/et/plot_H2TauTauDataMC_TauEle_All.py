import imp
import math
import copy
import time
import re
import os
import string
import ROOT

from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC import H2TauTauDataMC
from CMGTools.H2TauTau.proto.plotter.prepareComponents import prepareComponents
from CMGTools.H2TauTau.proto.plotter.rootutils import *
from CMGTools.H2TauTau.proto.plotter.categories_TauEle import *
from CMGTools.H2TauTau.proto.plotter.binning import binning_svfitMass_finer, binning_svfitMass, binning_svfitMass_mssm2013, binning_svfitMass_mssm, binning_svfitMass_mssm_nobtag
from CMGTools.H2TauTau.proto.plotter.titles import xtitles
from CMGTools.H2TauTau.proto.plotter.blind import blind
from CMGTools.H2TauTau.proto.plotter.plotmod import *
from CMGTools.H2TauTau.proto.plotter.datacards import *
from CMGTools.H2TauTau.proto.plotter.embed import setupEmbedding as setupEmbedding
from CMGTools.H2TauTau.proto.plotter.plotinfo import plots_All, plots_All_sorted_indices
# from CMGTools.RootTools.Style import *
from ROOT import kPink, TH1, TPaveText, TPad

from plot_helper import *


cp = copy.deepcopy
EWK = 'WJets'

    
NBINS = 100
XMIN  = 0
XMAX  = 200

def replaceShape(compName, shapeCut, osign, shapePlotInfo):
    var, anaDir, selComps, weights, nbins, xmin, xmax, weight, embed, shift = shapePlotInfo
    # print compName, 'replacing shape', shapeCut

    fileCompName = compName
    if 'Ztt_' in compName:
        fileCompName = 'Ztt'

    if osign.Hist(compName).Integral() == 0.:
        print 'INFO: in shape replacement, integral 0 for component', compName, 'returning'
        return

    pl = H2TauTauDataMC(var, anaDir,
                        {selComps[fileCompName].name:selComps[fileCompName]}, weights, nbins, xmin, xmax,
                        shapeCut, weight,
                        embed, shift,
                           treeName = 'H2TauTauTreeProducerTauEle' )

    print '\nDEBUG for', compName
    print 'DEBUG', shapeCut
    print 'DEBUG', selComps[fileCompName].name, '\n'

    shape = copy.deepcopy( pl.Hist(compName) ) 
    print compName, 'Int(shape)', shape.Integral(), 'Int(tight)', osign.Hist(compName).Integral()
    if shape.Integral() > 0. and osign.Hist(compName).Integral() > 0.:
        print 'Scaling new shape', osign.Hist(compName).Integral() / shape.Integral()
        shape.Scale( osign.Hist(compName).Integral() / shape.Integral())
        osign.Replace(compName, shape )
    else:
        print 'WARNING in shape replacement, integral 0 for component', compName, 'SHOULD NOT HAPPEN'

def replaceShapeInclusive(plot, var, anaDir,
                          comp, weights, 
                          cut, weight,
                          embed, shift):
    '''Replace WJets with the shape obtained using a relaxed selection'''

    print '\nINFO: Replacing W shape'

    if cat_VBF_loose in cut:
        print 'VBF loose: Relaxing VBF selection for W shape'
        cut = cut.replace(cat_VBF, cat_VBF_Rel_30)
    elif 'l1_pt>45. && pthiggs>100.' in cut or cat_VBF_tight in cut:
        print '1 jet high medium higgs and VBF tight: Relax tau isolation for W shape'
        cut = cut.replace('l1_threeHitIso<1.5', 'l1_threeHitIso<10.0')
    
    
    if cat_VBF_tight in cut or cat_VBF_loose in cut or 'nJets>=1' in cut:
        print 'VBF or 1-jet categories: Relaxing OS requirement for W shape'
        cut = cut.replace('diTau_charge==0', '1.')

    # cut = cut.replace('nBJets>=1', '1')
    print '[INCLUSIVE] estimate',comp.name,'with cut',cut
    plotWithNewShape = cp( plot )
    wjyield = plot.Hist(comp.name).Integral()
    nbins = plot.bins
    xmin = plot.xmin
    xmax = plot.xmax
    wjshape = shape(var, anaDir,
                    comp, weights, nbins, xmin, xmax,
                    cut, weight,
                    embed, shift,
                    treeName = 'H2TauTauTreeProducerTauEle')
    # import pdb; pdb.set_trace()
    wjshape.Scale( wjyield )
    
    plotWithNewShape.Replace(comp.name, wjshape) 
    # plotWithNewShape.Hist(comp.name).on = False 
    return plotWithNewShape

    

def makePlot( var, nbins, xmin, xmax, 
              anaDir, selComps, weights, wInfo,
              cut='', weight='weight', embed=False, shift=None, replaceW=False,
              VVgroup=None, TTgroup=None, antiEleIsoForQCD=False, antiEleRlxTauIsoForQCD=False, antiEleRlxTauIsoForQCDYield=False,
              subtractBGForQCDShape=False, embedForSS=False, relSelection={}, osForWExtrapolation=True, incQCDYield=99999., qcdYieldInclusiveExtrapolation=False, dataComps={}, cutName='', isZeroB=False, isOneJet=False):
    
    print '\nMaking the plot:', var, 'cut', cut
    print 'QCD setup', antiEleIsoForQCD, antiEleRlxTauIsoForQCD, subtractBGForQCDShape

    wJetScaleSS, wJetScaleOS, w_mt_ratio_ss, w_mt_ratio_os, w_mt_ratio = wInfo

    oscut = cut+' && diTau_charge==0'
    osign = H2TauTauDataMC(var, anaDir,
                           selComps, weights, nbins, xmin, xmax,
                           cut=oscut, weight=weight, shift=shift,
                           embed=embed,
                           treeName = 'H2TauTauTreeProducerTauEle')
    osign.Hist(EWK).Scale( wJetScaleOS )

    print 'USING wJetScaleOS', wJetScaleOS
    
    # No OS requirement for MT extrapolation in 1-jet categories
    if cut.find('mt<')!=-1 and not osForWExtrapolation:
        print 'correcting high->low mT extrapolation factor, OS', w_mt_ratio / w_mt_ratio_os
        osign.Hist(EWK).Scale( w_mt_ratio / w_mt_ratio_os )
    if replaceW:
        wweight = weight
        if shift and 'WShape' in shift:
            if 'Up' in shift:
                wweight = 'weight*tauFakeRateWeightUp/tauFakeRateWeight'
            elif 'Down' in shift:
                wweight = 'weight*tauFakeRateWeightDown/tauFakeRateWeight'
        osign = replaceShapeInclusive(osign, var, anaDir,
                                      selComps['WJets'], weights, 
                                      oscut, wweight,
                                      embed, shift)

    print '\nINFO Replacing shapes'
    shapePlotInfo = var, anaDir, selComps, weights, nbins, xmin, xmax, weight, embed, shift
    for sample in relSelection:
        if sample != 'QCD':
            print 'Replace shape', sample
            print 'Tight cut', cut+' && diTau_charge==0'
            print 'Relaxed cut', relSelection[sample]+' && diTau_charge==0'
            replaceShape(sample, relSelection[sample]+' && diTau_charge==0', osign, shapePlotInfo)

    sscut = cut+' && diTau_charge!=0'
    if not embedForSS and embed:
        print '\nINFO, as opposed to old default, not using embedded samples for SS region, but only for OS'

    ssign = H2TauTauDataMC(var, anaDir,
                           selComps, weights, nbins, xmin, xmax,
                           cut=sscut, weight=weight, shift=shift,
                           embed=embedForSS,
                           treeName = 'H2TauTauTreeProducerTauEle')
    ssign.Hist(EWK).Scale( wJetScaleSS )

    # if cut.find('mt<')!=-1:
    #     if w_mt_ratio_ss > 0.:
    #         print 'correcting high->low mT extrapolation factor, SS', w_mt_ratio / w_mt_ratio_ss
    #         ssign.Hist(EWK).Scale( w_mt_ratio / w_mt_ratio_ss  )
    #     else:
    #         print 'WARNING! Not correcting W mT ratio from SS to OS region: No events in SS high mT'

    # if replaceW:
    #     ssign = replaceShapeInclusive(ssign, var, anaDir,
    #                                   selComps['WJets'], weights, 
    #                                   sscut, weight,
    #                                   embed, shift)

    if VVgroup:
        ssign.Group('VV',VVgroup)
        osign.Group('VV',VVgroup)
    if TTgroup:
        ssign.Group('TTJets', TTgroup)
        osign.Group('TTJets', TTgroup)

    print '\nINFO, Estimating QCD'
    if subtractBGForQCDShape:
        print 'Subtracting BG for QCD shape'

    ssQCD, osQCD = getQCD( ssign, osign, 'Data', VVgroup, subtractBGForShape=subtractBGForQCDShape)

    # Calculate QCD yield for VBF and 1 jet high med higgs
    if qcdYieldInclusiveExtrapolation:
         # QCD, Inclusive, SS, anti-isolation, for QCD efficiency
        inc_qcd_cut = ' && '.join([cat_Inc, 'mt<30', 'diTau_charge!=0']) # FIXME: fixed for now, make configurable? guess from cat string?
        cat_qcd_cut = sscut
        if antiEleIsoForQCD:
            print 'INFO: Inverting ele iso for QCD yield extrapolation'
            inc_qcd_cut = inc_qcd_cut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5')
            cat_qcd_cut = cat_qcd_cut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5')
        if antiEleRlxTauIsoForQCDYield:
            print 'INFO: Inverting ele and relaxing tau iso for QCD yield extrapolation'
            inc_qcd_cut = inc_qcd_cut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5').replace('l1_threeHitIso<1.5', 'l1_threeHitIso<10.') + ' && diTau_charge!=0'
            cat_qcd_cut = cat_qcd_cut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5').replace('l1_threeHitIso<1.5', 'l1_threeHitIso<10.') + ' && diTau_charge!=0'


        inc_qcd_plot = buildPlot( var, anaDir, dataComps, weights, nbins, xmin, xmax,
                                  inc_qcd_cut, weight, embed, treeName = 'H2TauTauTreeProducerTauEle')
        inc_qcd_yield = inc_qcd_plot.Hist('Data').Integral()

        cat_qcd_plot = buildPlot( options.hist, anaDir,
                                  dataComps, weights, NBINS, XMIN, XMAX,
                                  cat_qcd_cut, weight, options.embed, treeName = 'H2TauTauTreeProducerTauEle')
        cat_qcd_yield = cat_qcd_plot.Hist('Data').Integral()
        
        qcd_cat_eff = cat_qcd_yield / inc_qcd_yield
        print 'Extrapolation factor:', qcd_cat_eff

        print 'Correcting QCD yield'
        print 'Old yield', osQCD.Hist('QCD').Integral()

        osQCD.Hist('QCD').Scale(qcd_cat_eff*incQCDYield/osQCD.Hist('QCD').Integral())
        print 'New yield', osQCD.Hist('QCD').Integral()

    if antiEleIsoForQCD or antiEleRlxTauIsoForQCD:
        qcdcut = cut
        if 'QCD' in relSelection:
            qcdcut = relSelection['QCD']
            print 'INFO, replacing QCD shape', qcdcut
        if antiEleIsoForQCD:
            print 'INFO: INVERTING ELE ISO FOR QCD SHAPE'
            sscut_qcdshape = qcdcut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5') + ' && diTau_charge!=0'
        if antiEleRlxTauIsoForQCD:
            print 'INFO: RELAXING TAU AND INVERTING ELE ISO FOR QCD SHAPE'
            sscut_qcdshape = qcdcut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5').replace('l1_threeHitIso<1.5', 'l1_threeHitIso<10.') + ' && diTau_charge!=0'

        qcd_yield = osQCD.Hist('QCD').Integral()
        
        ssign_qcdshape = H2TauTauDataMC(var, anaDir,
                                        dataComps, weights, nbins, xmin, xmax,
                                        cut=sscut_qcdshape, weight=weight,
                                        embed=embed,
                           treeName = 'H2TauTauTreeProducerTauEle')
        print var, anaDir, dataComps, weights, nbins, xmin, xmax, sscut_qcdshape, weight, embed
        qcd_shape = copy.deepcopy( ssign_qcdshape.Hist('Data') )   
        print ssign_qcdshape 
        # import pdb; pdb.set_trace()
        qcd_shape.Normalize()
        qcd_shape.Scale(qcd_yield)
        # qcd_shape.Scale( qcd_yield )
        old_qcd_shape = osQCD.Hist('QCD')
        osQCD.old_qcd_shape = copy.deepcopy(old_qcd_shape)
        osQCD.Replace('QCD', qcd_shape)

    # Scale QCD yield for mtt < 50 by factor 1.1 in one-jet categories unless tau iso is relaxed
    if (isOneJet or isZeroB) and not antiEleRlxTauIsoForQCD:
        for iBin in range(1, osQCD.Hist('QCD').weighted.GetNbinsX()):
            if osQCD.Hist('QCD').weighted.GetBinCenter(iBin) < 50.:
                osQCD.Hist('QCD').weighted.SetBinContent(iBin, osQCD.Hist('QCD').weighted.GetBinContent(iBin) * 1.1)
                osQCD.Hist('QCD').weighted.SetBinError(iBin, osQCD.Hist('QCD').weighted.GetBinError(iBin) * 1.1)

    # # Extra sausage for ZL in VBF loose: Normalise yield to yield in 2 jet
    # # plus extrapolation to VBF loose
    # if cat_VBF_loose in cut:
    #     dycomp = selComps['Ztt']
    #     cut2Jet = cut.replace(cat_VBF_loose, cat_J2)
    #     dyplot2Jet = buildPlot(var, anaDir, 
    #                    {dycomp.name:dycomp}, weights,
    #                    nbins, xmin, xmax,
    #                    cut2Jet, weight,
    #                    embed, shift, treeName = 'H2TauTauTreeProducerTauEle')
    #     dyplot = buildPlot(var, anaDir, 
    #                    {dycomp.name:dycomp}, weights,
    #                    nbins, xmin, xmax,
    #                    cut, weight,
    #                    embed, shift, treeName = 'H2TauTauTreeProducerTauEle')

    #     zlYield2Jet = dyplot2Jet.Hist('Ztt_ZL').Integral()
    #     zttYield2Jet = dyplot2Jet.Hist('Ztt').Integral()
    #     zlYieldVBFloose = dyplot.Hist('Ztt_ZL').Integral()
    #     zttYieldVBFloose = dyplot.Hist('Ztt').Integral()


    #     print 'INFO: Estimating ZL yield in VBF loose'
    #     print '      Yield in 2 jet', zlYield2Jet
    #     print '      Yield in VBF l', zlYieldVBFloose
    #     print '      Yield in 2 jet', zttYield2Jet
    #     print '      Yield in VBF l', zttYieldVBFloose

    #     print '      Scaling by', zttYieldVBFloose/zttYield2Jet * zlYield2Jet/zlYieldVBFloose

    #     osQCD.Hist('Ztt_ZL').Scale(zttYieldVBFloose/zttYield2Jet * zlYield2Jet/zlYieldVBFloose)
    if cat_J1B in cut:
        print 'INFO: Subtracting 1.5% of the Ztt yield from the ttbar yield in the 1 b-tag category'
        scaleZtt = osQCD.Hist('Ztt').Integral()
        scaleTT = osQCD.Hist('TTJets').Integral()
        scaleTTNew = scaleTT - scaleZtt * 0.015
        osQCD.Hist('TTJets').Scale(scaleTTNew/scaleTT)

    # osQCD.Group('electroweak', ['WJets', 'Ztt_ZL', 'Ztt_ZJ','VV'])
    osQCD.Group('electroweak', ['WJets', 'Ztt_ZJ','VV', 'Ztt_TL'])
    osQCD.Group('Higgs 125', ['HiggsVBF125', 'HiggsGGH125', 'HiggsVH125'])
    return ssign, osign, ssQCD, osQCD


def drawAll(plots, outDir, blind, parameters, isMSSM=False):
    '''See plotinfo for more information'''

    # for name, plot in plots.items():
    for iName, name in enumerate(plots_All_sorted_indices):
        plot = plots[name]
        print plot.var
        print '----------------', plot.xmin, plot.xmax, plot.nbins
        # print fwss, fwos
        ss, osign, ssQ, osQ = makePlot(plot.var, plot.nbins, plot.xmin, plot.xmax, **parameters)

        if blind and isMSSM:
            osQ.blindxmin = 100.
            osQ.blindxmax = 1000.
        elif blind and plot.var == 'visMass':
            osQ.blindxmin = 60.
            osQ.blindxmax = 120.

        varOutDir = outDir+parameters['cutName']+'/'
        if not os.path.exists(varOutDir):
            os.makedirs(varOutDir)
        varOutDir += 'v'+string.lowercase[iName] if iName < 26 else 'v'+str(iName)
        if name != plot.var:
            varOutDir += name + '_'
        # drawOfficial(osQ, blind, plotprefix=varOutDir)
        draw(osQ, blind, channel='TauEle', plotprefix=varOutDir)
        # osQ.ratioTotalHist.weighted.Fit('pol0', '', '', 0., 60.)
        # plot.ssign = cp(ss)
        # plot.osign = cp(osign)
        # plot.ssQCD = cp(ssQ)
        # plot.osQCD = cp(osQ)
        # time.sleep(1)


def handleW( anaDir, selComps, weights,
             cut, relCut, weight, embed, VVgroup, TTgroup=None, nbins=50, highMTMin=70., highMTMax=1070,
             lowMTMax=30.):
    print '\nHANDLING W'
    cut = cut.replace('mt<30', '1')
    cut = cut.replace('mt<20', '1')
    cut = cut.replace('mt<', 'mt<9999')
    # cut = cut.replace('mt>', 'mt>-0.00000')

    if 'mt>' in cut:
        lowMTMax = highMTMax

    relCut = relCut.replace('mt<30', '1')
    relCut = relCut.replace('mt<20', '1')

    fwss, fwos, ss, os = plot_W(
        anaDir, selComps, weights,
        nbins, highMTMin, highMTMax, cut,
        weight=weight, embed=embed,
        VVgroup=VVgroup, TTgroup=TTgroup, treeName='H2TauTauTreeProducerTauEle')


    w_mt_ratio_ss = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, relCut, weight, lowMTMax, highMTMin, highMTMax, 'diTau_charge!=0', treeName='H2TauTauTreeProducerTauEle')
    w_mt_ratio_os = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, relCut, weight, lowMTMax, highMTMin, highMTMax, 'diTau_charge==0', treeName='H2TauTauTreeProducerTauEle')
    w_mt_ratio = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, relCut, weight, lowMTMax, highMTMin, highMTMax, '1', treeName='H2TauTauTreeProducerTauEle')

    # Use relaxed cut for high-low ratio
    if relCut != cut:
        print 'Use relaxed cut for high-low MT ratio', relCut
        print 'Correct W high MT OS scale factor for relaxed high-low ratio'
        w_mt_ratio_ss_tight = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, cut, weight, lowMTMax, highMTMin, highMTMax, 'diTau_charge!=0', treeName='H2TauTauTreeProducerTauEle')
        w_mt_ratio_os_tight = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, cut, weight, lowMTMax, highMTMin, highMTMax, 'diTau_charge==0', treeName='H2TauTauTreeProducerTauEle')
        w_mt_ratio_tight = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, cut, weight, lowMTMax, highMTMin, highMTMax, '1', treeName='H2TauTauTreeProducerTauEle')

        print 'Factor OS', w_mt_ratio_os/w_mt_ratio_os_tight
        fwos *= w_mt_ratio_os/w_mt_ratio_os_tight

    # import pdb; pdb.set_trace()
    print 'FWSS, FWOS', fwss, fwos
    print 'W MT Ratios (SS, OS, all)', w_mt_ratio_ss, w_mt_ratio_os, w_mt_ratio
    return fwss, fwos, w_mt_ratio_ss, w_mt_ratio_os, w_mt_ratio


if __name__ == '__main__':

    import copy
    from optparse import OptionParser
    from CMGTools.RootTools.RootInit import *
    from CMGTools.H2TauTau.proto.plotter.officialStyle import officialStyle
    officialStyle(gStyle)


    parser = OptionParser()
    parser.usage = '''
    %prog <anaDir> <cfgFile>

    cfgFile: analysis configuration file, see CMGTools.H2TauTau.macros.MultiLoop
    anaDir: analysis directory containing all components, see CMGTools.H2TauTau.macros.MultiLoop.
    hist: histogram you want to plot
    '''
    parser.add_option("-H", "--hist", 
                      dest="hist", 
                      help="histogram list",
                      default=None)
    parser.add_option("-C", "--cut", 
                      dest="cut", 
                      help="cut to apply in TTree::Draw",
                      default='1')
    parser.add_option("-N", "--cutName", 
                      dest="cutName", 
                      help="name to prepend for output plots.",
                      default='')
    parser.add_option("-E", "--embed", 
                      dest="embed", 
                      help="Use embedd samples.",
                      action="store_true",
                      default=False)
    parser.add_option("-B", "--blind", 
                      dest="blind", 
                      help="Blind.",
                      action="store_true",
                      default=False)
    parser.add_option("-a", "--all", 
                      dest="allPlots", 
                      help="All plots.",
                      action="store_true",
                      default=False)
    parser.add_option("-b", "--batch", 
                      dest="batch", 
                      help="Set batch mode.",
                      action="store_true",
                      default=False)
    parser.add_option("-n", "--nbins", 
                      dest="nbins", 
                      help="Number of bins",
                      default=None)
    parser.add_option("-m", "--min", 
                      dest="xmin", 
                      help="xmin",
                      default=None)
    parser.add_option("-M", "--max", 
                      dest="xmax", 
                      help="xmax",
                      default=None)
    parser.add_option("-g", "--higgs", 
                      dest="higgs", 
                      help="Higgs mass: 125, 130,... or dummy",
                      default=None)
    parser.add_option("-k", "--mssmBinning", 
                      dest="mssmBinning", 
                      help="Binning for MSSM: 'fine', '2013', 'default'",
                      default='default')
    parser.add_option("-p", "--prefix", 
                      dest="prefix", 
                      help="Prefix for datacards",
                      default=None)
    parser.add_option("-s", "--shift", 
                      dest="shift", 
                      help="Shift to apply specific systematics",
                      default=None)
    
    (options,args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    if options.batch:
        gROOT.SetBatch()
    if options.nbins is None:
        NBINS = binning_svfitMass_finer
        XMIN = None
        XMAX = None
    else:
        NBINS = int(options.nbins)
        XMIN = float(options.xmin)
        XMAX = float(options.xmax)
        
    cutstring = options.cut

    anaDir = args[0].rstrip('/')
    shift = None
    if anaDir.endswith('_Down'):
        shift = 'Down'
    elif anaDir.endswith('_Up'):
        shift = 'Up'
    
    if options.shift:
        shift = options.shift

    cfgFileName = args[1]
    cfgFile = open( cfgFileName, 'r' )
    cfg = imp.load_source( 'cfg', cfgFileName, cfgFile)

    isVBF = cutstring.find('Xcat_VBF')!=-1 
    isOneJet = cutstring.find('Xcat_J1_')!=-1 or cutstring.find('Xcat_J1X')!=-1
    isMSSM = cutstring.find('Xcat_J1BX')!=-1 or cutstring.find('Xcat_0BX')!=-1
    isBtag = cutstring.find('Xcat_J1BX')!=-1
    isZeroB = cutstring.find('Xcat_0BX')!=-1

    # Adjust binning for VBF
    if options.nbins is None and isVBF:
        NBINS = binning_svfitMass

    if options.nbins is None and isMSSM:
        NBINS = binning_svfitMass_mssm
        if not isBtag:
            NBINS = binning_svfitMass_mssm_nobtag
        if options.mssmBinning == '2013':
            NBINS = binning_svfitMass_mssm2013
        elif options.mssmBinning == 'fine':
            NBINS = 400
            XMIN = 0.
            XMAX = 2000.

    # QCD handling
    antiEleIsoForQCD = cutstring.find('Xcat_VBF_looseX')!=-1 or cutstring.find('Xcat_VBFX')!=-1 or cutstring.find('Xcat_J0_highX')!=-1  or isBtag #or cutstring.find('Xcat_VBF_tightX')!=-1 
    antiEleRlxTauIsoForQCD = cutstring.find('Xcat_J1_high_mediumhiggsX')!=-1 or cutstring.find('Xcat_VBF_tightX')!=-1 or cutstring.find('Xcat_J1_mediumX')!=-1

    print 'QCD SETUP', antiEleIsoForQCD, antiEleRlxTauIsoForQCD

    antiEleRlxTauIsoForQCDYield = cutstring.find('Xcat_J1_high_mediumhiggsX')!=-1 or cutstring.find('Xcat_VBF_tightX')!=-1 

    qcdYieldInclusiveExtrapolation = isVBF or cutstring.find('Xcat_J1_high_mediumhiggsX')!=-1 

    subtractBGForQCDShape = not antiEleIsoForQCD and not antiEleRlxTauIsoForQCD

    # W handling: Cut for W yield differs for VBF and 1B categories
    wYieldCut = cutstring
    wYieldCut = wYieldCut.replace('Xcat_VBFX', 'Xcat_VBF_Rel_30X')
    wYieldCut = wYieldCut.replace('Xcat_VBF_tightX', 'Xcat_VBF_Rel_30X && pthiggs>100.')
    wYieldCut = wYieldCut.replace('Xcat_VBF_looseX', 'Xcat_VBF_Rel_30X')
    wYieldCut = wYieldCut.replace('Xcat_J1BX', 'Xcat_J1B_Rel_CSVLX')

    # Relaxed VBF selection (not for W, that's done separately)
    relSelection = {}
    if isVBF:
        relCut = cutstring.replace('Xcat_VBF_looseX', 'Xcat_VBF_Rel_30X')
        relCut = relCut.replace('Xcat_VBF_tightX', 'Xcat_VBF_Rel_30X && pthiggs>100.')

        relCut20 = cutstring.replace('Xcat_VBF_looseX', 'Xcat_VBF_Rel_20X')
        relCut20 = relCut20.replace('Xcat_VBF_tightX', 'Xcat_VBF_Rel_20X && pthiggs>100.')

        relSelection['Ztt_ZL'] = relCut20
        relSelection['Ztt_ZJ'] = relCut
        if not cfg.TTgroup:
            relSelection['TTJets'] = relCut
        else:
            for ttComp in cfg.TTgroup:
                relSelection[ttComp] = relCut
        for vvComp in cfg.VVgroup:
            relSelection[vvComp] = relCut
        relSelection['QCD'] = relCut20
    if isBtag:
        relCut = cutstring.replace('Xcat_J1BX', 'Xcat_J1B_Rel_CSVLX')
        relSamples = [c for c in cfg.VVgroup]
        relSamples += ['Ztt_ZL', 'Ztt_ZJ', 'WJets', 'QCD']
        for sample in relSamples:
            relSelection[sample] = relCut


    # Replace X..X by actual category cuts
    options.cut = replaceCategories(options.cut, categories)
    wYieldCut = replaceCategories(wYieldCut, categories)
    for sample in relSelection:
        relSelection[sample] = replaceCategories(relSelection[sample], categories)


    dataName = 'Data'
    weight = 'weight'
    replaceW = False
    osForWExtrapolation = True
    makeQCDIsoPlots = False

    # Replace W shape in VBF and 1-jet
    if isVBF or isOneJet or isBtag:
        replaceW = True

    if options.shift and 'WShape' in options.shift:
        replaceW = True

    # Use OS+SS for W extrapolation in VBF
    if isVBF:
        osForWExtrapolation = False

    embed = options.embed

    aliases = None
    selComps, weights, zComps = prepareComponents(anaDir, cfg.config, aliases, options.embed, 'TauEle', options.higgs, isMSSM=isMSSM)

    print 'SELECTED COMPONENTS', selComps
    print 'WEIGHTS', [weights[w].GetWeight() for w in weights]
    print 'Z COMPONENTS', zComps

    dataComps = dict( (comp.name, comp) for comp in selComps.values() if comp.isData )

    # can, pad, padr = buildCanvas()
    ocan = buildCanvasOfficial()

    embedForSS = False
    if options.embed:
        print 'INFO, not using embedded samples for W estimation in sideband as per Summer 13 twiki'

    nbins = 50
    highMTMin = 70.
    highMTMax = 1070.
    if isVBF:
        nbins = 30
        highMTMin = 60.
        highMTMax = 120.

    wInfo = handleW(
        anaDir, selComps, weights,
        options.cut, wYieldCut, weight, False, cfg.VVgroup, cfg.TTgroup,
        nbins=nbins, highMTMin=highMTMin, highMTMax=highMTMax
        )

    # Inclusive QCD yield for QCD estimation in VBF
    # Calculate externally, need to recalculate if samples change
    # NOTE: Recalculation requires full QCD estimation including 
    # BG subtraction and W estimation in SS region 
    # > python -i plot_H2TauTauDataMC_TauEle_All.py /data/steggema/Sep19EleTau/ tauEle_2012_cfg.py -C 'Xcat_IncX && mt<30' -H svfitMass -b
    # > print osQCD
    if isVBF:
        print 'WARNING, taking QCD yield from external calculation (Sep19EleTau)'
    # incQCDYield = 11251.0 (Aug06)
    incQCDYield = 3398.5 # with tau pt > 30 (Sep19EleTau)

    # The following parameters are the same for a given set of samples + a cut (= category)
    parameters = {'cut':options.cut, 'anaDir':anaDir, 'selComps':selComps, 'weights':weights, 
      'shift':shift, 'VVgroup':cfg.VVgroup, 'TTgroup':cfg.TTgroup, 'replaceW':replaceW, 
      'antiEleIsoForQCD':antiEleIsoForQCD, 'antiEleRlxTauIsoForQCD':antiEleRlxTauIsoForQCD, 'antiEleRlxTauIsoForQCDYield':antiEleRlxTauIsoForQCDYield,
      'subtractBGForQCDShape':subtractBGForQCDShape, 'embedForSS':embedForSS, 
      'embed':options.embed, 'wInfo':wInfo, 'relSelection':relSelection, 'osForWExtrapolation':osForWExtrapolation, 
      'incQCDYield':incQCDYield, 'qcdYieldInclusiveExtrapolation':qcdYieldInclusiveExtrapolation, 'dataComps':dataComps, 'cutName':options.cutName, 'isZeroB':isZeroB, 'isOneJet':isOneJet} #'blind':options.blind, 

    if options.allPlots:
        drawAll(plots_All, 'Summer13StackPlotsJul19/', options.blind, parameters, isMSSM)
    else:
        ssign, osign, ssQCD, osQCD = makePlot(options.hist, NBINS, XMIN, XMAX, **parameters)
        if makeQCDIsoPlots:
            qcdIsoPlots(options.hist, NBINS, XMIN, XMAX, parameters)
        if blind and isMSSM:
            osQCD.blindxmin = 100.
            osQCD.blindxmax = 1000.

        # Without ratio
        # drawOfficial(osQCD, options.blind)

        # With ratio
        draw(osQCD, options.blind, channel='TauEle')
        datacards(osQCD, cutstring, shift, 'eleTau', prefix=options.prefix)
        # printDataVsQCDInfo(osQCD, ssQCD)
