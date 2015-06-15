import imp
import math
import copy
import time
import re

from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC import H2TauTauDataMC
from CMGTools.H2TauTau.proto.plotter.prepareComponents import prepareComponents
from CMGTools.H2TauTau.proto.plotter.rootutils import *
from CMGTools.H2TauTau.proto.plotter.categories_TauMu import *
from CMGTools.H2TauTau.proto.plotter.binning import binning_svfitMass_finer
from CMGTools.H2TauTau.proto.plotter.titles import xtitles
from CMGTools.H2TauTau.proto.plotter.blind import blind
from CMGTools.H2TauTau.proto.plotter.plotmod import *
from CMGTools.H2TauTau.proto.plotter.datacards import *
from CMGTools.H2TauTau.proto.plotter.embed import *
from CMGTools.H2TauTau.proto.plotter.plotinfo import *
from CMGTools.RootTools.Style import *
from ROOT import kPink, TH1, TPaveText, TPad



cp = copy.deepcopy
EWK = 'WJets'

    
NBINS = 100
XMIN  = 0
XMAX  = 200


def replaceShapeInclusive(plot, var, anaDir,
                          comp, weights, 
                          cut, weight,
                          embed, shift):
    '''Replace WJets with the shape obtained using a relaxed tau iso'''
    cut = cut.replace('l1_looseMvaIso>0.5', 'l1_rawMvaIso>0.5')
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
                    embed, shift)
    # import pdb; pdb.set_trace()
    wjshape.Scale( wjyield )
    # import pdb; pdb.set_trace()
    plotWithNewShape.Replace(comp.name, wjshape) 
    # plotWithNewShape.Hist(comp.name).on = False 
    return plotWithNewShape

    

def makePlot( var, anaDir, selComps, weights, wJetScaleSS, wJetScaleOS,
              w_mt_ratio_ss, w_mt_ratio_os, w_mt_ratio,
              nbins=None, xmin=None, xmax=None,
              cut='', weight='weight', embed=False, shift=None, replaceW=False,
              VVgroup=None, antiMuIsoForQCD=False):
    
    print 'making the plot:', var, 'cut', cut

    oscut = cut+' && diTau_charge==0'
    osign = H2TauTauDataMC(var, anaDir,
                           selComps, weights, nbins, xmin, xmax,
                           cut=oscut, weight=weight, shift=shift,
                           embed=embed)
    osign.Hist(EWK).Scale( wJetScaleOS ) # * w_mt_ratio / w_mt_ratio_os )
    
    if cut.find('mt<')!=-1:
        print 'correcting high->low mT extrapolation factor, OS', w_mt_ratio / w_mt_ratio_os
        osign.Hist(EWK).Scale( w_mt_ratio / w_mt_ratio_os )
    if replaceW:
        osign = replaceShapeInclusive(osign, var, anaDir,
                                      selComps['WJets'], weights, 
                                      oscut, weight,
                                      embed, shift)
    sscut = cut+' && diTau_charge!=0'
    ssign = H2TauTauDataMC(var, anaDir,
                           selComps, weights, nbins, xmin, xmax,
                           cut=sscut, weight=weight, shift=shift,
                           embed=embed)
    ssign.Hist(EWK).Scale( wJetScaleSS )
    if cut.find('mt<')!=-1:
        print 'correcting high->low mT extrapolation factor, SS', w_mt_ratio / w_mt_ratio_ss
        ssign.Hist(EWK).Scale( w_mt_ratio / w_mt_ratio_ss  ) 
    if replaceW:
        ssign = replaceShapeInclusive(ssign, var, anaDir,
                                      selComps['WJets'], weights, 
                                      sscut, weight,
                                      embed, shift)
    if VVgroup:
        ssign.Group('VV',VVgroup)
        osign.Group('VV',VVgroup)
    
    ssQCD, osQCD = getQCD( ssign, osign, 'Data', VVgroup)
    if antiMuIsoForQCD:
        print 'WARNING RELAXING ISO FOR QCD SHAPE'
        # replace QCD with a shape obtained from data in an anti-iso control region
        qcd_yield = osQCD.Hist('QCD').Integral()
        
        # sscut_qcdshape = cut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5') + ' && diTau_charge!=0' 
        sscut_qcdshape = cut.replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5').replace('l1_looseMvaIso>0.5', 'l1_rawMvaIso>0.7') + ' && diTau_charge!=0'
        ssign_qcdshape = H2TauTauDataMC(var, anaDir,
                                        selComps, weights, nbins, xmin, xmax,
                                        cut=sscut_qcdshape, weight=weight,
                                        embed=embed)
        qcd_shape = copy.deepcopy( ssign_qcdshape.Hist('Data') )    
        qcd_shape.Normalize()
        qcd_shape.Scale(qcd_yield)
        # qcd_shape.Scale( qcd_yield )
        osQCD.Replace('QCD', qcd_shape)

    # osQCD.Group('VV', ['WW','WZ','ZZ'])
    osQCD.Group('electroweak', ['WJets', 'Ztt_ZL', 'Ztt_ZJ','VV'])
    osQCD.Group('Higgs 125', ['HiggsVBF125', 'HiggsGGH125', 'HiggsVH125'])
    return ssign, osign, ssQCD, osQCD


def drawAll(cut, plots, embed=True):
    '''See plotinfo for more information'''
    for plot in plots.values():
        print plot.var
        ss, os, ssQ, osQ = makePlot( plot.var, anaDir,
                                     selComps, weights, fwss, fwos,
                                     plot.nbins, plot.xmin, plot.xmax,
                                     cut, weight=weight, embed=embed)
        drawOfficial(osQ, False)
        plot.ssign = cp(ss)
        plot.osign = cp(os)
        plot.ssQCD = cp(ssQ)
        plot.osQCD = cp(osQ)
        time.sleep(1)


def handleW( anaDir, selComps, weights,
             cut, weight, embed, VVgroup, nbins=50, highMTMin=70., highMTMax=1070,
             lowMTMax=20.):
    
    cut = cut.replace('mt<20', '1')
    fwss, fwos, ss, os = plot_W(
        anaDir, selComps, weights,
        nbins, highMTMin, highMTMax, cut,
        weight=weight, embed=embed,
        VVgroup = VVgroup)

    w_mt_ratio_ss = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, cut, weight, lowMTMax, highMTMin, highMTMax, 'diTau_charge!=0');
    w_mt_ratio_os = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, cut, weight, lowMTMax, highMTMin, highMTMax, 'diTau_charge==0');
    w_mt_ratio = w_lowHighMTRatio('mt', anaDir, selComps['WJets'], weights, cut, weight, lowMTMax, highMTMin, highMTMax, '1');

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
    antiMuIsoForQCD = cutstring.find('l1_pt>40')!=-1 or cutstring.find('Xcat_J1X')!=-1
    options.cut = replaceCategories(options.cut, categories) 
    
    # TH1.AddDirectory(False)
    dataName = 'Data'
    weight='weight'
    replaceW = True
    useW11 = False
    
    anaDir = args[0].rstrip('/')
    shift = None
    if anaDir.endswith('_Down'):
        shift = 'Down'
    elif anaDir.endswith('_Up'):
        shift = 'Up'
        
    cfgFileName = args[1]
    file = open( cfgFileName, 'r' )
    cfg = imp.load_source( 'cfg', cfgFileName, file)
    embed = options.embed

    aliases = None
    selComps, weights, zComps = prepareComponents(anaDir, cfg.config, aliases, options.embed, 'TauMu', options.higgs)

    # can, pad, padr = buildCanvas()
    ocan = buildCanvasOfficial()
    
    fwss, fwos, w_mt_ratio_ss, w_mt_ratio_os, w_mt_ratio = handleW(
        anaDir, selComps, weights,
        options.cut, weight, options.embed, cfg.VVgroup
        )
    
    ssign, osign, ssQCD, osQCD = makePlot( options.hist, anaDir, selComps, weights, fwss, fwos, w_mt_ratio_ss, w_mt_ratio_os, w_mt_ratio, NBINS, XMIN, XMAX, options.cut, weight=weight, embed=options.embed, shift=shift, replaceW=replaceW, VVgroup=cfg.VVgroup, antiMuIsoForQCD=antiMuIsoForQCD);
    drawOfficial(osQCD, options.blind)
      
    datacards(osQCD, cutstring, shift)
