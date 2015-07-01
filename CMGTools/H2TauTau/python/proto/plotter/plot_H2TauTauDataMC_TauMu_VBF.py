import imp
import math
import copy
import time
import re
from numpy import array

from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC import H2TauTauDataMC
from CMGTools.H2TauTau.proto.plotter.prepareComponents import prepareComponents
from CMGTools.H2TauTau.proto.plotter.rootutils import *
from CMGTools.H2TauTau.proto.plotter.categories_TauMu import *
from CMGTools.H2TauTau.proto.plotter.binning import binning_svfitMass
from CMGTools.H2TauTau.proto.plotter.titles import xtitles
from CMGTools.H2TauTau.proto.plotter.blind import blind
from CMGTools.H2TauTau.proto.plotter.plotmod import *
from CMGTools.H2TauTau.proto.plotter.datacards import *
from CMGTools.H2TauTau.proto.plotter.plotinfo import plots_All, plots_J1
from CMGTools.H2TauTau.proto.plotter.plot_H2TauTauDataMC_TauMu_Inclusive import makePlot as makePlotInclusive
from CMGTools.H2TauTau.proto.plotter.plot_H2TauTauDataMC_TauMu_Inclusive import handleW as handleW

from CMGTools.RootTools.Style import *
from ROOT import kPink, TH1, TPaveText, TPad



cp = copy.deepcopy
EWK = 'WJets'

    
NBINS = 100
XMIN  = 0
XMAX  = 200

# cutwJ2 = ' && '.join([cat_Inc, cat_J2]) 

cut_VBF_Rel_W = ' && '.join( [cat_Inc, cat_VBF_Rel_30,'diTau_charge==0'])
cut_VBF_Rel_QCD =  ' && '.join( [cat_Inc, cat_VBF_Rel_20])


def makePlot( var, weights,
              w_yield,
              vbf_qcd_yield,
              vbf_zl_yield,
              vbf_zj_yield,
              nbins, xmin, xmax, cut,
              weight='weight', embed=False, shift=None, VVgroup=None):

    oscut = '&&'.join( [cat_Inc, cat_VBF, 'diTau_charge==0', cut])
    # oscut = str(inc_sig & Cut('l1_charge*l2_charge<0 && mt<40') & cat_VBF)
    print '[OS]', oscut
    osign = H2TauTauDataMC(var, anaDir,
                           selComps, weights, nbins, xmin, xmax,
                           cut=oscut, weight=weight, shift=shift,
                           embed=embed)

    # osign.Hist(EWK).Scale( wJetScaleOS ) 
    # import pdb; pdb.set_trace()
    # if cut.find('mt<')!=-1:
    #    print 'correcting high->low mT extrapolation factor, OS', w_mt_ratio / w_mt_ratio_os
    #    osign.Hist(EWK).Scale( w_mt_ratio / w_mt_ratio_os )
    osign.Hist('WJets').Normalize()
    osign.Hist('WJets').Scale(w_yield)

    cut_shape = ' && '.join([cut, cut_VBF_Rel_W])
    
    def replaceShape(compName):
        print compName, 'shape replaced', cut_shape
        nshape = shape(var, anaDir, 
                       selComps[compName], weights,
                       nbins, xmin, xmax,
                       cut_shape, weight,
                       embed, shift)
        nshape.Scale( osign.Hist(compName).Integral() )
        osign.Replace(compName, nshape )

    #  import pdb; pdb.set_trace()
    replaceShape('WJets')
    replaceShape('TTJets')
    for vvname in VVgroup:
        replaceShape(vvname)

    # ZL and ZJ shapes
    dycomp = selComps['Ztt']
    dyplot = buildPlot(var, anaDir, 
                       {dycomp.name:dycomp}, weights,
                       nbins, xmin, xmax,
                       cut_shape, weight,
                       embed, shift)
    zlshape = dyplot.Hist('Ztt_ZL')
    zlshape.Scale( vbf_zl_yield / zlshape.Integral())
    osign.Replace('Ztt_ZL', zlshape)
    zjshape = dyplot.Hist('Ztt_ZJ')
    zjshape.Scale( vbf_zj_yield / zjshape.Integral())
    osign.Replace('Ztt_ZJ', zjshape)

    # Ztt shape
    # from embedded samples, with relaxed cut

    embComps = dict( (comp.name, comp) for comp in selComps.values() if comp.isEmbed )
    emb_plot = buildPlot( var, anaDir,
                          embComps, weights,
                          nbins, xmin, xmax,
                          cut_shape, weight, embed)
    names = []
    for h in emb_plot.histos:
        h.stack = True
        names.append(h.name)
    emb_plot.Group('Embed', names)
    zttshape = emb_plot.Hist('Embed')
    zttshape.Scale( osign.Hist('Ztt').Integral() / zttshape.Integral())
    osign.Replace('Ztt', zttshape)
    
    
    qcd_cut = '&&'.join( [cat_Inc_AntiMuTauIsoJosh,
                          'diTau_charge!=0',
                          cat_VBF_Rel_20,
                          cut] )
    print 'QCD shape', qcd_cut
    qcd_plot = buildPlot( options.hist, anaDir,
                          dataComps, weights, NBINS, XMIN, XMAX,
                          qcd_cut, weight, options.embed)
    qcd_shape = qcd_plot.Hist('Data')
    qcd_shape.Scale( vbf_qcd_yield/qcd_shape.Integral() )
    

    osQCD = copy.deepcopy( osign )
    osQCD.AddHistogram('QCD', qcd_shape.weighted, 1.5)   
    osQCD.Hist('QCD').stack = True
    osQCD.Hist('QCD').SetStyle( sHTT_QCD )

    
    osQCD.Group('VV', VVgroup)
    osQCD.Group('EWK', ['WJets', 'Ztt_ZL', 'Ztt_ZJ', 'VV'])
    osQCD.Group('Higgs 125', ['HiggsVBF125', 'HiggsGGH125', 'HiggsVH125'])    

    return osign, osQCD 







if __name__ == '__main__':

    import copy
    from optparse import OptionParser
    from CMGTools.RootTools.RootInit import *

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
    parser.add_option("-b", "--batch", 
                      dest="batch", 
                      help="Set batch mode.",
                      action="store_true",
                      default=False)
    parser.add_option("-C", "--cut", 
                      dest="cut", 
                      help="cut",
                      default='1')
    parser.add_option("-E", "--embed", 
                      dest="embed", 
                      help="Use embedd samples.",
                      action="store_true",
                      default=False)
    parser.add_option("-g", "--higgs", 
                      dest="higgs", 
                      help="Higgs mass: 125, 130,... or dummy",
                      default=None)

    print '''
    IMPORTANT!! CALL THIS MACRO WITH ONLY THE MT CUT (OR WITHOUT IT IF YOU PLOT MT).
    SO IF THE 
    -C mt<40
    '''
    
    (options,args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)

    if options.batch:
        gROOT.SetBatch()

    NBINS = binning_svfitMass
    XMIN = None
    XMAX = None

    can = buildCanvasOfficial()
    
    weight='weight'
    # qcd_vbf_eff = 0.0025 # for 2012
    # qcd_vbf_eff = 0.001908 # for 2011
    # qcd_vbf_eff = 0.00142595233245 # 2011
    # emb_vbf_eff = 0.000828983358008 # 2011
    qcd_vbf_eff = None
    emb_vbf_eff = None
    
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
    
    selComps, weights, zComps = prepareComponents(anaDir, cfg.config, None, options.embed, 'TauMu', options.higgs)

    inc_fwss, inc_fwos, inc_w_mt_ratio_ss, inc_w_mt_ratio_os, inc_w_mt_ratio = handleW(
        anaDir, selComps, weights,
        cat_Inc, weight, options.embed, cfg.VVgroup
        )


    # inclusive QCD yield in signal region
    # this yield will be multiplied by the VBF efficiency
    insig_qcd_cut = '&&'.join([cat_Inc, options.cut])
    inc_ssign, inc_osign, inc_ssQCD, inc_osQCD = makePlotInclusive(
        options.hist, anaDir,
        selComps, weights,
        inc_fwss, inc_fwos, inc_w_mt_ratio_ss, inc_w_mt_ratio_os, inc_w_mt_ratio,
        NBINS, XMIN, XMAX, insig_qcd_cut,
        weight=weight, embed=options.embed, VVgroup=cfg.VVgroup
        )

    incsig_qcd_yield = inc_osQCD.Hist('QCD').Integral()
    incsig_zl_yield = inc_osQCD.Hist('Ztt_ZL').Integral()
    incsig_zj_yield = inc_osQCD.Hist('Ztt_ZJ').Integral()

    print 'Inclusive QCD yield =', incsig_qcd_yield
    print 'Inclusive ZL  yield =', incsig_zl_yield
    print 'Inclusive ZJ  yield =', incsig_zj_yield

    dataComps = dict( (comp.name, comp) for comp in selComps.values() if comp.isData )
    
    embComps = dict( (comp.name, comp) for comp in selComps.values() if comp.isEmbed )
    
    if qcd_vbf_eff is None:
        # computing VBF efficiency, in anti-isolated region ==================

        # QCD, Inclusive, SS, anti-isolation, for QCD efficiency
        inc_qcd_cut = ' && '.join([cat_Inc_AntiMuTauIsoJosh,
                                   options.cut,
                                   'diTau_charge!=0'])
        inc_qcd_plot = buildPlot( options.hist, anaDir,
                                  dataComps, weights, NBINS, XMIN, XMAX,
                                  inc_qcd_cut, weight, options.embed)
        inc_qcd_yield = inc_qcd_plot.Hist('Data').Integral()
        
        # QCD VBF, SS, anti-isolation, for QCD efficiency
        vbf_qcd_cut = '&&'.join( [inc_qcd_cut, cat_VBF] )
        
        vbf_qcd_plot = buildPlot( options.hist, anaDir,
                                  dataComps, weights, NBINS, XMIN, XMAX,
                                  vbf_qcd_cut, weight, options.embed)
        vbf_qcd_yield = vbf_qcd_plot.Hist('Data').Integral()
        
        qcd_vbf_eff = vbf_qcd_yield / inc_qcd_yield

        print 'QCD VBF Efficiency =', vbf_qcd_yield, '/', inc_qcd_yield, '=', qcd_vbf_eff

    if emb_vbf_eff is None:
        inc_emb_cut = ' && '.join([cat_Inc,
                                   options.cut,
                                   'diTau_charge==0'])
        inc_emb_plot = buildPlot( options.hist, anaDir,
                                  embComps, weights, NBINS, XMIN, XMAX,
                                  inc_emb_cut, weight, options.embed)
        names = []
        for h in inc_emb_plot.histos:
            h.stack = True
            names.append(h.name)
        inc_emb_plot.Group('Embed', names)
        inc_emb_yield = inc_emb_plot.Hist('Embed').Integral()


        vbf_emb_cut = '&&'.join( [inc_emb_cut, cat_VBF] )        
        vbf_emb_plot = buildPlot( options.hist, anaDir,
                                  embComps, weights, NBINS, XMIN, XMAX,
                                  vbf_emb_cut, weight, options.embed)
        names = []
        for h in vbf_emb_plot.histos:
            h.stack = True
            names.append(h.name)
        vbf_emb_plot.Group('Embed', names)        
        vbf_emb_yield = vbf_emb_plot.Hist('Embed').Integral()
        
        emb_vbf_eff = vbf_emb_yield / inc_emb_yield
                
        print 'Emb. VBF Efficiency =', vbf_emb_yield, '/', inc_emb_yield, '=', emb_vbf_eff
    
    vbf_w_cut = ' && '.join([cat_Inc, cat_VBF])
    vbf_fwss, vbf_fwos, vbf_w_mt_ratio_ss, vbf_w_mt_ratio_os, vbf_w_mt_ratio = handleW(
        anaDir, selComps, weights,
        vbf_w_cut, weight, options.embed, cfg.VVgroup,
        3, 60, 120
        )

    # relaxed vbf cut for the low / high mT extrapolation ratio
    vbf_w_cut_rel = ' && '.join([cat_Inc, cat_J2])
    w_lowhigh_plot = buildPlot( 'mt', anaDir,
                                {'WJets':selComps['WJets']}, weights, 200, 0, 200,
                                vbf_w_cut_rel, weight, options.embed)
    w_mt_hist = w_lowhigh_plot.Hist('WJets')
    w_low_yield = w_mt_hist.Integral(True, 0,20)
    w_high_yield = w_mt_hist.Integral(True, 60,120)
    w_lowhigh_ratio = w_low_yield / w_high_yield
    
    # full VBF cut for W normalization in high mT sideband
    vbf_w_cut = ' && '.join([cat_Inc, cat_VBF, 'diTau_charge==0'])
    w_plot = buildPlot( 'mt', anaDir,
                        {'WJets':selComps['WJets']}, weights, 200, 0, 200,
                        vbf_w_cut, weight, options.embed)
         
    # yield in high mT sideband
    w_high_yield_vbf = w_plot.Hist('WJets').Integral(True, 60, 120)
    # now normalized to the data
    w_high_yield_vbf *= vbf_fwos
    # extrapolating to low mt region
    w_low_yield_vbf = w_high_yield_vbf * w_lowhigh_ratio
    
    osign, osQCD  = makePlot( options.hist, weights,
                              w_low_yield_vbf,
                              qcd_vbf_eff * incsig_qcd_yield,
                              emb_vbf_eff * incsig_zl_yield,
                              emb_vbf_eff * incsig_zj_yield,                              
                              NBINS, XMIN, XMAX, options.cut, weight=weight,
                              embed=options.embed, shift=shift,
                              VVgroup = cfg.VVgroup);

    drawOfficial(osQCD, False)
    datacards(osQCD, 'Xcat_VBFX', shift)


