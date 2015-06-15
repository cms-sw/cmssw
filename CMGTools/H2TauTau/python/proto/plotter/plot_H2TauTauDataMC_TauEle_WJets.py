import imp
import math
import copy
import time
import re

#from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC import H2TauTauDataMC
from CMGTools.H2TauTau.proto.plotter.prepareComponents import prepareComponents #, readPickles
from CMGTools.H2TauTau.proto.plotter.rootutils import buildCanvas, draw
from CMGTools.H2TauTau.proto.plotter.categories_TauEle import *
from CMGTools.H2TauTau.proto.plotter.binning import binning_svfitMass
from CMGTools.H2TauTau.proto.plotter.titles import xtitles
from CMGTools.H2TauTau.proto.plotter.blind import blind
from CMGTools.H2TauTau.proto.plotter.plotmod import *
from CMGTools.H2TauTau.proto.plotter.datacards import *
from CMGTools.H2TauTau.proto.plotter.embed import *
from CMGTools.H2TauTau.proto.plotter.plotinfo import *
from PhysicsTools.HeppyCore.statistics.counter import Counters
from CMGTools.RootTools.Style import *
from ROOT import kGray, kPink, TH1, TPaveText, TPad, TCanvas

cp = copy.deepcopy

    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


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
    parser.add_option("-C", "--cut", 
                      dest="cut", 
                      help="cut to apply in TTree::Draw",
                      default='1')
    parser.add_option("-E", "--embed", 
                      dest="embed", 
                      help="Use embedd samples.",
                      action="store_true",
                      default=False)
    parser.add_option("-X", "--exclusiveVV", 
                      dest="useExcusiveVV", 
                      help="Use exclusive VV.",
                      action="store_true",
                      default=False)
    parser.add_option("-g", "--higgs", 
                      dest="higgs", 
                      help="Higgs mass: 125, 130,... or dummy",
                      default=None)
    
    (options,args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
        
    cutstring = options.cut
    options.cut = replaceCategories(options.cut, categories) 

    print 'CUT APPLIED:', options.cut
    
    dataName = 'Data'
    weight='weight'
    
    anaDir = args[0].rstrip('/')
    shift = None

    cfgFileName = args[1]
    file = open( cfgFileName, 'r' )
    cfg = imp.load_source( 'cfg', cfgFileName, file)

    #PG (STEP 0) prepare the samples on which to run
    #PG ---- ---- ---- ---- ---- ---- ---- ---- ----

    origComps = copy.deepcopy(cfg.config.components)

    comps = []
    for comp in cfg.config.components:
        if comp.name == 'W1Jets': continue
        if comp.name == 'W2Jets': continue
        if comp.name == 'W3Jets': continue
        if comp.name == 'W4Jets': continue
        if comp.name == 'TTJets11': continue #PG remove me
        if comp.name == 'WJets11': continue #PG remove me
        if options.useExcusiveVV :
            if comp.name == 'WW' : continue
            if comp.name == 'ZZ' : continue
            if comp.name == 'WZ' : continue
        else :
            if comp.name == 'WW2l2v' : continue
            if comp.name == 'WZ2l2q' : continue
            if comp.name == 'WZ3lv' : continue
            if comp.name == 'ZZ2l2q' : continue
            if comp.name == 'ZZ2l2v' : continue
            if comp.name == 'ZZ4l' : continue
        comps.append( comp )
        
    cfg.config.components = comps
    selComps, weights, zComps = prepareComponents(anaDir, cfg.config, None, 
                                                  options.embed, 'TauEle', options.higgs)


    cutw = options.cut.replace('mt<40', '1')
    # loosen electron isolation
    # cutw = cutw.replace('l2_relIso05<0.1', 'l2_relIso05<1')    
    # loosen tau isolation


    #PG (STEP 1) evaluate the WJets contribution from high mT sideband
    #PG ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    #PG as it is done in the analysis

    fw_ss, fw_ss_error, fw_os, fwos_error, ss, os = plot_W(anaDir, selComps, weights,
                                                        12, 60, 120, cutw,
                                                        weight = weight, 
                                                        embed = options.embed,
                                                        VVgroup = cfg.VVgroup,
                                                        treeName = 'H2TauTauTreeProducerTauEle')
    #PG fw_ss = W normalization factor for the same sign plots
    #PG fw_os = W normalization factor for the opposite sign plots
    #PG ss   = mt plot with the scaled W, according to fw_ss
    #PG os   = mt plot with the scaled W, according to fw_os

    #PG (TEST) remake the WJets plots over the full range for SS
    #PG ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    #PG with the binning I want

    cut_ss = '{cut} && diTau_charge!=0'.format(
        cut = cutw 
        )
    FULL_mt_ss = H2TauTauDataMC('mt', anaDir, selComps, weights,   #PG prepare the plot
                               30, 0, 200,
                               cut = cut_ss, weight = weight,
                               embed = options.embed, 
                               treeName = 'H2TauTauTreeProducerTauEle')
    FULL_mt_ss.Hist('WJets').Scale( fw_ss )

    if cfg.VVgroup != None :
        FULL_mt_ss.Group ('VV',cfg.VVgroup)
        
    # WJets_data = data - DY - TTbar
    FULL_mt_ss_wjet = copy.deepcopy(FULL_mt_ss.Hist(dataName))     #PG isolate data and subtract
    FULL_mt_ss_wjet.Add(FULL_mt_ss.Hist('Ztt'), -1)                #PG non-WJets bkgs (but QCD)
    removingFakes = False
    try:
        f1 = FULL_mt_ss.Hist('Ztt_ZL')
        f2 = FULL_mt_ss.Hist('Ztt_ZJ')
        FULL_mt_ss_wjet.Add(f1, -1)
        FULL_mt_ss_wjet.Add(f2, -1)
        removingFakes = True
    except:
        pass
    FULL_mt_ss_wjet.Add(FULL_mt_ss.Hist('TTJets'), -1)
    
    if FULL_mt_ss.histosDict.get('VV', None) != None :
        FULL_mt_ss_wjet.Add(FULL_mt_ss.Hist('VV'), -1)
    else:
        print 'VV group not found, VV not subtracted'

    # adding the WJets_data estimation to the stack
    FULL_mt_ss.AddHistogram( 'Data - DY - TT',                     #PG put it back in the 
                            FULL_mt_ss_wjet.weighted, 1010)
    FULL_mt_ss.Hist('Data - DY - TT').stack = False
    # with a nice pink color
    pink = kPink+7
    sPinkHollow = Style( lineColor=pink, 
                         markerColor=pink, markerStyle=4)
    FULL_mt_ss.Hist('Data - DY - TT').SetStyle( sPinkHollow )


    #PG compare the MC-subtracted data to the WJets MC only for SS
    #PG ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    can0 = TCanvas('can0','',100,100,600,600)
    p_main  = TPad ("p1","",0,0.3,1,1)
    p_ratio = TPad ("p1","",0,0,1,0.3)
    p_main.Draw()
    p_main.SetLogy ()
    p_ratio.Draw()

    p_main.cd()
    W_ss_WJets = FULL_mt_ss.Hist('WJets').weighted
    W_ss_Data  = FULL_mt_ss.Hist('Data - DY - TT').weighted
    W_ss_WJets.GetXaxis().SetTitle ('mt')
    W_ss_Data.GetXaxis().SetTitle ('mt')
    W_ss_WJets.Draw ('hist')
    W_ss_Data.Draw ('same')
    leg_W_ss = TLegend (0.6,0.6,0.9,0.9)
    leg_W_ss.AddEntry (W_ss_Data,  'data - DY - TT', 'pl')
    leg_W_ss.AddEntry (W_ss_WJets, 'WJets',          'pl')
    leg_W_ss.Draw ()

    p_ratio.cd()
    W_ss_Data_ratio = W_ss_Data.Clone ('W_ss_Data_ratio')
    W_ss_WJets_ratio = W_ss_WJets.Clone ('W_ss_WJets_ratio')
    W_ss_Data_ratio.Divide (W_ss_WJets)
    W_ss_WJets_ratio.Divide (W_ss_WJets)
    p_ratio.DrawFrame(W_ss_WJets_ratio.GetXaxis().GetXmin(),  0.5, W_ss_WJets_ratio.GetXaxis().GetXmax(), 2)
    W_ss_Data_ratio.Draw ('same')
    W_ss_WJets_ratio.SetFillStyle (4001)
    W_ss_WJets_ratio.SetFillColor (2)
    W_ss_WJets_ratio.SetMarkerStyle (9)
    W_ss_WJets_ratio.Draw ('samehist')
    W_ss_WJets_ratio.Draw ('sameE3')
    can0.Print ('compare_W_ss.png','png')

    FULL_mt_ss.Group('EWK', ['WJets', 'Ztt_ZJ','VV'])
    FULL_mt_ss.Group('Higgs 125', ['HiggsVBF125', 'HiggsGGH125', 'HiggsVH125'])
    draw(FULL_mt_ss, False, 'TauEle', plotprefix = 'MT_ss')


    #PG (TEST) remake the WJets plots over the full range for OS
    #PG ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    #PG with the binning I want

    cut_os = '{cut} && diTau_charge==0'.format(
        cut = cutw 
        )
    FULL_mt_os = H2TauTauDataMC('mt', anaDir, selComps, weights,   #PG prepare the plot
                               30, 0, 200,
                               cut = cut_os, weight = weight,
                               embed = options.embed, 
                               treeName = 'H2TauTauTreeProducerTauEle')
    FULL_mt_os.Hist('WJets').Scale( fw_os )

    if cfg.VVgroup != None :
        FULL_mt_os.Group ('VV',cfg.VVgroup)
        
    # WJets_data = data - DY - TTbar
    FULL_mt_os_wjet = copy.deepcopy(FULL_mt_os.Hist(dataName))     #PG isolate data and subtract
    FULL_mt_os_wjet.Add(FULL_mt_os.Hist('Ztt'), -1)                #PG non-WJets bkgs (but QCD)
    removingFakes = False
    try:
        f1 = FULL_mt_os.Hist('Ztt_ZL')
        f2 = FULL_mt_os.Hist('Ztt_ZJ')
        FULL_mt_os_wjet.Add(f1, -1)
        FULL_mt_os_wjet.Add(f2, -1)
        removingFakes = True
    except:
        pass
    FULL_mt_os_wjet.Add(FULL_mt_os.Hist('TTJets'), -1)
    
    if FULL_mt_os.histosDict.get('VV', None) != None :
        FULL_mt_os_wjet.Add(FULL_mt_os.Hist('VV'), -1)
    else:
        print 'VV group not found, VV not subtracted'

    # adding the WJets_data estimation to the stack
    FULL_mt_os.AddHistogram( 'Data - DY - TT',                     #PG put it back in the 
                            FULL_mt_os_wjet.weighted, 1010)
    FULL_mt_os.Hist('Data - DY - TT').stack = False
    # with a nice pink color
    pink = kPink+7
    sPinkHollow = Style( lineColor=pink, 
                         markerColor=pink, markerStyle=4)
    FULL_mt_os.Hist('Data - DY - TT').SetStyle( sPinkHollow )


    #PG compare the MC-subtracted data to the WJets MC only for OS
    #PG ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    p_main.cd()
    W_os_WJets = FULL_mt_os.Hist('WJets').weighted
    W_os_Data  = FULL_mt_os.Hist('Data - DY - TT').weighted
    W_os_WJets.GetXaxis().SetTitle ('mt')
    W_os_Data.GetXaxis().SetTitle ('mt')
    W_os_WJets.DrawCopy ('hist')
    W_os_Data.DrawCopy ('same')
    leg_W_os = TLegend (0.6,0.6,0.9,0.9)
    leg_W_os.AddEntry (W_os_Data,  'data - DY - TT', 'pl')
    leg_W_os.AddEntry (W_os_WJets, 'WJets',          'pl')
    leg_W_os.Draw ()

    p_ratio.cd()
    W_os_Data_ratio = W_os_Data.Clone ('W_os_Data_ratio')
    W_os_WJets_ratio = W_os_WJets.Clone ('W_os_WJets_ratio')
    W_os_Data_ratio.Divide (W_os_WJets)
    W_os_WJets_ratio.Divide (W_os_WJets)
    p_ratio.DrawFrame(W_os_WJets_ratio.GetXaxis().GetXmin(),  0.5, W_os_WJets_ratio.GetXaxis().GetXmax(), 2)
    W_os_Data_ratio.Draw ('same')
    W_os_WJets_ratio.SetFillStyle (4001)
    W_os_WJets_ratio.SetFillColor (2)
    W_os_WJets_ratio.SetMarkerStyle (9)
    W_os_WJets_ratio.Draw ('samehist')
    W_os_WJets_ratio.Draw ('sameE3')
    can0.Print ('compare_W_os.png','png')

    FULL_mt_os.Group('EWK', ['WJets', 'Ztt_ZJ','VV'])
    FULL_mt_os.Group('Higgs 125', ['HiggsVBF125', 'HiggsGGH125', 'HiggsVH125'])
    draw(FULL_mt_os, False, 'TauEle', plotprefix = 'MT_os')
