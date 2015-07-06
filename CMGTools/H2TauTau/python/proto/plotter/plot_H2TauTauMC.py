import imp
import math
import copy
import time
import re

from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC import H2TauTauDataMC
from CMGTools.H2TauTau.proto.plotter.prepareComponents import prepareComponents
from CMGTools.H2TauTau.proto.plotter.rootutils import *
from CMGTools.H2TauTau.proto.plotter.binning import binning_svfitMass_finer, binning_svfitMass, binning_svfitMass_mssm2013, binning_svfitMass_mssm, binning_svfitMass_mssm_nobtag
from CMGTools.H2TauTau.proto.plotter.titles import xtitles
from CMGTools.H2TauTau.proto.plotter.blind import blind
from CMGTools.H2TauTau.proto.plotter.plotmod import *
from CMGTools.H2TauTau.proto.plotter.datacards import *
from CMGTools.H2TauTau.proto.plotter.plotinfo import *
from CMGTools.H2TauTau.proto.plotter.categories_TauMu import *
from CMGTools.RootTools.Style import *
from ROOT import kPink, TH1, TPaveText, TPad



cp = copy.deepcopy
  
NBINS = 100
XMIN  = 0
XMAX  = 200



def makePlot( var, anaDir, selComps, weights,
              nbins=None, xmin=None, xmax=None,
              cut='', weight='weight', embed=False, shift=None,
              VVgroup=None, treeName='TauMu'):
    
    oscut = cut+' && diTau_charge==0'
    print 'making the plot:', var, 'cut', oscut
    osign = H2TauTauDataMC(var, anaDir,
                           selComps, weights, nbins, xmin, xmax,
                           cut=oscut, weight=weight, shift=shift,
                           embed=embed,
                           treeName = treeName)    
    # if VVgroup:
    #     osign.Group('VV',VVgroup)
    # osign.Group('electroweak', ['WJets', 'Ztt_ZJ','VV'])
    return osign


def filterComps(comps, filterString, embed): 
    filteredComps = copy.copy(comps)
    if filterString:
        filters = filterString.split(';')
        filteredComps = {}
        embedComps = {}
        zttPresent = False
        for comp in comps.values():
            if comp.name.startswith('embed'):
                embedComps[comp.name] = comp
            for filter in filters:
                pattern = re.compile( filter )
                if pattern.search( comp.name ):
                    filteredComps[comp.name] = comp
                    if comp.name.find('Ztt') != -1:
                        zttPresent = True
        if zttPresent and embed:
            filteredComps = dict( filteredComps.items() + embedComps.items() )
            
    return filteredComps

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
    parser.add_option("-c", "--channel", 
                      dest="channel", 
                      help="channel: TauEle or TauMu (default)",
                      default='TauMu')
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
    parser.add_option("-f", "--filter", 
                      dest="filter", 
                      help="Regexp filters to select components, separated by semicolons, e.g. Higgs;ZTT",
                      default=None)
    parser.add_option("-w", "--weight", 
                      dest="weight", 
                      help='Weight expression',
                      default='weight')
    parser.add_option("-s", "--shift", 
                      dest="shift", 
                      help='Shift expression',
                      default=None)
    parser.add_option("-k", "--mssmBinning", 
                      dest="mssmBinning", 
                      help="Binning for MSSM: 'fine', '2013', 'default'",
                      default='default')
    parser.add_option("-p", "--prefix", 
                      dest="prefix", 
                      help="Prefix for datacards",
                      default=None)
    
    (options,args) = parser.parse_args()

    weight = options.weight

    cutstring = options.cut
    isVBF = cutstring.find('Xcat_VBF')!=-1
    isMSSM = cutstring.find('Xcat_J1BX')!=-1 or cutstring.find('Xcat_0BX')!=-1

    options.cut = replaceCategories(options.cut, categories) 
    
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    if options.batch:
        gROOT.SetBatch()
    if options.nbins is None:
        NBINS = binning_svfitMass_finer
        XMIN = None
        XMAX = None

        if isVBF:
            NBINS = binning_svfitMass

        if isMSSM:
            NBINS = binning_svfitMass_mssm
            if cutstring.find('Xcat_0BX')!=-1:
                NBINS = binning_svfitMass_mssm_nobtag
            if options.mssmBinning == '2013':
                NBINS = binning_svfitMass_mssm2013
            elif options.mssmBinning == 'fine':
                NBINS = 400
                XMIN = 0.
                XMAX = 2000.
    else:
        NBINS = int(options.nbins)
        XMIN = float(options.xmin)
        XMAX = float(options.xmax)

    
    dataName = 'Data'
    
    anaDir = args[0].rstrip('/')
    shift = None
    if anaDir.endswith('Down'):
        shift = 'Down'
    elif anaDir.endswith('Up'):
        shift = 'Up'
    
    if options.shift:
        shift = options.shift

    cfgFileName = args[1]
    file = open( cfgFileName, 'r' )
    cfg = imp.load_source( 'cfg', cfgFileName, file)
    embed = options.embed

    aliases = None
    selComps, weights, zComps = prepareComponents(anaDir, cfg.config, aliases, options.embed,
                                                  channel=options.channel, higgsMass=options.higgs, isMSSM=isMSSM)

    filteredComps = filterComps(selComps, options.filter, options.embed)
    
    ocan = buildCanvasOfficial()

    treeName = 'H2TauTauTreeProducer' + options.channel
    osign = makePlot( options.hist, anaDir, filteredComps, weights,
                      NBINS, XMIN, XMAX,
                      options.cut, weight=weight, embed=options.embed,
                      shift=None, VVgroup=cfg.VVgroup, treeName=treeName);
    # drawOfficial(osign, options.blind)
    osign.Draw()
      
    dcchan = 'muTau'
    if options.channel == 'TauEle':
        dcchan = 'eleTau'
    datacards(osign, cutstring, shift, channel=dcchan, prefix=options.prefix)
