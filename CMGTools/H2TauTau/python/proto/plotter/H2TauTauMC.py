import os
from fnmatch import fnmatch
import copy

from ROOT import TFile, TH1F, TPaveText

from CMGTools.RootTools.DataMC.AnalysisDataMCPlot import AnalysisDataMC
from CMGTools.RootTools.fwlite.Weight import Weight
from CMGTools.RootTools.fwlite.Weight import printWeights
from CMGTools.RootTools.Style import *

class H2TauTauMC( AnalysisDataMC ):

    keeper = {}
    HINDEX = 0

    def __init__(self, varName, directory, selComps, weights,
                 bins = None, xmin = None, xmax=None, cut = '',
                 weight='weight', shift=None, treeName=None):
        '''Data/MC plotter adapted to the H->tau tau analysis.
        The plotter takes a collection of trees in input. The trees are found according
        to the dictionary of selected components selComps.
        The global weighting information for each component is read from the weights dictionary.
        The weight parameter is the name of an event weight variable that can be found in the tree.
        The default is "weight" (full event weight computed at python analysis stage),
        but you can build up the weight string you want before calling this constructor.
        To do an unweighted plot, choose weight="1" (the string, not the number).        
        '''
        if treeName is None:
            treeName = 'H2TauTauTreeProducerTauMu'
        self.treeName = treeName
        self.selComps = selComps
        self.varName = varName
        self.shift = shift
        self.cut = cut
        self.eventWeight = weight
        self.bins = bins
        self.xmin = xmin
        self.xmax = xmax
        # self.keeper = []
        
        super(H2TauTauMC, self).__init__(varName, directory, weights)

        self.legendBorders = 0.651, 0.463, 0.895, 0.892



    def _BuildHistogram(self, tree, comp, compName, varName, cut, layer ):
        '''Build one histogram, for a given component'''

        print 'filling', compName
        if not hasattr( comp, 'tree'):
            comp.tree = tree
                    
        histName = '_'.join( [compName, self.varName] )

        hist = None
        if self.xmin is not None and self.xmax is not None:
            hist = TH1F( histName, '', self.bins, self.xmin, self.xmax )
        else:
            hist = TH1F( histName, '', len(self.bins)-1, self.bins )
        hist.Sumw2()
        weight = self.eventWeight
## to do the following, modify the eventWeight before giving it to self
##         if not comp.isData:
##             weight = ' * '.join( [self.eventWeight, recEffId.weight(), recEffIso.weight()])
        if tree == None:
            raise ValueError('tree does not exist for component '+compName)
        var = varName
        if not comp.isData and self.shift:
            if self.shift == 'Up':
                if varName == 'visMass' or varName == 'svfitMass':
                    print 'Shifting visMass and svfitMass by sqrt(1.03) for', comp.name
                    var = varName + '* sqrt(1.03)'
            elif self.shift == 'Down':
                if varName == 'visMass' or varName == 'svfitMass':
                    print 'Shifting visMass and svfitMass by sqrt(0.97) for', comp.name
                    var = varName + '* sqrt(0.97)'
            else:
                raise ValueError( self.shift + ' is not recognized. Use None, "Up" or "Down".')
            
        tree.Project( histName, var, '{weight}*({cut})'.format(cut=cut,
                                                               weight=weight) )
        hist.SetStats(0)
        componentName = compName
        legendLine = compName
        self.AddHistogram( componentName, hist, layer, legendLine)
        self.Hist(componentName).realName = comp.realName
        if comp.isData:
            self.Hist(componentName).stack = False


    def _ReadHistograms(self, directory):
        '''Build histograms for all components.'''
        for layer, (compName, comp) in enumerate( sorted(self.selComps.iteritems()) ) : 
            fileName = '/'.join([ directory,
                                  comp.dir,
                                  self.treeName,
                                  '{treeName}_tree.root'.format(treeName=self.treeName)] )

            file = self.__class__.keeper[ fileName + str(self.__class__.HINDEX) ] = TFile(fileName) 
            self.__class__.HINDEX+=1

            tree = file.Get( self.treeName )
            
            if compName == 'Ztt':
                self._BuildHistogram(tree, comp, compName, self.varName,
                                     self.cut + ' && isFake==0', layer)
                fakeCompName = 'Ztt_ZL'
                self._BuildHistogram(tree, comp, fakeCompName, self.varName,
                                     self.cut + ' && isFake==1', layer)
                self.Hist(fakeCompName).realName =  comp.realName + '_ZL'
                self.weights[fakeCompName] = self.weights[compName]
                fakeCompName = 'Ztt_ZJ'
                self._BuildHistogram(tree, comp, fakeCompName, self.varName,
                                     self.cut + ' && isFake==2', layer)
                self.Hist(fakeCompName).realName =  comp.realName + '_ZJ'
                self.weights[fakeCompName] = self.weights[compName]

            else:
                self._BuildHistogram(tree, comp, compName, self.varName,
                                     self.cut, layer )     

        self._ApplyWeights()
        self._ApplyPrefs()


    def _InitPrefs(self):
        '''Definine preferences for each component'''
        self.histPref = {}
        self.histPref['Data'] = {'style':sData, 'layer':2999}
        self.histPref['data_*'] = {'style':sBlack, 'layer':2002}
        self.histPref['Ztt'] = {'style':sHTT_DYJets, 'layer':4}
        self.histPref['embed_*'] = {'style':sViolet, 'layer':4.1}
        self.histPref['TTJets*'] = {'style':sHTT_TTJets, 'layer':1} 
        self.histPref['WW'] = {'style':sBlue, 'layer':0.9} 
        self.histPref['WZ'] = {'style':sRed, 'layer':0.8} 
        self.histPref['ZZ'] = {'style':sGreen, 'layer':0.7} 
        self.histPref['QCD'] = {'style':sHTT_QCD, 'layer':2}
        self.histPref['WJets*'] = {'style':sHTT_WJets, 'layer':3}  
        self.histPref['Ztt_ZJ'] = {'style':sYellow, 'layer':3.1}
        self.histPref['Ztt_ZL'] = {'style':sBlue, 'layer':3.2}
        self.histPref['Higgs*'] = {'style':sHTT_Higgs, 'layer':1001}


def filterComps(comps, filterString=None): 
    filteredComps = copy.copy(comps)
    if filterString:
        filters = filterString.split(';')
        filteredComps = {}
        for comp in comps.values():
            for filter in filters:
                pattern = re.compile( filter )
                if pattern.search( comp.name ):
                    filteredComps[comp.name] = comp 
    return filteredComps


if __name__ == '__main__':


    import copy
    import imp
    import re 
    from optparse import OptionParser
    from CMGTools.RootTools.RootInit import *
    from CMGTools.H2TauTau.proto.plotter.binning import binning_svfitMass,binning_svfitMass_finer
    from CMGTools.H2TauTau.proto.plotter.prepareComponents import prepareComponents
    from CMGTools.H2TauTau.proto.plotter.categories_TauMu import *
    from CMGTools.H2TauTau.proto.plotter.rootutils import buildCanvas, draw
    from CMGTools.H2TauTau.proto.plotter.datacards import *

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
                      default='svfitMass')
    parser.add_option("-C", "--cut", 
                      dest="cut", 
                      help="cut to apply in TTree::Draw",
                      default=None)
    parser.add_option("-c", "--channel", 
                      dest="channel", 
                      help="channel: TauEle or TauMu (default)",
                      default='TauMu')
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
                      help="Regexp filter to select components",
                      default=None)
    parser.add_option("-b", "--batch", 
                      dest="batch", 
                      help="Set batch mode.",
                      action="store_true",
                      default=False)
    parser.add_option("-p", "--prefix", 
                      dest="prefix", 
                      help="Prefix for the root files, eg. MC to get MC_eleTau_vbf.root",
                      default=None)
    

    
    (options,args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)

    cutstring = options.cut
    isVBF = cutstring.find('Xcat_VBFX') != -1 
    options.cut = replaceCategories(options.cut, categories) 
        
    if options.batch:
        gROOT.SetBatch()
    if options.nbins is None:
        NBINS = binning_svfitMass_finer
        if isVBF:
            NBINS = binning_svfitMass
        XMIN = None
        XMAX = None
    else:
        NBINS = int(options.nbins)
        XMIN = float(options.xmin)
        XMAX = float(options.xmax)
        
    
    weight='weight'
    anaDir = args[0].rstrip('/')
    shift = None
    if anaDir.endswith('_Down'):
        shift = 'Down'
    elif anaDir.endswith('_Up'):
        shift = 'Up'

    treeName = 'H2TauTauTreeProducer' + options.channel
    
    cfgFileName = args[1]
    file = open( cfgFileName, 'r' )
    cfg = imp.load_source( 'cfg', cfgFileName, file)


    selComps, weights, zComps = prepareComponents(
        anaDir, cfg.config, None, True, options.channel,
        options.higgs)

    filteredComps = filterComps(selComps, options.filter)
    
    can, pad, padr = buildCanvas()
    # shift is now done at the skim level
    plot = H2TauTauMC( options.hist, anaDir, filteredComps,
                       weights, NBINS, XMIN, XMAX, options.cut,
                       weight=weight, shift=None,
                       treeName=treeName)
    plot.Draw()

    dcchan = None
    if options.channel == 'TauEle':
        dcchan = 'eleTau'
    datacards(plot, cutstring, shift, channel=dcchan, prefix=options.prefix)
