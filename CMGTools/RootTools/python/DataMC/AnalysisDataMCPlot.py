
import glob
import os
import fnmatch

from ROOT import TFile

from CMGTools.RootTools.DataMC.DataMCPlot import * 
from CMGTools.RootTools.DataMC.Weight import * 
from CMGTools.RootTools.Style import *

class AnalysisDataMCError( Exception ):
    def __init__(self, value):
        self.value = value
        pass
    def __str__(self):
        return str(value)


class AnalysisDataMC( DataMCPlot ):
    '''Handles a data vs MC plot for a concrete analysis,
    see __init__ for more information'''


    def __init__(self, histName, directory, weights = None, name = None ):
        '''The input should be the name of a directory containing:
        - a set of root files, e.g. data_whatever.root, TTJets_somethingelse.root,
        ZJets_blah.root... 
        - a Weights/ subdirectory, containing weight files named
        weight_TTJets.py, weight_ZJets.py, ... 

        Each weight file should contain a python dictionary such as:
        {
        "genNEvents":35035820,
        "xSection":3048,
        "genEff":1,
        "intLumi":1000
        }

        If you prefer, you can pass a dictionary containing the weights.
        
        This class is just an example. Feel Free to subclass it, or to subclass DataMCPlot
        directly. 
        '''
        if name == None:
            name = ':'.join( [directory, histName] )
        DataMCPlot.__init__(self, name)
        self.histName = histName
        self.directory = directory
        self._InitPrefs()
        if weights is None:
            self._ReadWeights( '/'.join( [directory, 'Weights'] ) )
        else:
            self.weights = weights
        self._ReadHistograms( directory )
        
    def _InitPrefs(self):
        '''Definine preferences for each component'''
        self.histPref = {}
        self.histPref['data'] = {'style':sBlack, 'layer':-99}
        self.histPref['PromptReco-v4'] = {'style':sBlack, 'layer':-1000}
        self.histPref['PromptReco-v6'] = {'style':sBlack, 'layer':-900}
        self.histPref['TTJets'] = {'style':sBlue, 'layer':1} 
        self.histPref['WJets'] = {'style':sRed, 'layer':2}  
        self.histPref['DYJets'] = {'style':sYellow, 'layer':3}  

    def _GetHistPref(self, name):
        '''Return the preference dictionary for a given component'''
        thePref = None
        for prefpat, pref in self.histPref.iteritems():
            if fnmatch.fnmatch( name, prefpat):
                if thePref is not None:
                    print 'several matching preferences for', name 
                thePref = pref
        if thePref is None:
            print 'cannot find preference for hist',name
            thePref = {'style':sBlack, 'layer':999}
        return thePref

    def _ReadWeights(self, dirName):
        '''Read weight information from the weight directory.

        Weight files should have a name like : weight_foo.py, where foo is e.g. WJets.
        Make sure the name of your weight files correspond to the name of your root files.'''
        self.weights = {}
        files = glob.glob( '/'.join( [dirName, 'weight_*.py' ] ) ) 
        for file in files:
            # getting what's after the _ : 
            componentName = os.path.basename( file ).split('_')[1]
            # removing extension: 
            componentName = os.path.splitext( componentName )[0]
            weight = Weight(componentName, file)
            self.weights[componentName] = weight
            # print 'read weight : ', weight
            
    def _ReadHistograms(self, directory):
        '''Get the histograms.

        Styles and weights are applied to each histogram.
        See AnalysisDataMC._GetFileNames().'''
        fileNames = self._GetFileNames(directory)
        for layer, compInfo in enumerate( sorted(fileNames) ):
            compName, fileName = compInfo
            file = TFile(fileName)
            hist = copy.deepcopy(file.Get(self.histName))
            if hist == None:
                raise ValueError(' '.join(['histogram', self.histName,
                                           'does not exist in file', fileName]))
            hist.SetStats(0)
            hist.Sumw2()
            componentName = compName
            legendLine = componentName
            self.AddHistogram( componentName, hist, layer, legendLine)
        self._ApplyWeights()
        self._ApplyPrefs()

    def _GetFileNames(self, directory):
        '''Returns a list of file names from a directory.

        AnalysisDataMC._GetFileNames returns all root file names.
        Feel free to overload this function.'''
        fileNames = glob.glob( '/'.join([directory,'*.root']))
        if len(fileNames) == 0:
            raise ValueError('no file matching pattern *.root in ' + directory)
        return fileNames

    def _ComponentName(self, name):
        '''If name is /dir1/dir2/foo_barblah.root, returns foo.
        Feel free to overload this function.'''
        base = os.path.splitext(os.path.basename(name))[0]
        return base.split('_')[0]

    def _GetWeight(self, histName ):
        '''Return the Weight corresponding to a given component.

        For "data", returns None.'''
        componentName = histName
        if componentName.lower() != 'data':
            try:
                return self.weights[ componentName ]
            except KeyError:
                print 'Weight not found for component', histName
                return None
        else:
            return None

    def _ApplyWeights(self, lumi=-1):
        '''Applies weights to all histograms. Can be used to set the integrated luminosity.'''
        for hist in self.histos:
            weight = self._GetWeight( hist.name )
            if weight != None:
                if lumi>0:
                    weight.SetIntLumi( lumi ) 
                hist.SetWeight( weight.GetWeight() )
            else:
                if hist.name.lower().find('data')==-1:
                    print '\tWARNING: no weight file found, setting weight to 1'

    def _ApplyPrefs(self):
        for hist in self.histos:
            pref = self._GetHistPref( hist.name )
            hist.layer = pref['layer']
            hist.SetStyle( pref['style'] )
            

if __name__ == '__main__':

    import sys
    from optparse import OptionParser
    from CMGTools.RootTools.RootInit import *
    
    parser = OptionParser()
    parser.usage = '''
    plot.py <hist_name> <directory>
    plot the same histogram from several root files in a given directory
    '''

    (options,args) = parser.parse_args()

    if len(args)!=2:
        parser.print_help()
        print
        print 'Please specify the correct arguments.'
        sys.exit(1)
    histName = args[0]
    dir = args[1]

    plot = AnalysisDataMC( histName, dir )

    print
    print 
    print plot
    print 
    print '''
now you can try the following:
plot.Draw("")

if the legend bothers you, just move it somewhere else.
plot.DrawNormalized("")
plot.DrawStack("HIST")
plot.DrawNormalizedStack("HIST")
plot.DrawNormalizedRatioStack("HIST")
'''
