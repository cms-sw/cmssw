
from ROOT import TH1, TDirectory, TCanvas

from CMGTools.RootTools.Style import *

import sys, re, math, pprint, string

class RootDir:
    """Manages a TDirectory in PyROOT, and allows easy access to the histograms and plotting"""
    
    def __init__( self, tDir, style=None):
        self.tDir = tDir
        self.histograms = {}
        self.subDirs = {}
        self.style = style
        self._Walk()
        
    def _LoadHistograms( self, tDir ):
        """Looks for all histograms in the directory, and stores them in a dictionary indexed by their key"""
        listOfKeys = tDir.GetListOfKeys()
        for key in listOfKeys:
            hist = tDir.Get( key.GetName() )
                
            if hist.InheritsFrom('TH1'):
                # print hist.GetName()
                
                if self.style != None:
                    hist = self.style.formatHisto( hist )
                self.histograms[key.GetName()] = hist 

    def SetStyle( self, style ):
        """Set style for all histograms, in this directory and its sub-directories. See Style module for more information"""
        self.style = style
        for key in self.histograms.iterkeys():
            self.histograms[key] = style.formatHisto( self.histograms[key] )
        for key in self.subDirs.iterkeys():
            self.subDirs[key].SetStyle( style )
            
    def _Walk( self ):        
        """loads histograms, create RootDirs for each subdirectory. RootDirs are stored in a dictionary, just like the histograms.""" 

        file = self.tDir
        # print 'file : ', file.GetName()
    
        self._LoadHistograms( self.tDir )
        # pattern = re.compile(regexp)
    
        listOfKeys = file.GetListOfKeys()
        for key in listOfKeys:
            keyname = key.GetName()
            subdir = file.GetDirectory( keyname )
            if subdir != None:
                rootDir = RootDir( subdir, self.style )
                rootDir._Walk()
                self.subDirs[subdir.GetName()] = rootDir

    def DrawAll( self, xsize=800, ysize=800, opt=''):
        """Draw all histograms in the RootDir canvas. Note that histograms in a given sub-directory can be drawn by doing: this.subDirs['theSubDir'].Draw()"""
        
        nPlots = len(self.histograms)

        if nPlots:
            self.canvas = TCanvas(self.tDir.GetName(),self.tDir.GetName(), xsize, ysize)

            # ny = int(math.sqrt (nPlots) ) 
            # nx = int(math.ceil( nPlots / float(ny) ))

            nx, ny = self._Pave( nPlots )
            print nPlots, ny, nx
        
            self.canvas.Divide(nx, ny)
            i = 1
            for key in sorted(self.histograms.iterkeys()):
                self.canvas.cd(i)
                self.histograms[key].Draw(opt)
                i = i+1

            self.canvas.Modified()
            self.canvas.Update()
        
        for key in sorted(self.subDirs.iterkeys()):
            self.subDirs[key].DrawAll(xsize, ysize, opt)


    def _Pave(self, nPlots):
        '''Most efficient use of the canvas space for a given number of plots.
        Trying to keep the canvas more or less square.'''
        nx = 1
        ny = 1
        lastIsNx = False
        while nx * ny < nPlots:
            if not lastIsNx:
                nx += 1
                lastIsNx = True
            else:
                ny += 1
                lastIsNx = False
        return nx, ny

    def Hist( self, histName ):
        """Returns an histogram in this RootDir or in its subdirectories. 
        as histName, give the absolute path, which can be obtained from the
        Print function
        """
        pathList = histName.split('/')
        if len(pathList) == 1:
            histKey = pathList[0]
            hist = self.histograms[ histKey ]
            if hist!=None:
                return hist
            else:
                return None
        else:
            subDirKey = pathList.pop(0)
            print subDirKey
            subDir = self.subDirs[subDirKey]
            if subDir!=None:
                return subDir.h( string.join(pathList, '/') )
            else:
                return None

    def h(self, histName):
        return self.histograms[ histName ]

    def SubDir(self, dirName):
        return self.subDirs[dirName]
    
    def Draw( self, histName):
        """Draws an histogram. Use the absolute path."""
        hist = self.h(histName).Draw()

    def Print( self, path=None, printout=None):
        """Print the contents of this RootDir, including its sub-directories."""
        if printout == None:
            printout = [] # list of lines
        for key in sorted(self.histograms.iterkeys()):
            if path!=None:
                printout.append( path+'/'+key )
            else:
                printout.append(key)
        for key in sorted(self.subDirs.iterkeys()):
            # printout.append( key )
            subPath = key
            if path!=None:
                subPath = path + '/' + key
            printout = self.subDirs[key].Print( subPath, printout )
        return printout

    def __str__(self):
        return '\n'.join( self.Print() ) 
        

if __name__ == '__main__':

    import os, sys

    filename = os.environ.get('PYTHONSTARTUP')
    if filename and os.path.isfile(filename):
        exec(open(filename).read())

    from ROOT import gROOT, TFile,gPad
    gROOT.Macro( os.path.expanduser( '~/rootlogon.C' ) )


    from optparse import OptionParser
    import sys
    
    parser = OptionParser()
    
    parser.usage = '''
    RootDir.py <root file name>
    '''
    parser.add_option("-l", "--list", 
                      dest="list", 
                      help="listing mode, no drawing.",
                      action='store_true',
                      default=False)
    parser.add_option("-d", "--dirname", 
                      dest="dirname", 
                      help="Draw a subdirectory",
                      default=None)

    (options,args) = parser.parse_args()

    if len(args) != 1:
        print 'ERROR: please provide an input root file'
        print args
        sys.exit(1)

    fileName = args[0]

    file = TFile( fileName )
    rootDir = RootDir( file, sRed )
    if options.dirname is not None:
        rootDir = rootDir.SubDir( options.dirname )
    print rootDir
    if not options.list:
        rootDir.DrawAll()
  
    
    # rootDir.subDirs_[dir].DrawAll()
