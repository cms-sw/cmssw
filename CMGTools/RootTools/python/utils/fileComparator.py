import time
import os
import re
from CMGTools.RootTools.HistComparator import *
from CMGTools.RootTools.PyRoot import *
from CMGTools.RootTools.html.DirectoryTree import Directory
from CMGTools.RootTools.utils.file_dir import file_dir, file_dir_names

def mkdir_p(path):
    '''equivalent to mkdir -p.

    If path exists, nothing is done.
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise


class Comparator(object):
    '''Compare the histograms in a TDirectory to the histograms in another TDirectory'''

    def __init__(self, info1, info2, outdir='Plots_Comparator', filter='.*',
                 title1=None, title2=None):
        '''

        info1 and info2 are of the form <root_file>:<directory_in_file>

        filter is a regexp pattern to select histograms to be compared according
        to their name.

        title1 and title2 are titles for both sets of histograms.

        outdir is the directory where all plots will be saved.
        '''
        self.info1 = info1
        self.info2 = info2
        self.outdir = outdir
        self.legend = None
        self.filter = re.compile(filter)
        self.hcomp = None
        self.title1 = title1
        self.title2 = title2

    def browse(self, wait = True):
        '''Browse the two directories and make the plots.

        if wait is True, waits for any key before moving to next histogram.
        '''
        # self.can = TCanvas ()
        # threshold = 0.3
        # self.pad_ratio = TPad ('ratio','ratio',0,0,1,threshold)
        # self.pad_ratio.Draw()
        # self.pad_main  = TPad ('main','main',0,threshold,1,1)
        # self.pad_main.Draw()
        maindir = self.outdir
        if os.path.isdir(maindir):
            os.system( 'rm -r ' + maindir)
        os.mkdir(maindir)
        for h1name, h1 in sorted(self.info1.hists.iteritems()):
            h2 = self.info2.hists.get(h1name, None)
            if h2 is None:
                pass
            # print h1name, 'not in', d2dir.GetName()
            else:
                plotdir = '/'.join([maindir,os.path.dirname(h1name)])
                try:
                    mkdir_p( plotdir )
                except:
                    pass
                h1.SetTitle(h1name)
                h2.SetTitle(h1name)
                if not self.filter.search( h1name ):
                    print 'Skipping', h1name
                    continue
                self._drawHists(h1, h2, h1name)
                if wait : res = raw_input('')
                
    def _drawHists(self, h1, h2, h1name):
        '''Compare 2 histograms'''
        
        h1.SetMarkerColor(1) 
        h1.SetMarkerStyle(21) 
        h1.SetLineColor(1) 
        h1.SetMarkerSize(0.8)
        
        h2.SetFillColor(16) 
        h2.SetFillStyle(1001)
        h2.SetMarkerColor(1)                    
        h2.SetMarkerStyle(4)                   
        h2.SetLineColor(1)
        h2.SetMarkerSize(0.8)
        
        title1=self.title1
        title2=self.title2
        if title1 is None:
            title1 = self.info1.name
        if title2 is None:
            title2 = self.info2.name
        # import pdb; pdb.set_trace()
        if not self.hcomp:
            self.hcomp = HistComparator(h1name,h1, h2, title1, title2)
        else:
            self.hcomp.set(h1name, h1, h2, title1, title2)
        self.hcomp.draw()
        print 'Draw', h1name, 'done'
        pngname = '/'.join([self.outdir,h1name+'.png'])
        print pngname
        self.hcomp.can.SaveAs(pngname)
        return True

        
class FlatFile(object):
    def __init__(self, tdir, name):
        self.tdir = tdir
        self.name = name
        self.hists = {}
        self.flatten( self.tdir, '.', self.hists )
        
    def flatten(self, dir, mothername, hists):
        for key in dir.GetListOfKeys():
            name = key.GetName()
            absname = '/'.join([mothername, name])
            obj = dir.Get(name)
            if type(obj) in dirTypes:
                self.flatten(obj, absname, self.hists)
            elif type(obj) in histTypes:
                self.hists[absname] = obj


if __name__ == '__main__':
    import sys

    from optparse import OptionParser
    
    parser = OptionParser()
    parser.usage = '''
    fileComparator.py <file1[:dir1]> <file2[:dir2]> 
    '''
    parser.add_option("-f", "--filter", 
                      dest="filter", 
                      help="Filtering regexp pattern to select histograms.",
                      default='.*')

    parser.add_option("-1", "--t1", 
                      dest="title1", 
                      help="Title for first set of histograms.",
                      default=None)

    parser.add_option("-2", "--t2", 
                      dest="title2", 
                      help="Title for second set of histograms.",
                      default=None)

    parser.add_option("-o", "--outdir", 
                      dest="outdir", 
                      help="Output directory for all plots.",
                      default='Plots_Comparator')


    parser.add_option("-w", "--nowait", 
                      dest="wait", 
                      help="not waiting for a keystroke between one plot and the following one.",
                      action="store_false",
                      default=True)

    parser.add_option("-b", "--batch", 
                      dest="batch", 
                      help="Set batch mode.",
                      action="store_true",
                      default=False)


    (options,args) = parser.parse_args()

    if len(args)!=2:
        parser.print_usage()
        print 'provide 2 sets of histograms'

    if options.batch:
        gROOT.SetBatch()
        options.wait=False
    
    f1, d1 = file_dir(args[0])
    f2, d2 = file_dir(args[1])
    name1 = '/'.join( [f1.GetName(), d1.GetName()])
    name2 = '/'.join( [f2.GetName(), d2.GetName()])
    file1 = FlatFile( d1, name1)
    file2 = FlatFile( d2, name2)
    comparator = Comparator(file1, file2, options.outdir, options.filter,
                            options.title1, options.title2)
    comparator.browse(wait = options.wait)
    dir = Directory(options.outdir)

