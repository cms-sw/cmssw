import os

from CMGTools.RootTools.treeComparator import main, draw, tree1, tree2, a1, a2
from CMGTools.RootTools.html.DirectoryTree import Directory
from ROOT import TCanvas

cut_inclusive = '1'
cut_J1 = 'njets>0'
cut_J2 = 'njets>1'

cut_tt = "1"

#cut_tt = "pt_1>45 && pt_2>45 && abs(eta_1)<2.1 && abs(eta_2)<2.1 && q_1+q_2==0 && (iso_1>0.884 && iso_2>0.884) && abs(jeta_1)<3.0 && jpt_1>50 && antiele_2==1"


class PlotInfo(dict):
    '''Just a dictionary to hold the arguments to be passed to the draw function.
    The resulting histogram comparator will be attached to this. 
    '''
    def __init__(self, **kwargs):
        super(PlotInfo, self).__init__(**kwargs)

#COLIN do we really need to duplicate code for tt?
plots_tt = {
    'pt_1': PlotInfo(var1='pt_1', cut=cut_tt, xmax=200),
    'pt_2': PlotInfo(var1='pt_2', cut=cut_tt, xmax=200),
    'jpt_1': PlotInfo(var1='jpt_1', cut=cut_tt, xmax=300),
    'jpt_2': PlotInfo(var1='jpt_2', cut=cut_tt, xmax=300),
    'jdeta': PlotInfo(var1='jdeta', cut=cut_tt, xmax=10),
    'eta_1': PlotInfo(var1='eta_1', cut=cut_tt, xmin=-3, xmax=3),
    'eta_2': PlotInfo(var1='eta_2', cut=cut_tt, xmin=-3, xmax=3),
    'iso_1': PlotInfo(var1='iso_1', cut=cut_tt, xmax=1.),
    'iso_2': PlotInfo(var1='iso_1', cut=cut_tt, xmax=1.),
    'antiele_2': PlotInfo(var1='antiele_2', cut=cut_tt, xmax=2.),
    'mvamet': PlotInfo(var1='mvamet', cut=cut_tt, xmax=200),
    'met': PlotInfo(var1='met', cut=cut_tt, xmax=200),
    'mvametphi': PlotInfo(var1='mvametphi', cut=cut_tt, xmin=-3.1416, xmax=3.1416),
    'mjj': PlotInfo(var1='mjj', cut=cut_tt, xmax=600),
    'njetingap': PlotInfo(var1='njetingap', cut=cut_tt, xmax=10),
    #'mvacov00': PlotInfo(var1='mvacov00', cut=cut_tt, xmax=1000),
    #'mvacov10': PlotInfo(var1='mvacov10', cut=cut_tt, xmax=1000),
    #'mvacov01': PlotInfo(var1='mvacov01', cut=cut_tt, xmax=1000),
    #'mvacov11': PlotInfo(var1='mvacov01', cut=cut_tt, xmax=1000),
    'mt_1': PlotInfo(var1='mt_1', cut=cut_tt, xmax=200),
    'm_vis': PlotInfo(var1='m_vis', cut=cut_tt, xmax=200),
    'm_sv': PlotInfo(var1='m_sv', cut=cut_tt, xmax=350),
    'njets': PlotInfo(var1='njets', cut=cut_tt, nbins=5, xmax=5),
    'nbtag': PlotInfo(var1='nbtag', cut=cut_tt, nbins=5, xmax=5),
    'puweight': PlotInfo(var1='puweight', cut=cut_tt, xmax=5),
    #'weight': PlotInfo(var1='weight', cut=cut_tt, xmax=5),
    }

plots_tt_weights = {
    'puweight': PlotInfo(var1='puweight', cut=cut_tt, xmax=5),
    'trigweight_1': PlotInfo(var1='trigweight_1', cut=cut_tt, nbins=200, xmin=0.7, xmax=1.3),
#    'idweight_1': PlotInfo(var1='idweight_1', cut=cut_tt, nbins=200, xmin=0.7, xmax=1.3),
#    'isoweight_1': PlotInfo(var1='isoweight_1', cut=cut_tt, nbins=200, xmin=0.7, xmax=1.3),
    'trigweight_2': PlotInfo(var1='trigweight_2', cut=cut_tt, nbins=200, xmin=0.7, xmax=1.3),
#    'idweight_2': PlotInfo(var1='idweight_2', cut=cut_tt, nbins=200, xmin=0.7, xmax=1.3),
#    'isoweight_2': PlotInfo(var1='isoweight_2', cut=cut_tt, nbins=200, xmin=0.7, xmax=1.3),
    }


plots_inclusive = {
    'pt_1': PlotInfo(var1='pt_1', cut=cut_inclusive, xmax=200),
    'pt_2': PlotInfo(var1='pt_2', cut=cut_inclusive, xmax=200),
    'eta_1': PlotInfo(var1='eta_1', cut=cut_inclusive, xmin=-3, xmax=3),
    'eta_2': PlotInfo(var1='eta_2', cut=cut_inclusive, xmin=-3, xmax=3),
    'iso_1': PlotInfo(var1='iso_1', cut=cut_inclusive, xmax=0.1),
    'mvamet': PlotInfo(var1='mvamet', cut=cut_inclusive, xmax=100),
    'mvametphi': PlotInfo(var1='mvametphi', cut=cut_inclusive, xmin=-3.1416, xmax=3.1416),
    'mvacov00': PlotInfo(var1='mvacov00', cut=cut_inclusive, xmax=1000),
    'mvacov10': PlotInfo(var1='mvacov10', cut=cut_inclusive, xmax=1000),
    'mvacov01': PlotInfo(var1='mvacov01', cut=cut_inclusive, xmax=1000),
    'mvacov11': PlotInfo(var1='mvacov01', cut=cut_inclusive, xmax=1000),
    'mt_1': PlotInfo(var1='mt_1', cut=cut_inclusive, xmax=200),
    'mvis': PlotInfo(var1='mvis', cut=cut_inclusive, xmax=200),
    'm_sv': PlotInfo(var1='m_sv', cut=cut_inclusive, xmax=350),
    'njets': PlotInfo(var1='njets', cut=cut_inclusive, nbins=5, xmax=5),
    'nbtag': PlotInfo(var1='nbtag', cut=cut_inclusive, nbins=5, xmax=5),
    'weight': PlotInfo(var1='weight', cut=cut_inclusive, xmax=5),
    'puweight': PlotInfo(var1='puweight', cut=cut_inclusive, xmax=5),
    'trigweight_1': PlotInfo(var1='trigweight_1', cut=cut_inclusive, nbins=200, xmin=0.7, xmax=1.3),
    'idweight_1': PlotInfo(var1='idweight_1', cut=cut_inclusive, nbins=200, xmin=0.7, xmax=1.3),
    'isoweight_1': PlotInfo(var1='isoweight_1', cut=cut_inclusive, nbins=200, xmin=0.7, xmax=1.3),
    'trigweight_2': PlotInfo(var1='trigweight_2', cut=cut_inclusive, nbins=200, xmin=0.7, xmax=1.3),
    'idweight_2': PlotInfo(var1='idweight_2', cut=cut_inclusive, nbins=200, xmin=0.7, xmax=1.3),
    'isoweight_2': PlotInfo(var1='isoweight_2', cut=cut_inclusive, nbins=200, xmin=0.7, xmax=1.3),
    }

plots_J1 = {
    'jpt_1': PlotInfo(var1='jpt_1', cut=cut_J1, xmax=200),
    'jeta_1': PlotInfo(var1='jeta_1', cut=cut_J1, xmin=-5, xmax=5),
    # 'jphi_1': PlotInfo(var1='jphi_1', cut=cut_J1, xmin=-3.1416, xmax=3.1416),
    }

plots_J2 = {
    'jpt_2': PlotInfo(var1='jpt_2', cut=cut_J2, xmax=200),
    'jeta_2': PlotInfo(var1='jeta_2', cut=cut_J2, xmin=-5, xmax=5),
    # 'jphi_2': PlotInfo(var1='jphi_2', cut=cut_J2, xmin=-3.1416, xmax=3.1416),
    'njetingap': PlotInfo(var1='njetingap', cut=cut_J2, nbins=5, xmax=5),
    'jdeta': PlotInfo(var1='jdeta', cut=cut_J2, nbins=20, xmin=-10, xmax=10),
    'mjj': PlotInfo(var1='mjj', cut=cut_J2, xmax=3000),
    }

def drawAll(plots):
    '''create and draw the comparators corresponding to each of the plot info in the input
    plots dictionary'''
    # comparators = []
    for plotName, plotInfo in plots.iteritems():
        plotInfo['t1'] = tree1
        plotInfo['t2'] = tree2
        plotInfo['name1'] = a1
        plotInfo['name2'] = a2
        c = draw( **plotInfo)
        plotInfo.comparator = c 


def saveAll(plots, outdir='Out'):
    '''save all the plots in png format in outdir.'''
    os.mkdir(outdir)
    oldpwd = os.getcwd()
    os.chdir(outdir)
    for plot in plots.values():
        plot.comparator.save()
    os.chdir(oldpwd)
    dir = Directory(outdir)
    
if __name__ == '__main__':

    tree1, tree2, options = main() 

    allplots = dict(plots_inclusive.items() + plots_J1.items() + plots_J2.items())
    drawAll(allplots)
    saveAll(allplots, options.outdir)
