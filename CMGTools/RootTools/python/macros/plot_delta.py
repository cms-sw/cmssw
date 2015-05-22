from CMGTools.RootTools.PyRoot import *

import copy
import sys
from optparse import OptionParser

parser = OptionParser()
parser.usage = '''
plot_delta.py [root_file]
'''

(options,args) = parser.parse_args()

file = None
if len(args)>1:
    parser.print_help()
    print
    print 'Maximum one argument (a root file)'
    sys.exit(1)
elif len(args)==1:
    file = TFile( args[0] )
tree = file.Get('DeltaTreeAnalyzer')


dRMax = 0.1
dR2Max = dRMax*dRMax

dR2_1 = '(col1Eta-genEta)*(col1Eta-genEta) + (col1Phi-genPhi)*(col1Phi-genPhi)'
dR2_2 = '(col2Eta-genEta)*(col2Eta-genEta) + (col2Phi-genPhi)*(col2Phi-genPhi)'

dRMatch1 = '{dR2}<{dR2Max}'.format(dR2 = dR2_1, dR2Max = dR2Max)
dRMatch2 = '{dR2}<{dR2Max}'.format(dR2 = dR2_2, dR2Max = dR2Max)

tree.SetAlias( 'dR2_1', dR2_1)
tree.SetAlias( 'dR2_2', dR2_2)
tree.SetAlias( 'dRMatch1', dRMatch1)
tree.SetAlias( 'dRMatch2', dRMatch2)

tree.SetAlias( 'col1Exists', 'col1Eta>-999' )
tree.SetAlias( 'col2Exists', 'col2Eta>-999' )
tree.SetAlias( 'match1', 'col1Exists && dR2_1 < 0.05' )
tree.SetAlias( 'match2', 'col2Exists && dR2_2 < 0.05' )

tree.SetAlias( 'dPt1', 'col1Pt - genPt' )
tree.SetAlias( 'dPt2', 'col2Pt - genPt' )

def plot2D(nEv, zoom=True):
    zoomCut = '1'
    if zoom:
        zoomCut = 'abs(col2Pt-genPt)<100 && abs(col1Pt-genPt)<100'
    tree.Draw('col2Pt-genPt : col1Pt - genPt', zoomCut + ' && match2 && col2Sel && match1', 'col', nEv)
    sBlack.formatHistoAxis( tree.GetHistogram() )
    tree.GetHistogram().SetStats(1)
    # tree.GetHistogram().SetTitle(';#Deltap_{T}(PF-gen) (GeV/c);#Deltap_{T}(std-gen) (GeV/c)')
    formatPad(gPad)
    gPad.SetLogz()

def plot1D(nEv, zoom=-1):
    zoomCut = '1'
    if zoom>0:
        zoomCut = 'abs(col2Pt-genPt)<{z} && abs(col1Pt-genPt)<{z}'.format(z=zoom)
    tree.Draw('abs(dPt2) - abs(dPt1)', zoomCut + ' && match2 && col2Sel && match1', '', nEv)
    sBlack.formatHisto( tree.GetHistogram() )
    tree.GetHistogram().SetStats(1)
    tree.GetHistogram().SetNdivisions(5)
    tree.GetHistogram().SetTitle(';|#Deltap_{T}(std-gen)| - |#Deltap_{T}(PF-gen)| (GeV/c)')
    formatPad(gPad)
    gPad.SetLogz()


keeper = []

def plotVsGen(nEv, select=1, xmax = 100):
    # dPt = 'dPt1'
    # if select == 2:
    #     dPt = 'dPt2'
    h1 = TH1F('h1','',200, -xmax, xmax)
    h2 = TH1F('h2','',200, -xmax, xmax)
    
    tree.Project('h1', 'dPt1', 'match2 && col2Sel && match1' , '', nEv)
    sBlue.formatHisto( h1)
    h1.SetStats(0)
    h1.SetFillStyle(0)
    h1.SetNdivisions(5)
    h1.SetTitle(';#Deltap_{T} (rec-gen) (GeV/c)')

    tree.Project('h2', 'dPt2', 'match2 && col2Sel && match1' , '', nEv)
    sBlack.formatHisto( h2 )
    h2.SetFillStyle(0)
    h2.SetStats(1)
    h2.SetNdivisions(5)
    h2.SetTitle(';#Deltap_{T} (rec-gen) (GeV/c)')

    
    h1.SetStats(0)
    h2.SetStats(0)

    legend = TLegend(0.6,0.7,0.89,0.89) 
    
    if select==1:
        h1.Draw()
        legend.AddEntry( h1, 'PF muon', 'l') 
        name = 'PF' 
    elif select==2:
        h2.Draw()
        legend.AddEntry( h2, 'tight muon', 'l') 
        name = 'Tight'
    else:
        h1.Draw()
        h2.Draw('same')
        legend.AddEntry( h1, 'PF muon') 
        legend.AddEntry( h2, 'tight muon')
        name = 'Both'
    legend.Draw('same')

    keeper.append(h1)
    keeper.append(h2)
    keeper.append(legend)
    
    formatPad(gPad)
    gPad.SetLogy()

    gPad.SaveAs('ptrecMgen_{name}.png'.format( name=name ))

