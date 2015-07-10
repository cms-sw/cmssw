import ROOT

from CMGTools.VVResonances.plotting.TreePlotter import TreePlotter
from CMGTools.VVResonances.plotting.MergedPlotter import MergedPlotter
from CMGTools.VVResonances.plotting.StackPlotter import StackPlotter
 

#create the W+jets plotters
wjPlotters=[]

for sample in ['WJetsToLNu_HT100to200','WJetsToLNu_HT200to400','WJetsToLNu_HT400to600','WJetsToLNu_HT600toInf']:
    wjPlotters.append(TreePlotter('samples/'+sample+'.root','tree'))
    wjPlotters[-1].setupFromFile('samples/'+sample+'.pck')
    wjPlotters[-1].addCorrectionFactor('xsec','xsec',0.0,'lnN')


WJets = MergedPlotter(wjPlotters)


WJets.setFillProperties(1001,ROOT.kAzure-9)

RSGWWLNuQQ = TreePlotter('samples/RSGravToWWToLNQQ_2000.root','tree')
RSGWWLNuQQ.setupFromFile('samples/RSGravToWWToLNQQ_2000.pck')
RSGWWLNuQQ.setFillProperties(0,ROOT.kWhite)
RSGWWLNuQQ.setLineProperties(1,ROOT.kOrange+10,3)
#RSGWWLNuQQ..addCorrectionFactor('xsec',0.001,0.0,'lnN')


#Stack
vvStack = StackPlotter()
vvStack.addPlotter(WJets,"W+jets","W+Jets","background")
vvStack.addPlotter(RSGWWLNuQQ,"RSG2000","RSGWW #rightarrow l#nu QQ","signal")
