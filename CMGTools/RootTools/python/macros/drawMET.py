# can draw MET or MHT
# check the effect of jet ID and of the cleaning filters
#
# just copy/paste to your python prompt all lines below, or part of them.

from ROOT import gDirectory, TLegend,gPad, TCanvas

class MyHistograms:
    def __init__(self, name, title, bins = 100, min = 0, max = 1000):
        self.name = name
        self.h1 = TH1F('h1_'+name, title, bins,min,max)
        self.h2 = TH1F('h2_'+name, title, bins,min,max)
        self.h2 = TH1F('h3_'+name, title, bins,min,max)
        self.canMet = TCanvas('canMet'+name, 'canMet'+name)
        self.canEff = TCanvas('canEff'+name, 'canEff'+name)
        self.canMet.SetLogy()
        self.canEff.SetLogy()
        self.h1.SetLineWidth(2)        
    def computeEff(self):
        self.eff = self.h2.Clone( 'eff_'+ self.name)
        self.eff.GetYaxis().SetTitle('efficiency')
        self.eff.Divide( self.h1 )
        return self.eff
    def setUpLegend(self, caption):
        self.legend = TLegend(0.5,0.5,0.85,0.8)
        self.legend.AddEntry(histos.h1,'all events')
        self.legend.AddEntry(histos.h2,caption)
    def draw(self):
        self.canMet.cd()
        self.h1.Draw()
        self.h2.Draw('same')
        if self.legend != None:
            self.legend.Draw()
        self.canMet.SaveAs(self.canMet.GetName()+'.png')
        self.canEff.cd()
        self.eff.Draw()
        self.canEff.SaveAs(self.canEff.GetName()+'.png')

nEvents = 1000000

met = 'met'

title = ';MET (GeV)'
if met == 'mht':
    title = ';MHT (GeV)'


plotJetId = True

notId99 = 'jetsVLId99Failed.@obj.size()>0'
notId95 = 'jetsVLId95Failed.@obj.size()>0'
beamHaloCSCLoose = 'beamHaloCSCLoose==1'
beamHaloCSCTight = 'beamHaloCSCTight==1'
hbheNoise2010 = 'hbheNoise2010.obj==0'
hbheNoise2011Iso = 'hbheNoise2011Iso.obj==0'
hbheNoise2011NonIso = 'hbheNoise2011NonIso.obj==0'

sel = notId95

# addCut = ' && met.obj[0].et()<500'
# addCut = ' && ht.obj.sumEt()>350 && mht.obj.et()<1000'
addCut = ''

histos = MyHistograms('histos', title = title, bins = 100, max = 500)

events.Draw(met + '.obj.et()>>'+histos.h1.GetName(), '1' + addCut, 'goff', nEvents)

events.Draw(met + '.obj.et()>>'+histos.h2.GetName(), sel + addCut, 'goff', nEvents)
if sel == notId99:
    histos.h2.SetFillColor(4)
    histos.setUpLegend('loose jet ID failed')
elif sel == notId95:
    histos.h2.SetFillColor(2)
    histos.setUpLegend('tight jet ID failed')
elif sel == beamHaloCSCTight:
    histos.h2.SetFillColor(5)
    histos.setUpLegend('CSCTightHaloId failed')    
elif sel == beamHaloCSCLoose:
    histos.h2.SetFillColor(8)
    histos.setUpLegend('CSCLooseHaloId failed')
elif sel == hbheNoise2010:
    histos.h2.SetFillColor(4)
    histos.setUpLegend('HBHENoise 2010 failed')
elif sel == hbheNoise2011Iso:
    histos.h2.SetFillColor(4)
    histos.setUpLegend('HBHENoise 2011 iso failed')
elif sel == hbheNoise2011NonIso:
    histos.h2.SetFillColor(4)
    histos.setUpLegend('HBHENoise 2011 non iso failed')
else:
    print 'What are you doing?'

histos.computeEff()
histos.draw()
