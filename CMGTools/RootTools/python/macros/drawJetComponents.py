# draws the jet photon and neutral hadron fractions

# just copy paste the lines below to your python prompt

from ROOT import gDirectory, TLegend,gPad, TCanvas
nEvents = 999999999999

jets = 'jets'

addCut = ''

gCan = TCanvas()
events.Draw(jets + '.obj.component(4).fraction()', '1' + addCut, '', nEvents)
h1 = events.GetHistogram()
h1.SetLineWidth(2)
h1.SetTitle(';f_{#gamma}')
gPad.SaveAs('fgamma.png')

nhCan = TCanvas()
events.Draw(jets + '.obj.component(5).fraction()', '1' + addCut, '', nEvents)
h2 = events.GetHistogram()
h2.SetLineWidth(2)
h2.SetTitle(';f_{nh}')
gPad.SaveAs('fnh.png')
