# first load cmstools and ROOT classes
from PhysicsTools.PythonAnalysis import *
from ROOT import *

gSystem.Load("libFWCoreFWLite.so")
FWLiteEnabler::enable()

# opening file
events = EventTree("simevent.root")

# prepare the histogram
histo = TH1F("tofhits", "Tof of hits", 100, -0.5, 50)

# loop over all events and filling the histogram
for event in events:
    simHits = event.getProduct("PSimHit_r_TrackerHitsTIBLowTof.obj")
    for hit in simHits:
        histo.Fill(hit.timeOfFlight())

hFile = TFile("histo.root", "RECREATE")
histo.Write()

gROOT.SetBatch()
gROOT.SetStyle("Plain")

c = TCanvas()
histo.Draw()
c.SaveAs("tofhits.jpg")
