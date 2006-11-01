# first load cmstools and ROOT classes
from cmstools import *
from ROOT import *

# opening file and accessing branches
print "Opening SimHit file"
file = TFile("simevent.root")
events = file.Get("Events")
branch = events.GetBranch("PSimHit_r_TrackerHitsTIBLowTof.obj")

simHit = std.vector(PSimHit)()
branch.SetAddress(simHit)

histo = TH1F("tofhits", "Tof of hits", 100, -0.5, 50)

# loop over all events 
for ev in all(events):
    branch.GetEntry(ev)
    for hit in all(simHit):
        histo.Fill(hit.timeOfFlight())

hFile = TFile("histo.root", "RECREATE")
histo.Write()

gROOT.SetBatch()
gROOT.SetStyle("Plain")

c = TCanvas()
histo.Draw()
c.SaveAs("tofhits.jpg")
