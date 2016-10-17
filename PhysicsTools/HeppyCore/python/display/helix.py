from ROOT import THelix, TH3F, gPad

helix = THelix(0, 0, 0, 2, 0, 1, 4)

hframe = TH3F("hframe","", 10, -2, 2, 10, -2, 2, 10, -2, 2)
hframe.Draw()
helix.SetRange(0, 0.1, 0)

helix.Draw("same")

gPad.Update()

