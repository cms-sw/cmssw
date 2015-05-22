import os, sys
from ROOT import * 

class response:
    def __init__(self, name='response'):
        self.name = name
        self.canvas = TCanvas(name,name,1000,500)
        self.canvas.Divide(3,1)

    def FitSlicesY(self):
        self.h2d.FitSlicesY()
        self.mean = gDirectory.Get( self.h2d.GetName() + '_1')
        self.mean.GetYaxis().SetRangeUser(0.5, 1.2)
        self.mean.Draw()
        self.sigma = gDirectory.Get( self.h2d.GetName() + '_2')
        self.sigma.GetYaxis().SetRangeUser(0.0, 0.5)
        self.sigma.Draw()

    def Draw(self):
        self.canvas.cd(1)
        self.h2d.Draw('col')
        self.canvas.cd(2)
        self.mean.Draw()
        self.canvas.cd(3)
        self.sigma.Draw()
        self.canvas.Modified()
        self.canvas.Update()
    
