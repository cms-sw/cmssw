import os, sys
from ROOT import * 


def plotJets( tree, canvas, jeth):
    # canvas.cd(1)
    # tree.Draw('(jet.pt()-genJet.pt())/genJet.pt():genJet().eta()>>'+ jeth.respEta_.GetName(),'genJet.pt()>30')
    
    canvas.cd(2)
    tree.Draw('(jet.pt()-genJet.pt())/genJet.pt():genJet().pt()>>'+ jeth.respPt_.GetName(),'genJet.pt()>0 && abs(genJet.eta())<1.5')

def showBin( hist, bin):
    hist.ProjectionY('',bin,bin,'').Fit('gaus')

from CMGTools.RootTools.RootTools import *

gROOT.Macro( os.path.expanduser( '~/rootlogon.C' ) )

file = TFile( sys.argv[1] )
events = file.Get('Events')

patMet = 'patMETs_patMETsPFlow__PAT'
events.SetAlias('met', patMet)

from response import response

def plotPtResponse( response ): 
    
    canvas = response.canvas
    canvas.cd(1)
    response.h2d = TH2F('responsePt','MET(gen) (GeV); MET(rec)/MET(gen)',50, 0, 1000, 50, 0,2)
    events.Draw('met.obj.pt()/met.obj.genMET().pt():met.obj.genMET().pt()>>'+ response.h2d.GetName(),'met.obj.genMET().pt()>20',"col")
    
    response.FitSlicesY()
    response.Draw()

    canvas.SaveAs(response.name + '.png')



if __name__ == '__main__':

    pt = response('pt')
    plotPtResponse( pt )
