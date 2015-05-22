import os, sys
from ROOT import * 

class jetHistos:
    def __init__(self, name):
        self.respEta_ = TProfile('respEta'+name,'',20, -5, 5)
        self.respPt_ = TH2F('respPt'+name,'',100, 0, 1000)



def showBin( hist, bin):
    hist.ProjectionY('',bin,bin,'').Fit('gaus')

from CMGTools.RootTools.RootTools import *

gROOT.Macro( os.path.expanduser( '~/rootlogon.C' ) )



def setAliases():
    patJets = 'patJets_selectedPatJetsAK5__PAT'
    if not allJets:
        patJets = 'patJets_patJetLead__ANA'
    
    events.SetAlias('jet', patJets)
    
    deltaPhi = '%s.obj.phi()-%s.obj.genJet().phi()' % (patJets, patJets)
    events.SetAlias('dPhi',deltaPhi )
    deltaEta = '%s.obj.eta()-%s.obj.genJet().eta()' % (patJets, patJets)
    events.SetAlias('dEta',deltaEta )
    deltaR = 'sqrt( dPhi*dPhi + dEta*dEta)'
    events.SetAlias('dR',deltaR )
    
    genJet = '%s.obj.genJet()' % patJets
    events.SetAlias('genJet',genJet )
    

def buildPrefix():
    basename = rootfile.split('/')[0]
    pref = os.path.splitext( basename)[0]
    pref += '_pt' + str(genJetPtCut)
    if raw:
        pref += '_raw'
    if dRCut:
        pref += '_dR'
    if allJets:
        pref += '_all'
    return pref

from CMGTools.RootTools.response import response

def plotPtResponse( response ): 
    
    prefix = buildPrefix()
    setAliases()

    print 'plotPtResponse : ', prefix, '...'

    canvas = response.canvas
    canvas.cd(1)
    response.h2d = TH2F('responsePt_' + response.name,';p_{T}(gen) (GeV);p_{T}(rec)/p_{T}(gen)',20, 0, 600, 50, 0,2)
    # events.Draw('jet.obj[0].pt()/jet.obj[0].genJet().pt():jet.obj[0].genJet().pt()>>'+ response.h2d.GetName(),'jet.obj[0].genJet().pt()>0 && abs(jet.obj[0].genJet().eta())<1.5',"col")

    var = 'jet.obj.pt()/jet.obj.genJet().pt():jet.obj.genJet().pt()>>' + response.h2d.GetName()
    if raw:
        var = 'jet.obj.pt()*jet.obj.jecFactor(0)/jet.obj.genJet().pt():jet.obj.genJet().pt()>>' + response.h2d.GetName()
    cut = 'jet.obj.genJet().pt()>%s && abs(jet.obj.genJet().eta())<1.5' % genJetPtCut
    if dRCut:
        cut +=  '&& dR<0.2'

    print var
    print cut
     
    events.Draw(var,cut,"col")
    
    response.FitSlicesY()
    response.Draw()
    
    canvas.SaveAs(prefix + '_' + response.name + '.png')


def plotEtaResponse( response ): 

    prefix = buildPrefix()
    setAliases()

    print 'plotEtaResponse : ', prefix, '...'
    
    canvas = response.canvas
    canvas.cd(1)
    response.h2d = TH2F('responseEta_'+response.name,';#eta(gen);p_{T}(rec)/p_{T}(gen)',50, -5, 5, 50, 0,2)

    var = 'jet.obj.pt()/jet.obj.genJet().pt():jet.obj.genJet().eta()>>'+ response.h2d.GetName()
    if raw:
        var = 'jet.obj.pt()*jet.obj.jecFactor(0)/jet.obj.genJet().pt():jet.obj.genJet().eta()>>'+ response.h2d.GetName()
        
    cut = 'jet.obj.genJet().pt()>%s' % genJetPtCut
    if dRCut:
        cut +=  '&& dR<0.2'
    
    # events.Draw('jet.obj[0].pt()/jet.obj[0].genJet().pt():jet.obj[0].genJet().eta()>>'+ response.h2d.GetName(),'jet.obj[0].genJet().pt()>0 && jet.obj[0].pt()>30',"col")
    events.Draw(var, cut ,"col")
    
    response.FitSlicesY()
    response.Draw()
    
    canvas.SaveAs( prefix + '_' + response.name + '.png')
    

def plotCor():
    global ptcor
    global etacor
    global raw
    
    raw = False
    
    ptcor = response('ptcor')
    plotPtResponse( ptcor )
    
    etacor = response('etacor')
    plotEtaResponse( etacor )
   
def plotRaw():
    global pt
    global eta
    global raw
    
    raw = True
    
    pt = response('pt')
    plotPtResponse( pt )
    
    eta = response('eta')
    plotEtaResponse( eta )
    
def plotAll():
    plotCor()
    plotRaw()

def plotDREffect():
    global dRon
    global dRoff

    global dRCut
    
    dRCut = False
    
    dRoff = response( 'dRoff' )
    plotPtResponse( dRoff )
    
    dRCut = True
    
    dRon = response( 'dRon' )
    plotPtResponse( dRon )

    
raw = True 
genJetPtCut = 20
dRCut = False
allJets = True

rootfile  = sys.argv[1]

events = Chain('Events', rootfile)
setAliases()

if __name__ == '__main__':

    # pass
    
    plotCor()

