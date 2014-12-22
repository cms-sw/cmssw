import os, sys
from ROOT import gSystem, gROOT, TFile, TCanvas, gPad, TBrowser, TH2F


gROOT.Macro( os.path.expanduser( '~/rootlogon.C' ) )

def loadFWLite():
    gSystem.Load("libFWCoreFWLite")
    gROOT.ProcessLine('AutoLibraryLoader::enable();')
    gSystem.Load("libFWCoreFWLite")



def init(events):
    
    events.SetAlias('vertex','recoVertexs_offlinePrimaryVertices__RECO')
    events.SetAlias('pu','recoPFCandidates_pfPileUp__PF2PAT')
    events.SetAlias('nopu','recoPFCandidates_pfNoPileUp__PF2PAT')

    events.SetAlias('tvnopu', 'nopu.obj.vertex()')
    events.SetAlias('tvrhonopu','sqrt(tvnopu.x()*tvnopu.x()+tvnopu.y()*tvnopu.y())')
    
    events.SetAlias('run','EventAuxiliary.id().run()')
    events.SetAlias('lumi','EventAuxiliary.id().luminosityBlock()')

    events.SetAlias('dzpu','pu.obj.vertex().z()-vertex.obj[0].z()')
    events.SetAlias('dznopu','nopu.obj.vertex().z()-vertex.obj[0].z()')

    return events

loadFWLite()

file = TFile( sys.argv[1] )
events = file.Get('Events')
init(events)

events.Draw('dznopu','abs(dznopu)<1')
