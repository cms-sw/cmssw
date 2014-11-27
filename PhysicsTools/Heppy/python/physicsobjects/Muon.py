from slow.Muon import Muon 
#comment next three lines for old objects
import ROOT
import FastObjects
FastObjects.decorate(ROOT.pat.Muon,Muon)
Muon=FastObjects.AddPhysObj
