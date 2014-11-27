from slow.Jet import Jet,GenJet

#comment for old
import ROOT
import FastObjects
FastObjects.decorate(ROOT.pat.Jet,Jet)
Jet=FastObjects.AddPhysObj
