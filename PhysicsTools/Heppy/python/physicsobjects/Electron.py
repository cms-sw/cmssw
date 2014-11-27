from slow.Electron import Electron
#comment for old
import ROOT
import FastObjects
FastObjects.decorate(ROOT.pat.Electron,Electron)
Electron=FastObjects.AddPhysObj
