import ROOT

file = ROOT.TFile.Open("alias.root")
events = file.Get("Events")

b = events.GetAlias("foo")
if not b:
    raise RuntimeError("foo missing")
