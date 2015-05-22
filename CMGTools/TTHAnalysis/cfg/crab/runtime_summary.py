import ROOT
from numpy import mean, std

# crab status --long -d crab_fullProd_test4/crab_TTJets_MT2_CMGTools-from-CMSSW_7_2_3/ | grep finished | grep T2_ > ttjets_summary.txt
filename = "ttjets_summary.txt"

def time2num(time): # time format is hh:mm:ss
    t = time.split(":")
    return float(t[0])+(float(t[1])+float(t[2])/60.)/60.


lines = open(filename).readlines()

maxtime = 0.
sites = {}
Ntotal = 0
for l in lines:
    Ntotal+=1
    k,t = l.split()[2],time2num(l.split()[3])
    maxtime = max(maxtime,t)
    if k not in sites:
        sites[k] = [t]
    else:
        sites[k].append(t)

Nbins = int(maxtime*6) # granularity of ~10min
h_timeAll = ROOT.TH1F("h_time_All","run time - all sites", Nbins, 0, maxtime)
h_timeSite = {}

h_timePerSite  = ROOT.TH1F("h_timePerSite", "Average time per site"           , len(sites),0,len(sites))
h_nSites       = ROOT.TH1F("h_nSites"     , "Jobs per site (total: %d)"%Ntotal, len(sites),0,len(sites))

for b,site in enumerate(sites.keys()):
    h_nSites.Fill(site, len(sites[site]))
    h_timePerSite.GetXaxis().SetBinLabel  (b+1, site)
    h_timePerSite.SetBinContent(b+1, mean(sites[site]))
    h_timePerSite.SetBinError  (b+1, std (sites[site]))
    h_timeSite[site] = ROOT.TH1F("h_time_"+site,"run time - "+site, Nbins, 0, maxtime)
    for time in sites[site]:
        h_timeAll.Fill(time)
        h_timeSite[site].Fill(time)

h_timePerSite.SetMinimum(0)
h_timePerSite.SetStats  (0)
h_nSites     .SetMinimum(0)
h_nSites     .SetStats  (0)

c = {}

c[h_timePerSite.GetName()] = ROOT.TCanvas()
h_timePerSite.Draw()

c[h_nSites.GetName()] = ROOT.TCanvas()
h_nSites.Draw()

c[h_timeAll.GetName()] = ROOT.TCanvas()
h_timeAll.Draw()

for h in h_timeSite.values():
    c[h.GetName()] = ROOT.TCanvas()
    h.Draw()
