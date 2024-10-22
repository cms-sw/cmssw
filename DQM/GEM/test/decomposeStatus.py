import ROOT
import os, sys
import argparse
import array


bitDigiOcc = 0

listBitConfFilter = [
    { "name": "AMC13", "error": 1, "warning": -1 }, 
    { "name": "AMC",   "error": 2, "warning": 3  }, 
    { "name": "OH",    "error": 4, "warning": 5  }, 
    { "name": "VFAT",  "error": 6, "warning": 7  }, 
]


parser = argparse.ArgumentParser()

parser.add_argument("src", default="", help="")
parser.add_argument("dst", default="", help="")

opts = parser.parse_args()

fDQM = ROOT.TFile.Open(opts.src)
dirMain = None

for k1 in fDQM.GetListOfKeys(): 
    d1 = fDQM.Get(k1.GetName())
    if not isinstance(d1, ROOT.TDirectoryFile): 
        continue
    
    for k2 in d1.GetListOfKeys(): 
        d2 = d1.Get(k2.GetName())
        if not isinstance(d1, ROOT.TDirectoryFile): 
            continue
        
        dirMain = d2
        break
    
    if dirMain: 
        break

dirEventInfo = dirMain.Get("GEM/Run summary/EventInfo")
listHistLumi = [ k.GetName() for k in dirEventInfo.GetListOfKeys() ]
listHistLumi = [ s for s in listHistLumi if s.startswith("chamberStatus_inLumi") ]

fOut = ROOT.TFile.Open(opts.dst, "RECREATE")

for name in listHistLumi: 
    histCurr = dirEventInfo.Get(name)
    
    listFillLumi = [ histCurr.GetBinContent(i + 1, 0) for i in range(histCurr.GetNbinsX()) ]
    numLumi = max([ i for i, x in enumerate(listFillLumi) if abs(x) > 0 ]) + 1
    
    listBinLumi = [ histCurr.GetXaxis().GetBinLowEdge(i + 1) for i in range(numLumi              + 1) ]
    listBinY    = [ histCurr.GetYaxis().GetBinLowEdge(i + 1) for i in range(histCurr.GetNbinsY() + 1) ]
    
    for dicConf in listBitConfFilter: 
        fOut.cd()
        
        numBinY = len(listBinY) - 1
        
        histNew = ROOT.TH2S(
            dicConf[ "name" ] + "_" + name, 
            histCurr.GetTitle() + " ({})".format(dicConf[ "name" ]), 
            numLumi, 
            array.array("d", listBinLumi), 
            numBinY, 
            array.array("d", listBinY), 
        )
        
        histNew.GetXaxis().SetTitle(histCurr.GetXaxis().GetTitle())
        histNew.GetYaxis().SetTitle(histCurr.GetYaxis().GetTitle())
        
        for i in range(len(listBinY) - 1): 
            histNew.GetYaxis().SetBinLabel(i + 1, histCurr.GetYaxis().GetBinLabel(i + 1))
        
        for j in range(numBinY): 
            for i in range(numLumi): 
                val  = int(histCurr.GetBinContent(i + 1, j + 1))
                occ  = val & ( 1 << bitDigiOcc           ) != 0
                err  = val & ( 1 << dicConf[ "error"   ] ) != 0 if dicConf[ "error"   ] >= 0 else False
                warn = val & ( 1 << dicConf[ "warning" ] ) != 0 if dicConf[ "warning" ] >= 0 else False
                print(dicConf[ "name" ], val, occ, err, warn)
                
                out = 0
                if err: 
                    out = 2
                elif warn: 
                    out = 3
                elif occ: 
                    out = 1
                
                histNew.SetBinContent(i + 1, j + 1, out)
        
        histNew.Write()

fOut.Write()
fOut.Close()
fDQM.Close()


