// To find the int lumi associated to runNumber
h1RunLumiDel->GetBinContent(h1RunLumiDel->GetXaxis()->FindBin(runNumber))

// To find the run closest to a lumi value (max 0.5 /fb)
h1RunLumiDel->GetBinWithContent(lumi,bin,0,h1RunLumiDel->GetNbinsX(),0.5)
h1RunLumiDel->GetXaxis()->GetBinLowEdge(bin)