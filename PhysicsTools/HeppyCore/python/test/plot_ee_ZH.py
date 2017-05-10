from ROOT import TFile, TCanvas, TH1F, gPad
import time

def plot(fname):
    root_file = TFile(fname)
    tree = root_file.Get('events')
    
    canvas = TCanvas("canvas", "canvas", 600,600)
    
    h = TH1F("h", "higgs di-jet mass;m_{jj} (GeV)", 50, 0, 200)
    tree.Draw('higgs_m>>h', 'zed_m>50') 
    # h.GetYaxis().SetRangeUser(0, 120)
    h.Fit("gaus")
    gPad.Update()
    gPad.SaveAs('ee_ZH_mjj.png')
    time.sleep(1)
    func = h.GetFunction("gaus")
    return func.GetParameter(1), func.GetParameter(2)


if __name__ == '__main__':

    import sys
    
    if len(sys.argv)!=2:
        print 'usage <ZHTreeProducer root file>'
        sys.exit(1)

    mean, sigma = plot(sys.argv[1])
    print mean, sigma
    
    
