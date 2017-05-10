from ROOT import TFile, TCanvas, TH1F, gPad
import time

holder = list()

def plot(fname):    
    root_file = TFile(fname)
    tree = root_file.Get('events')
    
    canvas = TCanvas("canvas", "canvas", 600,600)
    
    hist = TH1F("hist", ";mass of all particles (GeV)", 50, 0, 200)
    tree.Draw('sum_all_m>>hist', '', '') 
    hist.Fit("gaus")
    gPad.Update()
    gPad.SaveAs('sum_all_m.png')
    time.sleep(1)
    func = hist.GetFunction("gaus")

    holder.extend([root_file, tree, canvas, hist, func])
    
    return func.GetParameter(1), func.GetParameter(2)

if __name__ == '__main__':

    import sys
    
    if len(sys.argv)!=2:
        print 'usage <ZHTreeProducer root file>'
        sys.exit(1)

    mean, sigma = plot(sys.argv[1])
    print mean, sigma
    
    
