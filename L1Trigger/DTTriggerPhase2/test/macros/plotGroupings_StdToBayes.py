import pickle, os 
import math 
import ROOT 

ss = ["shared25", "shared50","shared75","shared100"]
samples = {"shared25":ROOT.kBlack, "shared50":ROOT.kRed, "shared75":ROOT.kBlue, "shared100":ROOT.kGreen+1}
tags = {"shared25": "shared 25% hits", "shared50": "shared 50% hits", "shared75":"shared 75% hits" , "shared100":"shared 100% hits"}

plots=[]
titles={}  

# Load histograms
for st in ["MB1","MB2","MB3","MB4"]:
    for wh in ["Wh-2", "Wh-1", "Wh0", "Wh1", "Wh2"]:
        for var in ["Phi","PhiB","Chi2","Bx","Time"]:
            for q in ["q1","q3","q5","q8"]:

                plots.append("h%sRes_%s_%s_%s" %(var, st, wh, q))
                titlename = '; %s^{std}-%s^{bay};' %(var,var)
                if "PhiB" in var: 
                    titlename = titlename.replace('PhiB','#phi_{B}')
                    titlename = titlename.replace('};','} (mrad);')
                elif "Phi" in var:
                    titlename = titlename.replace('Phi','#phi')    
                    titlename = titlename.replace('} ;','} (mrad) ;')            
                elif "Chi2" in var:
                    titlename = titlename.replace('Chi2','#chi^2')                
                titles["h%sRes_%s_%s_%s" %(var, st, wh, q)] = titlename

                
                # Histograms inclusive in wheel number: do them just once
                if wh == "Wh-2":
                    plots.append("h%sRes_%s_%s" %(var, st, q))
                    titlename = '; %s^{std}-%s^{bay};' %(var,var)
                    if "PhiB" in var: 
                        titlename = titlename.replace('PhiB','#phi_{B}')
                        titlename = titlename.replace('};','} (mrad);')
                    elif "Phi" in var:
                        titlename = titlename.replace('Phi','#phi')    
                        titlename = titlename.replace('} ;','} (mrad) ;')            
                    elif "Chi2" in var:
                        titlename = titlename.replace('Chi2','#chi^2')                
                    titles["h%sRes_%s_%s" %(var, st, q)] = titlename

        plots.append("hMatchingEff_%s_%s" %(st, wh))
        titles["hMatchingEff_%s_%s" %(st, wh)] = " ; muon quality ; Efficiency = N_{bayes}/N_{std}"  

        # Eff histogram inclusive in wheel number: do it just once
        if wh == "Wh-2":
            plots.append("hMatchingEff_%s" %(st))
            titles["hMatchingEff_%s" %(st)] = " ; muon quality ; Efficiency = N_{bayes}/N_{std}"  


outpath = "../../../../Groupings/"
if not os.path.exists(outpath):
    os.mkdir(outpath)
    print "cp /afs/cern.ch/user/n/ntrevisa/public/utils/index.php %s/" %outpath
    os.system("cp /afs/cern.ch/user/n/ntrevisa/public/utils/index.php %s/" %outpath)
os.system("cp EventDumpList_StdToBayes.log %s/" %outpath)

outpath = outpath + "StdToBayes/"
if not os.path.exists(outpath):
    os.mkdir(outpath)
    print "cp /afs/cern.ch/user/n/ntrevisa/public/utils/index.php %s/" %outpath
    os.system("cp /afs/cern.ch/user/n/ntrevisa/public/utils/index.php %s/" %outpath)


outFile = ROOT.TFile("GroupingComparison_StdToBayes.root","RECREATE")
outFile.cd()

ROOT.gROOT.ProcessLine('.L PlotTemplate.C+')
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

with open('GroupingComparison_StdToBayes.pickle', 'rb') as handle:
    b = pickle.load(handle)

#leg = ROOT.TLegend(0.6,0.6,0.85,0.26);
leg = ROOT.TLegend(0.6,0.6,0.88,0.4);
leg.SetTextSize(0.03);

for s in ss:        
    for plot in plots:
        print("s = {}, plot = {}".format(s, plot))
        b[s][plot].SetTitle(titles[plot])
        b[s][plot].Write()
        b[s][plot].SetLineColor(samples[s])
        b[s][plot].SetMarkerStyle(8)
        b[s][plot].SetMarkerColor(samples[s])
        b[s][plot].SetMarkerSize(0.8)
        b[s][plot].SetLineWidth(2)

    leg.AddEntry(b[s][plots[0]], tags[s],'l');

canvas = ROOT.CreateCanvas('name', False, True)

for plot in plots:
    drawn = False
    counts=0
    disp = ROOT.TLatex()
    disp.SetTextSize(0.025)
    for s in ss: 
        if not (drawn): 
            b[s][plot].Draw()
            drawn=True
        else:
            b[s][plot].Draw('same')            
        if (b[s][plot].InheritsFrom("TH1")):
            disp.DrawLatexNDC(0.13,0.87-counts*0.03,"#color[%d]{%s: %.1E (%.1E)}" %(samples[s],tags[s],b[s][plot].GetMean(),b[s][plot].GetRMS()))
        else :
            ROOT.gPad.Update()
            b[s][plot].GetPaintedGraph().SetMinimum(0.01)
            b[s][plot].GetPaintedGraph().SetMaximum(1.02)
            ROOT.gPad.Update()
        counts = counts+1

    disp.Draw("same")
    ROOT.DrawPrelimLabel(canvas)
    ROOT.DrawLumiLabel(canvas,'200 PU')
    leg.Draw("same")
    
    ROOT.SaveCanvas(canvas, outpath + plot)

outFile.Close()



