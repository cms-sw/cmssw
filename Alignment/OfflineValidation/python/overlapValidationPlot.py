import math
import ROOT
from TkAlStyle import TkAlStyle
from random import randint
def hist(tree_file_name, hist_name,subdet_id,module_direction,overlap_direction,l):
    f = ROOT.TFile(tree_file_name)
    t = f.Get("analysis/Overlaps")
    if (subdet_id==4 or subdet_id==6):
	    h = ROOT.TH1F(hist_name, hist_name, 100, -5000, 5000)
    else:
    	h = ROOT.TH1F(hist_name, hist_name, 100, -300, 300)
    h.SetDirectory(0)
    for entry in t:
        if not ((t.subdetID == subdet_id)):
            continue
	if module_direction not in ("r" ,"phi", "z"): 
	    raise ValueError("Invalid module direction")
	if module_direction != "z" and subdet_id%2 == 1 and ((abs(t.moduleZ[0] - t.moduleZ[1]) < 1)):
	    continue 
        modulePhi0 = math.atan2(t.moduleY[0], t.moduleX[0]) 
        modulePhi1 = math.atan2(t.moduleY[1], t.moduleX[1])
	phidiff = min(abs(modulePhi0-modulePhi1), abs(math.pi - abs(modulePhi0-modulePhi1)))
	moduleR0 = math.sqrt(t.moduleY[0]**2+t.moduleX[0]**2)
	moduleR1 = math.sqrt(t.moduleY[1]**2+t.moduleX[1]**2)
	if module_direction !="r" and subdet_id%2 == 0 and (abs(moduleR0 - moduleR1)<1):
            continue	
	if module_direction !="phi" and ((moduleR0*phidiff)<1):
	    continue
	if overlap_direction == "phi":
             if modulePhi0 > modulePhi1:
                 hitXA = t.hitX[1]
                 hitXB = t.hitX[0]
                 predXA = t.predX[1]
                 predXB = t.predX[0]
                 overlapSignA = t.localxdotglobalphi[1]
                 overlapSignB = t.localxdotglobalphi[0]
             else:
                 hitXA = t.hitX[0]
                 hitXB = t.hitX[1]
                 predXA = t.predX[0]
                 predXB = t.predX[1]
                 overlapSignA = t.localxdotglobalphi[0]
                 overlapSignB = t.localxdotglobalphi[1]
	if overlap_direction == "z":
             if t.moduleZ[0] > t.moduleZ[1]:
                 hitXA = t.hitY[1]
                 hitXB = t.hitY[0]
                 predXA = t.predY[1]
                 predXB = t.predY[0]
                 overlapSignA = t.localydotglobalz[1]
                 overlapSignB = t.localydotglobalz[0]
             else:
                 hitXA = t.hitY[0]
                 hitXB = t.hitY[1]
                 predXA = t.predY[0]
                 predXB = t.predY[1]
                 overlapSignA = t.localydotglobalz[0]
                 overlapSignB = t.localydotglobalz[1]
	if overlap_direction == "r":
             if moduleR0 > moduleR1:
                 hitXA = t.hitY[1]
                 hitXB = t.hitY[0]
                 predXA = t.predY[1]
                 predXB = t.predY[0]
                 overlapSignA = t.localydotglobalr[1]
                 overlapSignB = t.localydotglobalr[0]
             else:
                 hitXA = t.hitY[0]
                 hitXB = t.hitY[1]
                 predXA = t.predY[0]
                 predXB = t.predY[1]
                 overlapSignA = t.localydotglobalr[0]
                 overlapSignB = t.localydotglobalr[1]

        residualA = hitXA - predXA
        residualB = hitXB - predXB
        if overlapSignA < 0:
            residualA *= -1
        if overlapSignB < 0:
            residualB *= -1
        
        A = 10000*(residualA - residualB)
        if (l==0):
		h.Fill(A)
    	elif (l==1):
		h.Fill(10000*residualA)
	elif (l==2):
		h.Fill(10000*residualB)
	elif (l==3):
		h.Fill(10000*(residualA+((-1)^randint(0,1))*residualB))
    return h

def plot(file_name,subdet_id,module_direction,overlap_direction,m, *filesTitlesColorsStyles):
    hstack = ROOT.THStack("hstack","")
    legend = TkAlStyle.legend(len(filesTitlesColorsStyles), 1)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    hs = []
    
    for files, title, color, style in filesTitlesColorsStyles:
        h = hist(files,files.replace("/",""),subdet_id,module_direction,overlap_direction,m)
        h.SetLineColor(color)
        h.SetLineStyle(style)
	hMean = h.GetMean(1)
	hMeanError = h.GetMeanError(1)
#        legend.AddEntry(h, title, "l")
        legend.AddEntry(h, title + ", mean = {0}\pm {1}\mu m ".format(round(hMean,3),round(hMeanError,3)), "l")
        hstack.Add(h)
        hs.append(h)
    hstack.SetMaximum(hstack.GetMaximum("nostack") * 1.2)    
    c = ROOT.TCanvas()
    hstack.Draw("nostack")
    legend.Draw()
    if (m==0):
	xTitle = "hit_{A} - pred_{A} - (hit_{B} - pred_{B}) (#mum)"
    elif (m==1):
	xTitle = "hit_{A} - pred_{A} (#mum)"
    elif (m==2):
	xTitle = "(hit_{B} - pred_{B}) (#mum)"
    elif (m==3):
	xTitle = "hit_{A} - pred_{A} \pm  (hit_{B} - pred_{B}) (#mum)"

    hstack.GetXaxis().SetTitle(xTitle)
    hstack.GetYaxis().SetTitle("number of events")
    hstack.GetXaxis().SetNdivisions(404)

    TkAlStyle.drawStandardTitle()
        
    save_as_file_name = file_name+str(m)

    if (m==0):
	save_as_file_name = file_name

    for ext in "png", "eps", "root", "pdf":
        c.SaveAs(save_as_file_name+"." +ext)

