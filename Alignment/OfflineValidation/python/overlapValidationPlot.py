from __future__ import absolute_import
from __future__ import print_function

from builtins import range
import math

import ROOT

from TkAlStyle import TkAlStyle

dirNameList=["z","r","phi"]# in general directions are labeled z=0 r =1 phi =2 throughout this, I should probably think of something more elegant
detNameList = ("BPIX", "FPIX", "TIB", "TID", "TOB", "TEC")
def subdetCondition(subdet,module,overlap):
        if (subdet+1 == 1 and (module == 1 or overlap ==1))or (subdet+1== 2 and (module == 0 or overlap == 0))or ((subdet+1== 3 or subdet+1== 5) and (overlap != 2 or module == 1))or ((subdet+1== 4 or subdet+1== 6) and (overlap != 2 or module == 0)):
                return True
        else:
                return False
def hist(tree_file_name, hist_name,subdet_ids,module_directions,overlap_directions,profile_directions):
    f = ROOT.TFile(tree_file_name)
    t = f.Get("analysis/Overlaps")
    h = []
    for subdet in range(6):#Creates a 4-D list of empty histograms for permutations of subdetector, overlap direction, module direction and profile direction.
                h.append([])
                for module in range(3):
                        h[subdet].append([])
                        for overlap in range(3):
                                h[subdet][module].append([])
                                for profile in range(4):
                                        if subdetCondition(subdet,module,overlap):
                                                h[subdet][module][overlap].append(0)
                                                continue
                                        name = hist_name + "{0}_{1}_{2}".format(dirNameList[module],dirNameList[overlap],detNameList[subdet])
                                        if not ((subdet_ids[subdet]) and  (module_directions[module]) and (profile_directions[profile]) and overlap_directions[overlap]): #puts a 0 in any unwanted plots
                                                h[subdet][module][overlap].append(0)
                                                continue
                                        if (profile>0):
                                                if profile == 3: bins, xmin, xmax = 10, -math.pi, math.pi
                                                if subdet+1 == 1 and profile == 1: bins, xmin, xmax = 10, -30, -30
                                                if subdet+1 == 2 and profile == 1: bins, xmin, xmax = 40, -60, 60 
                                                if subdet+1 == 3 and profile == 1: bins, xmin, xmax = 10, -70, 70 
                                                if subdet+1 == 4 and profile == 1: bins, xmin, xmax = 40, -110, 110 
                                                if subdet+1 == 5 and profile == 1: bins, xmin, xmax = 10, -110, 110
                                                if subdet+1 == 6 and profile == 1: bins, xmin, xmax = 20, -280, 280 
                                                if subdet+1 == 1 and profile == 2: bins, xmin, xmax = 10, 0, 20 
                                                if subdet+1 == 2 and profile == 2: bins, xmin, xmax = 10, 0, 20
                                                if subdet+1 == 3 and profile == 2: bins, xmin, xmax = 10, 20, 55 
                                                if subdet+1 == 4 and profile == 2: bins, xmin, xmax = 10, 20, 55 
                                                if subdet+1 == 5 and profile == 2: bins, xmin, xmax = 10, 55, 110 
                                                if subdet+1 == 6 and profile == 2: bins, xmin, xmax = 10, 20, 110 
                                                h[subdet][module][overlap].append(ROOT.TProfile(hist_name, hist_name, bins, xmin, xmax)) 
                                        elif subdet+1==4 or subdet+1==6:
                                            h[subdet][module][overlap].append(ROOT.TH1F(name, name, 100, -5000, 5000))
                                        else:
                                            h[subdet][module][overlap].append(ROOT.TH1F(name, name, 100, -300, 300))                                            
                                        h[subdet][module][overlap][profile].SetDirectory(0)

    nentries = t.GetEntries()

    for i, entry in enumerate(t, start=1):#loops through the tree, filling in relevant histograms for each event
        if i % 10000 == 0 or i == nentries:
            print(i, "/", nentries)
        
        subdet_id = t.subdetID
        if subdet_ids[subdet_id-1]==False:
                continue

        modulePhi0 = math.atan2(t.moduleY[0], t.moduleX[0]) 
        modulePhi1 = math.atan2(t.moduleY[1], t.moduleX[1])
        phidiff = min(abs(modulePhi0-modulePhi1), abs(math.pi - abs(modulePhi0-modulePhi1)))
        moduleR0 = math.sqrt(t.moduleY[0]**2+t.moduleX[0]**2)
        moduleR1 = math.sqrt(t.moduleY[1]**2+t.moduleX[1]**2)

        #determines the direction the modules are in respect to each other for each event
        if subdet_id%2 == 1 and ((abs(t.moduleZ[0] - t.moduleZ[1]) > 1)): 
            module_direction = 0 #0 refers to Z, 1 to R and 2 to phi
        elif subdet_id%2 == 0 and (abs(moduleR0 - moduleR1)>1):
            module_direction = 1        
        elif ((moduleR0*phidiff)>1):
            module_direction = 2
        else: #prints if this method of selecting module_direction misses anything
            print(str(i)+" skipped")
            continue
        if module_directions[module_direction]==False:
                continue
        avgList=[(t.moduleZ[0]+t.moduleZ[1])/2,(moduleR0+moduleR1)/2,(modulePhi0+modulePhi1)/2]
        if overlap_directions[2]:
             overlap_direction = 2
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
             residualA = hitXA - predXA
             residualB = hitXB - predXB
             if overlapSignA < 0:
                 residualA *= -1
             if overlapSignB < 0:
                 residualB *= -1
             A = 10000*(residualA - residualB)
             h[subdet_id-1][module_direction][overlap_direction][0].Fill(A)
             for profile in range(3):
                 if profile_directions[profile+1]:
                        h[subdet_id-1][module_direction][overlap_direction][profile+1].Fill(avgList[profile],A)
                 
                        
        if subdet_id==1 and module_direction != 1 and overlap_directions[0]:
             overlap_direction = 0
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
             residualA = hitXA - predXA
             residualB = hitXB - predXB
             if overlapSignA < 0:
                residualA *= -1
             if overlapSignB < 0:
                residualB *= -1
             A = 10000*(residualA - residualB)
             h[subdet_id-1][module_direction][overlap_direction][0].Fill(A)
             for profile in range(3):
                 if profile_directions[profile+1]:
                        h[subdet_id-1][module_direction][overlap_direction][profile+1].Fill(avgList[profile],A)

        if subdet_id==2 and module_direction !=0 and overlap_directions[1]:
             overlap_direction = 1
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
             h[subdet_id-1][module_direction][overlap_direction][0].Fill(A)
             for profile in range(3):
                 if profile_directions[profile+1]:
                        h[subdet_id-1][module_direction][overlap_direction][profile+1].Fill(avgList[profile],A)
    return h

def plot(file_name,subdet_ids,module_directions,overlap_directions,profile_directions,*filesTitlesColorsStyles):

        legend=[]
        hstack=[]
        
        for subdet in range(6):#creates lists of empty THStacks and legends to be filled later
                hstack.append([])
                legend.append([])
                for module in range(3):
                        hstack[subdet].append([])
                        legend[subdet].append([])
                        for overlap in range(3):
                                hstack[subdet][module].append([])
                                legend[subdet][module].append([])
                                for profile in range(4):
                                        if subdetCondition(subdet,module,overlap):
                                                hstack[subdet][module][overlap].append(0)
                                                legend[subdet][module][overlap].append(0)
                                                continue
                                        if not ((subdet_ids[subdet]) and  (module_directions[module]) and (profile_directions[profile]) and overlap_directions[overlap]):
                                                legend[subdet][module][overlap].append(0)
                                                hstack[subdet][module][overlap].append(0)
                                                continue
                                        else:
                                                hstack[subdet][module][overlap].append(ROOT.THStack("hstack",""))
                                                legend[subdet][module][overlap].append(TkAlStyle.legend(len(filesTitlesColorsStyles), 1))
                                                legend[subdet][module][overlap][profile].SetBorderSize(0)
                                                legend[subdet][module][overlap][profile].SetFillStyle(0)    
        for files, title, color, style in filesTitlesColorsStyles:
                h = hist(files,files.replace("/",""),subdet_ids,module_directions,overlap_directions,profile_directions)
                for subdet in range(6):
                        for module in range(3):
                                for overlap in range(3):
                                        if subdetCondition(subdet,module,overlap):
                                                        continue
                                        for profile in range(4):
                                                if not((subdet_ids[subdet]) and  (module_directions[module]) and (profile_directions[profile]) and overlap_directions[overlap]):
                                                        continue
                                                g = h[subdet][module][overlap][profile]
                                                g.SetLineColor(color)
                                                g.SetLineStyle(style)
                                                hMean = g.GetMean(1)
                                                hMeanError = g.GetMeanError(1)
                                                if (profile==0):
                                                        legend[subdet][module][overlap][profile].AddEntry(g, title + ", mean = {0}#pm{1}#mum ".format(round(hMean,3),round(hMeanError,3)), "l")
                                                        g.Scale(1 / g.Integral())
                                                else:
                                                        legend[subdet][module][overlap][profile].AddEntry(g, title, "l")
                                                hstack[subdet][module][overlap][profile].Add(g)
        for subdet in range(6):
                for module in range(3):
                        for overlap in range(3):
                                if subdetCondition(subdet,module,overlap):
                                        continue
                                for profile in range(4):
                                        if not((subdet_ids[subdet]) and  (module_directions[module]) and (profile_directions[profile]) and overlap_directions[overlap]):
                                                continue
                                        currLegend = legend[subdet][module][overlap][profile]
                                        currhstack = hstack[subdet][module][overlap][profile]
                                        currhstack.SetMaximum(currhstack.GetMaximum("nostack") * 1.2)    
                                        c = ROOT.TCanvas()
                                        currhstack.Draw("hist nostack" if profile == 0 else "nostack")
                                        currLegend.Draw()
                                        xTitle = "hit_{A} - pred_{A} - (hit_{B} - pred_{B}) (#mum)"
                                        yTitle="fraction of events"
                                        save_as_file_name = file_name +  "{0}_{1}_{2}".format(dirNameList[module],dirNameList[overlap],detNameList[subdet])
                                        if profile>0:
                                                save_as_file_name = file_name +"Profiles/profile_{0}_{1}_{2}_{3}".format(dirNameList[module],dirNameList[overlap],detNameList[subdet],dirNameList[profile-1])
                                                yTitle= xTitle
                                                xTitle= dirNameList[profile-1]
                                        currhstack.GetXaxis().SetTitle(xTitle)
                                        currhstack.GetYaxis().SetTitle(yTitle)
                                        if profile==0:
                                                currhstack.GetXaxis().SetNdivisions(404)
                                                
                                        TkAlStyle.drawStandardTitle()        
                                                        
                                        for ext in "png", "eps", "root", "pdf":
                                                c.SaveAs(save_as_file_name+"." +ext)

