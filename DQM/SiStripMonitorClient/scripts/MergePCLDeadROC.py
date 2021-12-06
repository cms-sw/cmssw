#!/usr/bin/env python3

from __future__ import print_function
import ROOT
from ROOT import TBufferFile, TH1F, TProfile, TProfile2D, TH2F, TFile, TH1D, TH2D
import re
import os


def draw_line(lineList,x1,x2,y1,y2,width=1,style=1,color=1):
    from ROOT import TLine
    l=TLine(x1,y1,x2,y2)
    l.SetBit(ROOT.kCanDelete)
    l.SetLineWidth(width)
    l.SetLineStyle(style)
    l.SetLineColor(color)
    l.Draw()
    lineList.append(l)

def draw_box(boxList,xl,xr,yl,yr,opacity=1,color=1,style=1001,lstyle=1,lw=3):
    from ROOT import TBox
    b=TBox(xl,yl,xr,yr)
    b.SetBit(ROOT.kCanDelete)
    b.SetFillStyle(style)
    b.SetFillColorAlpha(color, opacity)
    b.SetLineColor(color)
    b.SetLineWidth(lw)
    b.SetLineStyle(lstyle)
    b.Draw()
    boxList.append(b)

def renderPluginBPIX(lineList,layer) :
    from ROOT import TCanvas,TLine
    nlad=[6,14,22,32]
    coordSign=[(-1,-1),(-1,1),(1,-1),(1,1)]
    for xsign,ysign in coordSign:
        xlow = xsign*0.5
        xhigh= xsign*(0.5+4)
        ylow = ysign*0.5
        yhigh= ysign*(0.5 + nlad[layer-1])
        # Outside Box
        draw_line(lineList,xlow,  xhigh,  ylow,  ylow) # bottom
        draw_line(lineList,xlow,  xhigh, yhigh, yhigh) # top
        draw_line(lineList,xlow,   xlow,  ylow, yhigh) # left
        draw_line(lineList,xhigh, xhigh,  ylow, yhigh) # right
        # Inner Horizontal lines
        for lad in range(nlad[layer-1]):
            lad+=1
            if lad != nlad[layer-1]:
                y = ysign * (lad+0.5)
                draw_line(lineList,xlow, xhigh,  y,  y)
            y = ysign * (lad);
            draw_line(lineList,xlow, xhigh,  y,  y, 1, 3);
        # Inner Vertical lines
        for mod in range(3) : 
            mod+=1
            x = xsign * (mod + 0.5);
            draw_line(lineList,x, x,  ylow,  yhigh);

        # Draw ROC0
        for mod in range(4):
            mod+=1
            for lad in range(nlad[layer-1]):
                lad+=1
                if ysign==1:
                    flipped = not(lad%2==0)
                else :
                    flipped = not(lad%2==1)
                if flipped : roc0_orientation = -1
                else : roc0_orientation = 1
                if xsign==-1 : roc0_orientation *= -1
                if ysign==-1 : roc0_orientation *= -1
                x1 = xsign * (mod+0.5)
                x2 = xsign * (mod+0.5 - 1./8);
                y1 = ysign * (lad)
                y2 = ysign * (lad + roc0_orientation*1./2)
                if layer == 1 and xsign == -1 :
                    x1 = xsign * (mod-0.5)
                    x2 = xsign * (mod-0.5 + 1./8)
                    y1 = ysign * (lad)
                    y2 = ysign * (lad - roc0_orientation*1./2)

                    draw_line(lineList,x1, x2, y1, y1, 1)
                    draw_line(lineList,x2, x2, y1, y2, 1)
                  
                else:
                    draw_line(lineList,x1, x2, y1, y1, 1)
                    draw_line(lineList,x2, x2, y1, y2, 1)

def maskBPixROC(boxList,xsign,ysign,layer,lad,mod,roc):
    if roc<8 : 
        rocShiftX=roc*1./8
        rocShiftY=0
    else : 
        rocShiftX=(15-roc)*1./8
        rocShiftY=1./2
    if ysign==1:
        flipped = not(lad%2==0)
    else :
        flipped = not(lad%2==1)
    if flipped : roc0_orientation = -1
    else : roc0_orientation = 1
    if xsign==-1 : roc0_orientation *= -1
    if ysign==-1 : roc0_orientation *= -1
    x1 = xsign * (mod+0.5-rocShiftX)
    x2 = xsign * (mod+0.5 - 1./8-rocShiftX);
    y1 = ysign * (lad-roc0_orientation*rocShiftY)
    y2 = ysign * (lad + roc0_orientation*1./2-roc0_orientation*rocShiftY)
    if layer == 1 and xsign == -1 :
        x1 = xsign * (mod-0.5)-rocShiftX
        x2 = xsign * (mod-0.5 + 1./8)-rocShiftX
        y1 = ysign * (lad +rocShiftY)
        y2 = ysign * (lad - roc0_orientation*1./2+rocShiftY)
    draw_box(boxList,min(x1,x2),max(x1,x2),min(y1, y2),max(y1,y2),0.75)


                  
def renderPluginFPIX(lineList,ring) :
    from ROOT import TCanvas,TLine
    coordSign=[(-1,-1),(-1,1),(1,-1),(1,1)]
    for dsk in range(3) :
        dsk+=1
        for xsign,ysign in coordSign:            
            for bld in range(5+ring*6):
                bld+=1
                # Panel 2 has dashed mid-plane
                x1      = xsign * (0.5 + dsk - 1)
                x2      = xsign * (0.5 + dsk)
                sign = ysign
                y1      = ysign * (bld + sign*0.5)
                y2      = ysign * (bld)
                yp2_mid = ysign * (bld - sign*0.25)
                y3      = ysign * (bld - sign*0.5)
                draw_line(lineList,x1, x2, y1, y1)
                draw_line(lineList,x1, x2, y2, y2)
                draw_line(lineList,x1, x2, yp2_mid, yp2_mid,1,2)
                draw_line(lineList,x1, x2, y3, y3)
                # Vertical lines
                x = xsign * (0.5 + dsk - 1)
                draw_line(lineList,x,  x,  y1,  y2)
                draw_line(lineList,x,  x,  y2,  y3)
                if ring==2 :
                    x = xsign * (0.5 + dsk)
                    draw_line(lineList,x,  x,  y1,  y2)
                    draw_line(lineList,x,  x,  y2,  y3)
                #Make a BOX around ROC 0
                x1 = xsign * (0.5 + dsk - 1/8.)
                x2 = xsign * (0.5 + dsk)
                y1_p1 = ysign * (bld + sign*0.25)
                y2_p1 = ysign * (bld + sign*0.25 + xsign*ysign*0.25)
                draw_line(lineList,x1, x2, y1_p1, y1_p1, 1)
                draw_line(lineList,x1, x1, y1_p1, y2_p1, 1)
                y1_p2 = ysign * (bld - sign*0.25)
                y2_p2 = ysign * (bld - sign*0.25 - xsign*ysign*0.25)
                draw_line(lineList,x1, x2, y1_p2, y1_p2)
                draw_line(lineList,x1, x1, y1_p2, y2_p2)

def maskFPixROC(boxList,xsign,ysign,dsk,bld,pnl,roc) :
    from ROOT import TCanvas,TLine       
    if roc<8 : 
        rocShiftX=roc*1./8
        rocShiftY=0
    else : 
        rocShiftX=(15-roc)*1./8
        rocShiftY=1./4
    sign=ysign
    x1 = xsign * (0.5 + dsk - 1/8.-rocShiftX)
    x2 = xsign * (0.5 + dsk-rocShiftX)
    if pnl==1:
        y1 = ysign * (bld + sign*0.25)-xsign*rocShiftY
        y2 = ysign * (bld + sign*0.25 + xsign*ysign*0.25)-xsign*rocShiftY
    else:
        y1 = ysign * (bld - sign*0.25)+xsign*rocShiftY
        y2 = ysign * (bld - sign*0.25 - xsign*ysign*0.25)+xsign*rocShiftY
    draw_box(boxList,min(x1,x2),max(x1,x2),min(y1,y2),max(y1,y2),0.75)


def dqm_get_dataset(server, match, run, type="offline_data"):
    datareq = urllib2.Request(('%s/data/json/samples?match=%s') % (server, match))
    datareq.add_header('User-agent', ident)
    # Get data                                                                                                                              
    data = eval(re.sub(r"\bnan\b", "0", urllib2.build_opener(X509CertOpen()).open(datareq).read()),
                       { "__builtins__": None }, {})
    ret = ""
    for l in data['samples']:
        if l['type'] == type:
            for x in l['items']:
                if int(x['run']) == int(run):
                    ret=x['dataset']
                    break
    print(ret)
    return ret




def main():
    import sys
    import os
    import ROOT

    if len(sys.argv) != 3:
        print("input files needed!")
        return
    else:
        filename=sys.argv[1]
        pclfile=sys.argv[2]
    print(filename+" -- "+filename[19:25])
    print(pclfile+" -- "+pclfile[19:25])
    runNum=filename[19:25]

    dir="DQMData/Run " + runNum + "/PixelPhase1/Run summary/Pahse1_MechanicalView/"
    dirERROR="DQMData/Run " + runNum + "/PixelPhase1/Run summary/SiPixelQualityPCL/BadROC_PCL/"


    dirBPix=dir + "PXBarrel/"
    dirFPix=dir + "PXForward/"

    hoccB="digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_"
    hoccF="digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_"
    hdeadB="Dead Channels per ROC_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_"
    hdeadF="Dead Channels per ROC_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_"
        
    ROOT.gROOT.SetBatch(1)    
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(52) #104 kTemperatureMap // 55 kRainBow // 97 kRust // 57 kBird
    color=[]
    for i in range(0,255):
        color.append(ROOT.TColor.GetColorPalette(i))
    ROOT.gStyle.SetPalette(1) #104 kTemperatureMap // 55 kRainBow
    ROOT.gStyle.SetNumberContours(128)
    rootf=ROOT.TFile(filename)
    rootp=ROOT.TFile(pclfile)
    


    c=ROOT.TCanvas("c","c",1250,1000)
    #BPIX
    print("----> Build maps for BPix")
    histOccList=[]
    histDeadList=[]
    for lyr in range(1,5):
        histOccList.append(rootf.FindObjectAny(hoccB+str(lyr)))
        histDeadList.append(rootp.Get(dirERROR+hdeadB+str(lyr)))
    for hist1, hist2 in zip(histOccList, histDeadList):
        if hist1 != None or hist2 !=None:
            hist1.Draw("colz")
            match=re.search('(?<=PXLayer_)[0-9]',hist1.GetName())
            if match != None and "per_SignedModuleCoord_per_SignedLadderCoord" in hist1.GetName():
                lyr=int(match.group(0))
                hist1.SetTitle("Digi Occupancy Layer {0}".format(lyr))
                boxList=[]
                lineList=[]
                renderPluginBPIX(lineList,lyr)
                lineWd=3
                if lyr==4 :
                    lineWd=2
                if lyr==1: 
                    tbmRoc=4
                else:
                    tbmRoc=8
                binTBM=[]
                singleROC=0
                maxx=hist2.GetMaximum()
                for biny in range(1,hist2.GetNbinsY()+1):
                    if len(binTBM)!=0:
                        x1=hist2.GetXaxis().GetBinLowEdge(binTBM[0])
                        x2=hist2.GetXaxis().GetBinUpEdge(binTBM[len(binTBM)-1])
                        y1=hist2.GetYaxis().GetBinLowEdge(biny-1)
                        y2=hist2.GetYaxis().GetBinUpEdge(biny-1)
                        draw_box(boxList,x1,x2,y1,y2,0.2,color[100+int((224-100)*(1-hist2.GetBinContent(binTBM[0],biny-1)/maxx))],0,1,lineWd)
                    binTBM=[]
                    singleROC=0
                    for binx in range(1,hist2.GetNbinsX()+1):
                        if len(binTBM)==0:
                            check=True
                        elif hist2.GetBinContent(binx,biny)==hist2.GetBinContent(binTBM[len(binTBM)-1],biny):
                            check=True
                        else:
                            check=False
                        if hist2.GetBinContent(binx,biny)!=0 and check:
                            if len(binTBM)==0:
                                binTBM.append(binx)
                            else:
                                if len(binTBM)==(tbmRoc): 
                                    x1=hist2.GetXaxis().GetBinLowEdge(binTBM[0])
                                    x2=hist2.GetXaxis().GetBinUpEdge(binTBM[len(binTBM)-1])
                                    y1=hist2.GetYaxis().GetBinLowEdge(biny)
                                    y2=hist2.GetYaxis().GetBinUpEdge(biny)
                                    draw_box(boxList,x1,x2,y1,y2,0.2,color[100+int((224-100)*(1-hist2.GetBinContent(binTBM[0],biny)/maxx))],0,1,lineWd)
                                    binTBM=[]
                                    singleROC=0
                                    binTBM.append(binx)
                                else:
                                    binTBM.append(binx)
                        else:
                            if len(binTBM)!=0:
                                x1=hist2.GetXaxis().GetBinLowEdge(binTBM[0])
                                x2=hist2.GetXaxis().GetBinUpEdge(binTBM[len(binTBM)-1])
                                y1=hist2.GetYaxis().GetBinLowEdge(biny)
                                y2=hist2.GetYaxis().GetBinUpEdge(biny)
                                draw_box(boxList,x1,x2,y1,y2,0.2,color[100+int((224-100)*(1-hist2.GetBinContent(binTBM[0],biny)/maxx))],0,1,lineWd)
                                binTBM=[]
                                if hist2.GetBinContent(binx,biny)!=0:
                                    binTBM.append(binx)
                c.SaveAs('MergedPCLDeadROC_BPix_Layer{0}_TBM.pdf'.format(lyr))
                os.system('gs -dBATCH -dNOPAUSE -sDEVICE=png16m -dUseCropBox -sOutputFile=MergedPCLDeadROC_BPix_Layer{0}_TBM.png -r144 -q MergedPCLDeadROC_BPix_Layer{0}_TBM.pdf'.format(lyr))
                os.system('rm -f MergedPCLDeadROC_BPix_Layer{0}_TBM.pdf'.format(lyr))
        else :
            print("Some Error in get the histograms for FPIX")
    #FPIX
    print("----> Build maps for FPix")
    for rng in range(1,3):
        histOccList.append(rootf.FindObjectAny(hoccF+str(rng)))
        histDeadList.append(rootp.Get(dirERROR+hdeadF+str(rng)))
    for hist1, hist2 in zip(histOccList, histDeadList):
        if hist1 != None or hist2 !=None:
            hist1.Draw("colz")
            match=re.search('(?<=PXRing_)[0-9]',hist1.GetName())
            if match != None and "per_SignedDiskCoord_per_SignedBladePanelCoord" in hist1.GetName():
                ring=int(match.group(0))
                hist1.SetTitle("Digi Occupancy Ring {0}".format(ring))
                boxList=[]
                lineList=[]
                renderPluginFPIX(lineList,ring)
                lineWd=3
                if ring==2 :
                    lineWd=2
                tbmRoc=8
                binTBM=[]
                maxx=hist2.GetMaximum()
                for biny in range(1,hist2.GetNbinsY()+1):
                    if len(binTBM)!=0:
                        x1=hist2.GetXaxis().GetBinLowEdge(binTBM[0])
                        x2=hist2.GetXaxis().GetBinUpEdge(binTBM[len(binTBM)-1])
                        y1=hist2.GetYaxis().GetBinLowEdge(biny-1)
                        y2=hist2.GetYaxis().GetBinUpEdge(biny-1)
                        draw_box(boxList,x1,x2,y1,y2,0.2,color[100+int((224-100)*(1-hist2.GetBinContent(binTBM[0],biny-1)/maxx))],0,1,lineWd)
                    binTBM=[]
                    for binx in range(1,hist2.GetNbinsX()+1):
                        if len(binTBM)==0:
                            check=True
                        elif hist2.GetBinContent(binx,biny)==hist2.GetBinContent(binTBM[len(binTBM)-1],biny):
                            check=True
                        else:
                            check=False
                        if hist2.GetBinContent(binx,biny)!=0 and check:
                            if len(binTBM)==0:
                                binTBM.append(binx)
                            else:
                                if len(binTBM)==tbmRoc: 
                                    x1=hist2.GetXaxis().GetBinLowEdge(binTBM[0])
                                    x2=hist2.GetXaxis().GetBinUpEdge(binTBM[len(binTBM)-1])
                                    y1=hist2.GetYaxis().GetBinLowEdge(biny)
                                    y2=hist2.GetYaxis().GetBinUpEdge(biny)
                                    draw_box(boxList,x1,x2,y1,y2,0.2,color[100+int((224-100)*(1-hist2.GetBinContent(binTBM[0],biny)/maxx))],0,1,lineWd)
                                    binTBM=[]
                                    binTBM.append(binx)
                                else:
                                    binTBM.append(binx)
                        else:
                            if len(binTBM)!=0:
                                x1=hist2.GetXaxis().GetBinLowEdge(binTBM[0])
                                x2=hist2.GetXaxis().GetBinUpEdge(binTBM[len(binTBM)-1])
                                y1=hist2.GetYaxis().GetBinLowEdge(biny)
                                y2=hist2.GetYaxis().GetBinUpEdge(biny)
                                draw_box(boxList,x1,x2,y1,y2,0.2,color[100+int((224-100)*(1-hist2.GetBinContent(binTBM[0],biny)/maxx))],0,1,lineWd)
                                binTBM=[]
                                if hist2.GetBinContent(binx,biny)!=0:
                                    binTBM.append(binx)
                c.SaveAs('MergedPCLDeadROC_FPix_Ring{0}_TBM.pdf'.format(ring))
                os.system('gs -dBATCH -dNOPAUSE -sDEVICE=png16m -dUseCropBox -sOutputFile=MergedPCLDeadROC_FPix_Ring{0}_TBM.png -r144 -q MergedPCLDeadROC_FPix_Ring{0}_TBM.pdf'.format(ring))
                os.system('rm -f MergedPCLDeadROC_FPix_Ring{0}_TBM.pdf'.format(ring))

        else :
            print("Some Error in get the histograms for FPIX")




if __name__ == '__main__':
    main()
