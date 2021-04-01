import ROOT
ROOT.gROOT.SetBatch(True)



def generate(tfile,name,outname,bins,mini,maxi,zeroSup=False,constant=-1,absol=False,factor=1,rebin=False):
    xaxis=ROOT.TAxis(512*factor,0,512*factor)
    axis=ROOT.TAxis(bins,mini,maxi)
    h=tfile.Get(name)
    if rebin:
        h.Rebin()
    histo = h.ProfileX().ProjectionX()
    for i in range(1,h.GetNbinsX()+1):
        proj = h.ProjectionY("q",i,i)
        mean=proj.GetMean()
        rms = proj.GetRMS()
        if proj.Integral()>15:
            histo.SetBinContent(i,mean)
            histo.SetBinError(i,rms)
        else:    
            histo.SetBinContent(i,0.0)
            histo.SetBinError(i,0.0)

    lastx=10000000000;        
    if zeroSup:
        g=ROOT.TGraphErrors()
        N=0
        for i in range(1,histo.GetNbinsX()+1):
            content = histo.GetBinContent(i)
            x = histo.GetXaxis().GetBinLowEdge(i)
            if content==0.0:
                continue;
            g.SetPoint(N,x,content)
            g.SetPointError(N,0.0,histo.GetBinError(i))
            lastx=x
            N=N+1
    else:
        g = ROOT.TGraphErrors(histo)
        x1=ROOT.Double(0.0)
        y1=ROOT.Double(0.0)
        g.GetPoint(g.GetN()-1,x1,y1)
        lastx=x1
    newH = ROOT.TH1D(outname,outname,512*factor,0,512*factor)    
    for i in range(1,xaxis.GetNbins()+1):
        x = xaxis.GetBinLowEdge(i)
        if x>lastx:
            x=lastx
        if constant>0 and x>constant:
            content=g.Eval(constant,0,"")
        else: 
            content=g.Eval(x,0,"")
        if absol:
            content=abs(content)
        intCont = axis.GetBinLowEdge(axis.FindBin(content))    
        newH.SetBinContent(i,round(content,5))
    c1 = ROOT.TCanvas("temp", "", 600, 600)
    newH.Draw("p")
    histo.Draw("same")
    newH.GetYaxis().SetRangeUser(min(newH.GetMinimum(), histo.GetMinimum())-.1, max(newH.GetMaximum(), histo.GetMaximum())+.1)
    c1.SaveAs(outname+".png")
    return newH,h.ProfileX().ProjectionX(outname+'_orig')
    

def printLUT(h,f,name,N):
    typ = 'updateLUT'+name[-1]+'_t'
    arr=[]
    for i in range(1,h.GetNbinsX()+1):
        arr.append("{}".format(h.GetBinContent(i)))
    st = "const "+typ+" "+name+"["+str(N)+"]={"+','.join(arr)+"};\n"
    f.write(st)

def printLUT2(h,f,name,N):
    typ = 'update2LUT'+name[-4]+"_t"
    arr=[]
    if name[-2]=="L":
        st = "const "+typ+" "+name+"={}".format(h.GetBinContent(1))+";\n"
        f.write(st)
    else:
        for i in range(1,h.GetNbinsX()+1):
            arr.append("{}".format(h.GetBinContent(i)))
        st = "const "+typ+" "+name+"["+str(N)+"]={"+','.join(arr)+"};\n"
        f.write(st)

def printLUTV(h,f,name,N):
    typ = 'updateLUTV'+name[-1]+'_t'
    arr=[]
    for i in range(1,h.GetNbinsX()+1):
        arr.append("{}".format(h.GetBinContent(i)))
    st = "const "+typ+" "+name+"["+str(N)+"]={"+','.join(arr)+"};\n"
    f.write(st)



fileio=open("gainLUTs.h","w")
fileio.write('#include common.h\n')

f=ROOT.TFile("gains.root")

fout = ROOT.TFile("gainLUTs.root","RECREATE")
fout.cd()


def parse(fileio,f,fout,ele,bins,mini,maxi,zeroSup=False,constant=-1,absol=False,factor=1,rebin=False):
    fout.cd()
    stele=str(ele)
    h,hO=generate(f,"gain_8_3_"+stele,"G_8_3_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_1000_3_"+stele,1024)
    h.Write()
    hO.Write()

    h,hO=generate(f,"gain_8_2_"+stele,"G_8_2_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_1000_2_"+stele,1024)
    h.Write()
    hO.Write()


    h,hO=generate(f,"gain_12_2_"+stele,"G_12_2_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_1100_2_"+stele,1024)
    h.Write()
    hO.Write()

    h,hO=generate(f,"gain_12_1_"+stele,"G_12_1_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_1100_1_"+stele,1024)
    h.Write()
    hO.Write()

    h,hO=generate(f,"gain_4_2_"+stele,"G_4_2_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_0100_2_"+stele,1024)
    h.Write()
    hO.Write()

    h,hO=generate(f,"gain_10_1_"+stele,"G_10_1_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_1010_1_"+stele,1024)
    h.Write()
    hO.Write()

    h,hO=generate(f,"gain_6_1_"+stele,"G_6_1_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_0110_1_"+stele,1024)
    h.Write()
    hO.Write()

    h,hO=generate(f,"gain_14_1_"+stele,"G_14_1_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUT(h,fileio,"gain_1110_1_"+stele,1024)
    h.Write()
    hO.Write()



def parse2(fileio,f,fout,ele,bins,mini,maxi,zeroSup=False,constant=-1,absol=False,factor=1):
    fout.cd()
    stele=str(ele)
    for q1 in ['H', 'L']:
        for q2 in ['H', 'L']:
            stele = str(ele)
            stele = stele+"_"+q1+q2
            h,hO=generate(f,"gain2_8_3_"+stele,"G2_8_3_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor)
            printLUT2(h,fileio,"gain2_1000_3_"+stele,512)
            h.Write()
            hO.Write()

            h,hO=generate(f,"gain2_8_2_"+stele,"G2_8_2_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor)
            printLUT2(h,fileio,"gain2_1000_2_"+stele,512)
            h.Write()
            hO.Write()

            h,hO=generate(f,"gain2_8_1_"+stele,"G2_8_1_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor)
            printLUT2(h,fileio,"gain2_1000_1_"+stele,512)
            h.Write()
            hO.Write()

            h,hO=generate(f,"gain2_4_2_"+stele,"G2_4_2_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor)
            printLUT2(h,fileio,"gain2_0100_2_"+stele,512)
            h.Write()
            hO.Write()

            h,hO=generate(f,"gain2_4_1_"+stele,"G2_4_1_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor)
            printLUT2(h,fileio,"gain2_0100_1_"+stele,512)
            h.Write()
            hO.Write()

            h,hO=generate(f,"gain2_2_1_"+stele,"G2_2_1_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor)
            printLUT2(h,fileio,"gain2_0010_1_"+stele,512)
            h.Write()
            hO.Write()





def parseV(fileio,f,fout,ele,bins,mini,maxi,zeroSup=False,constant=-1,absol=False,factor=2,rebin=False):
    fout.cd()
    stele=str(ele)
    h,hO=generate(f,"gain_15_0_"+stele,"G_15_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_1111_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_14_0_"+stele,"G_14_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_1110_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_13_0_"+stele,"G_13_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_1101_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_12_0_"+stele,"G_12_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_1100_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_11_0_"+stele,"G_11_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_1011_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_10_0_"+stele,"G_10_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_1010_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_9_0_"+stele,"G_9_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_1001_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_7_0_"+stele,"G_7_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_0111_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_6_0_"+stele,"G_6_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_0110_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_5_0_"+stele,"G_5_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_0101_0_"+stele,1024)
    h.Write()
    hO.Write()
    h,hO=generate(f,"gain_3_0_"+stele,"G_3_0_"+stele,bins,mini,maxi,zeroSup,constant,absol,factor,rebin)
    printLUTV(h,fileio,"gain_0011_0_"+stele,1024)
    h.Write()
    hO.Write()



parse(fileio,f,fout,0,512*2,0,64*2,True,-1,False,2,False)    
parse(fileio,f,fout,4,512,0,16,True,-1,True,2,False)    

parse2(fileio,f,fout,0,512,0,64,True,-1,False,1)    
parse2(fileio,f,fout,1,512,-8,8,True,-1,True,1)    
parse2(fileio,f,fout,4,512,0,16,True,-1,True,1)    
parse2(fileio,f,fout,5,512,0,1,True,-1,False,1)    

parseV(fileio,f,fout,0,512,0,4,True,-1,True,2,True)    
parseV(fileio,f,fout,1,512,0,4,True,-1,True,2,True)    

fout.Close()
f.Close()
fileio.close()
