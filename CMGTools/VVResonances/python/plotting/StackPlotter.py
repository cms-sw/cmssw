import ROOT
from TreePlotter import TreePlotter


def convertToPoisson(h):
    graph = ROOT.TGraphAsymmErrors()
    q = (1-0.6827)/2.

    for i in range(1,h.GetNbinsX()+1):
        x=h.GetXaxis().GetBinCenter(i)
        xLow =h.GetXaxis().GetBinLowEdge(i) 
        xHigh =h.GetXaxis().GetBinUpEdge(i) 
        y=h.GetBinContent(i)
        yLow=0
        yHigh=0
        if y !=0.0:
            yLow = y-ROOT.Math.chisquared_quantile_c(1-q,2*y)/2.
            yHigh = ROOT.Math.chisquared_quantile_c(q,2*(y+1))/2.-y
            graph.SetPoint(i-1,x,y)
            graph.SetPointEYlow(i-1,yLow)
            graph.SetPointEYhigh(i-1,yHigh)
            graph.SetPointEXlow(i-1,0.0)
            graph.SetPointEXhigh(i-1,0.0)


    graph.SetMarkerStyle(20)
    graph.SetLineWidth(2)
    graph.SetMarkerSize(1.)
    graph.SetMarkerColor(ROOT.kBlack)
    

    return graph    

class StackPlotter(object):
    def __init__(self):
        self.plotters = []
        self.types    = []
        self.labels   = []
        self.names    = []
        self.log=False

    def setLog(self,doLog):
        self.log=doLog
    def addPlotter(self,plotter,name="",label = "label",typeP = "background"):
        self.plotters.append(plotter)
        self.types.append(typeP)
        self.labels.append(label)
        self.names.append(name)

    def drawStack(self,var,cut,lumi,bins,mini,maxi,titlex = "", units = ""):
        canvas = ROOT.TCanvas("canvas","")
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)
        canvas.Range(-68.75,-7.5,856.25,42.5)
        canvas.SetFillColor(0)
        canvas.SetBorderMode(0)
        canvas.SetBorderSize(2)
        canvas.SetTickx(1)
        canvas.SetTicky(1)
        canvas.SetLeftMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetTopMargin(0.05)
        canvas.SetBottomMargin(0.15)
        canvas.SetFrameFillStyle(0)
        canvas.SetFrameBorderMode(0)
        canvas.SetFrameFillStyle(0)
        canvas.SetFrameBorderMode(0)


        canvas.cd()
        hists=[]
        stack = ROOT.THStack("stack","")
        
        signal=0
        background=0
        
        data=None
        dataG=None
        for (plotter,typeP,label,name) in zip(self.plotters,self.types,self.labels,self.names):
            if typeP == "signal" or typeP =="background":
                hist = plotter.drawTH1(var,cut,lumi,bins,mini,maxi,titlex,units)
                hist.SetName(name)
                stack.Add(hist)
                hists.append(hist)
                print label+" : %f\n" % hist.Integral()
 
                if typeP == "signal" :
                    signal+=hist.Integral()
                if typeP == "background" :
                    background+=hist.Integral()

       
            if typeP =="data":
                hist = plotter.drawTH1(var,cut,"1",bins,mini,maxi,titlex,units)
                hist.SetName(hist.GetName()+label)
                hists.append(hist)
                data=hist
                dataG=convertToPoisson(hist)
                dataG.SetLineWidth(1)
                print label+" : %f\n" % hist.Integral()
                
       
        #if data not found plot stack only

        if data != None:                  
            datamax = ROOT.Math.chisquared_quantile_c((1-0.6827)/2.,2*(data.GetMaximum()+1))/2.

        else: 
            datamax = stack.GetMaximum()

        if not self.log:
            frame = canvas.DrawFrame(mini,0.0,maxi,max(stack.GetMaximum(),datamax)*1.20)
        else:    
            frame = canvas.DrawFrame(mini,0.1,maxi,max(stack.GetMaximum(),datamax)*100)

        frame.GetXaxis().SetLabelFont(42)
        frame.GetXaxis().SetLabelOffset(0.007)
        frame.GetXaxis().SetLabelSize(0.045)
        frame.GetXaxis().SetTitleSize(0.05)
        frame.GetXaxis().SetTitleOffset(1.15)
        frame.GetXaxis().SetTitleFont(42)
        frame.GetYaxis().SetLabelFont(42)
        frame.GetYaxis().SetLabelOffset(0.007)
        frame.GetYaxis().SetLabelSize(0.045)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetTitleOffset(1.4)
        frame.GetYaxis().SetTitleFont(42)
        frame.GetZaxis().SetLabelFont(42)
        frame.GetZaxis().SetLabelOffset(0.007)
        frame.GetZaxis().SetLabelSize(0.045)
        frame.GetZaxis().SetTitleSize(0.05)
        frame.GetZaxis().SetTitleFont(42)


        if len(units)>0:
            frame.GetXaxis().SetTitle(titlex + " (" +units+")")
            frame.GetYaxis().SetTitle("Events / "+str((maxi-mini)/bins)+ " "+units)
        else:    
            frame.GetXaxis().SetTitle(titlex)
            frame.GetYaxis().SetTitle("Events")

        frame.Draw()
        stack.Draw("A,HIST,SAME")
        if data !=None:
            dataG.Draw("Psame")              

        legend = ROOT.TLegend(0.62,0.6,0.92,0.90,"","brNDC")
	legend.SetBorderSize(0)
	legend.SetLineColor(1)
	legend.SetLineStyle(1)
	legend.SetLineWidth(1)
	legend.SetFillColor(0)
	legend.SetFillStyle(0)
	legend.SetTextFont(42)

        legend.SetFillColor(ROOT.kWhite)
        for (histo,label,typeP) in reversed(zip(hists,self.labels,self.types)):
            if typeP != "data" and typeP !='signal':
                legend.AddEntry(histo,label,"f")
            elif typeP == 'data':
                legend.AddEntry(histo,label,"p")

        for (histo,label,typeP) in reversed(zip(hists,self.labels,self.types)):
            if typeP == "signal":
                legend.AddEntry(histo,label,"f")


 #       ROOT.SetOwnership(legend,False)

        legend.Draw()
        if self.log:
            canvas.SetLogy()
        canvas.SetLeftMargin(canvas.GetLeftMargin()*1.15)
        canvas.Update()




        print"---------------------------"
        print "Signal = %f" %(signal)
        print "Bkg    = %f" %(background)
        if data is not None:
            print "Observed = %f"%(data.Integral())

	pt =ROOT.TPaveText(0.1577181,0.9562937,0.9580537,0.9947552,"brNDC")
	pt.SetBorderSize(0)
	pt.SetTextAlign(12)
	pt.SetFillStyle(0)
	pt.SetTextFont(42)
	pt.SetTextSize(0.03)
	text = pt.AddText(0.01,0.3,"CMS Preliminary")
#	text = pt.AddText(0.25,0.3,"#sqrt{s} = 7 TeV, L = 5.1 fb^{-1}  #sqrt{s} = 8 TeV, L = 19.7 fb^{-1}")
	text = pt.AddText(0.25,0.3,"#sqrt{s} = 13 TeV")
	pt.Draw()   
        

#        latex1 = ROOT.TLatex(frame.GetXaxis().GetXmin()*1.01,frame.GetYaxis().GetXmax()*1.01,'CMS Preliminary 2011-2012, #sqrt{s} = 7+8 TeV')
#        latex1.SetTextSize(0.037)
#        latex1.Draw()

        plot={'canvas':canvas,'stack':stack,'legend':legend,'data':data,'dataG':dataG,'latex1':pt}


        

        canvas.RedrawAxis()
        canvas.Update()


        return plot



    def drawComp(self,var,cut,bins,mini,maxi,titlex = "", units = ""):
        canvas = ROOT.TCanvas("canvas","")
        ROOT.SetOwnership(canvas,False)
        canvas.cd()
        hists=[]
        stack = ROOT.THStack("stack","")
        ROOT.SetOwnership(stack,False)

        canvas.Range(-68.75,-7.5,856.25,42.5)
        canvas.SetFillColor(0)
        canvas.SetBorderMode(0)
        canvas.SetBorderSize(2)
        canvas.SetTickx(1)
        canvas.SetTicky(1)
        canvas.SetLeftMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetTopMargin(0.05)
        canvas.SetBottomMargin(0.15)
        canvas.SetFrameFillStyle(0)
        canvas.SetFrameBorderMode(0)
        canvas.SetFrameFillStyle(0)
        canvas.SetFrameBorderMode(0)


        for (plotter,typeP,label) in zip(self.plotters,self.types,self.labels):
                hist = plotter.drawTH1(var,cut,"1",bins,mini,maxi,titlex,units)
#                hist.SetFillStyle(0)
                hist.SetName(hist.GetName()+label)
                hist.Scale(1./hist.Integral())
                stack.Add(hist)
                hists.append(hist)


        stack.Draw("HIST,NOSTACK")
        canvas.SetLeftMargin(canvas.GetLeftMargin()*1.15)

        if len(units):
            stack.GetXaxis().SetTitle(titlex + " [" +units+"]")
        else:
            stack.GetXaxis().SetTitle(titlex)
    
        stack.GetYaxis().SetTitle("a.u")
        stack.GetYaxis().SetTitleOffset(1.2)


        legend = ROOT.TLegend(0.6,0.6,0.9,0.9)
        legend.SetFillColor(ROOT.kWhite)
        for (histo,label,typeP) in zip(hists,self.labels,self.types):
                legend.AddEntry(histo,label,"lf")
        ROOT.SetOwnership(legend,False)
        legend.Draw()


	pt =ROOT.TPaveText(0.1577181,0.9562937,0.9580537,0.9947552,"brNDC")
	pt.SetBorderSize(0)
	pt.SetTextAlign(12)
	pt.SetFillStyle(0)
	pt.SetTextFont(42)
	pt.SetTextSize(0.03)
	text = pt.AddText(0.01,0.5,"CMS simulation")
	pt.Draw()   


        canvas.Update()

        return canvas

        
        

