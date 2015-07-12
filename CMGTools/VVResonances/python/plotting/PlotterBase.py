import ROOT
class PlotterBase(object):

    def __init__(self):
        self.fillstyle=1001
        self.linestyle=1
        self.linecolor=1
        self.linewidth=2
        self.fillcolor=ROOT.kOrange-3
        self.markerstyle=20
        self.corrFactors=[]

    def addCorrectionFactor(self,name,value,error,model):
        corr=dict()
        corr['value']=value
        corr['error']=error
        corr['name']=name
        corr['model']=model
        self.corrFactors.append(corr)

    def setLineProperties(self,linestyle,linecolor,linewidth):
        self.linestyle=linestyle
        self.linecolor=linecolor
        self.linewidth=linewidth 

    def setFillProperties(self,fillstyle,fillcolor):
        self.fillstyle=fillstyle
        self.fillcolor=fillcolor

    def setMarkerProperties(self,markerstyle):
        self.markerstyle=markerstyle

    def drawTH2(self,var,cuts,lumi,binsx,minx,maxx,binsy,miny,maxy,titlex = "",unitsx = "",titley="",unitsy="", drawStyle = "COLZ"):
        return None

    def drawTH3(self,var,cuts,lumi,binsx,minx,maxx,binsy,miny,maxy,binsz,minz,maxz,titlex = "",unitsx = "",titley="",unitsy="", drawStyle = "COLZ"):
        return None

    def drawTH1(self,var,cuts,lumi,bins,min,max,titlex = "",units = "",drawStyle = "HIST"):
        return None

