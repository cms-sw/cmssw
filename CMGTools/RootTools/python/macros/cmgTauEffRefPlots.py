from ROOT import *

##define the sample here and just add new lines for later productions
input = "DYToTauTau_M_20_TuneZ2_7TeV_pythia6_tauola.Summer11_PU_S3_START42_V11_v2.AODSIM.V2.PAT_CMG_V2_0_1"
####################################################################


#########generic function
def efficiency(input,type):
    file = TFile("CmgTauEffHistos_"+input+".root")
    histogen = file.Get("genTau"+type+"Histograms/pt")
    historec = file.Get("cmgTauSel"+type+"TrueHistogramsGen/pt")

    histogen.SetTitle(input)
    histogen.GetXaxis().SetTitle("Gen. Tau p_{T}     (GeV)")
    histogen.GetXaxis().SetRangeUser(0,100)
    histogen.GetYaxis().SetTitle("(1 GeV bins)")

    historec.SetTitle(input)
    historec.GetXaxis().SetTitle("Gen. Tau p_{T}     (GeV)")
    historec.GetXaxis().SetRangeUser(0,100)
    historec.GetYaxis().SetTitle("(1 GeV bins)")

    ##Compute Efficiency
    histoeff=historec.Clone("histoeff")
    histoeff.Divide(histogen)


    canv = TCanvas()
    canv.SetFillColor(0)
    canv.SetGrid(1)
    histoeff.SetTitle("")
    histoeff.SetStats(0)
    histoeff.GetYaxis().SetRangeUser(0,1)
    histoeff.GetYaxis().SetTitle("Tau Reco. Efficiency")
    if type=="Plus":
        histoeff.GetYaxis().SetTitle("TauPlus Reco. Efficiency")
    if type=="Minus":
        histoeff.GetYaxis().SetTitle("TauMinus Reco. Efficiency")
    histoeff.Draw()
    label = input.split(".",2)
    title = TText()
    title.SetTextSize(.025)
    title.DrawTextNDC(.4,.98,label[0])
    title.DrawTextNDC(.4,.96,label[1])
    title.DrawTextNDC(.4,.94,label[2])
    canv.Print("CmgTauEffHistos_"+input+type+".png")


#####Efficiency for total taus
efficiency(input=input,type="")
efficiency(input=input,type="Plus")
efficiency(input=input,type="Minus")
