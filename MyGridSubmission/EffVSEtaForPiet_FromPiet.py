from ROOT import *
import numpy as np

gROOT.ProcessLine(".L tdrstyle.C")
#setTDRStyle()
gStyle.SetOptStat(0)


def ReturnEff(tree_):
    _file0 = TFile.Open(tree_,"READ")

    ThisEff = TH1F(""      , ""   ,8, 2.0, 2.8  )

    ThisMatched=_file0.Get("MatchedME0Muon_WideBinning_Eta")
    ThisGen = _file0.Get("GenMuon_WideBinning_Eta")

    ThisMatched.Sumw2()
    ThisGen.Sumw2()

    ThisEff.Divide(ThisMatched, ThisGen, 1, 1, "B")

    ThisMatched.SetDirectory(0)
    ThisGen.SetDirectory(0)
    ThisEff.SetDirectory(0)
    _file0.Close()

    return ThisEff

def MakeEfficiencyPlot(BaseFile,TimingFile,FileName,Label,SecondLabel):
    ThisEff=ReturnEff(BaseFile)
    
    ThisEff.GetXaxis().SetTitle("Muon |#eta|")
    ThisEff.GetYaxis().SetTitle("ME0Muon Efficiency")
    
    
    #ThisEff.GetYaxis().SetTitleOffset(2.0)
    ThisEff.GetXaxis().SetTitleSize(0.04)
    ThisEff.GetYaxis().SetTitleSize(0.04)
    ThisEff.SetMarkerStyle(21)
    
    ThisEff.SetMinimum(0.1)
    ThisEff.SetMaximum(1.05)
    
    ThisEff.SetMarkerColor(kBlue)
    ThisEff.SetLineColor(kBlue)
    ThisEff.SetLineWidth(2)
    ThisEff.Draw("E1")

    gPad.SetRightMargin(0.05)

    leg= TLegend(0.2,0.2,0.5,0.4,"","brNDC")
    leg.SetTextFont(42)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetHeader("Z/#gamma* #rightarrow #mu#mu, p_{T} > 5 GeV")
    leg.AddEntry(ThisEff,"Original result, "+Label)
    leg.Draw()

    ThisEff_WithTiming=ReturnEff(TimingFile)
    ThisEff_WithTiming.SetMarkerStyle(20)
    ThisEff_WithTiming.SetMarkerColor(2)
    ThisEff_WithTiming.SetLineColor(2)
    ThisEff_WithTiming.SetLineWidth(2)
    ThisEff_WithTiming.Draw("SAMEE1")
    leg.AddEntry(ThisEff_WithTiming,"With timing, "+Label+", "+SecondLabel)
    leg.Draw(TimingFile)

    c1.Print(FileName+".png")
    c1.Print(FileName+".pdf")


def ReturnResolutions(tree_):
    _file0 = TFile.Open(tree_,"READ")

    QOverPt_Resolution_VSEta_WithTiming = TH1F(""      , ""   ,4, 2.0, 2.8  )

    ThisPtDiff_s=_file0.Get("PtDiff_s")


    for i in range(ThisPtDiff_s.GetNbinsX()):
        test= TH1D("test"   , "pt resolution"   , 200, -5.0, 5.0 )
        ThisPtDiff_s.ProjectionY("test",i+1,i+1,"")
        
        gaus = TF1("gaus","gaus",-0.3,0.3)
        test.Fit(gaus,"R")
        
        w2_2  = gaus.GetParameter(2)
        e_w2_2  = gaus.GetParError(2)
        
        QOverPt_Resolution_VSEta_WithTiming.SetBinContent(i+1, w2_2) 
        QOverPt_Resolution_VSEta_WithTiming.SetBinError(i+1, e_w2_2) 

    ThisPtDiff_s.SetDirectory(0)
    QOverPt_Resolution_VSEta_WithTiming.SetDirectory(0)
    _file0.Close()

    return QOverPt_Resolution_VSEta_WithTiming

def MakeResolutionPlot(BaseFile,TimingFile,FileName,Label,SecondLabel):

    QOverPt_Resolution_VSEta = ReturnResolutions(BaseFile)
    QOverPt_Resolution_VSEta_WithTiming = ReturnResolutions(TimingFile)

    QOverPt_Resolution_VSEta.SetMinimum(0.01)
    QOverPt_Resolution_VSEta.SetMaximum(0.1)


    QOverPt_Resolution_VSEta.SetMarkerStyle(21) 
    QOverPt_Resolution_VSEta.SetMarkerColor(kBlue) 
    QOverPt_Resolution_VSEta.SetLineColor(kBlue) 
    QOverPt_Resolution_VSEta.SetLineWidth(2)

    QOverPt_Resolution_VSEta.GetXaxis().SetTitle("Gen Muon |#eta|")
    QOverPt_Resolution_VSEta.GetYaxis().SetTitle("Resolution")
    QOverPt_Resolution_VSEta.GetYaxis().SetTitleSize(.04)
    QOverPt_Resolution_VSEta.Draw("E1")  
    

    leg= TLegend(0.2,0.6,0.5,0.8,"","brNDC")
    leg.SetTextFont(42)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetHeader("Z/#gamma* #rightarrow #mu#mu, p_{T} > 5 GeV")
    leg.AddEntry(QOverPt_Resolution_VSEta,"Original result, "+Label)
    leg.Draw()



    QOverPt_Resolution_VSEta_WithTiming.SetMarkerStyle(22) 
    QOverPt_Resolution_VSEta_WithTiming.SetMarkerColor(2) 
    QOverPt_Resolution_VSEta_WithTiming.SetLineColor(2) 
    QOverPt_Resolution_VSEta_WithTiming.SetLineWidth(2)

    QOverPt_Resolution_VSEta_WithTiming.Draw("SAMEE1")  
    leg.AddEntry(QOverPt_Resolution_VSEta_WithTiming,"With timing, "+Label+", "+SecondLabel)
    leg.Draw()
    #gROOT.ProcessLine(".L CMS_lumi.C")
    #writeExtraText=false
    #CMS_lumi(c1,113,0)

    c1.Print(FileName+".png")
    c1.Print(FileName+".pdf")



def ReturnYields(tree_):
    varbins=[0.,5.,10.,20.,40.,70.]
    _file0 = TFile.Open(tree_,"READ")

    ThisUnmatchedME0Muon_Pt=_file0.Get("UnmatchedME0Muon_Pt")
    ThisNevents_h=_file0.Get("Nevents_h")

    UnmatchedME0Muon_VariableBins_Pt_Rebinned = ThisUnmatchedME0Muon_Pt.Rebin(5,"UnmatchedME0Muon_VariableBins_Pt_Rebinned",np.array(varbins))
    
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.Sumw2()

    N=ThisNevents_h.Integral()

    UnmatchedME0Muon_VariableBins_Pt_Rebinned.Scale(1/N)

    ForBins=UnmatchedME0Muon_VariableBins_Pt_Rebinned
    for i in range(UnmatchedME0Muon_VariableBins_Pt_Rebinned.GetNbinsX()):
        UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetBinContent(i+1,(ForBins.GetBinContent(i+1)/varbins[i+1]) )
        UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetBinError(i+1,(ForBins.GetBinError(i+1)/varbins[i+1]) )

    ThisUnmatchedME0Muon_Pt.SetDirectory(0)
    ThisNevents_h.SetDirectory(0)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetDirectory(0)
    _file0.Close()

    return UnmatchedME0Muon_VariableBins_Pt_Rebinned
    

def MakeYieldPlot(BaseFile,TimingFile,FileName,Label,SecondLabel):

    UnmatchedME0Muon_VariableBins_Pt_Rebinned= ReturnYields(BaseFile)

    UnmatchedME0Muon_VariableBins_Pt_Rebinned.GetXaxis().SetTitle("Muon p_{T} (GeV)")
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.GetYaxis().SetTitle("Average ME0Muon Background Multiplicity / GeV")

    UnmatchedME0Muon_VariableBins_Pt_Rebinned.GetYaxis().SetTitleOffset(2.0)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.GetXaxis().SetTitleSize(0.04)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.GetXaxis().SetTitleOffset(1.2)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.GetYaxis().SetTitleSize(0.04)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetMarkerStyle(21)
    
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetMinimum(0.000001)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetMaximum(50.0)

    UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetLineWidth(2)


    UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetMarkerColor(kBlue)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.SetLineColor(kBlue)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned.Draw("E1")
    
    gPad.SetRightMargin(0.05)
    
    c1.SetLogy()
    leg= TLegend(0.5,0.5,0.8,0.7,"","brNDC")
    leg.SetTextFont(42)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetHeader("Z/#gamma* #rightarrow #mu#mu")
    leg.AddEntry(UnmatchedME0Muon_VariableBins_Pt_Rebinned,"Original result, "+Label)
    leg.Draw()


    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming= ReturnYields(TimingFile)
    
    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming.SetMarkerStyle(20)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming.SetMarkerColor(2)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming.SetLineColor(2)
    
    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming.SetMarkerColor(kRed)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming.SetLineColor(kRed)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming.SetLineWidth(2)
    UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming.Draw("SAMEE1")

    leg.AddEntry(UnmatchedME0Muon_VariableBins_Pt_Rebinned_WithTiming,"With timing, "+Label+", "+SecondLabel)
    leg.Draw()
    #gROOT.ProcessLine(".L CMS_lumi.C")
    
    #writeExtraText=false
    #CMS_lumi(c1,113,0)

    c1.Print(FileName+".png")
    c1.Print(FileName+".pdf")



def main():

    BaseTightSample="/afs/cern.ch/work/d/dnash/ME0Segments/ForRealSegmentsOnly/CMSSW_6_2_0_SLHC23_patch1/src/DY_HitsLoose_ForDelphes/DY_HitsLoose_ForDelphes.root"
    BaseLooseSample="/afs/cern.ch/work/d/dnash/ME0Segments/ForRealSegmentsOnly/ForDoingLooseInParallel/CMSSW_6_2_0_SLHC23_patch1/src/DY_HitsLoose_ForDelphes/DY_HitsLoose_ForDelphes.root"

    ListOfPietsSamples=['/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_Loose_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_Tight_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_Tight_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_Loose_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_Tight_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_Loose_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_Tight_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_Loose_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_Tight_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_Loose_Analysis_David.root',
                        '/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_Tight_Analysis_David.root']

    for thissample in ListOfPietsSamples:
        if "Tight" in thissample:
            Resolution=thissample.split('HGCALGS_PU140_')[1].split('_Tight')[0]
            MakeEfficiencyPlot(BaseTightSample,thissample,"EffVSEta_Tight_"+Resolution,"Tight ID",Resolution)

            MakeResolutionPlot(BaseTightSample,thissample,"PtResolution_Tight_"+Resolution,"Tight ID",Resolution)

            ###Seems there isn't a record of number of events stored for the run over the timing samples, in the histo Nevent_h
            #MakeYieldPlot(BaseTightSample,thissample,"BkgYield_Tight_"+Resolution,"Tight ID",Resolution)
        elif "Loose" in thissample:
            Resolution=thissample.split('HGCALGS_PU140_')[1].split('_Loose')[0]

            MakeEfficiencyPlot(BaseLooseSample,thissample,"EffVSEta_Loose_"+Resolution,"Loose ID",Resolution)

            MakeResolutionPlot(BaseLooseSample,thissample,"PtResolution_Loose_"+Resolution,"Loose ID",Resolution)

            ###Seems there isn't a record of number of events stored for the run over the timing samples, in the histo Nevent_h
            #MakeYieldPlot(BaseLooseSample,thissample,"BkgYield_Loose_"+Resolution,"Loose ID",Resolution)


        ###Seems there isn't a record of number of events stored for the run over the timing samples, in the histo Nevent_h
        #MakeYieldPlot("DY_HitsLoose_ForDelphes/DY_HitsLoose_ForDelphes.root","/afs/cern.ch/user/p/piet/public/ForDavid/ME0MuonAnalyzerOutput_crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_Tight_Analysis_David.root","QOverPt_Resolution_VSEta_fromPiet")


main()
