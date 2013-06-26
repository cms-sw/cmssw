{
  gROOT->ProcessLine(".L fit2Dprojection.cc+");

  macroPlot("hRecBestResVSMu_MassVSEta",      "0_MuScleFit.root", "2_MuScleFit.root", "massVsEta",      "JPsi", 2, 1, "JPsi_massVsEta.root");
  macroPlot("hRecBestResVSMu_MassVSPt",       "0_MuScleFit.root", "2_MuScleFit.root", "massVsPt",       "JPsi", 2, 1, "JPsi_massVsPt.root");
  macroPlot("hRecBestResVSMu_MassVSPhiMinus", "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiMinus", "JPsi", 2, 1, "JPsi_massVsPhiMinus.root");
  macroPlot("hRecBestResVSMu_MassVSPhiPlus",  "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiPlus",  "JPsi", 2, 1, "JPsi_massVsPhiPlus.root");

  macroPlot("hRecBestResVSMu_MassVSEta",      "0_MuScleFit.root", "2_MuScleFit.root", "massVsEta",      "Upsilon1S", 2, 1, "Upsilon1S_massVsEta.root");
  macroPlot("hRecBestResVSMu_MassVSPt",       "0_MuScleFit.root", "2_MuScleFit.root", "massVsPt",       "Upsilon1S", 2, 1, "Upsilon1S_massVsPt.root");
  macroPlot("hRecBestResVSMu_MassVSPhiMinus", "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiMinus", "Upsilon1S", 2, 1, "Upsilon1S_massVsPhiMinus.root");
  macroPlot("hRecBestResVSMu_MassVSPhiPlus",  "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiPlus",  "Upsilon1S", 2, 1, "Upsilon1S_massVsPhiPlus.root");

  macroPlot("hRecBestResVSMu_MassVSEta",      "0_MuScleFit.root", "2_MuScleFit.root", "massVsEta",      "Upsilon2S", 2, 1, "Upsilon2S_massVsEta.root");
  macroPlot("hRecBestResVSMu_MassVSPt",       "0_MuScleFit.root", "2_MuScleFit.root", "massVsPt",       "Upsilon2S", 2, 1, "Upsilon2S_massVsPt.root");
  macroPlot("hRecBestResVSMu_MassVSPhiMinus", "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiMinus", "Upsilon2S", 2, 1, "Upsilon2S_massVsPhiMinus.root");
  macroPlot("hRecBestResVSMu_MassVSPhiPlus",  "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiPlus",  "Upsilon2S", 2, 1, "Upsilon2S_massVsPhiPlus.root");

  macroPlot("hRecBestResVSMu_MassVSEta",      "0_MuScleFit.root", "2_MuScleFit.root", "massVsEta",      "Upsilon3S", 2, 1, "Upsilon3S_massVsEta.root");
  macroPlot("hRecBestResVSMu_MassVSPt",       "0_MuScleFit.root", "2_MuScleFit.root", "massVsPt",       "Upsilon3S", 2, 1, "Upsilon3S_massVsPt.root");
  macroPlot("hRecBestResVSMu_MassVSPhiMinus", "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiMinus", "Upsilon3S", 2, 1, "Upsilon3S_massVsPhiMinus.root");
  macroPlot("hRecBestResVSMu_MassVSPhiPlus",  "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiPlus",  "Upsilon3S", 2, 1, "Upsilon3S_massVsPhiPlus.root");

  macroPlot("hRecBestResVSMu_MassVSEta",      "0_MuScleFit.root", "2_MuScleFit.root", "massVsEta",      "Z", 2, 1, "Z_massVsEta.root");
  macroPlot("hRecBestResVSMu_MassVSPt",       "0_MuScleFit.root", "2_MuScleFit.root", "massVsPt",       "Z", 2, 1, "Z_massVsPt.root");
  macroPlot("hRecBestResVSMu_MassVSPhiMinus", "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiMinus", "Z", 2, 1, "Z_massVsPhiMinus.root");
  macroPlot("hRecBestResVSMu_MassVSPhiPlus",  "0_MuScleFit.root", "2_MuScleFit.root", "massVsPhiPlus",  "Z", 2, 1, "Z_massVsPhiPlus.root");
}
