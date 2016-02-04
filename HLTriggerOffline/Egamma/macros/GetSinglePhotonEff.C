{
#include <strstream.h>
#include <iomanip.h>

TH1F::SetDefaultSumw2();

/* Parameters to be modified based on cuts! */
Double_t MCEtCut = 0.;                // Cut to eliminate unmatched electrons in plots of generated variables
Double_t EtCut = 30.;                 // Et Cut
Double_t IEcalCut = 1.5;              // Ecal Isolation (Ecal Cluster Et in dR < 0.3 not part of photon SC)
Double_t IHcalBarrelCut = 6.;         // Hcal Isolation in barrel (Hcal Et in dR < 0.3)
Double_t IHcalEndcapCut = 4.;         // "    "         "  endcap
Double_t ItrackCut = 1.;              // Track Isolation (Number of tracks in dR < 0.3)

/* Strings for various cuts formed as TCuts */
TString MCEtCutString = "PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEt > ";
MCEtCutString += MCEtCut;
TString l1MatchCutString = "PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.l1Match";
TString EtCutString = "PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > ";
EtCutString +=  EtCut;
TString IEcalCutString = "PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IEcal < ";
IEcalCutString += IEcalCut;
TString IHcalCutString = "(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IHcal < ";
IHcalCutString += IHcalBarrelCut; 
IHcalCutString += " && fabs(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.eta) < 1.5) || (PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.IHcal < ";
IHcalCutString +=  IHcalEndcapCut;
IHcalCutString += " && fabs(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.eta) > 1.5 && fabs(PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.eta) < 2.5)";
TString ItrackCutString = "PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Itrack < ";
ItrackCutString += ItrackCut;

TString l1MatchCutPathString = "(" + l1MatchCutString + ")";
TString EtCutPathString = "(" + l1MatchCutPathString + ")&&(" + EtCutString + ")";
TString IEcalCutPathString = "(" + EtCutPathString + ")&&(" + IEcalCutString + ")";
TString IHcalCutPathString = "(" + IEcalCutPathString + ")&&(" + IHcalCutString + ")";
TString ItrackCutPathString = "(" + IHcalCutPathString + ")&&(" + ItrackCutString + ")";

TString l1MatchCutPathMCString = "(" + l1MatchCutString + ")&&(" + MCEtCutString + ")";
TString EtCutPathMCString = "(" + EtCutPathString + ")&&(" + MCEtCutString + ")";
TString IEcalCutPathMCString = "(" + IEcalCutPathString + ")&&(" + MCEtCutString + ")";
TString IHcalCutPathMCString = "(" + IHcalCutPathString + ")&&(" + MCEtCutString + ")";
TString ItrackCutPathMCString = "(" + ItrackCutPathString + ")&&(" + MCEtCutString + ")";

TFile *file = new TFile("../test/ZEE-HLTEgamma.root");
TTree *allEvents = Events->CloneTree();

TH1F *l1NumEt = new TH1F("l1NumEt", "Efficiency vs. Et", 100, 0, 150);
allEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEt>>l1NumEt",MCEtCutString);
TH1F *l1NumEta = new TH1F("l1NumEta", "Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
allEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEta>>l1NumEta",MCEtCutString);
TH1F *l1NumPhi = new TH1F("l1NumPhi", "Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
allEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcPhi>>l1NumPhi",MCEtCutString);
Long64_t total = allEvents->GetEntries();
Long64_t pass0 = allEvents->GetEntries("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.Et > -999.");

//TFile *file1 = new TFile("../test/ZEE-HLTEgamma-l1Match.root");
TTree *l1MatchEvents = allEvents->CopyTree(l1MatchCutPathString);

TH1F *l1MatchNumEt = new TH1F("l1MatchNumEt", "L1 Match Efficiency vs. Et", 100, 0, 150);
l1MatchEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEt>>l1MatchNumEt",l1MatchCutPathMCString);
TH1F *l1MatchHistEt = new TH1F("l1MatchHistEt", "L1 Match Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
l1MatchHistEt->Divide(l1MatchNumEt, l1NumEt, 1, 1, "B");

TH1F *l1MatchNumEta = new TH1F("l1MatchNumEta", "L1 Match Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
l1MatchEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEta>>l1MatchNumEta",l1MatchCutPathMCString);
TH1F *l1MatchHistEta = new TH1F("l1MatchHistEta", "L1 Match Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
l1MatchHistEta->Divide(l1MatchNumEta, l1NumEta, 1, 1, "B");

TH1F *l1MatchNumPhi = new TH1F("l1MatchNumPhi", "L1 Match Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
l1MatchEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcPhi>>l1MatchNumPhi",l1MatchCutPathMCString);
TH1F *l1MatchHistPhi = new TH1F("l1MatchHistPhi", "L1 Match Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
l1MatchHistPhi->Divide(l1MatchNumPhi, l1NumPhi, 1, 1, "B");
Long64_t pass1 = l1MatchEvents->GetEntries();

//TFile *file2 = new TFile("../test/ZEE-HLTEgamma-Et.root");
TTree *EtEvents = allEvents->CopyTree(EtCutPathString);

TH1F *EtNumEt = new TH1F("EtNumEt", "Et Cut Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
EtEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEt>>EtNumEt",EtCutPathMCString);
TH1F *EtHistEt = new TH1F("EtHistEt", "Et Cut Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
EtHistEt->Divide(EtNumEt, l1MatchNumEt, 1, 1, "B");

TH1F *EtNumEta = new TH1F("EtNumEta", "Et Cut Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
EtEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEta>>EtNumEta",EtCutPathMCString);
TH1F *EtHistEta = new TH1F("EtHistEta", "Et Cut Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
EtHistEta->Divide(EtNumEta, l1MatchNumEta, 1, 1, "B");

TH1F *EtNumPhi = new TH1F("EtNumPhi", "Et Cut Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
EtEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcPhi>>EtNumPhi",EtCutPathMCString);
TH1F *EtHistPhi = new TH1F("EtHistPhi", "Et Cut Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
EtHistPhi->Divide(EtNumPhi, l1MatchNumPhi, 1, 1, "B");
Long64_t pass2 = EtEvents->GetEntries();

TTree *IEcalEvents = allEvents->CopyTree(IEcalCutPathString);

TH1F *IEcalNumEt = new TH1F("IEcalNumEt", "Ecal Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
IEcalEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEt>>IEcalNumEt",IEcalCutPathMCString);
TH1F *IEcalHistEt = new TH1F("IEcalHistEt", "Ecal Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
IEcalHistEt->Divide(IEcalNumEt, EtNumEt, 1, 1, "B");

TH1F *IEcalNumEta = new TH1F("IEcalNumEta", "Ecal Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
IEcalEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEta>>IEcalNumEta",IEcalCutPathMCString);
TH1F *IEcalHistEta = new TH1F("IEcalHistEta", "Ecal Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
IEcalHistEta->Divide(IEcalNumEta, EtNumEta, 1, 1, "B");

TH1F *IEcalNumPhi = new TH1F("IEcalNumPhi", "Ecal Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
IEcalEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcPhi>>IEcalNumPhi",IEcalCutPathMCString);
TH1F *IEcalHistPhi = new TH1F("IEcalHistPhi", "Ecal Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
IEcalHistPhi->Divide(IEcalNumPhi, EtNumPhi, 1, 1, "B");
Long64_t pass3 = IEcalEvents->GetEntries();

TTree *IHcalEvents = allEvents->CopyTree(IHcalCutPathString);

TH1F *IHcalNumEt = new TH1F("IHcalNumEt", "Hcal Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
IHcalEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEt>>IHcalNumEt",IHcalCutPathMCString);
TH1F *IHcalHistEt = new TH1F("IHcalHistEt", "Hcal Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
IHcalHistEt->Divide(IHcalNumEt, EtNumEt, 1, 1, "B");

TH1F *IHcalNumEta = new TH1F("IHcalNumEta", "Hcal Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
IHcalEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEta>>IHcalNumEta",IHcalCutPathMCString);
TH1F *IHcalHistEta = new TH1F("IHcalHistEta", "Hcal Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
IHcalHistEta->Divide(IHcalNumEta, EtNumEta, 1, 1, "B");

TH1F *IHcalNumPhi = new TH1F("IHcalNumPhi", "Hcal Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
IHcalEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcPhi>>IHcalNumPhi",IHcalCutPathMCString);
TH1F *IHcalHistPhi = new TH1F("IHcalHistPhi", "Hcal Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
IHcalHistPhi->Divide(IHcalNumPhi, EtNumPhi, 1, 1, "B");
Long64_t pass4 = IHcalEvents->GetEntries();

TTree *ItrackEvents = allEvents->CopyTree(ItrackCutPathString);

TH1F *ItrackNumEt = new TH1F("ItrackNumEt", "Track Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
ItrackEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEt>>ItrackNumEt",ItrackCutPathMCString);
TH1F *ItrackHistEt = new TH1F("ItrackHistEt", "Track Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
ItrackHistEt->Divide(ItrackNumEt, EtNumEt, 1, 1, "B");

TH1F *ItrackNumEta = new TH1F("ItrackNumEta", "Track Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
ItrackEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcEta>>ItrackNumEta",ItrackCutPathMCString);
TH1F *ItrackHistEta = new TH1F("ItrackHistEta", "Track Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
ItrackHistEta->Divide(ItrackNumEta, EtNumEta, 1, 1, "B");

TH1F *ItrackNumPhi = new TH1F("ItrackNumPhi", "Track Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
ItrackEvents->Draw("PhotHLTCutVarss_hltCutVars_SinglePhots_EGAMMAHLT.obj.mcPhi>>ItrackNumPhi",ItrackCutPathMCString);
TH1F *ItrackHistPhi = new TH1F("ItrackHistPhi", "Track Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
ItrackHistPhi->Divide(ItrackNumPhi, EtNumPhi, 1, 1, "B");
Long64_t pass5 = ItrackEvents->GetEntries();

Double_t eff0 = (Double_t)pass0 / (Double_t)total;
Double_t eff1 = (Double_t)pass1 / (Double_t)pass0;
Double_t eff2 = (Double_t)pass2 / (Double_t)pass1;
Double_t eff3 = (Double_t)pass3 / (Double_t)pass2;
Double_t eff4 = (Double_t)pass4 / (Double_t)pass3;
Double_t eff5 = (Double_t)pass5 / (Double_t)pass4;
Double_t eff = (Double_t)pass5 / (Double_t)pass0;

cout.setf(ios::left);
cout<<setw(25)<<"L1 Pass: "<<setw(15)<<pass0<<setw(25)<<"MC eta + L1 Efficiency: "<<eff0<<endl;
cout<<setw(25)<<"L1 Match Pass: "<<setw(15)<<pass1<<setw(25)<<"L1 Match Efficiency: "<<eff1<<endl;
cout<<setw(25)<<"Et Pass: "<<setw(15)<<pass2<<setw(25)<<"Et Efficiency: "<<setw(15)<<eff2<<endl;
cout<<setw(25)<<"IEcal Pass: "<<setw(15)<<pass3<<setw(25)<<"IEcal Efficiency: "<<setw(15)<<eff3<<endl;
cout<<setw(25)<<"IHcal Pass: "<<setw(15)<<pass4<<setw(25)<<"IHcal Efficiency: "<<setw(15)<<eff4<<endl;
cout<<setw(25)<<"Itrack Pass: "<<setw(15)<<pass5<<setw(25)<<"Itrack Efficiency: "<<setw(15)<<eff5<<endl;
cout<<"--"<<endl;
cout<<"HLT Efficiency: "<<eff<<endl;

TCanvas *myCanvas = new TCanvas("myCanvas", "Single Photon Efficiencies vs. Et", 1500, 1000);
myCanvas->Divide(3,2);
myCanvas->cd(1);
l1MatchHistEt->Draw("e");
myCanvas->cd(2);
EtHistEt->Draw("e");
myCanvas->cd(3);
IEcalHistEt->Draw("e");
myCanvas->cd(4);
IHcalHistEt->Draw("e");
myCanvas->cd(5);
ItrackHistEt->Draw("e");
myCanvas->Print("images/EffVEtSP.gif");
myCanvas->cd(1);
l1MatchHistEta->Draw("e");
myCanvas->cd(2);
EtHistEta->Draw("e");
myCanvas->cd(3);
IEcalHistEta->Draw("e");
myCanvas->cd(4);
IHcalHistEta->Draw("e");
myCanvas->cd(5);
ItrackHistEta->Draw("e");
myCanvas->Print("images/EffVEtaSP.gif");
myCanvas->cd(1);
l1MatchHistPhi->Draw("e");
myCanvas->cd(2);
EtHistPhi->Draw("e");
myCanvas->cd(3);
IEcalHistPhi->Draw("e");
myCanvas->cd(4);
IHcalHistPhi->Draw("e");
myCanvas->cd(5);
ItrackHistPhi->Draw("e");
myCanvas->Print("images/EffVPhiSP.gif");

file->Close();
gROOT->Reset();
}
