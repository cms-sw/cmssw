{
#include <strstream.h>
#include <iomanip.h>

TH1F::SetDefaultSumw2();

/* Parameters to be modified based on cuts! */
Double_t MCEtCut = 0.;                // Cut to eliminate unmatched electrons in plots of generated variables
Double_t EtCut = 18.;                 // Et Cut
Double_t IHcalCut = 3.;               // Hcal Isolation (Hcal Et in dR < 0.15)
Double_t pixMatchCut = 1;             // Pixel Match Cut
Double_t EoverpBarrelCut = 1.5;       // E/p Cut for electrons in barrel
Double_t EoverpEndcapCut = 2.45;      // "   "   "   "         "  endcap
Double_t ItrackCut = 0.06;            // Track Isolation (Sum of track pt's for tracks in dR < 0.15, not matching the electron's track)

TFile *file = new TFile("../test/ZEE-HLTEgamma.root");
TTree *allEvents = Events->CloneTree();

/* Strings for various cuts formed as TCuts */
TString MCEtPTCutString = "ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEt > ";
MCEtPTCutString += MCEtCut;
TString l1MatchPTCutString = "ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.l1Match";
TString EtPTCutString = "ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.Et > ";
EtPTCutString +=  EtCut;
TString IHcalPTCutString = "ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.IHcal < ";
IHcalPTCutString += IHcalCut;
TString pixMatchPTCutString = "ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.pixMatch >= ";
pixMatchPTCutString += pixMatchCut;

TString MCEtCutString = "ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.mcEt > ";
MCEtCutString += MCEtCut;
TString l1MatchCutString = "ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.l1Match";
TString EtCutString = "ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Et > ";
EtCutString +=  EtCut;
TString IHcalCutString = "ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.IHcal < ";
IHcalCutString += IHcalCut;
TString pixMatchCutString = "ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.pixMatch >= ";
pixMatchCutString += pixMatchCut;
TString EoverpCutString = "(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Eoverp < ";
EoverpCutString += EoverpBarrelCut; 
EoverpCutString += " && fabs(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.eta) < 1.5) || (ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Eoverp < ";
EoverpCutString +=  EoverpEndcapCut;
EoverpCutString += " && fabs(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.eta) > 1.5 && fabs(ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.eta) < 2.5)";
TString ItrackCutString = "ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.Itrack < ";
ItrackCutString += ItrackCut;

TString l1MatchCutPathString = "(" + l1MatchPTCutString + ")";
TString EtCutPathString = "(" + l1MatchCutPathString + ")&&(" + EtPTCutString + ")";
TString IHcalCutPathString = "(" + EtCutPathString + ")&&(" + IHcalPTCutString + ")";
TString pixMatchCutPathString = "(" + IHcalCutPathString + ")&&(" + pixMatchPTCutString + ")";
TString EoverpCutPathString = "(" + l1MatchCutString + ")&&(" + EtCutString + ")&&(" + IHcalCutString + ")&&(" + pixMatchCutString + ")&&(" + EoverpCutString + ")";
TString ItrackCutPathString = "(" + EoverpCutPathString + ")&&(" + ItrackCutString + ")"; 

TString l1MatchCutPathMCString = "(" + l1MatchCutString + ")&&(" + MCEtPTCutString + ")";
TString EtCutPathMCString = "(" + EtCutPathString + ")&&(" + MCEtPTCutString + ")";
TString IHcalCutPathMCString = "(" + IHcalCutPathString + ")&&(" + MCEtPTCutString + ")";
TString pixMatchCutPathMCString = "(" + pixMatchCutPathString + ")&&(" + MCEtPTCutString + ")";
TString EoverpCutPathMCString = "(" + EoverpCutPathString + ")&&(" + MCEtCutString + ")";
TString ItrackCutPathMCString = "(" + ItrackCutPathString + ")&&(" + MCEtCutString + ")"; 

TH1F *l1NumEt = new TH1F("l1NumEt", "Efficiency vs. Et", 100, 0, 150);
allEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEt>>l1NumEt",MCEtPTCutString);
TH1F *l1NumEta = new TH1F("l1NumEta", "Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
allEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEta>>l1NumEta",MCEtPTCutString);
TH1F *l1NumPhi = new TH1F("l1NumPhi", "Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
allEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcPhi>>l1NumPhi",MCEtPTCutString);
Long64_t total = allEvents->GetEntries();
Long64_t pass0 = allEvents->GetEntries("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.Et > -999.");

TTree *l1MatchEvents = allEvents->CopyTree(l1MatchCutPathString);

TH1F *l1MatchNumEt = new TH1F("l1MatchNumEt", "L1 Match Efficiency vs. Et", 100, 0, 150);
l1MatchEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEt>>l1MatchNumEt",l1MatchCutPathMCString);
TH1F *l1MatchHistEt = new TH1F("l1MatchHistEt", "L1 Match Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
l1MatchHistEt->Divide(l1MatchNumEt, l1NumEt, 1, 1, "B");

TH1F *l1MatchNumEta = new TH1F("l1MatchNumEta", "L1 Match Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
l1MatchEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEta>>l1MatchNumEta",l1MatchCutPathMCString);
TH1F *l1MatchHistEta = new TH1F("l1MatchHistEta", "L1 Match Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
l1MatchHistEta->Divide(l1MatchNumEta, l1NumEta, 1, 1, "B");

TH1F *l1MatchNumPhi = new TH1F("l1MatchNumPhi", "L1 Match Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
l1MatchEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcPhi>>l1MatchNumPhi",l1MatchCutPathMCString);
TH1F *l1MatchHistPhi = new TH1F("l1MatchHistPhi", "L1 Match Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
l1MatchHistPhi->Divide(l1MatchNumPhi, l1NumPhi, 1, 1, "B");
Long64_t pass1 = l1MatchEvents->GetEntries();

TTree *EtEvents = allEvents->CopyTree(EtCutPathString);

TH1F *EtNumEt = new TH1F("EtNumEt", "Et Cut Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
EtEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEt>>EtNumEt",EtCutPathMCString);
TH1F *EtHistEt = new TH1F("EtHistEt", "Et Cut Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
EtHistEt->Divide(EtNumEt, l1MatchNumEt, 1, 1, "B");

TH1F *EtNumEta = new TH1F("EtNumEta", "Et Cut Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
EtEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEta>>EtNumEta",EtCutPathMCString);
TH1F *EtHistEta = new TH1F("EtHistEta", "Et Cut Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
EtHistEta->Divide(EtNumEta, l1MatchNumEta, 1, 1, "B");

TH1F *EtNumPhi = new TH1F("EtNumPhi", "Et Cut Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
EtEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcPhi>>EtNumPhi",EtCutPathMCString);
TH1F *EtHistPhi = new TH1F("EtHistPhi", "Et Cut Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
EtHistPhi->Divide(EtNumPhi, l1MatchNumPhi, 1, 1, "B");
Long64_t pass2 = EtEvents->GetEntries();

TTree *IHcalEvents = allEvents->CopyTree(IHcalCutPathString);

TH1F *IHcalNumEt = new TH1F("IHcalNumEt", "Hcal Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
IHcalEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEt>>IHcalNumEt",IHcalCutPathMCString);
TH1F *IHcalHistEt = new TH1F("IHcalHistEt", "Hcal Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
IHcalHistEt->Divide(IHcalNumEt, EtNumEt, 1, 1, "B");

TH1F *IHcalNumEta = new TH1F("IHcalNumEta", "Hcal Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
IHcalEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEta>>IHcalNumEta",IHcalCutPathMCString);
TH1F *IHcalHistEta = new TH1F("IHcalHistEta", "Hcal Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
IHcalHistEta->Divide(IHcalNumEta, EtNumEta, 1, 1, "B");

TH1F *IHcalNumPhi = new TH1F("IHcalNumPhi", "Hcal Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
IHcalEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcPhi>>IHcalNumPhi",IHcalCutPathMCString);
TH1F *IHcalHistPhi = new TH1F("IHcalHistPhi", "Hcal Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
IHcalHistPhi->Divide(IHcalNumPhi, EtNumPhi, 1, 1, "B");
Long64_t pass3 = IHcalEvents->GetEntries();

TTree *PixMatchEvents = allEvents->CopyTree(pixMatchCutPathString);

TH1F *PixMatchNumEt = new TH1F("PixMatchNumEt", "Pixel Match Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
PixMatchEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEt>>PixMatchNumEt",pixMatchCutPathMCString);
TH1F *PixMatchHistEt = new TH1F("PixMatchHistEt", "Pixel Match Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
PixMatchHistEt->Divide(PixMatchNumEt, IHcalNumEt, 1, 1, "B");

TH1F *PixMatchNumEta = new TH1F("PixMatchNumEta", "Pixel Match Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
PixMatchEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcEta>>PixMatchNumEta",pixMatchCutPathMCString);
TH1F *PixMatchHistEta = new TH1F("PixMatchHistEta", "Pixel Match Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
PixMatchHistEta->Divide(PixMatchNumEta, IHcalNumEta, 1, 1, "B");

TH1F *PixMatchNumPhi = new TH1F("PixMatchNumPhi", "Pixel Match Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
PixMatchEvents->Draw("ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.mcPhi>>PixMatchNumPhi",pixMatchCutPathMCString);
TH1F *PixMatchHistPhi = new TH1F("PixMatchHistPhi", "Pixel Match Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
PixMatchHistPhi->Divide(PixMatchNumPhi, IHcalNumPhi, 1, 1, "B");
Long64_t pass4 = PixMatchEvents->GetEntries();

TTree *EoverpEvents = allEvents->CopyTree(EoverpCutPathString);

TH1F *EoverpNumEt = new TH1F("EoverpNumEt", "E/p Cut Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
EoverpEvents->Draw("ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.mcEt>>EoverpNumEt",EoverpCutPathMCString);
TH1F *EoverpHistEt = new TH1F("EoverpHistEt", "E/p Cut Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
EoverpHistEt->Divide(EoverpNumEt, PixMatchNumEt, 1, 1, "B");

TH1F *EoverpNumEta = new TH1F("EoverpNumEta", "E/p Cut Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
EoverpEvents->Draw("ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.mcEta>>EoverpNumEta",EoverpCutPathMCString);
TH1F *EoverpHistEta = new TH1F("EoverpHistEta", "E/p Cut Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
EoverpHistEta->Divide(EoverpNumEta, PixMatchNumEta, 1, 1, "B");

TH1F *EoverpNumPhi = new TH1F("EoverpNumPhi", "E/p Cut Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
EoverpEvents->Draw("ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.mcPhi>>EoverpNumPhi",EoverpCutPathMCString);
TH1F *EoverpHistPhi = new TH1F("EoverpHistPhi", "E/p Cut Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
EoverpHistPhi->Divide(EoverpNumPhi, PixMatchNumPhi, 1, 1, "B");
Long64_t pass5 = EoverpEvents->GetEntries();

TTree *HLTEvents = allEvents->CopyTree(ItrackCutPathString);

TH1F *ItrackNumEt = new TH1F("ItrackNumEt", "Track Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
HLTEvents->Draw("ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.mcEt>>ItrackNumEt",ItrackCutPathMCString);
TH1F *ItrackHistEt = new TH1F("ItrackHistEt", "Track Isolation Efficiency vs. Et;Et (GeV);Eff.", 100, 0, 150);
ItrackHistEt->Divide(ItrackNumEt, EoverpNumEt, 1, 1, "B");

TH1F *ItrackNumEta = new TH1F("ItrackNumEta", "Track Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
HLTEvents->Draw("ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.mcEta>>ItrackNumEta",ItrackCutPathMCString);
TH1F *ItrackHistEta = new TH1F("ItrackHistEta", "Track Isolation Efficiency vs. eta;eta;Eff.", 100, -2.5, 2.5);
ItrackHistEta->Divide(ItrackNumEta, EoverpNumEta, 1, 1, "B");

TH1F *ItrackNumPhi = new TH1F("ItrackNumPhi", "Track Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
HLTEvents->Draw("ElecHLTCutVarss_hltCutVars_RelaxedSingleElecs_EGAMMAHLT.obj.mcPhi>>ItrackNumPhi",ItrackCutPathMCString);
TH1F *ItrackHistPhi = new TH1F("ItrackHistPhi", "Track Isolation Efficiency vs. phi;phi;Eff.", 100, 0, 6.2832);
ItrackHistPhi->Divide(ItrackNumPhi, EoverpNumPhi, 1, 1, "B");
Long64_t pass6 = HLTEvents->GetEntries();

Double_t eff0 = (Double_t)pass0 / (Double_t)total;
Double_t eff1 = (Double_t)pass1 / (Double_t)pass0;
Double_t eff2 = (Double_t)pass2 / (Double_t)pass1;
Double_t eff3 = (Double_t)pass3 / (Double_t)pass2;
Double_t eff4 = (Double_t)pass4 / (Double_t)pass3;
Double_t eff5 = (Double_t)pass5 / (Double_t)pass4;
Double_t eff6 = (Double_t)pass6 / (Double_t)pass5;
Double_t eff = (Double_t)pass6 / (Double_t)pass0;
cout.setf(ios::left);
cout<<setw(25)<<"L1 Pass: "<<setw(15)<<pass0<<setw(25)<<"MC eta + L1 Efficiency: "<<eff0<<endl;
cout<<setw(25)<<"L1 Match Pass: "<<setw(15)<<pass1<<setw(25)<<"L1 Match Efficiency: "<<eff1<<endl;
cout<<setw(25)<<"Et Pass: "<<setw(15)<<pass2<<setw(25)<<"Et Efficiency: "<<setw(15)<<eff2<<endl;
cout<<setw(25)<<"IHcal Pass: "<<setw(15)<<pass3<<setw(25)<<"IHcal Efficiency: "<<setw(15)<<eff3<<endl;
cout<<setw(25)<<"Pixel Match Pass: "<<setw(15)<<pass4<<setw(25)<<"Pixel Match Efficiency: "<<setw(15)<<eff4<<endl;
cout<<setw(25)<<"E / p Pass: "<<setw(15)<<pass5<<setw(25)<<"E / p Efficiency: "<<setw(15)<<eff5<<endl;
cout<<setw(25)<<"Itrack Pass: "<<setw(15)<<pass6<<setw(25)<<"Itrack Efficiency: "<<setw(15)<<eff6<<endl;
cout<<"--"<<endl;
cout<<"HLT Efficiency: "<<eff<<endl;

TCanvas *myCanvas = new TCanvas("myCanvas", "Relaxed Single Electron Efficiencies vs. Et", 1500, 1000);
myCanvas->Divide(3,2);
myCanvas->cd(1);
l1MatchHistEt->Draw("e");
myCanvas->cd(2);
EtHistEt->Draw("e");
myCanvas->cd(3);
IHcalHistEt->Draw("e");
myCanvas->cd(4);
PixMatchHistEt->Draw("e");
myCanvas->cd(5);
EoverpHistEt->Draw("e");
myCanvas->cd(6);
ItrackHistEt->Draw("e");
myCanvas->Print("images/EffVEtRSE.gif");
myCanvas->cd(1);
l1MatchHistEta->Draw("e");
myCanvas->cd(2);
EtHistEta->Draw("e");
myCanvas->cd(3);
IHcalHistEta->Draw("e");
myCanvas->cd(4);
PixMatchHistEta->Draw("e");
myCanvas->cd(5);
EoverpHistEta->Draw("e");
myCanvas->cd(6);
ItrackHistEta->Draw("e");
myCanvas->Print("images/EffVEtaRSE.gif");
myCanvas->cd(1);
l1MatchHistPhi->Draw("e");
myCanvas->cd(2);
EtHistPhi->Draw("e");
myCanvas->cd(3);
IHcalHistPhi->Draw("e");
myCanvas->cd(4);
PixMatchHistPhi->Draw("e");
myCanvas->cd(5);
EoverpHistPhi->Draw("e");
myCanvas->cd(6);
ItrackHistPhi->Draw("e");
myCanvas->Print("images/EffVPhiRSE.gif");

file->Close();
gROOT->Reset();
}
