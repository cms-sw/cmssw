#include <cmath>
#include <cstdio>
#include <TROOT.h>
#include <TSystem.h>
#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TCut.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>

#define TRIGGERING 1

//TString basePath="/afs/cern.ch/work/g/gpetrucc/ttH/TREES_250513_FR/%s/ttHLepFRAnalyzer/ttHLepFRAnalyzer_tree.root";
TString basePath = "/afs/cern.ch/user/g/gpetrucc/w/TREES_250513_FR_v4/%s/ttHLepFRAnalyzer/ttHLepFRAnalyzer_tree.root";


void fillFRSimple(TString comp="QCDMuPt15", int selection = 0, int selbin = 0) {
    gROOT->ProcessLine(".L ../../python/plotter/fakeRate.cc+");

    TString baseFileName   = Form(basePath.Data(),comp.Data());
    baseFileName.ReplaceAll("_LooseTag","");
    baseFileName.ReplaceAll("_SingleMu","");
    baseFileName.ReplaceAll("_TagMu8","");
    baseFileName.ReplaceAll("_TagMu12","");
    baseFileName.ReplaceAll("_TagMu17","");
    baseFileName.ReplaceAll("_TagMu24","");
    baseFileName.ReplaceAll("_TagMu40","");
    baseFileName.ReplaceAll("_TagMuL","");
    baseFileName.ReplaceAll("_TagMu","");
    TFile *fileIn = TFile::Open(baseFileName);
    TTree *t = (TTree *) fileIn->Get("ttHLepFRAnalyzer");
    TString mvaVar = "mva";

    TString postFix = "";
    if (selection == 1) postFix = "LooseTightDen";
    if (selection == 2) postFix = "JustIso";
    if (selection == 3) postFix = "BTight";
    if (selection == 4) postFix = "LLSS";
    if (selection == 5) postFix = "LLOS";
    if (selection == 6) postFix = "LessB";
    if (selection == 7) postFix = "MoreP";
    if (selection == 10) postFix = "CatSIP";
    if (selection == 11) postFix = "CatID";
    if (selection == 12) postFix = "SIP4";
    if (selection == 13) postFix = "BTightSIP4";
    if (selection == 14) postFix = "IsoSUS13";
    if (selection == 15) postFix = "BTightIsoSUS13";
    if (selection == 16) postFix = "IsoSUS13C";
    if (selection == 17) postFix = "SB";
    if (selection == 18) postFix = "BTightSB";
    const float mvas[6] = { 0.5, 0.3, 0.0, -0.3, -0.5, -0.7 };
    if (selection >= 20 && selection < 32) {
        postFix = Form("MVA%s%02d%s", mvas[(selection-20)/2] >= 0 ? "" : "m", abs(int(10*mvas[(selection-20)/2])), selection % 2 == 0 ? "" : "BTight");
        std::cout << "Will work at " << postFix << std::endl;
    }

    TFile *fOut = TFile::Open(selbin ? Form("frDistsSimple%s_%s.%d.root",postFix.Data(),comp.Data(),selbin) : Form("frDistsSimple%s_%s.root",postFix.Data(),comp.Data()),"RECREATE");
    
    const int npt_mu = 8, npt_el = 7, neta_mu = 2, neta_el = 3;
    const int npt_muj = 9;
    double ptbins_mu[npt_mu+1] = { 5.0, 7.0, 8.5, 10, 15, 20, 25, 35, 80 };
    double ptbins_el[npt_el+1] = {        7, 8.5, 10, 15, 20, 25, 35, 80 };
    double etabins_mu[neta_mu+1] = { 0.0, 1.5,   2.5 };
    double etabins_el[neta_el+1] = { 0.0, 0.8, 1.479, 2.5 };
    double ptbins_muj[npt_muj+1] = { 5.0, 7.0, 8.5, 13, 18, 25, 35, 45, 80 };
    const int npt2_mu = 5, npt2_el = 4;
    const int npt2_muj = 5;
    double ptbins2_mu[npt_mu+1] = { 5.0, 8.5, 15, 25, 45, 80 };
    double ptbins2_el[npt_el+1] = {        7, 10, 20, 35, 80 };
    double ptbins2_muj[npt_muj+1] = { 5.0, 8.5, 15, 25, 45, 80 };



    gROOT->ProcessLine(".x /afs/cern.ch/user/g/gpetrucc/cpp/tdrstyle.cc");
    TCanvas *c1 = new TCanvas("c1","c1");
    gStyle->SetOptStat("emr");

    TString pdir = "ttH_plots/250513/FR_QCD_Simple/"+comp+postFix+"/";
    gSystem->Exec("mkdir -p "+pdir);
    gSystem->Exec("cp /afs/cern.ch/user/g/gpetrucc/php/index.php "+pdir);
    for (int itype = 1; itype <= 13; itype += 12) {
        TCut cut0 = (itype == 1 ? "tagType == 1  && TagJet_pt > 40 && TagJet_btagCSV >= 0.898 && abs(dphi_tp) > 2.5 && Probe_pt/TagJet_pt < 1.0" 
                                : "tagType == 13 && TagLepton_sip3d > 7 && TagLepton_relIso > 0.5 && Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1 && abs(dphi_tp) > 2.5");
        if (postFix.Contains("BTight")) {
            cut0 = (itype == 1 ? "tagType == 1  && hasSecondB == 1 && SecondJet_btagCSV > 0.679 && nJet25 <= 2 && nJet25Fwd == 0 && abs(TagJet_eta-SecondJet_eta) > 1.5*(Probe_pt/50)" 
                               : "tagType == 13 && hasSecondB == 1 && SecondJet_btagCSV > 0.679 && nJet25 <= 2 && abs(TagJet_eta-SecondJet_eta) > 1.5*(Probe_pt/50)");
                               //: "tagType == 13 && hasSecondB == 1 && SecondJet_btagCSV > 0.679 && nJet25 <= 2 && pt_3(Probe_pt,Probe_phi,SecondJet_pt,SecondJet_phi,TagLepton_pt,TagLepton_phi) < 30");
        }
        TString name0 = (itype == 13 ? "FR_MuTag_" : "FR_JetTag_");
        if (selection == 4 || selection == 5) {
            if (itype == 1) continue;
            cut0 += Form("TagLepton_charge*Probe_charge %c 0", (selection == 4 ? '>' : '<'));
        }
        if (itype == 11 && (selection == 4 || selection == 5)) continue;
        if (selection == 6) {
            cut0 = (itype == 1 ? "tagType == 1  && TagJet_pt > 40      && TagJet_btagCSV >= 0.679 && abs(dphi_tp) > 2.5 && Probe_pt/TagJet_pt < 1.0" 
                               : "tagType == 13 && TagLepton_sip3d > 4 && TagLepton_relIso > 0.5 && Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1 && abs(dphi_tp) > 2.5");
        } else if (selection == 7) {
            cut0 = (itype == 1 ? "tagType == 1  && TagJet_pt > 40      && TagJet_btagCSV >= 0.898 && abs(dphi_tp) > 2.0 && Probe_pt/TagJet_pt < 1.5" 
                               : "tagType == 13 && TagLepton_sip3d > 7 && TagLepton_relIso > 0.5  && Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1.5 && abs(dphi_tp) > 2.0");
        } else if (selection == 8) {
            TCut cut0 = (itype == 1 ? "tagType == 1  && TagJet_pt > 40      && TagJet_btagCSV >= 0.898 && abs(dphi_tp) > 2.5 && Probe_pt/TagJet_pt < 1.0" 
                                    : "tagType == 13 && TagLepton_sip3d > 7 && TagLepton_relIso > 0.5  && Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1 && abs(dphi_tp) > 2.5");
        } else if (selection == 9) {
            TCut cut0 = (itype == 1 ? "tagType == 1  && TagJet_pt > 40      && TagJet_btagCSV >= 0.898 && abs(dphi_tp) > 2.5 && Probe_pt/TagJet_pt < 1.0" 
                                    : "tagType == 13 && TagLepton_sip3d > 7 && TagLepton_relIso > 0.5  && Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1 && abs(dphi_tp) > 2.5");
        }
        if (comp.Contains("_TagMu")) {
            if (itype != 13) continue;
            if      (comp.Contains("_TagMu8"))  cut0 += "Trig_Tag_Mu8";
            else if (comp.Contains("_TagMu12")) cut0 += "Trig_Tag_Mu12";
            else if (comp.Contains("_TagMu17")) cut0 += "Trig_Tag_Mu17";
            else if (comp.Contains("_TagMu24")) cut0 += "Trig_Tag_Mu24";
            else if (comp.Contains("_TagMu40")) cut0 += "Trig_Tag_Mu40";
            else if (comp.Contains("_TagMuL"))   {
                if      (comp.Contains("DoubleMu")) cut0 += "(Trig_Tag_Mu8 || Trig_Tag_Mu17)";
                else if (comp.Contains("SingleMu")) cut0 += "(Trig_Tag_Mu12 || Trig_Tag_Mu24)";
            } else if (comp.Contains("_TagMu"))   {
                if      (comp.Contains("DoubleMu")) cut0 += "(Trig_Tag_Mu8 || Trig_Tag_Mu17)";
                else if (comp.Contains("SingleMu")) cut0 += "(Trig_Tag_Mu12 || Trig_Tag_Mu24 || Trig_Tag_Mu40)";
            }
        } else if (comp.Index("SingleMu") == 0) {
            if (itype == 13) continue;
        } else if (comp.Contains("DoubleMu")) {
            if (itype != 1) cut0 += "Trig_Pair_2Mu";
        } else if (comp.Contains("MuEG")) {
           cut0 += "Trig_Pair_MuEG";
        //} else if (comp.Contains("DoubleElectron")) {
        //    cut0 += "Trig_Probe_1ElL";
        }
        for (int ipdg = 11; ipdg <= 13; ipdg += 2) {
            if(comp.Contains("DoubleElectron") && (itype != 1  || ipdg != 11)) continue;
            if(comp.Contains("MuEG")           && (itype != 13 || ipdg != 11)) continue;
            if (!comp.Contains("TagMu")) {
                if(comp.Contains("DoubleMu")       && ipdg != 13) continue;
                //if(comp.Index("SingleMu") == 0     && (itype == 1  && ipdg == 11)) continue;
            }

            TCut    cut1  = cut0  + Form("abs(Probe_pdgId) == %d",ipdg);
            TString name1 = name0 + (ipdg == 13 ? "mu_" : "el_");
            double *ptBins  = (ipdg == 11 ? ptbins_el  : ptbins_mu);
            double *etaBins = (ipdg == 11 ? etabins_el : etabins_mu);
            int    npt      = (ipdg == 11 ? npt_el : npt_mu);
            int    neta     = (ipdg == 11 ? neta_el : neta_mu);
            if (itype == 1) {
                ptBins  = (ipdg == 11 ? ptbins_el  : ptbins_muj);
                etaBins = (ipdg == 11 ? etabins_el : etabins_mu);
                npt     = (ipdg == 11 ? npt_el : npt_muj);
            }
            if (postFix.Contains("BTight")) { // need different binning
                ptBins  = (ipdg == 11 ? ptbins2_el  : (itype == 13 ? ptbins2_mu : ptbins2_muj));
                npt     = (ipdg == 11 ? npt2_el     : (itype == 13 ? npt2_mu    : npt2_muj));
            }
            if (selection == 10 || selection == 11) {
                if (ipdg == 11) continue; // and use also wider bins
                ptBins  = (ipdg == 11 ? ptbins2_el  : (itype == 13 ? ptbins2_mu : ptbins2_muj));
                npt     = (ipdg == 11 ? npt2_el     : (itype == 13 ? npt2_mu    : npt2_muj));
            }
            for (int ieta = 0; ieta < neta; ++ieta) {
                TCut    cut2  = cut1  + Form("%g <= abs(Probe_eta) && abs(Probe_eta) < %g", etaBins[ieta],etaBins[ieta+1]);
                TString name2 = name1 + Form("eta_%.1f-%.1f_",etaBins[ieta],etaBins[ieta+1]);
                for (int ipt = 0; ipt < npt; ++ipt) {
                    TCut    cut  = cut2  + (ipt != npt-1 ? Form("%g <= Probe_pt && Probe_pt < %g", ptBins[ipt],ptBins[ipt+1]) :
                                                           Form("%g <= Probe_pt", ptBins[ipt]));
                    TString name = name2 + Form("pt_%.0f-%.0f",ptBins[ipt],ptBins[ipt+1]);
                    int ibin = 100000 * itype + 10000 * ieta + 100 * ipt + ipdg;
                    printf(" %s -> %d\n",name.Data(),ibin);
                    if (!TString(cut0).Contains("Trig")) {
                        if (!comp.Contains("_SingleMu")) {
                            if (itype == 13 && ipdg == 11) cut += " Trig_Pair_MuEG";
                            if (itype == 13 && ipdg == 13) cut += " Trig_Pair_2Mu";
                        }
                        if (itype == 1) {
                            if (ipdg == 11) {
                                if (comp.Contains("DoubleElectron")) cut += "Trig_Probe_1ElT";
                                else                                 cut += "Trig_Event_Mu40";
                            } else if (ipdg == 13) {
                                bool sm = !comp.Contains("DoubleMu");
                                bool dm = !comp.Contains("SingleMu");
                                if      (ptBins[ipt] > 40) cut += (sm ? "Trig_Probe_Mu40" : "0");
                                else if (ptBins[ipt] > 24) cut += (sm ? "Trig_Probe_Mu24" : "0");
                                else if (ptBins[ipt] > 17) cut += (dm ? "Trig_Probe_Mu17" : "Trig_Event_Mu40");
                                else if (ptBins[ipt] > 12) cut += (sm ? "(Trig_Probe_Mu12 || Trig_Event_Mu40)" : "0");
                                else                       cut += (dm ? "Trig_Probe_Mu8"  : "Trig_Probe_Mu5 || Trig_Probe_RelIso1p0Mu5 || Trig_Event_Mu40");
                            }
                        }
                    }
                    if (selection == 0 && comp.Contains("QCD") && !comp.Contains("_LooseTag")) {
                        cut += " Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1 && abs(dphi_tp) > 2.5";
                    }
                    if (selbin != 0 && selbin != ibin) continue;
                    //printf(" %s: %s\n",name.Data(),(const char *)cut);

                    
                    TCut looseDen = "";
                    //TCut tightDen = (ipdg == 11 ? "Probe_innerHits == 0 && Probe_tightCharge > 1 && Probe_convVeto" : "Probe_tightCharge > 0");
                    TCut tightDen = (ipdg == 11 ? "Probe_innerHits == 0 && Probe_convVeto && Probe_tightCharge > 1" : "Probe_tightCharge > 0");
                    TCut looseNum = Form("Probe_%s >= -0.30",mvaVar.Data());
                    TCut tightNum = Form("Probe_%s >= +0.70",mvaVar.Data());

                    if (selection == -1) {
                        // Numerator, cb loose
                        looseDen = (ipdg == 11 ? "(abs(Probe_eta)<1.4442 || abs(Probe_eta)>1.5660)" : "");
                        tightDen = tightDen + looseDen + (ipdg == 13 ? "abs(Probe_eta) < 2.1" : "");
                        looseNum = (ipdg == 13 ? "Probe_relIso < 0.2" : 
                                                 "Probe_relIso03/Probe_pt < 0.20 && Probe_tightId > 0.0 && abs(Probe_dxy) < 0.04 && abs(Probe_innerHits) <= 0");
                        tightNum = (ipdg == 13 ? "Probe_relIso < 0.12 && Probe_tightId       && abs(Probe_dxy) < 0.2 && abs(Probe_dz) < 0.5" :
                                                 "Probe_relIso03/Probe_pt < 0.10 && Probe_tightId > 0.0 && abs(Probe_dxy) < 0.02 && abs(Probe_innerHits) <= 0");
                    } else if (selection == 1) {
                        looseDen = tightDen;
                    } else if (selection == 2) {
                        tightDen = (ipdg == 11 ? "Probe_innerHits == 0 && Probe_convVeto && Probe_tightCharge > 1 &&  passEgammaTightMVA(Probe_pt,Probe_eta,Probe_tightId)" : 
                                                 "Probe_tightId && Probe_tightCharge > 0 ");
                        tightDen += "Probe_sip3d < 4";
                        tightNum = "Probe_relIso < 0.12";
                        looseNum = "";
                        looseDen = "";
                    } else if (selection == 3) {
                        looseDen = tightDen;
                    } else if (selection > 5 && selection < 10) {
                        looseNum = "";
                        looseDen = "";
                    } else if (selection == 10) {
                        looseNum = tightNum;
                        looseDen = tightDen + "Probe_sip3d >  3.5";
                        tightDen = tightDen + "Probe_sip3d <= 3.5";
                    } else if (selection == 11) {
                        looseNum = tightNum;
                        looseDen = tightDen + "Probe_tightId == 0";
                        tightDen = tightDen + "Probe_tightId == 1";
                    } else if (selection == 12) {
                        looseNum = "";
                        looseDen = "";
                        tightDen = tightDen + "Probe_sip3d < 4";
                    } else if (selection == 13) {
                        looseNum = "";
                        looseDen = "";
                        tightDen = tightDen + "Probe_sip3d < 4";
                    } else if (selection == 14 || selection == 15) {
                        tightDen = (ipdg == 11 ? "Probe_innerHits == 0 && Probe_convVeto && Probe_tightCharge > 1 &&  passEgammaTightMVA(Probe_pt,Probe_eta,Probe_tightId) && abs(Probe_dxy) < 0.0100  && abs(Probe_dz) < 0.1 && (abs(Probe_eta) < 1.4442 || abs(Probe_eta) > 1.566)" : 
                                                 "Probe_tightId && Probe_tightCharge > 0  && abs(Probe_dxy) < 0.0050 && abs(Probe_dz) < 0.1");
                        tightNum = "Probe_relIso < 0.1";
                        looseNum = "";
                        looseDen = "";
                    } else if (selection == 16) {
                        tightDen = (ipdg == 11 ? "Probe_innerHits == 0 && Probe_convVeto && Probe_tightCharge > 1 &&  passEgammaTightMVA(Probe_pt,Probe_eta,Probe_tightId) && abs(Probe_dxy) < 0.0100  && abs(Probe_dz) < 0.1 && (abs(Probe_eta) < 1.4442 || abs(Probe_eta) > 1.566)" : 
                                                 "Probe_tightId && Probe_tightCharge > 0  && abs(Probe_dxy) < 0.0050 && abs(Probe_dz) < 0.1");
                        tightNum = "Probe_chargedIso/Probe_pt < 0.05";
                        looseNum = "";
                        looseDen = "";
                    } else if (selection == 17 || selection == 18) {
                        looseNum = "";
                        looseDen = "";
                        tightDen += "(Probe_mva > -0.7 && Probe_mva < 0.5 || Probe_mva > 0.7)";
                    } else if (selection >= 20 && selection < 32) {
                        if (ipdg == 11) continue;
                        looseNum = "";
                        looseDen = "";
                        tightNum = Form("Probe_mva > %.1f", mvas[(selection-20)/2]);
                    }

                    //std::cout << "looseNum: " << looseNum << std::endl;
                    //std::cout << "looseDen: " << looseDen << std::endl;
                    //std::cout << "tightNum: " << tightNum << std::endl;
                    //std::cout << "tightDen: " << tightDen << std::endl;
                    //return;

                    if (looseNum != "") {
                    // Denominator
                    t->Draw(Form("met>>%s_den(60,0,100)",name.Data()), cut + looseDen);                
                    TH1 *den = (TH1*) gROOT->FindObject(Form("%s_den",name.Data())); den->Write();
                    c1->Print(pdir + den->GetName() + ".png");

                    // Numerator, loose
                    t->Draw(Form("met>>%s_numL(60,0,100)",name.Data()), cut + looseDen + looseNum );                
                    TH1 *numL = (TH1*) gROOT->FindObject(Form("%s_numL",name.Data())); numL->Write();
                    c1->Print(pdir + numL->GetName() + ".png");
                    }

                    if (tightNum != "") {
                    // Denominator for tight
                    t->Draw(Form("met>>%s_denT(60,0,100)",name.Data()), cut + tightDen);                
                    TH1 *denT = (TH1*) gROOT->FindObject(Form("%s_denT",name.Data())); denT->Write();
                    c1->Print(pdir + denT->GetName() + ".png");

                     // Numerator, tight
                    t->Draw(Form("met>>%s_numT(60,0,100)",name.Data()), cut + tightDen + tightNum );                
                    TH1 *numT = (TH1*) gROOT->FindObject(Form("%s_numT",name.Data())); numT->Write();
                    c1->Print(pdir + numT->GetName() + ".png");

                    // fudge
                    if (looseNum == "") {
                       TH1 *num = (TH1*) numT->Clone(Form("%s_numL",name.Data())); num->Write();
                       TH1 *den = (TH1*) denT->Clone(Form("%s_den",name.Data())); den->Write();
                    }
                    }
                }
            }
        }
    }
    fOut->Write();
    fOut->Close();
}
