#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <sys/stat.h>
#include <algorithm>

using std::vector;
using std::string;
using std::cout;


typedef vector<string> vstring;

void figs()
{
    gStyle->SetOptStat(0);
    gStyle->SetErrorX(0.);
        
    ofstream out("to_twiki"); // file with the code to insert figures to twiki
    
    TFile f6("DQM_V0001_R000000001__CMSSW_test__RelVal__TrigVal_710pre6.root","read");
    TFile f7("DQM_V0001_R000000001__CMSSW_test__RelVal__TrigVal_710pre7.root","read");
    TFile f8("DQM_V0001_R000000001__CMSSW_test__RelVal__TrigVal_710pre8.root","read");
    string f_path = "DQMData/Run 1/HLT/Run summary/BTag/";
    
    vstring triggers; // trigger in DQM file
    triggers.push_back("HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v6");
    triggers.push_back("HLT_QuadJet75_55_35_20_BTagIP_VBF_v9");
    //triggers.push_back("HLT_QuadJet75_55_38_20_BTagIP_VBF_v9");
    string current_trigger;
    
    vstring levels; // trigger systems in DQM file
    levels.push_back("L3");
    levels.push_back("L25");
    string current_level;
    
    for (size_t i = 0; i < triggers.size(); ++i)
    {
	current_trigger = triggers[i];
        cout << "Processing " << current_trigger << '\n';
        out << "\n" << current_trigger << "\n" << endl;
        
        struct stat dirStat; // create the directory for each trigger
        if (stat(current_trigger.c_str(), &dirStat) != 0)  // the directory does not exist
            mkdir(current_trigger.c_str(), 0755);
        
        for (size_t l = 0; l < levels.size(); ++l)
        {
            current_level = levels[l];
            
            if ((current_trigger == "HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v6") and (current_level == "L25")) //no this level for this trigger
                continue;
            
            out << "\n" << current_level << "\n" << endl;
            
            vstring profiles; // efficiencies in DQM file
            profiles.push_back("JetTag_" + current_level + "_b_efficiency_vs_disc");  
            profiles.push_back("JetTag_" + current_level + "_c_efficiency_vs_disc");
            profiles.push_back("JetTag_" + current_level + "_uds_efficiency_vs_disc");
            profiles.push_back("JetTag_" + current_level + "_g_efficiency_vs_disc");
            profiles.push_back("JetTag_" + current_level + "_light_efficiency_vs_disc");
            profiles.push_back("JetTag_" + current_level + "_b_disc_pT_efficiency_vs_pT");
            profiles.push_back("JetTag_" + current_level + "_c_disc_pT_efficiency_vs_pT"); 
            profiles.push_back("JetTag_" + current_level + "_uds_disc_pT_efficiency_vs_pT");
            profiles.push_back("JetTag_" + current_level + "_g_disc_pT_efficiency_vs_pT"); 
            profiles.push_back("JetTag_" + current_level + "_light_disc_pT_efficiency_vs_pT");
            profiles.push_back(current_level + "_b_c_mistagrate");
            profiles.push_back(current_level + "_b_g_mistagrate");
            profiles.push_back(current_level + "_b_light_mistagrate");
            string current_profile;
            
            vstring histos; // histograms in DQM file
            histos.push_back("JetTag_" + current_level);
            histos.push_back("JetTag_" + current_level + "_b");
            histos.push_back("JetTag_" + current_level + "_c");
            histos.push_back("JetTag_" + current_level + "_uds");
            histos.push_back("JetTag_" + current_level + "_g");
            histos.push_back("JetTag_" + current_level + "_light");
            string current_histo;

            for (size_t p = 0; p < profiles.size(); ++p)
            {
                current_profile = profiles[p];
                TProfile* p6 = (TProfile*)f6.Get((f_path+current_trigger+"/efficiency/"+current_profile).c_str());
                TProfile* p7 = (TProfile*)f7.Get((f_path+current_trigger+"/efficiency/"+current_profile).c_str());
                TProfile* p8 = (TProfile*)f8.Get((f_path+current_trigger+"/efficiency/"+current_profile).c_str());
                
                double hist_max = std::max(p6->GetMaximum(),p7->GetMaximum());
                hist_max = std::max(hist_max, p8->GetMaximum());
                p6->SetMaximum(1.1 * hist_max);
                
                TCanvas canvas("canvas", "", 700, 500);
                canvas.cd();
                p6->SetLineColor(kRed+1);
                p7->SetLineColor(kBlue);
                p8->SetLineColor(kGreen+2);
                p6->SetMarkerColor(kRed+1);
                p7->SetMarkerColor(kBlue);
                p8->SetMarkerColor(kGreen+2);
                p6->SetMarkerSize(0.5);
                p7->SetMarkerSize(0.5);
                p8->SetMarkerSize(0.5);
                p6->Draw("p0 hist");
                p7->Draw("p0 hist same");
                p8->Draw("p0 hist same");
                canvas.SetFillColor(kWhite);
                TLegend legend(0.25, 0.25, 0.45, 0.4);
                TLegend legend1(0.6, 0.75, 0.8, 0.9);               
                legend.SetFillColor(kWhite);
                legend.AddEntry(p6, "CMSSW_7_1_0_pre6", "p");
                legend.AddEntry(p7, "CMSSW_7_1_0_pre7", "p");
                legend.AddEntry(p8, "CMSSW_7_1_0_pre8", "p");
                legend1.SetFillColor(kWhite);
                legend1.AddEntry(p6, "CMSSW_7_1_0_pre6", "p");
                legend1.AddEntry(p7, "CMSSW_7_1_0_pre7", "p");
                legend1.AddEntry(p8, "CMSSW_7_1_0_pre8", "p");
                if (i == 0) legend.Draw();
                    else legend1.Draw();
                
                canvas.Print((current_trigger + "/" + current_trigger + "___" + current_profile + ".png").c_str());
                out << "%ATTACHURL%/" << current_trigger << "___" << current_profile << ".png" << endl;
                
            }
            
            out << endl;
            
            for (size_t h = 0; h < histos.size(); ++h)
            {
                current_histo = histos[h];
                TH1F* h6 = (TH1F*)f6.Get((f_path+current_trigger+"/"+current_histo).c_str());  
                TH1F* h7 = (TH1F*)f7.Get((f_path+current_trigger+"/"+current_histo).c_str());
                TH1F* h8 = (TH1F*)f8.Get((f_path+current_trigger+"/"+current_histo).c_str());
                
                double hist_max = std::max(h6->GetMaximum(),h7->GetMaximum());
                hist_max = std::max(hist_max, h8->GetMaximum());
                h6->SetMaximum(1.1 * hist_max);
                
                TCanvas canvas("canvas", "", 700, 500);
                canvas.cd();
                h6->SetLineColor(kRed+1);
                h7->SetLineColor(kBlue);
                h8->SetLineColor(kGreen+2);
                h6->Draw();
                h7->Draw("same");
                h8->Draw("same");
                canvas.SetFillColor(kWhite);
                
                TLegend legend(0.25, 0.25, 0.45, 0.4);
                legend.SetFillColor(kWhite);
                legend.AddEntry(h6, "CMSSW_7_1_0_pre6", "l");
                legend.AddEntry(h7, "CMSSW_7_1_0_pre7", "l");
                legend.AddEntry(h8, "CMSSW_7_1_0_pre8", "l");
                
                TLegend legend1(0.6, 0.75, 0.8, 0.9);
                legend1.SetFillColor(kWhite);
                legend1.AddEntry(h6, "CMSSW_7_1_0_pre6", "l");
                legend1.AddEntry(h7, "CMSSW_7_1_0_pre7", "l");
                legend1.AddEntry(h8, "CMSSW_7_1_0_pre8", "l");
                
                if (i == 0) legend.Draw();
                    else legend1.Draw();
                
                canvas.Print((current_trigger + "/" + current_trigger + "___" + current_histo + ".png").c_str());
                out << "%ATTACHURL%/" << current_trigger << "___" << current_histo << ".png" << endl;
            }
            out << endl;
        }
        out << endl;
        
    }
    out.close();
}

