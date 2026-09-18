// compareClusterFields.C
// ROOT macro to compare multiple fields from two different ROOT files
// Usage: root -l 'compareClusterFields.C("file1.root", "file2.root")'

#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPad.h>
#include <TLine.h>
#include <TSystem.h>
#include <iostream>
#include <vector>
#include <string>

// Helper function to manage output directory
void manageOutputDirectory() {
    const char* dirName = "packer_unpacker_performance";
    
    // Check if directory exists
    if (gSystem->AccessPathName(dirName) == false) {
        // Directory exists, delete it
        std::cout << "Directory '" << dirName << "' exists. Removing it..." << std::endl;
        gSystem->Exec(Form("rm -rf %s", dirName));
    }
    
    // Create new directory
    gSystem->Exec(Form("mkdir -p %s", dirName));
    std::cout << "Created directory: " << dirName << std::endl;
}

void compareDistribution(TTree* tree1, TTree* tree2, const char* fieldName, const char* file1, const char* file2, 
                         int nbins = 100, double xmin = 0, double xmax = 1000) {
    // Create canvas for distribution comparison
    TCanvas* c = new TCanvas(Form("c_dist_%s", fieldName), Form("%s Distribution", fieldName), 800, 600);
    
    // Create histograms
    TH1D* h1 = new TH1D(Form("h1_%s", fieldName), Form("%s from File 1", fieldName), nbins, xmin, xmax);
    TH1D* h2 = new TH1D(Form("h2_%s", fieldName), Form("%s from File 2", fieldName), nbins, xmin, xmax);
    
    // Fill histograms
    tree1->Draw(Form("%s>>h1_%s", fieldName, fieldName));
    tree2->Draw(Form("%s>>h2_%s", fieldName, fieldName));
    
    // Normalize histograms
    if (h1->Integral() > 0) h1->Scale(1.0 / h1->Integral());
    if (h2->Integral() > 0) h2->Scale(1.0 / h2->Integral());
    
    // Set styles with transparency
    h1->SetLineColor(kBlue);
    h1->SetLineWidth(2);
    h1->SetFillColor(kBlue);
    h1->SetFillStyle(3001);
    h1->SetFillColorAlpha(kBlue, 0.3);
    
    h2->SetLineColor(kRed);
    h2->SetLineWidth(2);
    h2->SetFillColor(kRed);
    h2->SetFillStyle(3001);
    h2->SetFillColorAlpha(kRed, 0.3);
    
    // Draw histograms
    h1->Draw("hist");
    h2->Draw("histsame");
    
    // Add labels
    h1->SetTitle(Form("%s Distribution (Normalized)", fieldName));
    h1->GetXaxis()->SetTitle(fieldName);
    h1->GetYaxis()->SetTitle("Fraction of Entries");
    h1->GetYaxis()->SetTitleOffset(1.4);
    
    // Legend
    TLegend* leg = new TLegend(0.12, 0.12, 0.35, 0.28);
    leg->AddEntry(h1, file1, "f");
    leg->AddEntry(h2, file2, "f");
    leg->SetBorderSize(1);
    leg->SetFillColor(0);
    leg->Draw();
    
    // Save canvas as PDF in the output directory
    c->Print(Form("packer_unpacker_performance/%s_distribution.pdf", fieldName));
    
    // Clean up
    delete leg;
    delete c;
}

void compareRatio(TTree* tree1, TTree* tree2, const char* fieldName, const char* file1, const char* file2,
                  int nbins = 100, double xmin = 0, double xmax = 1000) {
    // Create canvas for ratio plot
    TCanvas* c = new TCanvas(Form("c_ratio_%s", fieldName), Form("%s Ratio", fieldName), 800, 600);
    
    // Create histograms
    TH1D* h1 = new TH1D(Form("h1_ratio_%s", fieldName), Form("%s from File 1", fieldName), nbins, xmin, xmax);
    TH1D* h2 = new TH1D(Form("h2_ratio_%s", fieldName), Form("%s from File 2", fieldName), nbins, xmin, xmax);
    
    // Fill histograms
    tree1->Draw(Form("%s>>h1_ratio_%s", fieldName, fieldName));
    tree2->Draw(Form("%s>>h2_ratio_%s", fieldName, fieldName));
    
    // Normalize histograms
    if (h1->Integral() > 0) h1->Scale(1.0 / h1->Integral());
    if (h2->Integral() > 0) h2->Scale(1.0 / h2->Integral());
    
    // Create ratio histogram (no error bars)
    TH1D* ratio = (TH1D*)h1->Clone(Form("ratio_%s", fieldName));
    ratio->SetTitle(Form("%s Ratio: File1 / File2", fieldName));
    ratio->SetLineColor(kBlue);
    ratio->SetLineWidth(2);
    ratio->SetFillColor(kBlue);
    ratio->SetFillStyle(3001);
    ratio->SetFillColorAlpha(kBlue, 0.3);
    ratio->GetXaxis()->SetTitle(fieldName);
    ratio->GetYaxis()->SetTitle("Ratio (File1 / File2)");
    ratio->GetYaxis()->SetTitleOffset(1.4);
    ratio->SetStats(0);
    ratio->SetMarkerStyle(0);  // No markers
    
    // Calculate ratio (no errors)
    for (int i = 1; i <= ratio->GetNbinsX(); ++i) {
        double val1 = h1->GetBinContent(i);
        double val2 = h2->GetBinContent(i);
        
        if (val2 > 0) {
            ratio->SetBinContent(i, val1 / val2);
        } else if (val1 > 0) {
            ratio->SetBinContent(i, 0);
        } else {
            ratio->SetBinContent(i, 1.0);
        }
    }
    
    // Draw ratio as histogram (no error bars)
    ratio->Draw("hist");
    
    // Add line at y=1
    TLine* line = new TLine(ratio->GetXaxis()->GetXmin(), 1, ratio->GetXaxis()->GetXmax(), 1);
    line->SetLineColor(kBlack);
    line->SetLineStyle(kDashed);
    line->Draw("same");
    
    // Save canvas as PDF in the output directory
    c->Print(Form("packer_unpacker_performance/%s_ratio.pdf", fieldName));
    
    // Clean up
    delete line;
    delete c;
    delete ratio;
}

void compareFieldDiscreteDistribution(TTree* tree1, TTree* tree2, const char* fieldName, const char* file1, const char* file2) {
    // Special handling for discrete fields - Distribution only
    TCanvas* c = new TCanvas(Form("c_dist_%s", fieldName), Form("%s Distribution", fieldName), 800, 600);
    
    // For discrete fields, determine bins automatically
    TH1D* h1_temp = new TH1D(Form("h1_temp_%s", fieldName), "", 100, 0, 100);
    TH1D* h2_temp = new TH1D(Form("h2_temp_%s", fieldName), "", 100, 0, 100);
    tree1->Draw(Form("%s>>h1_temp_%s", fieldName, fieldName));
    tree2->Draw(Form("%s>>h2_temp_%s", fieldName, fieldName));
    
    int nbins = std::max(h1_temp->GetNbinsX(), h2_temp->GetNbinsX());
    double xmin = std::min(h1_temp->GetXaxis()->GetXmin(), h2_temp->GetXaxis()->GetXmin());
    double xmax = std::max(h1_temp->GetXaxis()->GetXmax(), h2_temp->GetXaxis()->GetXmax());
    
    // Recreate histograms with proper binning
    TH1D* h1 = new TH1D(Form("h1_%s", fieldName), Form("%s from File 1", fieldName), nbins, xmin, xmax);
    TH1D* h2 = new TH1D(Form("h2_%s", fieldName), Form("%s from File 2", fieldName), nbins, xmin, xmax);
    
    tree1->Draw(Form("%s>>h1_%s", fieldName, fieldName));
    tree2->Draw(Form("%s>>h2_%s", fieldName, fieldName));
    
    // Normalize histograms
    if (h1->Integral() > 0) h1->Scale(1.0 / h1->Integral());
    if (h2->Integral() > 0) h2->Scale(1.0 / h2->Integral());
    
    // Set styles with transparency
    h1->SetLineColor(kBlue);
    h1->SetLineWidth(2);
    h1->SetFillColor(kBlue);
    h1->SetFillStyle(3001);
    h1->SetFillColorAlpha(kBlue, 0.3);
    
    h2->SetLineColor(kRed);
    h2->SetLineWidth(2);
    h2->SetFillColor(kRed);
    h2->SetFillStyle(3001);
    h2->SetFillColorAlpha(kRed, 0.3);
    
    // Draw histograms
    h1->Draw("hist");
    h2->Draw("histsame");
    
    // Add labels
    h1->SetTitle(Form("%s Distribution (Normalized)", fieldName));
    h1->GetXaxis()->SetTitle(fieldName);
    h1->GetYaxis()->SetTitle("Fraction of Entries");
    h1->GetYaxis()->SetTitleOffset(1.4);
    
    // Legend
    TLegend* leg = new TLegend(0.12, 0.12, 0.35, 0.28);
    leg->AddEntry(h1, file1, "f");
    leg->AddEntry(h2, file2, "f");
    leg->SetBorderSize(1);
    leg->SetFillColor(0);
    leg->Draw();
    
    // Save canvas as PDF in the output directory
    c->Print(Form("packer_unpacker_performance/%s_distribution.pdf", fieldName));
    
    // Clean up
    delete leg;
    delete c;
    delete h1_temp;
    delete h2_temp;
}

void compareFieldDiscreteRatio(TTree* tree1, TTree* tree2, const char* fieldName, const char* file1, const char* file2) {
    // Special handling for discrete fields - Ratio only
    TCanvas* c = new TCanvas(Form("c_ratio_%s", fieldName), Form("%s Ratio", fieldName), 800, 600);
    
    // For discrete fields, determine bins automatically
    TH1D* h1_temp = new TH1D(Form("h1_temp_%s", fieldName), "", 100, 0, 100);
    TH1D* h2_temp = new TH1D(Form("h2_temp_%s", fieldName), "", 100, 0, 100);
    tree1->Draw(Form("%s>>h1_temp_%s", fieldName, fieldName));
    tree2->Draw(Form("%s>>h2_temp_%s", fieldName, fieldName));
    
    int nbins = std::max(h1_temp->GetNbinsX(), h2_temp->GetNbinsX());
    double xmin = std::min(h1_temp->GetXaxis()->GetXmin(), h2_temp->GetXaxis()->GetXmin());
    double xmax = std::max(h1_temp->GetXaxis()->GetXmax(), h2_temp->GetXaxis()->GetXmax());
    
    // Recreate histograms with proper binning
    TH1D* h1 = new TH1D(Form("h1_ratio_%s", fieldName), Form("%s from File 1", fieldName), nbins, xmin, xmax);
    TH1D* h2 = new TH1D(Form("h2_ratio_%s", fieldName), Form("%s from File 2", fieldName), nbins, xmin, xmax);
    
    tree1->Draw(Form("%s>>h1_ratio_%s", fieldName, fieldName));
    tree2->Draw(Form("%s>>h2_ratio_%s", fieldName, fieldName));
    
    // Normalize histograms
    if (h1->Integral() > 0) h1->Scale(1.0 / h1->Integral());
    if (h2->Integral() > 0) h2->Scale(1.0 / h2->Integral());
    
    // Create ratio histogram (no error bars)
    TH1D* ratio = (TH1D*)h1->Clone(Form("ratio_%s", fieldName));
    ratio->SetTitle(Form("%s Ratio: File1 / File2", fieldName));
    ratio->SetLineColor(kBlue);
    ratio->SetLineWidth(2);
    ratio->SetFillColor(kBlue);
    ratio->SetFillStyle(3001);
    ratio->SetFillColorAlpha(kBlue, 0.3);
    ratio->GetXaxis()->SetTitle(fieldName);
    ratio->GetYaxis()->SetTitle("Ratio (File1 / File2)");
    ratio->GetYaxis()->SetTitleOffset(1.4);
    ratio->SetStats(0);
    ratio->SetMarkerStyle(0);  // No markers
    
    // Calculate ratio (no errors)
    for (int i = 1; i <= ratio->GetNbinsX(); ++i) {
        double val1 = h1->GetBinContent(i);
        double val2 = h2->GetBinContent(i);
        
        if (val2 > 0) {
            ratio->SetBinContent(i, val1 / val2);
        } else if (val1 > 0) {
            ratio->SetBinContent(i, 0);
        } else {
            ratio->SetBinContent(i, 1.0);
        }
    }
    
    // Draw ratio as histogram (no error bars)
    ratio->Draw("hist");
    
    // Add line at y=1
    TLine* line = new TLine(ratio->GetXaxis()->GetXmin(), 1, ratio->GetXaxis()->GetXmax(), 1);
    line->SetLineColor(kBlack);
    line->SetLineStyle(kDashed);
    line->Draw("same");
    
    // Save canvas as PDF in the output directory
    c->Print(Form("packer_unpacker_performance/%s_ratio.pdf", fieldName));
    
    // Clean up
    delete line;
    delete c;
    delete ratio;
    delete h1_temp;
    delete h2_temp;
}

void TrackerPackerUnpackerPerformance(const char* file1, const char* file2) {
    // Manage output directory
    manageOutputDirectory();
    
    // Open both files
    TFile* f1 = TFile::Open(file1);
    TFile* f2 = TFile::Open(file2);
    
    if (!f1 || f1->IsZombie()) {
        std::cerr << "Error: Cannot open file " << file1 << std::endl;
        return;
    }
    if (!f2 || f2->IsZombie()) {
        std::cerr << "Error: Cannot open file " << file2 << std::endl;
        return;
    }
    
    // Get the trees
    TTree* tree1 = (TTree*)f1->Get("ClusterAnalyzer/ClusterTree");
    TTree* tree2 = (TTree*)f2->Get("ClusterAnalyzer/ClusterTree");
    
    if (!tree1 || !tree2) {
        std::cerr << "Error: ClusterTree not found in one of the files" << std::endl;
        return;
    }
    
    // Get the number of entries
    Long64_t nEntries1 = tree1->GetEntries();
    Long64_t nEntries2 = tree2->GetEntries();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "File 1: " << file1 << " has " << nEntries1 << " events" << std::endl;
    std::cout << "File 2: " << file2 << " has " << nEntries2 << " events" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // List of fields to compare
    std::vector<std::string> continuousFields = {
        "clusterRow",
        "clusterCol",
        "clusterSize",
        "clusterCenter",
        "clusterR",
        "clusterZ",
        "clusterLocalX",
        "clusterLocalY",
        "clusterGlobalX",
        "clusterGlobalY",
        "clusterGlobalZ"
    };
    
    std::vector<std::string> discreteFields = {
        "dtcID",
        "isPSModulePixel",
        "isPSModuleStrip",
        "is2SModule"
    };
    
    // Compare continuous fields - Distribution and Ratio separately
    for (const auto& field : continuousFields) {
        std::cout << "Processing: " << field << std::endl;
        // Determine binning based on field
        if (field == "clusterRow" || field == "clusterCol") {
            compareDistribution(tree1, tree2, field.c_str(), file1, file2, 100, 0, 1000);
            compareRatio(tree1, tree2, field.c_str(), file1, file2, 100, 0, 1000);
        } else if (field == "clusterSize") {
            compareDistribution(tree1, tree2, field.c_str(), file1, file2, 50, 0, 50);
            compareRatio(tree1, tree2, field.c_str(), file1, file2, 50, 0, 50);
        } else if (field == "clusterCenter") {
            compareDistribution(tree1, tree2, field.c_str(), file1, file2, 100, 0, 2000);
            compareRatio(tree1, tree2, field.c_str(), file1, file2, 100, 0, 2000);
        } else if (field == "clusterR" || field == "clusterZ") {
            compareDistribution(tree1, tree2, field.c_str(), file1, file2, 100, 0, 1500);
            compareRatio(tree1, tree2, field.c_str(), file1, file2, 100, 0, 1500);
        } else if (field.find("Local") != std::string::npos) {
            compareDistribution(tree1, tree2, field.c_str(), file1, file2, 100, -10, 10);
            compareRatio(tree1, tree2, field.c_str(), file1, file2, 100, -10, 10);
        } else if (field.find("Global") != std::string::npos) {
            compareDistribution(tree1, tree2, field.c_str(), file1, file2, 100, -1000, 1000);
            compareRatio(tree1, tree2, field.c_str(), file1, file2, 100, -1000, 1000);
        } else {
            compareDistribution(tree1, tree2, field.c_str(), file1, file2);
            compareRatio(tree1, tree2, field.c_str(), file1, file2);
        }
    }
    
    // Compare discrete fields - Distribution and Ratio separately
    for (const auto& field : discreteFields) {
        std::cout << "Processing: " << field << std::endl;
        compareFieldDiscreteDistribution(tree1, tree2, field.c_str(), file1, file2);
        compareFieldDiscreteRatio(tree1, tree2, field.c_str(), file1, file2);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All comparisons completed!" << std::endl;
    std::cout << "Generated PDF files in 'packer_unpacker_performance/':" << std::endl;
    std::cout << "  Distribution plots:" << std::endl;
    for (const auto& field : continuousFields) {
        std::cout << "    - " << field << "_distribution.pdf" << std::endl;
    }
    for (const auto& field : discreteFields) {
        std::cout << "    - " << field << "_distribution.pdf" << std::endl;
    }
    std::cout << "  Ratio plots:" << std::endl;
    for (const auto& field : continuousFields) {
        std::cout << "    - " << field << "_ratio.pdf" << std::endl;
    }
    for (const auto& field : discreteFields) {
        std::cout << "    - " << field << "_ratio.pdf" << std::endl;
    }
    std::cout << "========================================" << std::endl;
}