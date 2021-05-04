#include <TROOT.h>
#include <TFile.h>
#include <TH2D.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

struct record {
  record(int ser = 0, int id = 0, double x = 0, double y = 0, double z = 0)
      : serial_(ser), id_(id), x_(x), y_(y), z_(z) {}

  int serial_, id_;
  double x_, y_, z_;
};

struct recordLess {
  bool operator()(const record& a, const record& b) {
    return ((a.z_ < b.z_) || ((a.z_ == b.z_) && (a.y_ < b.y_)) || ((a.z_ == b.z_) && (a.y_ == b.y_) && (a.x_ < b.x_)));
  }
};

void checkDuplicate(std::string fname, std::string outFile) {
  std::vector<record> records;
  ifstream infile(fname.c_str());
  if (!infile.is_open()) {
    std::cout << "Cannot open " << fname << std::endl;
  } else {
    while (1) {
      int ser, id;
      double x, y, z;
      infile >> ser >> id >> x >> y >> z;
      if (!infile.good())
        break;
      record rec(ser, id, x, y, z);
      records.push_back(rec);
    }
    infile.close();
    std::cout << "Reads " << records.size() << " records from " << fname << std::endl;

    std::ofstream fileout;
    fileout.open(outFile.c_str(), std::ofstream::out);
    std::cout << "Opens " << outFile << " in output mode" << std::endl;

    // Use std::sort
    std::sort(records.begin(), records.end(), recordLess());
    unsigned int bad1(0), bad2(0);
    for (unsigned k = 0; k < records.size(); ++k) {
      unsigned int dup(0);
      for (unsigned k1 = k + 1; k1 < records.size(); ++k1) {
        if ((fabs(records[k].x_ - records[k1].x_) < 0.0001) && (fabs(records[k].y_ - records[k1].y_) < 0.0001) &&
            (fabs(records[k].z_ - records[k1].z_) < 0.0001)) {
          ++dup;
        } else {
          break;
        }
      }
      fileout << "[" << records[k].serial_ << "] ID " << std::hex << records[k].id_ << std::dec << std::setprecision(6)
              << " (" << records[k].x_ << ", " << records[k].y_ << ", " << records[k].z_ << ") with " << dup
              << " Duplicates:";
      if (dup > 0) {
        for (unsigned k1 = 0; k1 < dup; ++k1)
          fileout << std::hex << " " << records[k + k1 + 1].id_ << std::dec;
        fileout << " ****** Check *****" << std::endl;
        k += dup;
        ++bad1;
        bad2 += (dup + 1);
      } else {
        fileout << std::endl;
      }
    }
    fileout.close();
    std::cout << "Bad records " << bad1 << ":" << bad2 << std::endl;
  }
}

void plotHist(const char* fname, bool save = false) {
  std::string name[4] = {"hist01", "hist02", "hist11", "hist12"};
  std::string titl[4] = {"z = -1, layer = 1", "z = -1, layer = 2", "z = 1, layer = 1", "z = 1, layer = 2"};
  gStyle->SetCanvasBorderMode(0);
  TFile* file = new TFile(fname);
  if (file != nullptr) {
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptStat(1110);
    gStyle->SetOptTitle(0);
    for (int k = 0; k < 4; ++k) {
      TH2D* hist = (TH2D*)file->FindObjectAny(name[k].c_str());
      if (hist != nullptr) {
        char namep[20];
        sprintf(namep, "c_%s", name[k].c_str());
        TCanvas* pad = new TCanvas(namep, namep, 700, 700);
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        hist->GetXaxis()->SetTitleSize(0.04);
        hist->GetXaxis()->SetLabelSize(0.035);
        hist->GetXaxis()->SetTitle("x (cm)");
        hist->GetYaxis()->SetTitle("y (cm)");
        hist->GetYaxis()->SetLabelOffset(0.005);
        hist->GetYaxis()->SetTitleSize(0.04);
        hist->GetYaxis()->SetLabelSize(0.035);
        hist->GetYaxis()->SetTitleOffset(1.10);
        hist->SetMarkerColor(2);
        hist->Draw();
        pad->Update();
        TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          st1->SetY1NDC(0.75);
          st1->SetY2NDC(0.90);
          st1->SetX1NDC(0.65);
          st1->SetX2NDC(0.90);
        }
        TPaveText* txt1 = new TPaveText(0.25, 0.91, 0.75, 0.96, "blNDC");
        txt1->SetFillColor(0);
        txt1->AddText(titl[k].c_str());
        txt1->Draw("same");
        pad->Modified();
        pad->Update();
        if (save) {
          sprintf(namep, "%s.pdf", pad->GetName());
          pad->Print(namep);
        }
      }
    }
  }
}
