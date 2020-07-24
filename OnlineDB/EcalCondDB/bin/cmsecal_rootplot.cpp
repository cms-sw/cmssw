#include "boost/program_options.hpp"
#include "boost/tokenizer.hpp"
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TGraph.h"
#include "TLine.h"
#include "TPad.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cfloat>

using namespace std;
using namespace boost;

class RootPlot {
public:
  static const int DEFAULT_AXIS = 111;
  static const int TIME_AXIS = 222;

  RootPlot(string type, string format, string file, float hmin = 0., float hmax = 0.) {
    m_isInit = false;
    m_type = type;
    m_outputFormat = format;
    m_outputFile = file + "." + format;
    m_outputRoot = file + ".root";
    m_title = "rootplot Plot";
    m_xtitle = "X";
    m_ytitle = "Y";
    m_xAxisType = DEFAULT_AXIS;
    m_debug = 1;
    m_T0 = TDatime(2005, 01, 01, 00, 00, 00);
    m_hmin = hmin;
    m_hmax = hmax;
  };

  ~RootPlot(){};

  void init() {
    m_nbins[0] = m_nbins[1] = 100;
    m_mins[0] = m_mins[1] = FLT_MAX;
    m_maxs[0] = m_maxs[1] = FLT_MIN;
    m_data[0] = 0;
    m_data[1] = 0;
    m_data[2] = 0;
    m_data[3] = 0;
    m_data[4] = 0;

    gROOT->SetStyle("Plain");
    gStyle->SetOptStat(111111);
    gStyle->SetOptFit();
    gStyle->SetPalette(1, nullptr);

    int pCol[2] = {2, 3};
    if ((m_type == "Map" || m_type == "EBEEMap") && (TString(m_title).Contains("status"))) {
      gStyle->SetPalette(2, pCol);
    }

    m_rootfile = new TFile(m_outputRoot.c_str(), "RECREATE");
    m_tree = new TTree("t1", "rootplot tree");

    if (m_type == "TH1F") {
      m_nfields = 1;
      m_tree->Branch("x", &m_data[0], "x/F");
    } else if (m_type == "TH2F") {
      m_nfields = 2;
      m_tree->Branch("x", &m_data[0], "x/F");
      m_tree->Branch("y", &m_data[1], "y/F");
    } else if (m_type == "TGraph") {
      m_nfields = 2;
      m_tree->Branch("x", &m_data[0], "x/F");
      m_tree->Branch("y", &m_data[1], "y/F");
    } else if (m_type == "Map") {
      m_nfields = 2;
      m_tree->Branch("x", &m_data[0], "x/F");  // channel number
      m_tree->Branch("y", &m_data[1], "y/F");  // variable var
    } else if (m_type == "EBEEMap") {
      m_nfields = 5;
      m_tree->Branch("ism_z", &m_data[0], "ism_z/F");      // SM number (EB), z (EE)
      m_tree->Branch("chnum_x", &m_data[1], "chnum_x/F");  // channel number (EB) , x (EE)
      m_tree->Branch("var", &m_data[2], "var/F");          // variable var
      m_tree->Branch("null_y", &m_data[3], "z/F");         // null value (EB), y (EE)
      m_tree->Branch("isEB", &m_data[4], "isEB/F");        // 1 (EB), 0 (EE)
    }
  };

  void setTitle(string title) { m_title = title; }
  void setXTitle(string xtitle) { m_xtitle = xtitle; }
  void setYTitle(string ytitle) { m_ytitle = ytitle; }
  void setDebug(int debug) { m_debug = debug; }
  void setXAxisType(int code) { m_xAxisType = code; }

  void parseAndFill(std::string str) {
    if (!m_isInit) {
      this->init();
      m_isInit = true;
    }

    if (str[0] == '#') {
      // skip header
      return;
    }

    if (m_debug) {
      cout << "[data] " << flush;
    }

    typedef boost::tokenizer<boost::escaped_list_separator<char> > tokenizer;
    escaped_list_separator<char> sep('\\', ' ', '\"');
    tokenizer tokens(str, sep);
    float datum;
    int cnt = 0;

    for (tokenizer::iterator tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter) {
      if (cnt > m_nfields) {
        continue;
      }

      if (m_debug) {
        cout << "<" << *tok_iter << ">" << flush;
      }

      if (m_xAxisType == TIME_AXIS && cnt == 0) {
        TDatime d((*tok_iter).c_str());
        datum = (Float_t)(d.Convert() - m_T0.Convert());
      } else {
        datum = atof((*tok_iter).c_str());
      }

      if (m_type != "EBEEMap") {
        if (datum < m_mins[cnt]) {
          m_mins[cnt] = datum;
        }
        if (datum > m_maxs[cnt]) {
          m_maxs[cnt] = datum;
        }
      }

      m_data[cnt] = datum;

      cnt++;
    }

    if (m_debug) {
      cout << endl;
    }
    m_tree->Fill();
  };

  void draw() {
    for (int i = 0; i < m_nfields; i++) {
      if (m_mins[i] == m_maxs[i]) {
        m_mins[i] -= m_mins[i] * (0.05);
        m_maxs[i] += m_mins[i] * (0.05);
      }
    }

    if (m_debug) {
      cout << "[draw()]:" << endl;
      cout << "  m_type:          " << m_type << endl;
      cout << "  m_outputFormat:  " << m_outputFormat << endl;
      cout << "  m_outputFile:    " << m_outputFile << endl;
      cout << "  m_outputRoot:    " << m_outputRoot << endl;
      cout << "  m_title:         " << m_title << endl;
      cout << "  m_xtitle:        " << m_xtitle << endl;
      cout << "  m_ytitle:        " << m_ytitle << endl;
      cout << "  m_xAxisType:     " << m_xAxisType << endl;
      cout << "  m_nfields:       " << m_nfields << endl;
      cout << "  m_nbins[]:       " << m_nbins[0] << " " << m_nbins[1] << endl;
      cout << "  m_mins[]:        " << m_mins[0] << " " << m_mins[1] << endl;
      cout << "  m_maxs[]:        " << m_maxs[0] << " " << m_maxs[1] << endl;
    }

    m_tree->Write();

    //std::cout << "m_type = " << m_type << std::endl;

    if (m_type == "TH1F") {
      this->drawTH1F();
    } else if (m_type == "TH2F") {
      this->drawTH2F();
    } else if (m_type == "TGraph") {
      this->drawTGraph();
    } else if (m_type == "Map") {
      this->drawMap();
    } else if (m_type == "EBEEMap") {
      this->drawEBEEMap();
    }

    m_isInit = false;
  };

  void drawTH1F() {
    TCanvas c1("c1", "rootplot", 200, 10, 450, 300);
    c1.SetGrid();

    if ((TString(m_title).Contains("status"))) {
      m_mins[0] = -0.001;
      m_maxs[0] = 1.001;
    }

    TH1F* plot = new TH1F("rootplot", m_title.c_str(), m_nbins[0], m_mins[0], m_maxs[0]);
    plot->GetXaxis()->SetTitle(m_xtitle.c_str());
    plot->GetYaxis()->SetTitle(m_ytitle.c_str());

    if (m_xAxisType == TIME_AXIS) {
      TAxis* axis = plot->GetXaxis();
      setTimeAxis(axis);
    }

    m_tree->Draw("x >> rootplot");

    plot->Draw();
    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());

    plot->Write();
  };

  void drawTH2F() {
    TCanvas c1("c1", "rootplot", 200, 10, 600, 400);
    c1.SetGrid();

    TH2F* plot =
        new TH2F("rootplot", m_title.c_str(), m_nbins[0], m_mins[0], m_maxs[0], m_nbins[1], m_mins[1], m_maxs[1]);
    plot->GetXaxis()->SetTitle(m_xtitle.c_str());
    plot->GetYaxis()->SetTitle(m_ytitle.c_str());

    if (m_xAxisType == TIME_AXIS) {
      TAxis* axis = plot->GetXaxis();
      setTimeAxis(axis);
    }

    m_tree->Draw("x:y >> rootplot");

    plot->Draw();
    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());

    plot->Write();
  }

  void drawTGraph() {
    TCanvas c1("c1", "rootplot", 200, 10, 600, 400);
    c1.SetGrid();

    Int_t n = (Int_t)m_tree->GetEntries();
    TGraph* plot = new TGraph(n);

    Float_t x, y;
    m_tree->SetBranchAddress("x", &x);
    m_tree->SetBranchAddress("y", &y);
    for (Int_t i = 0; i < n; i++) {
      m_tree->GetEntry(i);
      plot->SetPoint(i, x, y);
    }

    if (m_xAxisType == TIME_AXIS) {
      TAxis* axis = plot->GetXaxis();
      setTimeAxis(axis);
    }

    plot->SetTitle(m_title.c_str());
    plot->GetXaxis()->SetTitle(m_xtitle.c_str());
    plot->GetYaxis()->SetTitle(m_ytitle.c_str());
    plot->SetMarkerStyle(21);
    plot->Draw("AP");

    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());
    plot->Write();
  };

  void drawMap() {
    gStyle->SetOptStat(0);

    const Int_t csize = 250;
    TCanvas c1("c1", "rootplot", Int_t(85. / 20. * csize), csize);
    TH2F* plot = new TH2F("rootplot", m_title.c_str(), 85, 0.0001, 85.0001, 20, 0.0001, 20.0001);
    plot->GetXaxis()->SetTitle(m_xtitle.c_str());
    plot->GetYaxis()->SetTitle(m_ytitle.c_str());

    Float_t x, y;
    m_tree->SetBranchAddress("x", &x);
    m_tree->SetBranchAddress("y", &y);

    // now fill the map...
    Int_t n = (Int_t)m_tree->GetEntries();
    for (Int_t i = 0; i < n; i++) {
      m_tree->GetEntry(i);
      //      while(x>1700) x-=1700;
      Float_t xmap = Float_t(Int_t(x - 1) / 20) + 1;
      Float_t ymap = Float_t(Int_t(x - 1) % 20) + 1;
      plot->Fill(xmap, ymap, y);
    }

    // draw the map
    plot->SetTitle(m_title.c_str());
    if ((TString(m_title).Contains("status"))) {
      plot->SetMinimum(-0.001);
      plot->SetMaximum(1.001);
    } else if (!(m_hmin == 0 && m_hmax == 0)) {
      plot->SetMinimum(m_hmin);
      plot->SetMaximum(m_hmax);
    }

    plot->GetXaxis()->SetTitle("#phi");
    plot->GetYaxis()->SetTitle("#eta");
    plot->GetZaxis()->SetTitle(m_ytitle.c_str());
    plot->GetXaxis()->SetNdivisions(17);
    plot->GetYaxis()->SetNdivisions(4);
    c1.SetGridx();
    c1.SetGridy();
    plot->Draw("colz");

    // and draw the grid upon the map...
    TH2C* labelGrid = new TH2C("labelGrid", "label grid for SM", 85, 0., 85., 20, 0., 20.);
    for (Int_t i = 0; i < 68; i++) {
      Float_t X = (i / 4) * 5 + 2;
      Float_t Y = (i % 4) * 5 + 2;
      labelGrid->Fill(X, Y, i + 1);
    }
    labelGrid->SetMinimum(0.1);
    labelGrid->SetMarkerSize(4);
    labelGrid->Draw("text,same");

    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());
    plot->Write();
  };

  void drawEBEEMap() {
    const Int_t csize = 900;

    TCanvas c1("c1", "rootplot", csize, csize);
    TPad p1("EBPad", "EBPad", 0., 0.5, 1., 1.);
    p1.Draw();
    TPad p2("EEPlusPad", "EEPlusPad", 0., 0., 0.48, 0.5);
    p2.Draw();
    TPad p3("EEPlusPad", "EEPlusPad", 0.50, 0., 0.98, 0.5);
    p3.Draw();

    //EB
    TH2F* EBPlot = new TH2F("rootplotEB", m_title.c_str(), 360, 0., 360., 170, -85., 85.);
    EBPlot->GetXaxis()->SetTitle("i#phi");
    EBPlot->GetYaxis()->SetTitle("i#eta");
    EBPlot->GetZaxis()->SetTitle(m_ytitle.c_str());
    EBPlot->GetXaxis()->SetNdivisions(18, kFALSE);
    EBPlot->GetYaxis()->SetNdivisions(2);

    //EE+
    TH2F* EEPlot_plus = new TH2F("rootplotEE+", m_title.c_str(), 100, 0., 100., 100, 0., 100.);
    EEPlot_plus->GetXaxis()->SetTitle("ix");
    EEPlot_plus->GetYaxis()->SetTitleOffset(1.3);
    EEPlot_plus->GetYaxis()->SetTitle("iy");
    EEPlot_plus->GetZaxis()->SetTitle(m_ytitle.c_str());
    EEPlot_plus->GetXaxis()->SetNdivisions(10, kTRUE);
    EEPlot_plus->GetYaxis()->SetNdivisions(10);

    //EE-
    TH2F* EEPlot_minus = new TH2F("rootplotEE-", m_title.c_str(), 100, 0., 100., 100, 0., 100.);
    EEPlot_minus->GetXaxis()->SetTitle("ix");
    EEPlot_minus->GetYaxis()->SetTitle("iy");
    EEPlot_minus->GetYaxis()->SetTitleOffset(1.3);
    EEPlot_minus->GetZaxis()->SetTitle(m_ytitle.c_str());
    EEPlot_minus->GetXaxis()->SetNdivisions(10, kTRUE);
    EEPlot_minus->GetYaxis()->SetNdivisions(10);

    Float_t chnum_x, var, ism_z, isEB, null_y;
    m_tree->SetBranchAddress("ism_z", &ism_z);
    m_tree->SetBranchAddress("chnum_x", &chnum_x);
    m_tree->SetBranchAddress("var", &var);
    m_tree->SetBranchAddress("isEB", &isEB);
    m_tree->SetBranchAddress("null_y", &null_y);

    // now fill the maps...
    Int_t n = (Int_t)m_tree->GetEntries();
    for (Int_t i = 0; i < n; i++) {
      m_tree->GetEntry(i);

      if (isEB) {
        Float_t iex = -1;
        Float_t ipx = -1;
        for (unsigned int i = 1; i <= 36; i++) {
          if (i == ism_z) {
            Float_t ie = Float_t(Int_t(chnum_x - 1) / 20) + 1;
            Float_t ip = Float_t(Int_t(chnum_x - 1) % 20) + 1;

            if (ism_z <= 18) {
              iex = ie - 1;
              ipx = (20 - ip) + 20 * (ism_z - 1);
            } else {
              iex = -1 * ie;
              ipx = ip + 20 * (ism_z - 19) - 1;
            }

            EBPlot->Fill(ipx, iex, var);
          }
        }
      }  //end loop on EB

      //assuming: if not EB, it's EE (TODO: check strings)...
      else {
        //EE+
        if (ism_z == 1)
          EEPlot_plus->Fill(chnum_x - 0.5, null_y - 0.5, var);
        //EE-
        if (ism_z == -1)
          EEPlot_minus->Fill(chnum_x - 0.5, null_y - 0.5, var);
      }  //end loop on EE

    }  //end loop on entries

    // draw the map
    //setting
    gStyle->SetOptStat("e");
    gStyle->SetPaintTextFormat("+g");

    EBPlot->SetTitle(m_title.c_str());
    EEPlot_plus->SetTitle(m_title.c_str());
    EEPlot_minus->SetTitle(m_title.c_str());

    if ((TString(m_title).Contains("status"))) {
      EBPlot->SetMinimum(-0.001);
      EBPlot->SetMaximum(1.001);
      EEPlot_plus->SetMinimum(-0.001);
      EEPlot_plus->SetMaximum(1.001);
      EEPlot_minus->SetMinimum(-0.001);
      EEPlot_minus->SetMaximum(1.001);

    } else if (!(m_hmin == 0 && m_hmax == 0)) {
      EBPlot->SetMinimum(m_hmin);
      EBPlot->SetMaximum(m_hmax);
      EEPlot_plus->SetMinimum(m_hmin);
      EEPlot_plus->SetMaximum(m_hmax);
      EEPlot_minus->SetMinimum(m_hmin);
      EEPlot_minus->SetMaximum(m_hmax);
    }

    p1.cd();
    gPad->SetGridx();
    gPad->SetGridy();
    EBPlot->Draw("colz");

    // and draw the grid upon the map...
    TH2C* labelGrid = new TH2C("labelGrid", "label grid for SM", 18, 0., 360., 2, -85., 85.);
    for (Int_t sm = 1; sm <= 36; sm++) {
      int X = (sm <= 18) ? sm : sm - 18;
      int Y = (sm <= 18) ? 2 : 1;
      double posSM = (sm <= 18) ? sm : -1 * (sm - 18);
      labelGrid->SetBinContent(X, Y, posSM);
    }
    labelGrid->SetMarkerSize(2);
    labelGrid->Draw("text,same");

    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());
    EBPlot->Write();

    //END OF EBPLOT

    int ixSectorsEE[202] = {
        61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41,  40,  40,  39,  39, 40, 40,
        41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61,  0,   100, 100, 97, 97, 95,
        95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25,  25,  20,  20,  15, 15, 13,
        13, 8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8,  13, 13, 15, 15, 20,  20,  25,  25,  35, 35, 40,
        40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97, 100, 100, 0,   61,  65, 65, 70,
        70, 80, 80, 90, 90, 92, 0,  61, 65, 65, 90, 90, 97, 0,  57, 60, 60, 65, 65, 70,  70,  75,  75,  80, 80, 0,
        50, 50, 0,  43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20, 0,  39, 35, 35, 10, 10,  3,   0,   39,  35, 35, 30,
        30, 20, 20, 10, 10, 8,  0,  45, 45, 40, 40, 35, 35, 0,  55, 55, 60, 60, 65, 65};

    int iySectorsEE[202] = {
        50, 55,  55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60,  59,  59, 58, 58, 57, 57, 55, 55, 45, 45, 43,
        43, 42,  42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42,  43,  43, 45, 45, 50, 0,  50, 60, 60, 65, 65,
        75, 75,  80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97, 100, 100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85,
        80, 80,  75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20,  20,  15, 15, 13, 13, 8,  8,  5,  5,  3,  3,
        0,  0,   3,  3,  5,  5,  8,  8,  13, 13, 15, 15, 20, 20, 25,  25,  35, 35, 40, 40, 50, 0,  45, 45, 40, 40,
        35, 35,  30, 30, 25, 25, 0,  50, 50, 55, 55, 60, 60, 0,  60,  60,  65, 65, 70, 70, 75, 75, 85, 85, 87, 0,
        61, 100, 0,  60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87, 0,   50,  50, 55, 55, 60, 60, 0,  45, 45, 40, 40,
        35, 35,  30, 30, 25, 25, 0,  39, 30, 30, 15, 15, 5,  0,  39,  30,  30, 15, 15, 5};

    //grid
    TH2C labelGrid1("labelGrid1", "label grid for EE -", 10, 0., 100., 10, 0., 100.);
    for (int i = 1; i <= 10; i++) {
      for (int j = 1; j <= 10; j++) {
        labelGrid1.SetBinContent(i, j, -10);
      }
    }

    labelGrid1.SetBinContent(2, 5, -3);
    labelGrid1.SetBinContent(2, 7, -2);
    labelGrid1.SetBinContent(4, 9, -1);
    labelGrid1.SetBinContent(7, 9, -9);
    labelGrid1.SetBinContent(9, 7, -8);
    labelGrid1.SetBinContent(9, 5, -7);
    labelGrid1.SetBinContent(8, 3, -6);
    labelGrid1.SetBinContent(6, 2, -5);
    labelGrid1.SetBinContent(3, 3, -4);
    labelGrid1.SetMarkerSize(2);
    labelGrid1.SetMinimum(-9.01);
    labelGrid1.SetMaximum(-0.01);

    TH2C labelGrid2("labelGrid2", "label grid for EE +", 10, 0., 100., 10, 0., 100.);

    for (int i = 1; i <= 10; i++) {
      for (int j = 1; j <= 10; j++) {
        labelGrid2.SetBinContent(i, j, -10);
      }
    }

    labelGrid2.SetBinContent(2, 5, +3);
    labelGrid2.SetBinContent(2, 7, +2);
    labelGrid2.SetBinContent(4, 9, +1);
    labelGrid2.SetBinContent(7, 9, +9);
    labelGrid2.SetBinContent(9, 7, +8);
    labelGrid2.SetBinContent(9, 5, +7);
    labelGrid2.SetBinContent(8, 3, +6);
    labelGrid2.SetBinContent(5, 2, +5);
    labelGrid2.SetBinContent(3, 3, +4);

    labelGrid2.SetMarkerSize(2);
    labelGrid2.SetMinimum(+0.01);
    labelGrid2.SetMaximum(+9.01);

    //EE+
    p2.cd();
    gPad->SetGridx();
    gPad->SetGridy();
    EEPlot_plus->Draw("colz");
    labelGrid2.Draw("text,same");

    //drawing sector grid

    TLine l;
    l.SetLineWidth(1);
    for (int i = 0; i < 201; i = i + 1) {
      if ((ixSectorsEE[i] != 0 || iySectorsEE[i] != 0) && (ixSectorsEE[i + 1] != 0 || iySectorsEE[i + 1] != 0)) {
        l.DrawLine(ixSectorsEE[i], iySectorsEE[i], ixSectorsEE[i + 1], iySectorsEE[i + 1]);
      }
    }

    //EE-
    p3.cd();
    gPad->SetGridx();
    gPad->SetGridy();
    EEPlot_minus->Draw("colz");
    labelGrid1.Draw("text,same");

    //drawing sector grid
    for (int i = 0; i < 201; i = i + 1) {
      if ((ixSectorsEE[i] != 0 || iySectorsEE[i] != 0) && (ixSectorsEE[i + 1] != 0 || iySectorsEE[i + 1] != 0)) {
        l.DrawLine(ixSectorsEE[i], iySectorsEE[i], ixSectorsEE[i + 1], iySectorsEE[i + 1]);
      }
    }

    //drawing everything & printing
    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());
  };

  void setTimeAxis(TAxis* axis) {
    axis->SetTimeOffset(m_T0.Convert());
    axis->SetTimeDisplay(1);
    axis->SetTimeFormat("#splitline{%d/%m}{%H:%M:%S}");
    axis->SetLabelOffset(0.02);
    axis->SetLabelSize(0.03);
  }

private:
  RootPlot(){};  // hidden default constructor

  bool m_isInit;
  TFile* m_rootfile;
  TTree* m_tree;
  string m_type;
  string m_outputFormat;
  string m_outputFile;
  string m_outputRoot;
  string m_title;
  string m_xtitle;
  string m_ytitle;
  float m_hmin;
  float m_hmax;
  int m_xAxisType;
  int m_debug;
  int m_nfields;
  int m_nbins[2];
  float m_mins[2];
  float m_maxs[2];
  Float_t m_data[5];

  TDatime m_T0;
};

void arg_error(string msg) {
  cerr << "ERROR:  " << msg << endl;
  cerr << "Use 'ECALrootPlotter -h' for help" << endl;
  exit(1);
}

int main(int argc, char* argv[]) {
  // Parse command line
  program_options::options_description desc("options");
  program_options::options_description visible(
      "Usage:  ECALrootPlotter [options] [Plot Type] [Output Format] [Output File] \noptions");
  visible.add_options()("time,t", "X axis takes time values")(
      "file,f", program_options::value<string>(), "input file name")(
      "title,T", program_options::value<string>(), "Set plot title")(
      "xtitle,X", program_options::value<string>(), "X axis title")(
      "ytitle,Y", program_options::value<string>(), "Y axis title")("debug", "Print debug information")("help,h",
                                                                                                        "help message");
  program_options::options_description hidden("argument");
  hidden.add_options()("type", program_options::value<string>(), "Type of ROOT plot")(
      "format", program_options::value<string>(), "Output format")(
      "output", program_options::value<string>(), "Output file")("hmin", program_options::value<float>(), "histo_min")(
      "hmax", program_options::value<float>(), "histo_max");
  desc.add(visible).add(hidden);
  program_options::positional_options_description pd;
  pd.add("type", 1);
  pd.add("format", 1);
  pd.add("output", 1);
  pd.add("hmin", 1);
  pd.add("hmax", 1);
  program_options::variables_map vm;
  try {
    program_options::store(program_options::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);
    program_options::notify(vm);
  } catch (const program_options::error& e) {
    cerr << e.what() << endl;
    return 1;
  }

  if (vm.count("help")) {
    cout << visible << endl;
    return 0;
  }

  string type, outputFile, outputFormat;
  string file = "";
  string title = "";
  string xtitle = "";
  string ytitle = "";
  float histo_min = 0;
  float histo_max = 0;
  int axisCode = RootPlot::DEFAULT_AXIS;
  int debug = 0;

  if (vm.count("type")) {
    type = vm["type"].as<string>();
    if (type != "TH1F" && type != "TH2F" && type != "TGraph" && type != "Map" && type != "EBEEMap") {
      cerr << "ERROR:  Plot type " << type << " is not valid." << endl;
      arg_error(
          "Valid types are:\n"
          "  'TH1F'   (1 col data)\n"
          "  'TH2F'   (2 col data)\n"
          "  'TGraph' (2 col data)\n"
          "  'Map'    (2 col data)\n"
          "  'EBEEMap'  (5 col data)\n");
    }
  } else {
    arg_error(
        "type is required.\n"
        "Valid types are:\n"
        "  'TH1F'   (1 col data)\n"
        "  'TH2F'   (2 col data)\n"
        "  'TGraph' (2 col data)\n"
        "  'Map'    (2 col data)\n"
        "  'EBEEMap'  (5 col data)");
  }
  if (vm.count("format")) {
    outputFormat = vm["format"].as<string>();
  } else {
    arg_error("format is required");
  }
  if (vm.count("output")) {
    outputFile = vm["output"].as<string>();
  } else {
    arg_error("output is required");
  }
  if (vm.count("hmin")) {
    histo_min = vm["hmin"].as<float>();
  }
  if (vm.count("hmax")) {
    histo_max = vm["hmax"].as<float>();
  }
  if (vm.count("file")) {
    file = vm["file"].as<string>();
  }

  if (vm.count("time")) {
    axisCode = RootPlot::TIME_AXIS;
  }
  if (vm.count("title")) {
    title = vm["title"].as<string>();
  }
  if (vm.count("xtitle")) {
    xtitle = vm["xtitle"].as<string>();
  }
  if (vm.count("ytitle")) {
    ytitle = vm["ytitle"].as<string>();
  }
  if (vm.count("debug")) {
    debug = 1;
  }

  string path = "";
  if ((int)file.find('/') >= 0) {
    path = file.substr(0, file.rfind('/'));
  }
  outputFile = path + "/" + outputFile;

  // substitute _ with spaces
  size_t t;
  while ((t = title.find('_')) != string::npos)
    title[t] = ' ';
  while ((t = xtitle.find('_')) != string::npos)
    xtitle[t] = ' ';
  while ((t = ytitle.find('_')) != string::npos)
    ytitle[t] = ' ';

  if (debug) {
    cout << "Debug info:    " << endl;
    cout << "  type:        " << type << endl;
    cout << "  format:      " << outputFormat << endl;
    cout << "  output:      " << outputFile << endl;
    cout << "  axisCode:    " << axisCode << endl;
    cout << "  title:       " << title << endl;
    cout << "  xtitle:      " << xtitle << endl;
    cout << "  ytitle:      " << ytitle << endl;
    cout << "  map min:     " << histo_min << endl;
    cout << "  map max:     " << histo_max << endl;
    cout << "  input:       " << file << endl;
  }

  // Read data from stdin
  try {
    RootPlot rootplot(type, outputFormat, outputFile, histo_min, histo_max);
    rootplot.setXAxisType(axisCode);
    rootplot.setTitle(title);
    rootplot.setXTitle(xtitle);
    rootplot.setYTitle(ytitle);
    rootplot.setDebug(debug);

    string line;

    ifstream* finput = (ifstream*)&cin;
    if (file.length() > 0) {
      finput = new ifstream(file.c_str());
    }
    while (getline(*finput, line) && finput->good() && !finput->eof()) {
      rootplot.parseAndFill(line);
    }

    if (finput->bad() || !finput->eof()) {
      cerr << "Input error." << endl;
    }

    finput->close();

    if (file.length() > 0) {
      delete finput;
    }
    rootplot.draw();

  } catch (std::exception& e) {
    cerr << "ERROR:  " << e.what() << endl;
  }
}
