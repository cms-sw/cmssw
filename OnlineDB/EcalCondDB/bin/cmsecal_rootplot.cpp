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

#include <iostream>
#include <string>
#include <vector>
#include <cfloat>

using namespace std;
using namespace boost;

class RootPlot {
public:
  static const int DEFAULT_AXIS = 111;
  static const int TIME_AXIS = 222;

  RootPlot(string type, string format, string file, float hmin=0., float hmax=0.) 
  {
    m_isInit = 0;
    m_type = type;
    m_outputFormat = format;
    m_outputFile = file+"."+format;
    m_outputRoot = file+".root";
    m_title = "rootplot Plot";
    m_xtitle = "X";
    m_ytitle = "Y";
    m_xAxisType = DEFAULT_AXIS;
    m_debug = 1;
    m_T0 = TDatime(2005,01,01,00,00,00);
    m_hmin = hmin;
    m_hmax = hmax;
  };

  ~RootPlot() {};

  void init()
  {
    m_nbins[0] = m_nbins[1] = 100;
    m_mins[0] = m_mins[1] = FLT_MAX;
    m_maxs[0] = m_maxs[1] = FLT_MIN;
    m_data[0] = 0;
    m_data[1] = 0;
    m_data[2] = 0;
    m_data[3] = 0;

    gROOT->SetStyle("Plain");
    gStyle->SetOptStat(111111);
    gStyle->SetOptFit();
    gStyle->SetPalette(1,0);

    int pCol[2] = { 2, 3 };
    if((m_type == "Map" || m_type == "EBMap") && (TString(m_title).Contains("status")) ) {
      gStyle->SetPalette(2,pCol);
    }

    m_rootfile = new TFile(m_outputRoot.c_str(), "RECREATE");
    m_tree = new TTree("t1", "rootplot tree");

    if (m_type == "TH1F") {
      m_nfields = 1;
      m_tree->Branch("x", &m_data[0], "x/F");
    } else if(m_type == "TH2F") {
      m_nfields = 2;
      m_tree->Branch("x", &m_data[0], "x/F");
      m_tree->Branch("y", &m_data[1], "y/F");
    } else if (m_type == "TGraph") {
      m_nfields = 2;
      m_tree->Branch("x", &m_data[0], "x/F");
      m_tree->Branch("y", &m_data[1], "y/F");
    } else if (m_type == "Map") {
      m_nfields = 2;
      m_tree->Branch("x", &m_data[0], "x/F"); // channel number
      m_tree->Branch("y", &m_data[1], "y/F"); // variable var
    } else if (m_type == "EBMap") {
      m_nfields = 3;
      m_tree->Branch("ism", &m_data[0], "ism/F"); // SM number
      m_tree->Branch("x", &m_data[1], "x/F"); // channel number
      m_tree->Branch("y", &m_data[2], "y/F"); // variable var
    } 

  };

  void setTitle(string title) { m_title = title; }
  void setXTitle(string xtitle) { m_xtitle = xtitle; }
  void setYTitle(string ytitle) { m_ytitle = ytitle; }
  void setDebug(int debug) { m_debug = debug; }
  void setXAxisType(int code) { m_xAxisType = code; }

  void parseAndFill(std::string str)
  {
    if (!m_isInit) {
      this->init();
      m_isInit = 1;
    }

    if (m_debug) { cout << "[data] " << flush; }
    
    typedef boost::tokenizer<boost::escaped_list_separator<char> > tokenizer;
    escaped_list_separator<char> sep('\\', ' ', '\"');
    tokenizer tokens(str, sep);
    float datum;
    int cnt = 0;

    for (tokenizer::iterator tok_iter = tokens.begin();
	 tok_iter != tokens.end(); ++tok_iter) {
      if (cnt > m_nfields) { continue; }

      if (m_debug) {
	cout << "<" << *tok_iter << ">" << flush;
      }
      
      if (m_xAxisType == TIME_AXIS && cnt == 0) {
	TDatime d((*tok_iter).c_str());
	datum = (Float_t)(d.Convert() - m_T0.Convert());
      } else {
	datum = atof((*tok_iter).c_str());
      }

      if(m_type != "EBMap") {
	if (datum < m_mins[cnt]) { m_mins[cnt] = datum; }
	if (datum > m_maxs[cnt]) { m_maxs[cnt] = datum; }
      }

      m_data[cnt] = datum;

      cnt++;
    }

    if (m_debug) { cout << endl; }
    m_tree->Fill();
  };
  
  void draw()
  {
    for (int i=0; i<m_nfields; i++) {
      if (m_mins[i] == m_maxs[i]) {
	m_mins[i] -= m_mins[i]*(0.05);
	m_maxs[i] += m_mins[i]*(0.05);
      }
    }

    if (m_debug) {
      cout << "[draw()]:" << endl;
      cout << "  m_type:          " << m_type << endl;
      cout << "  m_outputFormat:  " <<  m_outputFormat << endl;
      cout << "  m_outputFile:    " <<  m_outputFile << endl;
      cout << "  m_outputRoot:    " <<  m_outputRoot << endl;
      cout << "  m_title:         " <<  m_title << endl;
      cout << "  m_xtitle:        " <<  m_xtitle << endl;
      cout << "  m_ytitle:        " << m_ytitle << endl;
      cout << "  m_xAxisType:     " <<  m_xAxisType << endl;
      cout << "  m_nfields:       " <<  m_nfields << endl;
      cout << "  m_nbins[]:       " <<  m_nbins[0] << " " << m_nbins[1] << endl;
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
    } else if (m_type == "EBMap") {
      this->drawEBMap();
    }
    
    m_isInit = 0;
  };

  void drawTH1F()
  {
    TCanvas c1("c1","rootplot",200,10,600,400);
    c1.SetGrid();

    if((TString(m_title).Contains("status"))) {
      m_mins[0]=-0.001;
      m_maxs[0]=1.001;
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

  void drawTH2F()
  {
    TCanvas c1("c1","rootplot",200,10,600,400);
    c1.SetGrid();

    TH2F* plot = new TH2F("rootplot", m_title.c_str(), m_nbins[0], m_mins[0], m_maxs[0], m_nbins[1], m_mins[1], m_maxs[1]);
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

  void drawTGraph()
  {
    TCanvas c1("c1","rootplot",200,10,600,400);
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

  void drawMap()
  {
    gStyle->SetOptStat(0);

    const Int_t csize = 250;
    TCanvas c1("c1","rootplot",Int_t(85./20.*csize),csize);
    TH2F* plot = new TH2F("rootplot",m_title.c_str(),85,0.0001,85.0001,20,0.0001,20.0001);
    plot->GetXaxis()->SetTitle(m_xtitle.c_str());
    plot->GetYaxis()->SetTitle(m_ytitle.c_str());

    Float_t x, y;
    m_tree->SetBranchAddress("x", &x);
    m_tree->SetBranchAddress("y", &y);

    // now fill the map...
    Int_t n = (Int_t)m_tree->GetEntries();
    for(Int_t i=0; i<n; i++) {
      m_tree->GetEntry(i);
      //      while(x>1700) x-=1700;
      Float_t xmap = Float_t(Int_t(x-1)/20)+1;
      Float_t ymap = Float_t(Int_t(x-1)%20)+1;
      plot->Fill(xmap,ymap,y);
    }

    // draw the map
    plot->SetTitle(m_title.c_str());
    if((TString(m_title).Contains("status"))) {
      plot->SetMinimum(-0.001);
      plot->SetMaximum(1.001);
    }
    else if(!(m_hmin==0 && m_hmax==0)) {
      plot->SetMinimum(m_hmin);
      plot->SetMaximum(m_hmax);
    }

    plot->GetXaxis()->SetTitle("#eta");
    plot->GetYaxis()->SetTitle("#phi");
    plot->GetZaxis()->SetTitle(m_ytitle.c_str());
    plot->GetXaxis()->SetNdivisions(17);
    plot->GetYaxis()->SetNdivisions(4);
    c1.SetGridx();
    c1.SetGridy();
    plot->Draw("colz");

    // and draw the grid upon the map...
    TH2C* labelGrid = new TH2C("labelGrid", "label grid for SM", 85, 0., 85., 20, 0., 20.);
    for(Int_t i=0; i<68; i++){
      Float_t X = (i/4)*5+2;
      Float_t Y = (i%4)*5+2;
      labelGrid->Fill(X,Y,i+1);
    }
    labelGrid->SetMinimum(0.1);
    labelGrid->SetMarkerSize(4);
    labelGrid->Draw("text,same");

    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());
    plot->Write();

  };

  void drawEBMap()
  {
    gStyle->SetOptStat(0);

    const Int_t csize = 400;
    TCanvas c1("c1","rootplot",Int_t(360./170.*csize),csize);
    TH2F* plot = new TH2F("rootplot",m_title.c_str(), 360, 0., 360., 170, -85., 85.);
    plot->GetXaxis()->SetTitle(m_xtitle.c_str());
    plot->GetYaxis()->SetTitle(m_ytitle.c_str());

    Float_t x, y, ism;
    m_tree->SetBranchAddress("ism", &ism);
    m_tree->SetBranchAddress("x", &x);
    m_tree->SetBranchAddress("y", &y);

    // now fill the map...
    Int_t n = (Int_t)m_tree->GetEntries();
    for(Int_t i=0; i<n; i++) {
      m_tree->GetEntry(i);

      Float_t iex = -1;
      Float_t ipx = -1;
      for ( unsigned int i=1; i<=36; i++ ) {
	
	if(i == ism) {

	  Float_t ie = Float_t(Int_t(x-1)/20)+1;       
	  Float_t ip = Float_t(Int_t(x-1)%20)+1;       

	  if ( ism <= 18 ) {
	    iex = ie-1;
	    ipx = 20*(ism-1)+(20-ip);
	  } else {
	    iex = -1*ie;
	    ipx = ip + (ism-19)*20-1;
	  }

	  plot->Fill(ipx,iex,y);  

	}

      }

    }

    // draw the map
    plot->SetTitle(m_title.c_str());
    if((TString(m_title).Contains("status"))) {
      plot->SetMinimum(-0.001);
      plot->SetMaximum(1.001);
    }
    else if(!(m_hmin==0 && m_hmax==0)) {
      plot->SetMinimum(m_hmin);
      plot->SetMaximum(m_hmax);
    }

    plot->GetXaxis()->SetTitle("#phi");
    plot->GetYaxis()->SetTitle("#eta");
    plot->GetZaxis()->SetTitle(m_ytitle.c_str());
    plot->GetXaxis()->SetNdivisions(18, kFALSE);
    plot->GetYaxis()->SetNdivisions(2);
    c1.SetGridx();
    c1.SetGridy();
    plot->Draw("colz");

    // and draw the grid upon the map...
    TH2C* labelGrid = new TH2C("labelGrid", "label grid for SM", 18, 0., 360., 2, -85., 85.);
    for(Int_t sm=1; sm<=36; sm++) {
      int X = (sm<=18) ? sm : sm-18;
      int Y = (sm<=18) ? 2 : 1;
      double posSM = (sm<=18) ? sm : -1*(sm-18);
      labelGrid->SetBinContent(X,Y,posSM);
    }
    labelGrid->SetMarkerSize(2);
    labelGrid->Draw("text,same");

    c1.Print(m_outputFile.c_str(), m_outputFormat.c_str());
    plot->Write();

  };

  void setTimeAxis(TAxis* axis) {
    axis->SetTimeOffset(m_T0.Convert());
    axis->SetTimeDisplay(1);
    axis->SetTimeFormat("#splitline{%d/%m}{%H:%M:%S}");
    axis->SetLabelOffset(0.02);
    axis->SetLabelSize(0.03);
  }
  
private:
  RootPlot() {};  // hidden default constructor

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
  Float_t m_data[4];

  TDatime m_T0;
};

void arg_error(string msg)
{
  cerr << "ERROR:  " << msg << endl;
  cerr << "Use 'rootplot -h' for help" << endl;
  exit(1);
}

int main (int argc, char* argv[])
{
  // Parse command line
  program_options::options_description desc("options");
  program_options::options_description visible("Usage:  rootplot [options] [Plot Type] [Output Format] [Output File]\noptions");
  visible.add_options()
    ("time,t", "X axis takes time values")
    ("title,T", program_options::value<string>(), "Plot title")
    ("xtitle,X", program_options::value<string>(), "X axis title")
    ("ytitle,Y", program_options::value<string>(), "Y axis title")
    ("debug","Print debug information")
    ("help,h", "help message")
    ;
  program_options::options_description hidden("argument");
  hidden.add_options()
    ("type", program_options::value<string>(),"Type of ROOT plot")
    ("format", program_options::value<string>(), "Output format")
    ("output", program_options::value<string>(), "Output file")
    ("hmin", program_options::value<float>(), "histo_min")
    ("hmax", program_options::value<float>(), "histo_max")
    ;
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
  string title = "";
  string xtitle = "";
  string ytitle = "";
  float histo_min = 0;
  float histo_max = 0;
  int axisCode = RootPlot::DEFAULT_AXIS;
  int debug = 0;

  if (vm.count("type")) { 
    type = vm["type"].as<string>(); 
    if (type != "TH1F" && type != "TH2F" && type != "TGraph" && type != "Map" && type != "EBMap" ) {
      cerr << "ERROR:  Plot type " << type << " is not valid." << endl;
      arg_error("Valid types are:\n"
		"  'TH1F'   (1 col data)\n"
		"  'TH2F'   (2 col data)\n"
		"  'TGraph' (2 col data)\n"
		"  'Map'    (2 col data)\n"
		"  'EBMap'  (3 col data)\n"
		); 
    }
  } else {  
   arg_error("type is required.\n"
	      "Valid types are:\n"
	      "  'TH1F'   (1 col data)\n"
	      "  'TH2F'   (2 col data)\n"
	      "  'TGraph' (2 col data)\n"
              "  'Map'    (2 col data)\n"
              "  'EBMap'  (3 col data)"
	      ); 
  }
  if (vm.count("format")) { outputFormat = vm["format"].as<string>(); }
  else { arg_error("format is required"); }
  if (vm.count("output")) { outputFile = vm["output"].as<string>(); }
  else { arg_error("output is required"); }
  if (vm.count("hmin")) {histo_min = vm["hmin"].as<float>(); }
  if (vm.count("hmax")) {histo_max = vm["hmax"].as<float>(); }

  if (vm.count("time")) { axisCode = RootPlot::TIME_AXIS; }
  if (vm.count("title")) {
    title = vm["title"].as<string>();
  }
  if (vm.count("xtitle")) {
    xtitle = vm["xtitle"].as<string>();
  }
  if (vm.count("ytitle")) {
    ytitle = vm["ytitle"].as<string>();
  }
  if (vm.count("debug")) { debug = 1; }
  

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
    
    while (getline(cin, line) && cin.good() && !cin.eof()) {
      rootplot.parseAndFill(line);
    }

    if ( cin.bad() || !cin.eof() ) {
      cerr << "Input error." << endl;
    }

    rootplot.draw();

  } catch (std::exception &e) {
    cerr << "ERROR:  " << e.what() << endl;
  }
}
