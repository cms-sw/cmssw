#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

#include <TDirectory.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH1I.h>


using namespace std;

vector<string> get_entries(const TDirectory & file) {
  vector<string> entries;

  for (int i = 0; i < file.GetListOfKeys()->GetSize(); ++i) {
    TString name( file.GetListOfKeys()->At(i)->GetName() );
    entries.push_back(string((const char *) name));
  }
  sort(entries.begin(), entries.end());
  return entries;
}

void ls(TDirectory & file) {
  vector<string> entries = get_entries(file);
  
  for (unsigned int i = 0; i < entries.size(); ++i)
    cout << entries[i] << endl;
}


void make_plots(TDirectory & file, const string & name, ostream & log = cout) {
  log << "*** " << name << " **********" << endl << endl;
  
  vector<string> step(5);
  step[0] = "Total";
  step[1] = "L1";
  step[2] = "L2";
  step[3] = "L2.5";
  step[4] = "L3";

  TH1I * rates = (TH1I *) file.Get("Event_rates");

  log << name << " cumulative efficiencies: " << endl;
  TH1F * efficiencies = new TH1F("Event_cum_efficiencies", "Event cumulative efficiencies", rates->GetXaxis()->GetNbins(), rates->GetXaxis()->GetXmin(), rates->GetXaxis()->GetXmax());
  for (int i = 1; i <= rates->GetXaxis()->GetNbins(); i++) {
    (*efficiencies)[i] = (double) (*rates)[i] / (double) (*rates)[1];
    log << "\t" << setw(6) << step[i-1] << ": " << setw(6) << (*rates)[i] << " / " << setw(6) << (*rates)[1] << " = " << (*efficiencies)[i] << endl;
  }
  log << endl;
  efficiencies->Draw("");

  log << name << " step-by-step efficiencies: " << endl;
  TH1F * stepeffs = new TH1F("Event_step_effs", "Event step-by-step efficienciess", rates->GetXaxis()->GetNbins(), rates->GetXaxis()->GetXmin(), rates->GetXaxis()->GetXmax());
  (*stepeffs)[1] = 1.;
  for (int i = 2; i <= rates->GetXaxis()->GetNbins(); i++) {
    (*stepeffs)[i] = (double) (*rates)[i] / (double) (*rates)[i-1];
    log << "\t" << setw(6) << step[i-1] << ": " << setw(6) << (*rates)[i] << " / " << setw(6) << (*rates)[i-1] << " = " << (*stepeffs)[i] << endl;
  }
  log << endl;
  stepeffs->Draw("");
}


void plot(TFile & file) {
  ofstream log("log.txt");

  vector<string> path_names = get_entries(file);
  vector<TDirectory *> paths( path_names.size() );

  for (unsigned int i = 0; i < path_names.size(); i++)
    paths[i] = file.GetDirectory(path_names[i].c_str());

  for (unsigned int i = 0; i < path_names.size(); i++)
    make_plots(*paths[i], path_names[i], log);
}
