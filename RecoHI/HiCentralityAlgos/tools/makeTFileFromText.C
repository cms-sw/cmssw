#include "TFile.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

void makeTFileFromText(char* fname = "bins20_4TeV_CMSSW_3_5_2.root", char* tag = "tag", int nbins = 20){   

   TFile * outf = new TFile(fname,"update");
   TDirectoryFile* dir = (TDirectoryFile*)outf->mkdir(tag);
   CentralityBins* bins = new CentralityBins("HFhitBins","Test tag", nbins);
   bins->table_.reserve(nbins);

   string file = "./bins20_4TeV_CMSSW_3_5_2.txt";
   ifstream in( file.c_str() );
   string line;

   int i = 0;
   while ( getline( in, line ) ) {
      if ( !line.size() || line[0]=='#' ) { continue; }
      istringstream ss(line);
      string binLabel;
      ss>>binLabel
	>>bins->table_[i].n_part_mean
	>>bins->table_[i].n_part_var
	>>bins->table_[i].n_coll_mean
	>>bins->table_[i].n_coll_var
	>>bins->table_[i].b_mean
	>>bins->table_[i].b_var
	>>bins->table_[i].bin_edge;
      bins->table_[i].n_hard_mean = 0;
      bins->table_[i].n_hard_var = 0;
      
      i++;
   }

   bins->Write();
   outf->Write();

}



