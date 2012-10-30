#include <iostream>
#include <stdio.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TKey.h>
#include <TH1.h>
#include <Riostream.h>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <string>
#include <stdint.h>
#include <cstdlib>
#include <cstdio>

#include "listbadmodule.h"

//using namespace std;

int main(int argc , char *argv[]) {

  if(argc==3) {
    char* filename = argv[1];
    char* pclfilename = argv[2];

    std::cout << "ready to prepare list of bad modules " << filename << std::endl;

    listbadmodule(filename,pclfilename);

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }
  return 0;

}


void listbadmodule(std::string filename, std::string pclfilename) {

  int debug = 1;

  // extract fully bad modules from PCLBadComponents txt file

  std::set<unsigned int> pclbadmods;

  ifstream pclfile(pclfilename);

  char line[400];
  unsigned int pcldetid;
  char sixapvs[]="1 1 1 1 1 1";
  char fourapvs[]="1 1 x x 1 1";

  while(pclfile.getline(line,400)) {
    if(strstr(line,sixapvs) || strstr(line,fourapvs)) {
      stringstream linestream;
      linestream << line;
      linestream >> pcldetid;
      //      std::cout << pcldetid << endl;
      pclbadmods.insert(pcldetid);
    }
  }

  std::vector<std::string> subdet;
  subdet.push_back("TIB");
  subdet.push_back("TID/side_1"); 
  subdet.push_back("TID/side_2");
  subdet.push_back("TOB");
  subdet.push_back("TEC/side_1");
  subdet.push_back("TEC/side_2");

  std::string nrun = filename.substr(filename.find("_R000")+5, 6);
  int fileNum = atoi(nrun.c_str()); 
  cout << " ------   Run " << fileNum << endl;
  
  ofstream outfile;
  std::string namefile;
  namefile = "QualityTest_run" + nrun + ".txt";
  outfile.open(namefile.c_str());
 
  TFile *myfile = TFile::Open(filename.c_str());
  if (debug == 1){	  
    std::cout <<" Opened "<< filename << std::endl; 
  }
  std::string topdir = "DQMData/Run " + nrun + "/SiStrip/Run summary/MechanicalView";
  gDirectory->cd(topdir.c_str());
  TDirectory* mec1 = gDirectory;

  //get the summary first                                                                                                                                    
  vector <int> nbadmod;
  for (unsigned int i=0; i < subdet.size(); i++){
    int nbad = 0;
    string badmodule_dir = subdet[i] + "/BadModuleList";
    if (gDirectory->cd(badmodule_dir.c_str())){
      TIter next(gDirectory->GetListOfKeys());
      TKey *key;
      while  ( (key = dynamic_cast<TKey*>(next())) ) {
        string sflag = key->GetName();
        if (sflag.size() == 0) continue;
        nbad++;
      }
    }
    nbadmod.push_back(nbad);
    mec1->cd();
  }

  outfile << "Number of bad modules in total excluding PCL-only bad modules:" << std::endl;
  outfile << "--------------------------------------------------------------" << std::endl;
  outfile << subdet.at(0) << ": " << nbadmod.at(0) << std::endl;
  outfile << subdet.at(1) << ": " << nbadmod.at(1) << std::endl;
  outfile << subdet.at(2) << ": " << nbadmod.at(2) << std::endl;
  outfile << subdet.at(3) << ": " << nbadmod.at(3) << std::endl;
  outfile << subdet.at(4) << ": " << nbadmod.at(4) << std::endl;
  outfile << subdet.at(5) << ": " << nbadmod.at(5) << std::endl;
  outfile << "-------------------------------" << std::endl;

  outfile << std::endl
          << "List of bad modules per partition:" << std::endl;
  outfile << "----------------------------------" << std::endl;

  std::set<unsigned int>::const_iterator pclbadmod = pclbadmods.begin();

  for (unsigned int i=0; i < subdet.size(); i++){
    std::string badmodule_dir = subdet[i] + "/BadModuleList";
    outfile << " " << endl;
    outfile << "SubDetector " << subdet[i] << endl;
    outfile << " " << endl;
    cout << badmodule_dir.c_str() << endl;
    if (gDirectory->cd(badmodule_dir.c_str())){
    //
    // Loop to find bad module for each partition
    //
      TIter next(gDirectory->GetListOfKeys());
      TKey *key;
      
      while  ( (key = dynamic_cast<TKey*>(next())) ) {
	std::string sflag = key->GetName();
	if (sflag.size() == 0) continue;
	std::string detid = sflag.substr(sflag.find("<")+1,9); 
	size_t pos1 = sflag.find("/");
	sflag = sflag.substr(sflag.find("<")+13,pos1-2);
	int flag = atoi(sflag.c_str());	
	sscanf(detid.c_str(),"%u",&pcldetid);
	// the following loop add modules which are bad only for the PCL to the list. 
	// It requires that the bad modules from QT are sorted as the one of the PCL
	while(pclbadmod!= pclbadmods.end() && pcldetid > *pclbadmod) {
	  outfile << "Module " << *pclbadmod << " PCLBadModule " << std::endl;
	  pclbadmod++;
	}
	std::string message;
	message = "Module " + detid;
	if (((flag >> 0) & 0x1) > 0) message += " Fed BadChannel : ";
	if (((flag >> 1) & 0x1) > 0) message += " # of Digi : ";  
	if (((flag >> 2) & 0x1) > 0) message += " # of Clusters :";
	if (((flag >> 3) & 0x1) > 0) message += " Excluded FED Channel ";
	if (((flag >> 4) & 0x1) > 0) message += " DCSError "; 
	if (pclbadmods.find(pcldetid) != pclbadmods.end()) {
	  message += " PCLBadModule ";
	  pclbadmod = pclbadmods.find(pcldetid);
	  pclbadmod++;
	}
	outfile << message.c_str() << std::endl;
	
      }
    }
    mec1->cd();
  }
  myfile->Close();
  outfile.close();
}



