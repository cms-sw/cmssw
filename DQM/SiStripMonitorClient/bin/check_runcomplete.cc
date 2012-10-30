#include <Riostream.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TKey.h>
#include <TH1.h>
#include <TH2.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TPaveStats.h>
#include <TText.h>
#include <TLegend.h>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <algorithm>
#include <TString.h>
#include <TColor.h>
#include <dirent.h>

#include "check_runcomplete.h"

//using namespace std;

int main(int argc , char *argv[]) {

  if(argc==2) {
    char* cfile = argv[1];

    std::cout << "ready to check file " << cfile << std::endl;

    int returncode = check_runcomplete(cfile);
    if(returncode==0) std::cout << "DQM file is ok" << std::endl;

    return  returncode;

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }

  return -9;

}

int check_runcomplete (std::string filename )
{
  int runflag = read_runflag ( filename );
  if ( runflag == 1 )
    {
      printf("************************************\n");
      printf("**\n");
      printf("** W A R N I N G: the DQM file %s does not exist" , filename.c_str() );
      printf("**\n");
      printf("************************************\n");
    }
  else if ( runflag == 2 )
    {
      printf("************************************\n");
      printf("**\n");
      printf("** W A R N I N G: the DQM file %s is incomplete" , filename.c_str() );
      printf("**\n");
      printf("************************************\n");
    }
  else if ( runflag != 0 )
    {
      printf("************************************\n");
      printf("**\n");
      printf("** W A R N I N G: problems found in the DQM file %s"  , filename.c_str() );
      printf("**\n");
      printf("************************************\n");
    }
  return runflag;
}

int read_runflag (std::string filename )
{
  std::string nrun = filename.substr ( filename.find( "_R000" ) + 5 , 6 );

  TFile *dqmfile = TFile::Open ( filename.c_str() , "READ" );

  if(dqmfile==0) return 1;

  std::string infodir = "DQMData/Run " + nrun + "/Info/Run summary/ProvInfo";
  gDirectory->cd(infodir.c_str());

  TIter next ( gDirectory->GetListOfKeys() );
  TKey *key;

  int isruncomplete = -1;

  while  ( ( key = dynamic_cast<TKey*> ( next() ) ) ) 
    {
      std::string svar = key->GetName();
      if ( svar.size() == 0 ) continue;
      
      if ( svar.find( "runIsComplete" ) != std::string::npos )
	{
	  std::string statusflag = svar.substr ( svar.rfind ( "<" ) -1 , 1 );
	  isruncomplete = atoi ( statusflag.c_str() );
	}
    }

  dqmfile->Close();

  if(isruncomplete == -1) return 3;
  if(isruncomplete == 0) return 2;
  if(isruncomplete == 1) return 0;

  return -8;
}




