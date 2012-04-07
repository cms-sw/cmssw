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

  if(argc==3) {
    char* crun = argv[1];
    char* repro_type = argv[2];

    int run = 0;
    sscanf(crun,"%d",&run);

    std::cout << "ready to check run " << run << " repro type " << repro_type << std::endl;

    check_runcomplete(run,repro_type);

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }
  return 0;

}

void check_runcomplete ( int run , std::string repro_type )
{
  int runflag = read_runflag ( run , repro_type );
  if ( runflag == 0 )
    {
      printf("************************************\n");
      printf("**\n");
      printf("** W A R N I N G: the DQM file with run %i (%s) is not fully archived yet on /afs,\n" , run , repro_type.c_str() );
      printf("** it is strongly recommended that you analyze it later.\n" );
      printf("**\n");
      printf("************************************\n");
    }
}

int read_runflag ( int run , std::string repro_type )
{
  std::string filename;
  int flag = get_filename ( run , repro_type , filename );
  
  if ( flag < 0 )
    {
      //cout << "reading problem" << endl;
      return -1; 
    }

  std::string nrun = filename.substr ( filename.find( "_R000" ) + 5 , 6 );

  TFile *dqmfile = TFile::Open ( filename.c_str() , "READ" );
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
  return isruncomplete;
}


int get_filename  ( int run , std::string repro_type , std::string& filename )
{
  std::stringstream runstr;
  runstr << run;
  
  std::stringstream rundirprefix;
  rundirprefix << "000" << run / 100 << "xx/";

  std::stringstream thisdir;
  thisdir << "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OfflineData/Run2011/" << repro_type.c_str() << "/" << rundirprefix.str();
  
  std::string thisdir2 = thisdir.str();
  DIR *dp;
  
  if ( ( dp = opendir( thisdir2.c_str() ) ) == NULL )
    {
      //cout << "dir " << thisdir2.c_str() << " not found" << endl;
      return -1;
    }

  struct dirent *dirp;

  std::string dqmfile;

  while ( ( dirp = readdir ( dp ) ) != NULL )
    {
      std::string dirfile = std::string ( dirp->d_name );
      if ( 
	  dirfile.find ( "__DQM" ) != std::string::npos &&
	  dirfile.find ( runstr.str() ) != std::string::npos
	  )
	{
	  dqmfile = dirfile;
	  break;
	}
    }

  closedir( dp );

  if ( dqmfile.size() < 10 )
    {
      //cout << "file " << dqmfile << " not found" << endl;
      return -1;
    }
  
  filename = thisdir.str() + dqmfile;
  
  return 0;
}


