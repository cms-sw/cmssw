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
#include <string.h>
#include <utility>
#include <vector>
#include <sstream>
#include <algorithm>
#include <TString.h>
#include <TColor.h>
#include <dirent.h>

using namespace std;

int  read_badmodlist     ( int            , string         , vector < int >& );
void get_difference      ( vector < int > , vector < int > , vector < int >& );
int  get_filename        ( int            , string         , string& );
int  search_closest_run  ( int , string );
void modulediff ( int run_2 , string repro_run2 );

int main(int argc , char *argv[]) {

  if(argc==3) {
    char* crun2 = argv[1];
    char* repro_run2 = argv[2];

    int run2 = 0;
    sscanf(crun2,"%d",&run2);

    std::cout << "ready to run modulediff " << run2 << " repro run " << repro_run2 << std::endl;

    modulediff(run2,repro_run2);

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }
  return 0;

}

void modulediff ( int run_2 , string repro_run2 )
{
  vector < int > badmodlist_run1;
  vector < int > badmodlist_run2;
  vector < int > modules_recovered;
  vector < int > modules_malformed;
  int run_1;
  string repro_run1;

  int res = search_closest_run ( run_2 , repro_run2 );
  if ( res > 0 )
    {
      run_1 = res;
      repro_run1 = repro_run2;
    }
  else 
    {
      printf("closest run not found, exiting.\n");
      return;
    }
  

  int flag1 = read_badmodlist ( run_1 , repro_run1 , badmodlist_run1 );
  int flag2 = read_badmodlist ( run_2 , repro_run2 , badmodlist_run2 );

  if ( flag1 < 0 || flag2 < 0 )
    {
      cout << "Error: file not found." << endl;
      return;
    }

  get_difference ( badmodlist_run1 , badmodlist_run2 , modules_recovered );  
  get_difference ( badmodlist_run2 , badmodlist_run1 , modules_malformed );  

  //save into file
  ofstream outfile;
  string namefile = "modulediff_emailbody.txt";
  outfile.open(namefile.c_str());

  outfile << "Recovered modules in run " << run_2 << ": " << (int)modules_recovered.size() << endl;
  outfile << "New bad modules in run   " << run_2 << ": " << (int)modules_malformed.size() << endl;
  outfile << "Using reference run " << run_1 << endl << endl;

  outfile << "Recovered modules in run " << run_2 << ":" << endl;
  if ( modules_recovered.size() == 0 ) 
    outfile << " -" << endl;
  for ( unsigned int i = 0; i < modules_recovered.size() ; i++ )
    outfile << " " << modules_recovered[ i ] << endl;
  
  outfile << "New bad modules that appeared in run " << run_2 << ":" << endl;
  if ( modules_malformed.size() == 0 ) 
    outfile << " -" << endl;
  for ( unsigned int i = 0; i < modules_malformed.size() ; i++ )
    outfile << " " << modules_malformed[ i ] << endl;

  outfile.close();

  //create two flat files to run the locatemodule script on later
  
  if ( modules_recovered.size() > 0 )
    {
      ofstream outfile_good;
      outfile_good.open("modulediff_good.txt");
      for ( unsigned int i = 0; i < modules_recovered.size() ; i++ )
	outfile_good << " " << modules_recovered[ i ] << endl;
      outfile_good.close();
    }

  if ( modules_malformed.size() > 0 )
    {
      ofstream outfile_bad;
      outfile_bad.open("modulediff_bad.txt");
      for ( unsigned int i = 0; i < modules_malformed.size() ; i++ )
	outfile_bad << " " << modules_malformed[ i ] << endl;
      outfile_bad.close();
    }
}

void get_difference  ( vector < int > badlist1 , vector < int > badlist2 , vector < int > &difflist )
{
  //check if any element of badlist1 is in badlist2
  for ( unsigned int i1 = 0; i1 < badlist1.size() ; i1++ )
    {
      bool thisrecovered = true;
      for ( unsigned int i2 = 0; i2 < badlist2.size() ; i2++ )
	if ( badlist1[ i1 ] == badlist2[ i2 ] )
	  {
	    thisrecovered = false;
	    break;
	  }
      if ( thisrecovered )
	difflist.push_back ( badlist1[ i1 ] );
    }
}


int read_badmodlist ( int run , string repro_type , vector < int >& badlist )
{
  string filename;
  int flag = get_filename ( run , repro_type , filename );
  
  if ( flag < 0 )
    {
      cout << "reading problem" << endl;
      return -1; 
    }

  vector<string> subdet;
  subdet.push_back("TIB");
  subdet.push_back("TID/side_1"); 
  subdet.push_back("TID/side_2");
  subdet.push_back("TOB");
  subdet.push_back("TEC/side_1");
  subdet.push_back("TEC/side_2");

  string nrun = filename.substr ( filename.find( "_R000" ) + 5 , 6 );

  TFile *dqmfile = TFile::Open ( filename.c_str() , "READ" );
  string topdir = "DQMData/Run " + nrun + "/SiStrip/Run summary/MechanicalView";
  gDirectory->cd(topdir.c_str());
  TDirectory* mec1 = gDirectory;

  for ( unsigned int i=0; i < subdet.size(); i++ )
    {
      string badmodule_dir = subdet[ i ] + "/BadModuleList";
      if ( gDirectory->cd ( badmodule_dir.c_str() ) ) 
	{
	  TIter next ( gDirectory->GetListOfKeys() );
	  TKey *key;

	  while  ( ( key = dynamic_cast<TKey*> ( next() ) ) ) 
	    {
	      string sflag = key->GetName();
	      if ( sflag.size() == 0 ) continue;
	      
	      string detid = sflag.substr ( sflag.find ( "<" ) + 1 , 9 ); 
	      badlist.push_back ( atoi ( detid.c_str() ) );
	    }
	}
      else
	{
	  //cout << "no dir " << badmodule_dir << " in filename " << filename << endl;
	}
      mec1->cd();
    }
  dqmfile->Close();
  return 0;
}


int get_filename  ( int run , string repro_type , string& filename )
{
  stringstream runstr;
  runstr << run;
  
  stringstream rundirprefix;
  rundirprefix << "000" << run / 100 << "xx/";

  stringstream thisdir;
  //thisdir << "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OfflineData/Run2011/" << repro_type.c_str() << "/" << rundirprefix.str();
  thisdir << "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OfflineData/HIRun2011/" << repro_type.c_str() << "/" << rundirprefix.str();
  
  string thisdir2 = thisdir.str();
  DIR *dp;
  
  if ( ( dp = opendir( thisdir2.c_str() ) ) == NULL )
    {
      cout << "dir " << thisdir2.c_str() << " not found" << endl;
      return -1;
    }

  struct dirent *dirp;

  string dqmfile;

  while ( ( dirp = readdir ( dp ) ) != NULL )
    {
      string dirfile = string ( dirp->d_name );
      if ( 
	  dirfile.find ( "__DQM" ) != string::npos &&
	  dirfile.find ( runstr.str() ) != string::npos
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


int  search_closest_run  ( int thisrun , string repro_type )
{
  string filename;

  for ( int test_run = thisrun - 1; test_run > thisrun - 1000; test_run-- )
    {
      int res = get_filename  ( test_run , repro_type , filename );
      if ( res == 0 )
	  return test_run;
    }

  return -1;
}
