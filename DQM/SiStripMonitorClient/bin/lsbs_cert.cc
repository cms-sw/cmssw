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

using namespace std;

//global vars
bool debug = false;
int numlumis = -1;

int     nlumis     ( string filename ); //get number of run lumisections
string  runnum_str ( string filename ); //read the run number, return in string
int     getplot    ( string filename , string iDir , string strplot , TH1F& plot ); //read given plot
void    Cleaning   ( vector<int> & );
string  ListOut    ( vector<int> & );
void    vector_AND ( vector<int> & , vector<int> );
void    lsbs_cert( string filename ); 

int main(int argc , char *argv[]) {

  if(argc==2) {
    char* filename = argv[1];

    std::cout << "ready to run lsbs filename " << filename << std::endl;

    lsbs_cert(filename);

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }
  return 0;

}

void    lsbs_cert( string filename ) 
{
  void check_offset ( string filename , string iDir , string plot , float limit_min , float limit_max , vector <int>& );
  void check_sigma  ( string filename , string iDir , string plot , float limit_err , vector <int>& );
  bool check_isgood ( vector<int> & , int ls ); //check if this LS is good

  //presets
  numlumis = -1;

  float limit_x = 0.002; 
  float limit_y = 0.002; 
  float limit_z = 0.5; 
  float limit_dx = 0.002;
  float limit_dy = 0.002;
  float limit_dz = 0.5;
  float limit_errdx = 0.002;
  float limit_errdy = 0.002;
  float limit_errdz = 0.5;


  //LS certification
  vector <int> ls_x_bad;
  vector <int> ls_y_bad;
  vector <int> ls_z_bad;

  vector <int> ls_xsc_bad;
  vector <int> ls_ysc_bad;
  vector <int> ls_zsc_bad;

  vector <int> ls_dx_bad;
  vector <int> ls_dy_bad;
  vector <int> ls_dz_bad;

  vector <int> ls_dxsc_bad;
  vector <int> ls_dysc_bad;
  vector <int> ls_dzsc_bad;

  vector <int> ls_errdx_bad;
  vector <int> ls_errdy_bad;
  vector <int> ls_errdz_bad;

  vector <int> ls_errdxsc_bad;
  vector <int> ls_errdysc_bad;
  vector <int> ls_errdzsc_bad;

  vector <int> ls_good;
  vector <int> ls_bad;

  //beamspot vs primary vertex
  check_offset ( filename, "Validation" , "hxLumibased PrimaryVertex-DataBase"          , -limit_x , limit_x , ls_x_bad );
  check_offset ( filename, "Validation" , "hyLumibased PrimaryVertex-DataBase"          , -limit_y , limit_y , ls_y_bad );
  check_offset ( filename, "Validation" , "hzLumibased PrimaryVertex-DataBase"          , -limit_z , limit_z , ls_z_bad );

  //beamspot vs scalers
  check_offset ( filename, "Validation" , "hxLumibased Scalers-DataBase fit"          , -limit_x , limit_x , ls_xsc_bad );
  check_offset ( filename, "Validation" , "hyLumibased Scalers-DataBase fit"          , -limit_y , limit_y , ls_ysc_bad );
  check_offset ( filename, "Validation" , "hzLumibased Scalers-DataBase fit"          , -limit_z , limit_z , ls_zsc_bad );

  check_offset ( filename, "Debug"      , "hsigmaXLumibased PrimaryVertex-DataBase fit" , -limit_dx , limit_dx , ls_dx_bad );
  check_offset ( filename, "Debug"      , "hsigmaYLumibased PrimaryVertex-DataBase fit" , -limit_dy , limit_dy , ls_dy_bad );
  check_offset ( filename, "Debug"      , "hsigmaZLumibased PrimaryVertex-DataBase fit" , -limit_dz , limit_dz , ls_dz_bad );

  check_offset ( filename, "Validation" , "hsigmaXLumibased Scalers-DataBase fit" , -limit_dx , limit_dx , ls_dxsc_bad );
  check_offset ( filename, "Validation" , "hsigmaYLumibased Scalers-DataBase fit" , -limit_dy , limit_dy , ls_dysc_bad );
  check_offset ( filename, "Validation" , "hsigmaZLumibased Scalers-DataBase fit" , -limit_dz , limit_dz , ls_dzsc_bad );

  check_sigma  ( filename, "Debug"      , "hsigmaXLumibased PrimaryVertex-DataBase fit" ,  limit_errdx , ls_errdx_bad );
  check_sigma  ( filename, "Debug"      , "hsigmaYLumibased PrimaryVertex-DataBase fit" ,  limit_errdy , ls_errdy_bad );
  check_sigma  ( filename, "Debug"      , "hsigmaZLumibased PrimaryVertex-DataBase fit" ,  limit_errdz , ls_errdz_bad );
  
  check_sigma  ( filename, "Validation" , "hsigmaXLumibased Scalers-DataBase fit" ,  limit_errdx , ls_errdxsc_bad );
  check_sigma  ( filename, "Validation" , "hsigmaYLumibased Scalers-DataBase fit" ,  limit_errdy , ls_errdysc_bad );
  check_sigma  ( filename, "Validation" , "hsigmaZLumibased Scalers-DataBase fit" ,  limit_errdz , ls_errdzsc_bad );

  //BAD LS only if bad in both histos (wrt PV, Scalers)
  vector_AND ( ls_x_bad , ls_xsc_bad );
  vector_AND ( ls_y_bad , ls_ysc_bad );
  vector_AND ( ls_z_bad , ls_zsc_bad );
  vector_AND ( ls_dx_bad , ls_dxsc_bad );
  vector_AND ( ls_dy_bad , ls_dysc_bad );
  vector_AND ( ls_dz_bad , ls_dzsc_bad );
  vector_AND ( ls_errdx_bad , ls_errdxsc_bad );
  vector_AND ( ls_errdy_bad , ls_errdysc_bad );
  vector_AND ( ls_errdz_bad , ls_errdzsc_bad );

  //good LS = all LS minus BAD LS
  for ( int i = 1; i <= nlumis ( filename ) ; i++ )
    {
      if ( !check_isgood ( ls_x_bad , i ) && !check_isgood ( ls_xsc_bad , i ) ) 
	{
	  ls_bad.push_back ( i );
	  continue;
	}
      else
	if ( !check_isgood ( ls_y_bad , i ) && !check_isgood ( ls_ysc_bad , i ) ) 
	  {
	    ls_bad.push_back ( i );
	    continue;
	  }
	else
	  if ( !check_isgood ( ls_z_bad , i ) && !check_isgood ( ls_zsc_bad , i ) ) 
	    {
	      ls_bad.push_back ( i );
	      continue;
	    }
	  else
	    if ( !check_isgood ( ls_dx_bad , i ) && !check_isgood ( ls_dxsc_bad , i ) ) 
	      {
		ls_bad.push_back ( i );
		continue;
	      }
	    else
	      if ( !check_isgood ( ls_dy_bad , i ) && !check_isgood ( ls_dysc_bad , i ) ) 
		{
		  ls_bad.push_back ( i );
		  continue;
		}
	      else
		if ( !check_isgood ( ls_dz_bad , i ) && !check_isgood ( ls_dzsc_bad , i ) ) 
		  {
		    ls_bad.push_back ( i );
		    continue;
		  }
		else
		  if ( !check_isgood ( ls_errdx_bad , i ) && !check_isgood ( ls_errdxsc_bad , i ) ) 
		    {
		      ls_bad.push_back ( i );
		      continue;
		    }
		  else
		    if ( !check_isgood ( ls_errdy_bad , i ) && !check_isgood ( ls_errdysc_bad , i ) ) 
		      {
			ls_bad.push_back ( i );
			continue;
		      }
		    else
		      if ( !check_isgood ( ls_errdz_bad , i ) && !check_isgood ( ls_errdzsc_bad , i ) ) 
			{
			  ls_bad.push_back ( i );
			  continue;
			}
      
      //check also that LS is not missing!!!
      ls_good.push_back( i );
    }

  std::ofstream outfile;
  string namefile = "Certification_BS_run_" + runnum_str( filename ) + ".txt";
  outfile.open(namefile.c_str());
  outfile << "Lumibased BeamSpot Calibration Certification for run " << runnum_str( filename ) << ":" << endl << endl;

  char line[2000];
  sprintf( line, "    GOOD Lumisections (values within limits): %s" , ListOut( ls_good ).c_str() );
  outfile << line << endl;

  if ( ls_bad.size() > 0 )
    {
      sprintf( line, "    BAD Lumisections (values outside limits): %s" , ListOut( ls_bad ).c_str() );
      outfile << line << endl;

      sprintf( line, "      --- histogram name ---                                --- bad lumisection list(*) ---" );
      outfile << line << endl;
      sprintf( line, "      hxLumibased PrimaryVertex-DataBase (mean):            %s" , ListOut( ls_x_bad ).c_str() );
      if ( ls_x_bad.size() > 0 ) outfile << line << endl;
      sprintf( line, "      hyLumibased PrimaryVertex-DataBase (mean):            %s" , ListOut( ls_y_bad ).c_str() );
      if ( ls_y_bad.size() > 0 ) outfile << line << endl;
      sprintf( line, "      hzLumibased PrimaryVertex-DataBase (mean):            %s" , ListOut( ls_z_bad ).c_str() );
      if ( ls_z_bad.size() > 0 ) outfile << line << endl;
            
      sprintf( line, "      hsigmaXLumibased PrimaryVertex-DataBase fit (mean):   %s" , ListOut( ls_dx_bad ).c_str() );
      if ( ls_dx_bad.size() > 0 ) outfile << line << endl;
      sprintf( line, "      hsigmaYLumibased PrimaryVertex-DataBase fit (mean):   %s" , ListOut( ls_dy_bad ).c_str() );
      if ( ls_dy_bad.size() > 0 ) outfile << line << endl;
      sprintf( line, "      hsigmaZLumibased PrimaryVertex-DataBase fit (mean):   %s" , ListOut( ls_dz_bad ).c_str() );
      if ( ls_dz_bad.size() > 0 ) outfile << line << endl;
      
      sprintf( line, "      hsigmaXLumibased PrimaryVertex-DataBase fit (error):  %s" , ListOut( ls_errdx_bad ).c_str() );
      if ( ls_errdx_bad.size() > 0 ) outfile << line << endl;
      sprintf( line, "      hsigmaYLumibased PrimaryVertex-DataBase fit (error):  %s" , ListOut( ls_errdy_bad ).c_str() );
      if ( ls_errdy_bad.size() > 0 ) outfile << line << endl;
      sprintf( line, "      hsigmaZLumibased PrimaryVertex-DataBase fit (error):  %s" , ListOut( ls_errdz_bad ).c_str() );
      if ( ls_errdz_bad.size() > 0 ) outfile << line << endl;

      sprintf( line, "    (*) also bad in the corresponding 'Scalers-Database fit' histograms" );
      outfile << line << endl;
    }

  outfile.close();
  std::cout << "Lumibased BeamSpot Calibration Certification summary saved in " << namefile << endl;
}

void check_offset ( string filename , string iDir , string plot , float limit_min , float limit_max , vector <int>& badLS ) 
{  
  TH1F checkPlot;
  if ( getplot ( filename , iDir , plot , checkPlot ) < 0 ) return;

  
  //look at each LS, save the bad one
  for ( int i = 1; i <= checkPlot.GetNbinsX() ; i++ )
    {
      float value = checkPlot.GetBinContent( i );
      if ( value < limit_min || value > limit_max )
	{
	  badLS.push_back( (int)checkPlot.GetBinCenter( i ) );
	}
    }
}


void check_sigma ( string filename , string iDir , string plot , float limit_err , vector <int>& badLS ) 
{  
  TH1F checkPlot;
  if ( getplot ( filename , iDir , plot , checkPlot ) < 0 ) return;

  //look at each LS
  for ( int i = 1; i <= checkPlot.GetNbinsX() ; i++ )
    {
      float value = checkPlot.GetBinError( i );
      if ( value > limit_err )
	{
	  badLS.push_back( (int)checkPlot.GetBinCenter( i ) );
	}
    }
  
}

bool check_isgood ( vector<int> & ls_badlist, int ls ) 
{
  //check if this LS is found in the BAD list
  for ( unsigned int i = 0; i < ls_badlist.size() ; i++ )
    {
      if ( ls == ls_badlist[i] ) return false;
    }
  return true;
}

int nlumis( string filename )
{
  if ( numlumis > -1 ) 
    return numlumis;

  TFile* file = TFile::Open(filename.c_str());
  if (!file->IsOpen()) {
    cerr << "Failed to open " << filename << endl;
    return -1;
  }
  
  string run = runnum_str( filename );
  string EventInfoDir = "DQMData/Run " + run + "/SiStrip/Run summary/EventInfo";
  TDirectory *rsEventInfoDir = dynamic_cast<TDirectory*>( file->Get(EventInfoDir.c_str()));
  rsEventInfoDir->cd();
  TIter eiKeys(rsEventInfoDir->GetListOfKeys());
  TKey *eiKey;
  while  ( (eiKey = dynamic_cast<TKey*>(eiKeys())) )
    {
      string classname(eiKey->GetClassName());
      if (classname=="TObjString" )
        {
          string sflag = eiKey->GetName();
          string tempname = sflag.substr(sflag.find("i=")+2);
          size_t pos1 = tempname.find("<");
          size_t pos2 = sflag.find_first_of(">");
          string detvalue = tempname.substr(0,pos1);
          string numlumisec = sflag.substr(1,pos2-1);
          if ( numlumisec.c_str() == (string)"iLumiSection" )
            {
              numlumis = atoi( detvalue.c_str() );
              break;
            }
        }
    }

  return numlumis;
}

string runnum_str( string filename )
{
  return filename.substr(filename.find("_R000")+5, 6);  
}

int getplot( string filename , string iDir , string strplot , TH1F& plot )
{
  string run = runnum_str( filename );
  if (debug) std::cout << filename.c_str() << endl;
  
  TFile* file = TFile::Open(filename.c_str());
  if (!file->IsOpen()) {
    cerr << "Failed to open " << filename << endl; 
    return -1;
  }

  string dir = "DQMData/Run " + run + "/AlcaBeamMonitor/Run summary/" + iDir;

  file->cd( dir.c_str() );
  
  string theplot = strplot + ";1";
  TH1F* thisplot;
  gDirectory->GetObject ( theplot.c_str() , thisplot );

  if ( !thisplot )
    {
      std::cout << "Error: plot " << dir << "/" << theplot.c_str() << " not found!" << endl;
      return -2;
    }

  plot = *thisplot;
  thisplot = NULL;
  delete thisplot;
  
  return 0;  
}

void Cleaning( vector<int> &LSlist)
{
  if ( LSlist.size() == 0 ) return;

  //cleaning: keep only 1st and last lumisection in the range
  int refLS = LSlist[0];
  for (unsigned int at = 1; at < LSlist.size() - 1; at++) 
    {
      //delete LSnums in between a single continuous range
      if ( refLS + 1 == LSlist[at] && LSlist[at] + 1 == LSlist[at+1] )
        {
          refLS = LSlist[at];
          LSlist[at] = -1;
        }
      else
        {
          refLS = LSlist[at];
        }
    }
}

string ListOut(vector<int> &LSlist)
{
  
  Cleaning( LSlist );

  string strout = "";
  bool rangeset = false;
  for (unsigned int at = 0; at < LSlist.size(); at++)
    {
      if ( LSlist[at] != -1 ) 
        {
          if ( at > 0 && LSlist[at-1] != -1 ) strout += ",";
          stringstream lsnum;
          lsnum << LSlist[at];
          strout += lsnum.str();
          rangeset = false;
        }
      if ( LSlist[at] == -1 && !rangeset )
        {
          strout += "-";
          rangeset = true;
        }
    }

  return strout;
}

void vector_AND ( vector<int> & bad_def, vector<int> bad_sc)
{
  vector <int> temp;

  int def_size = bad_def.size();
  for ( int i = 0; i < def_size; i++ )
    for ( unsigned int j = 0; j < bad_sc.size(); j++ )
      if ( bad_def[ i ] == bad_sc[ j ] )
	{
	  temp.push_back( bad_def[ i ] );
	  break;
	}
  
  bad_def = temp;
}
