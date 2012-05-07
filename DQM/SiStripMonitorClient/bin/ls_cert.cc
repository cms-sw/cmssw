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
int numlumis = -1;

int     nlumis     ( string filename ); //get number of run lumisections
string  runnum_str ( string filename ); //read the run number, return in string
void    ls_cert( float threshold_pixel , float threshold , string filename ) ;

int main(int argc , char *argv[]) {

  if(argc==4) {
    char* cpixel = argv[1];
    char* cthr = argv[2];
    char* filename = argv[3];

    float threshold_pixel = 0;
    sscanf(cpixel,"%f",&threshold_pixel);
    float threshold = 0;
    sscanf(cthr,"%f",&threshold);

    std::cout << "ready to run ls_cert: pixel thr  " << threshold_pixel 
	      << " threshold " << threshold
	      << " filename " << filename << std::endl;

    ls_cert(threshold_pixel,threshold,filename);

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }
  return 0;

}

void    ls_cert( float threshold_pixel , float threshold , string filename ) 
{
  void ls_cert_type ( string iDir , float threshold , string filename , vector <string>& , vector<pair<string,vector<float> > >& , vector<pair<string,vector<float> > >& , vector<pair<string,vector<float> > >& );
  void cert_plot    ( float threshold_pixel , float threshold , string filename , 
		      vector <string>& , vector <string>& , vector <string>& , vector<pair<string,vector<float> > >& , vector<pair<string,vector<float> > >& , vector<pair<string,vector<float> > >& );

  //presets
  numlumis = -1;

  //certifications
  vector <string> cert_strip;
  vector <string> cert_track;
  vector <string> cert_pixel;
  
  //good lumisections
  vector < pair < string,vector <float> > > gLS_strip;
  vector < pair < string,vector <float> > > gLS_track;
  vector < pair < string,vector <float> > > gLS_pixel;

  //bad lumisections
  vector < pair < string,vector <float> > > bLS_strip;
  vector < pair < string,vector <float> > > bLS_track;
  vector < pair < string,vector <float> > > bLS_pixel;

  //missing lumisections
  vector < pair < string,vector <float> > > mLS_strip;
  vector < pair < string,vector <float> > > mLS_track;
  vector < pair < string,vector <float> > > mLS_pixel;

  ls_cert_type( "SiStrip"  , threshold       , filename , cert_strip , gLS_strip , bLS_strip , mLS_strip );
  ls_cert_type( "Tracking" , threshold       , filename , cert_track , gLS_track , bLS_track , mLS_track );
  ls_cert_type( "Pixel"    , threshold_pixel , filename , cert_pixel , gLS_pixel , bLS_pixel , mLS_pixel );

  ofstream outfile;
  string namefile = "Certification_run_" + runnum_str( filename ) + ".txt";
  outfile.open(namefile.c_str());
  outfile << "Lumisection Certification (GOOD: >= " << threshold_pixel << " [Pixel]; >= " << threshold << " [SiStrip,Tracking] " 
	  << ", otherwise BAD):" << endl << endl;
  outfile << "GOOD Lumisections:" << endl;
  char line[200];
  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( line, " Pixel    %*sSummary: %s" , 13 , cert_pixel[ityp].c_str() , gLS_pixel[ityp].first.c_str() );
      outfile << line << endl;
    }
  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( line, " SiStrip  %*sSummary: %s" , 13 , cert_strip[ityp].c_str() , gLS_strip[ityp].first.c_str() );
      outfile << line << endl;
    }
  for (int ityp = 0; ityp < 1; ityp++)
    {
      sprintf( line, " Tracking %*sSummary: %s" , 13 , cert_track[ityp].c_str() , gLS_track[ityp].first.c_str() );
      outfile << line << endl;
    }

  outfile << "\nBAD Lumisections:" << endl;
  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( line, " Pixel    %*sSummary: %s" , 13 , cert_pixel[ityp].c_str() , bLS_pixel[ityp].first.c_str() );
      outfile << line << endl;
    }
  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( line, " SiStrip  %*sSummary: %s" , 13 , cert_strip[ityp].c_str() , bLS_strip[ityp].first.c_str() );
      outfile << line << endl;
    }
  for (int ityp = 0; ityp < 1; ityp++)
    {
      sprintf( line, " Tracking %*sSummary: %s" , 13 , cert_track[ityp].c_str() , bLS_track[ityp].first.c_str() );
      outfile << line << endl;
    }
  
  outfile << "\nMISSING Lumisections:" << endl;
  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( line, " Pixel    %*sSummary: %s" , 13 , cert_pixel[ityp].c_str() , mLS_pixel[ityp].first.c_str() );
      outfile << line << endl;
    }
  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( line, " SiStrip  %*sSummary: %s" , 13 , cert_strip[ityp].c_str() , mLS_strip[ityp].first.c_str() );
      outfile << line << endl;
    }
  for (int ityp = 0; ityp < 1; ityp++)
    {
      sprintf( line, " Tracking %*sSummary: %s" , 13 , cert_track[ityp].c_str() , mLS_track[ityp].first.c_str() );
      outfile << line << endl;
    }
  
  outfile.close();
  cout << "Lumisection Certification summary saved in " << namefile << endl;

  cert_plot ( threshold_pixel , threshold , filename , cert_strip , cert_track , cert_pixel , gLS_strip , gLS_track , gLS_pixel );
}

void ls_cert_type(string iDir, float threshold, string filename, vector <string>& cert, vector<pair<string,vector<float> > >& gLS, vector<pair<string,vector<float> > >& bLS, vector<pair<string,vector<float> > >& mLS ) 
{  
  void Cleaning(vector<int> &);
  string ListOut(vector<int> &);

  bool debug = false;
  string run = runnum_str( filename );
  if (debug) cout << filename.c_str() << endl;

  TDirectory* topDir; 
  vector<float> ls;
  
  TFile* file = TFile::Open(filename.c_str());
  if (!file->IsOpen()) {
    cerr << "Failed to open " << filename << endl; 
    return;
  }

  string dir = "DQMData/Run " + run + "/" + iDir;
  topDir = dynamic_cast<TDirectory*>( file->Get(dir.c_str()));
  topDir->cd();
  if (debug) cout << topDir->GetTitle() << endl;
  //
  // Reading the LS directory    
  //
  TIter next(topDir->GetListOfKeys());
  TKey *key;
  while  ( (key = dynamic_cast<TKey*>(next())) ) {
    string clName(key->GetClassName());
    if (clName == "TDirectoryFile") {
      TDirectory *curr_dir = dynamic_cast<TDirectory*>(key->ReadObj());
      string name = curr_dir->GetName();
      if (name == "Run summary") continue;
      name = name.substr(name.find("-")+1);
      float temp1 = atof(name.c_str()); 
      ls.push_back(temp1);
      //cout << temp1 << endl;
    }
  }
  sort(ls.begin(),ls.end());   
  int vecsize = ls.size();

  //
  // Definition of vectors for LS certification
  //
  Float_t * lsd = new Float_t[vecsize];
  
  Float_t ** v = new Float_t*[4];
  for(int k=0;k<4;k++) {
    v[k] = new Float_t[vecsize];
  }
  //string certflag[4] = {"CertificationSummary","DAQSummary","DCSSummary","reportSummary"};
  //string certflagPrint[4] = {"Certification","DAQ","DCS","DQM"};
  string certflag[4] = {"DAQSummary","DCSSummary","reportSummary","CertificationSummary"};
  string certflagPrint[4] = {"DAQ","DCS","DQM","Certification"};

  int smax = 2;
  if ( iDir == "SiStrip" || iDir == "Pixel" ) smax = 4;
  
  if ( iDir == "Tracking" )
    {
      certflag[0] = "CertificationSummary";
      certflagPrint[0] = "Certification";
      certflag[1] = "reportSummary";
      certflagPrint[1] = "DQM";
    }

  for (int icert_type = 0; icert_type < smax; icert_type++)
    {
      cert.push_back( certflagPrint[icert_type] );
    }

  if (debug) cout << gDirectory->GetName() << endl;
  
  for (int i=0; i < vecsize; i++){
    stringstream lsdir;
    lsdir << dir << "/By Lumi Section " << ls[i] <<"-"<<ls[i]<<"/EventInfo";
    if (debug) cout << lsdir.str().c_str() << endl;
    float templs = ls[i];
    lsd[i] = templs;
    TDirectory *tempDir = dynamic_cast<TDirectory*>( file->Get(lsdir.str().c_str()));
    tempDir->cd();
    int j = 0;      
    TIter nextTemp(tempDir->GetListOfKeys());
    TKey *keyTemp;
    while  ( (keyTemp = dynamic_cast<TKey*>(nextTemp())) ) {
      float tempvalue = -1.;
      string classname(keyTemp->GetClassName());
      if (classname=="TObjString" ){
	string sflag = keyTemp->GetName();
	string tempname = sflag.substr(sflag.find("f=")+2);
	size_t pos1 = tempname.find("<");
	size_t pos2 = sflag.find_first_of(">");
	string detvalue = tempname.substr(0,pos1);
	string typecert = sflag.substr(1,pos2-1);
	if (debug) cout << typecert.c_str() << endl;
	tempvalue = atof(detvalue.c_str());
	
	for (j=0; j<smax; j++){
	  //	    cout << "j " << j << endl;
	  if ( strstr(typecert.c_str(),certflag[j].c_str())!=NULL)
	    v[j][i] = tempvalue;
	  if (debug) cout << "Entering value " << tempvalue << " " << v[j][i] << " for " << certflag[j].c_str() << endl;
	}
	j = j + 1;
      }
    }
  }

  int nLS_run = nlumis ( filename );

  for (int iS = 0; iS < smax; iS++)
    {
      
      vector<int> goodLS;
      vector<int> badLS;
      vector<int> missingLS;
      vector<float> allLSthr;

      //loop over all available lumisections and fill good/bad lists
      for (int iLS = 0; iLS < vecsize; iLS++)
	{
	  if ( v[iS][iLS] >= threshold ) 
	    goodLS.push_back(lsd[iLS]);
	  else					
	    if ( v[iS][iLS] > -1 ) //protect from flagging non-tested LS as bad
	      badLS.push_back(lsd[iLS]);
	}

      int last_idx = 0;
      for (int i_ls = 1; i_ls <= nLS_run; i_ls++)
	{
	  for (int j = last_idx; j < vecsize; j++)
	    {
	      if ( lsd[j] == i_ls ) 
		{
		  last_idx = j+1;
		  if ( v[iS][j] == 0 ) allLSthr.push_back(0.00001);
		  else
		    allLSthr.push_back(v[iS][j]);
		  break;
		}
	      if ( lsd[j] > i_ls )
		{
		  last_idx = j;
		  missingLS.push_back(i_ls);
		  allLSthr.push_back(-1);
		  break;
		}
	    }
	}

      Cleaning( goodLS    );
      Cleaning( badLS     );
      Cleaning( missingLS );

      string goodList    = ListOut( goodLS    );
      string badList     = ListOut( badLS     );
      string missingList = ListOut( missingLS );
      
      //save lumisections for this certification type
      gLS.push_back ( make_pair ( goodList    , allLSthr ) );
      bLS.push_back ( make_pair ( badList     , allLSthr ) );
      mLS.push_back ( make_pair ( missingList , allLSthr ) );      
    }
}

void cert_plot( float threshold_pixel , float threshold , string filename , vector <string>& cert_strip , vector <string>& cert_track , 
		vector <string>& cert_pixel , vector<pair<string,vector<float> > >& LS_strip , vector<pair<string,vector<float> > >& LS_track , 
		vector<pair<string,vector<float> > >& LS_pixel )
{
  int nLumiSections = nlumis( filename );

  char plottitles[200];
  sprintf( plottitles , "Lumisection Certification: Run %s;Luminosity Section;" , runnum_str( filename ).c_str() );
  TH2D *cert_plot = new TH2D( "cert_plot" , plottitles , nLumiSections , 1 , nLumiSections + 1 , 5 , 1 , 6 );
  cert_plot->SetStats(0);
  char label[100];
  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( label , "SiStrip %s" , cert_strip[ityp].c_str() );
      cert_plot->GetYaxis()->SetBinLabel( 5 - ityp , label );

      for (unsigned int idx = 0; idx <  LS_strip[ityp].second.size() ; idx++)
        if ( LS_strip[ityp].second[idx] > -1 ) cert_plot->SetBinContent(idx+1, 5 - ityp, LS_strip[ityp].second[idx]);
    }
  for (int ityp = 0; ityp < 1; ityp++)
    {
      sprintf( label , "Tracking %s" , cert_track[ityp].c_str() );
      cert_plot->GetYaxis()->SetBinLabel( 1 - ityp , label );
      for (unsigned int idx = 0; idx <  LS_track[ityp].second.size() ; idx++)
        if ( LS_track[ityp].second[idx] > -1 ) cert_plot->SetBinContent(idx+1, 1 - ityp, LS_track[ityp].second[idx]);
    }

  const Int_t colNum = 20; // defining of a new palette
  Int_t palette[colNum];
  float rgb[colNum][3];
  int col_thr = colNum * threshold;
  for (Int_t i=0; i<colNum; i++)
    {
      if ( i >= col_thr )
        {
          // green
          rgb[i][0] = 0.00;
          rgb[i][1] = 0.80;
          rgb[i][2] = 0.00;
        }
      else
        {
          // red to yellow   //yellow red
          rgb[i][0] = 0.80 + ( 0.98 - 0.80 ) / ( col_thr - 1 ) * i ;  //0.98   0.80
          rgb[i][1] = 0.00 + ( 0.79 - 0.00 ) / ( col_thr - 1 ) * i ;  //0.79   0.00
          rgb[i][2] = 0.00;  //0.00
        }

      palette[i] = 9001+i;

      TColor *color = gROOT->GetColor(9001+i);
      if (!color) color = new TColor(9001 + i, 0, 0, 0, "");
      color->SetRGB(rgb[i][0], rgb[i][1], rgb[i][2]);
    }
  gStyle->SetPalette(colNum,palette);
  gROOT->SetStyle("Plain");

  TCanvas *cc = new TCanvas( "name" , "title" , 1000 , 600 );
  cert_plot->Draw("colz");
  gPad->SetLeftMargin(0.17);
  string plotfilename = "Certification_run_" +  runnum_str( filename ) + ".png";
  cc->Print( plotfilename.c_str() );


  //PIXEL plot

  TH2D *cert_plot_pixel = new TH2D( "cert_plot_pixel" , plottitles , nLumiSections , 1 , nLumiSections + 1 , 4 , 1 , 5 );
  cert_plot_pixel->SetStats(0);

  for (int ityp = 0; ityp < 4; ityp++)
    {
      sprintf( label , "Pixel %s" , cert_pixel[ityp].c_str() );
      cert_plot_pixel->GetYaxis()->SetBinLabel( 4 - ityp , label );

      for (unsigned int idx = 0; idx <  LS_pixel[ityp].second.size() ; idx++)
        if ( LS_pixel[ityp].second[idx] > -1 ) cert_plot_pixel->SetBinContent(idx+1, 4 - ityp, LS_pixel[ityp].second[idx]);
    }

  int col_thr_pixel = colNum * threshold_pixel;

  for (Int_t i=0; i<colNum; i++)
    {
      if ( i >= col_thr_pixel )
        {
          // green
          rgb[i][0] = 0.00;
          rgb[i][1] = 0.80;
          rgb[i][2] = 0.00;
        }
      else
        {
          // red to yellow   //yellow red
          rgb[i][0] = 0.80 + ( 0.98 - 0.80 ) / ( col_thr - 1 ) * i ;  //0.98   0.80
          rgb[i][1] = 0.00 + ( 0.79 - 0.00 ) / ( col_thr - 1 ) * i ;  //0.79   0.00
          rgb[i][2] = 0.00;  //0.00
        }

      palette[i] = 10001+i;

      TColor *color = gROOT->GetColor(10001+i);
      if (!color) color = new TColor(10001 + i, 0, 0, 0, "");
      color->SetRGB(rgb[i][0], rgb[i][1], rgb[i][2]);
    }
  gStyle->SetPalette(colNum,palette);
  gROOT->SetStyle("Plain");

  cert_plot_pixel->Draw("colz");
  //gPad->SetLeftMargin(0.17);
  string plotfilename_pixel = "Certification_run_" +  runnum_str( filename ) + "_pixel.png";
  cc->Print( plotfilename_pixel.c_str() );

  delete cc;
}


int nlumis( string filename )
{
  if ( numlumis > -1 ) 
    return numlumis;

  //TDirectory* topDir;
  vector<float> ls;

  TFile* file = TFile::Open(filename.c_str());
  if (!file->IsOpen()) {
    cerr << "Failed to open " << filename << endl;
    return -1;
  }
  
  string run = runnum_str( filename );

  //check if HIRun or pp run
  bool isHIRun = false;
  if ( filename.find ( "HIRun" ) != string::npos )
    isHIRun = true;

  //valid up to the end of 2011 pp collisions
  if ( !isHIRun )
    {
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
    }
  else
    {
      //valid since 2011 HI running (iLumiSection variable not there anymore)  
      string EventInfoDirHist = "DQMData/Run " + run + "/Info/Run summary/EventInfo/ProcessedLS";
      TH1F* allLS = (TH1F*) file->Get( EventInfoDirHist.c_str() );
      numlumis = allLS->GetEntries() - 1;
      delete allLS;
    }

  return numlumis;
}

string runnum_str( string filename )
{
  return filename.substr(filename.find("_R000")+5, 6);  
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
  
  string strout = "";
  bool rangeset = false;
  for (unsigned int at = 0; at < LSlist.size(); at++)
    {
      if ( LSlist[at] != -1 ) 
	{
	  if ( LSlist[at-1] != -1 && at > 0 ) strout += ",";
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
