#include <cmath>

#include <utility>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

using namespace std;

#include "TROOT.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TNtuple.h"

//#include "gnuplot_i.hpp"
//Gnuplot * g;

// Total loss, loss in one direction is Loss/2 
#define Loss 0.005

// FIXME
#define MinEntries 100

TH1D * line;

#define exMax 10
#define eyMax 15

const char *subName[2] = {"barrel","endcap"};

/****************************************************************************/
void printToFile(TH2F *h2, const char* fileName)
{
  char name[256];

  sprintf(name,"%s.dat",fileName);
  ofstream file(name);

  for(int i = 1; i <= h2->GetNbinsX(); i++)
  {
    for(int j = 1; j <= h2->GetNbinsY(); j++)
      file << " " << h2->GetXaxis()->GetBinLowEdge(i)
           << " " << h2->GetYaxis()->GetBinLowEdge(j)
           << " " << h2->GetBinContent(i,j)
           << endl;

      file << " " << h2->GetXaxis()->GetBinLowEdge(i)
           << " " << h2->GetYaxis()->GetXmax()
           << " 0" << endl;
    file << endl;
  }

  for(int j = 1; j <= h2->GetNbinsY(); j++)
    file << " " << h2->GetXaxis()->GetXmax()
         << " " << h2->GetYaxis()->GetBinLowEdge(j)
         << " 0" << endl;

    file << " " << h2->GetXaxis()->GetXmax()
         << " " << h2->GetYaxis()->GetXmax()
         << " 0" << endl;

  file.close();
}

/****************************************************************************/
class Point 
{
  public:
    float x[2]; // position
    int w;      // point weight
    int n;      // cluster index
};

class Center
{
  public:
    float x[2]; // position
    int n;      // cluster index
    float area, lim[2][2];
};

/****************************************************************************/
class ClusterAnalyzer
{
  public:
	void analyze(const std::string& inputFileName, const std::string& outputFileName);

  private:
    float distance(Center& c, Point& p);
    void getVectors(vector<Point>& points, TH1D * x[2][2]);
    double getFraction(TH1D * line, double fraction);
    void kMeans    (vector<Point>& points, vector<Center>& c);
    void findLimits(vector<Point>& points, vector<Center>& c);
    void printOut  (TH2F * histo, vector<Point>& points, vector<Center>& c, char* flag);
    void process   (TH2F * histo,          vector<Center>& c, char* flag);
};

/****************************************************************************/
float ClusterAnalyzer::distance(Center& c, Point& p)
{
  float x[2];

  for(int d=0; d<2; d++)
    x[d] = fabs(p.x[d] - c.x[d]);

  return x[0]*x[0] + x[1]*x[1];
}

/****************************************************************************/
void ClusterAnalyzer::getVectors(vector<Point>& points, TH1D * x[2][2])
{
  for(int n = 0 ; n < 2; n++)
  for(int d = 0 ; d < 2; d++)
  {
    char name[256]; sprintf(name,"%d_%d",n,d);
    x[n][d] = (TH1D *) line->Clone(name);
  }

  for(vector<Point>::iterator ip = points.begin();
                              ip!= points.end(); ip++)
    for(int d=0; d<2; d++)
      x[ip->n][d]->Fill(ip->x[d], ip->w);
}

/****************************************************************************/
double ClusterAnalyzer::getFraction(TH1D * line, double fraction)
{
  double n = line->Integral();
  double s = 0;

  for(int i = 0 ; i <= line->GetNbinsX()+1; i++)
  {
    s += line->GetBinContent(i);
    if(s > n * fraction)
    {
      return line->GetXaxis()->GetBinUpEdge(i) - 
             (s - n * fraction) / line->GetBinContent(i) *
             line->GetXaxis()->GetBinWidth(i);
    } 
  }

  cerr << " exit!" << endl;
  return 0.;
}

/****************************************************************************/
void ClusterAnalyzer::kMeans(vector<Point>& points, vector<Center>& c)
{
  int changes;

  do
  {
    changes = 0;

    // Sort point into clusters
    for(vector<Point>::iterator ip = points.begin();
                                ip!= points.end(); ip++)
    {
      int n;
      if(distance(c[0],*ip) < distance(c[1],*ip)) n = 0; else n = 1;

      if(n != ip->n)
      { ip->n = n; changes++; }
    } 

    if(changes != 0)
    {
      // Re-compute centers
      TH1D * x[2][2];
      getVectors(points, x);

      for(int n=0; n<2; n++) // which cluster
      for(int d=0; d<2; d++) // which direction
      {
        c[n].n = int(x[n][d]->Integral() + 0.5);

        if(x[n][d]->Integral() > 0)
          c[n].x[d] = getFraction(x[n][d], 0.5);
      }
    }
  }
  while(changes != 0);
}

/****************************************************************************/
void ClusterAnalyzer::findLimits(vector<Point>& points, vector<Center>& c)
{
  TH1D * x[2][2];
  getVectors(points, x);

  for(int b=0; b<2; b++) // branch
  {
    for(int d=0; d<2; d++) // direction
    if(x[b][d]->Integral() > 0)
    {
      double limits[2] = {0, 0};
      double dmin = 1e+9;

      for(double f = (Loss/2)/100; f < Loss/2 - (Loss/2)/200; f+=(Loss/2)/100)
      {
        double x0 = getFraction(x[b][d],               f );
        double x1 = getFraction(x[b][d], 1 - (Loss/2 - f));

        if(fabs(x1 - x0) < dmin)
        {
          limits[0] = x0;
          limits[1] = x1;

          dmin = fabs(x1 - x0);
        }
      }

      c[b].lim[d][0] = limits[0];
      c[b].lim[d][1] = limits[1];
//      c[b].lim[d][0] = getFraction(x[b][d],     Loss/2);
//      c[b].lim[d][1] = getFraction(x[b][d], 1 - Loss/2);
    }

    // Area
    c[b].area = (c[b].lim[0][1] - c[b].lim[0][0]) *
                (c[b].lim[1][1] - c[b].lim[1][0]);
  }
}

/****************************************************************************/
void ClusterAnalyzer::printOut
  (TH2F * histo, vector<Point>& points, vector<Center>& c, char* flag)
{
  char fileName[256];
  ofstream outFile;

  //////////////////////////////////
  sprintf(fileName,"../out/points_%s",flag);
  printToFile(histo, fileName);

  //////////////////////////////////
  sprintf(fileName,"../out/limits_%s.par",flag);
  outFile.open(fileName);

  char label[128];
  if(histo->Integral() >= MinEntries)
    sprintf(label," ");
  else
    sprintf(label,"Too few entries, not used");
  

  for(int b=0; b<2; b++)
  for(int d=0; d<2; d++)
  {
    outFile << " l" << b << d << "=" << c[b].lim[d][0] << endl;
    outFile << " h" << b << d << "=" << c[b].lim[d][1] << endl;
  }

  outFile.close(); 

  //////////////////////////////////
  sprintf(fileName,"../out/centers_%s.dat",flag);
  outFile.open(fileName);

  for(int b=0; b<2; b++)
    outFile << " " << c[b].x[0] << " " << c[b].x[1] << endl;

  outFile.close();

  //////////////////////////////////
//  *g << "call '../gnu/pixelShape.gnu' '" << flag << "' '" << label << "'\n";
}

/****************************************************************************/
void ClusterAnalyzer::process(TH2F * histo, vector<Center>& c,
                              char* flag)
{
  vector<Point> points;

  for(int i = 1 ; i <= histo->GetNbinsX(); i++)
  for(int j = 1 ; j <= histo->GetNbinsY(); j++)
    if(histo->GetBinContent(i,j) > 0)
    {
      Point p;
      p.x[0] = histo->GetXaxis()->GetBinCenter(i);
      p.x[1] = histo->GetYaxis()->GetBinCenter(j);
      p.w    = int(histo->GetBinContent(i,j) + 0.5);

      points.push_back(p);
    }

  line = histo->ProjectionY("_py",0,0);
  line->Reset();

  kMeans    (points,c);
  findLimits(points,c);
  printOut  (histo, points,c, flag);
}

/****************************************************************************/
void ClusterAnalyzer::analyze(const std::string& inputFileName, const std::string& outputFileName)
{
  // Open file
  //TFile resFile("../data/clusterShape.root","read");
  TFile resFile(inputFileName.c_str(), "READ");

  //ofstream outFile("../res/pixelShape.par");
  ofstream outFile(outputFileName.c_str());
//  ofstream mapFile("../res/pixelShape.map");
  char flag[256];

  for(int is=0; is<2; is++)
  {
    for(int ix=  0; ix<= exMax; ix++)
    {
      cerr << " " << subName[is] << " dx=" << ix << " ";
  
      for(int iy=0; iy<=eyMax; iy++)
      {
        char histName[256];
        sprintf(histName,"hspc_%d_%d_%d",is,ix,iy);
        TH2F * histo = (TH2F *) resFile.Get(histName);
  
        cerr << ".";
      
        // Initial guess
        vector<Center> c(2);
        c[0].n = 0;
        c[0].x[0] = ix+1;
        c[0].x[1] = iy+1;
    
        c[1].n = 1;
        c[1].x[0] =-ix-1;
        c[1].x[1] =-iy-1;
  
        sprintf(flag,"%s_%d_%d", subName[is],ix,iy);
      
        process(histo,c, flag);
  
        if(histo->Integral() >= MinEntries)
        {
          // Fix barrel_0_0
          if(is==0 && ix==0 && iy==0)
          {
            c[0].lim[1][0] = 0.; 
            c[1].lim[1][1] = 0.; 
          }
      
          outFile << is << " " << ix << " " << iy;
      
          for(int b=0; b<2; b++)
          for(int d=0; d<2; d++)
            outFile << " " << c[b].lim[d][0] 
                    << " " << c[b].lim[d][1];
    
          for(int b=0; b<2; b++)
          { 
            if(c[b].area > 0 )
              outFile << " " << c[b].n / c[b].area;
            else
              outFile << " " << 0.;
  
            outFile << " " << c[b].n;
          }
      
          outFile << endl; 
        }

//        mapFile << ix << " " << iy << " " << histo->Integral() << endl;
      }
      outFile << endl;

//      mapFile << ix << " " << eyMax+1 << " " << 0 << endl;
//      mapFile << endl;
    
      cerr << endl;
    }

//    for(int iy=0; iy<=eyMax+1; iy++)
//      mapFile << exMax+1 << " " << iy << " " << 0 << endl;

    outFile << endl;
//    mapFile << endl << endl;
  }

  outFile.close();
//  mapFile.close();
}

/****************************************************************************/
int main(int argc, char* argv[])
{
//  g = new Gnuplot();
  if(argc != 3) {
	std::cout << "Usage: pixelAnalyzer <input_file> <output_file>\nExample: pixelAnalyzer clusterShape.root pixelShape.par\n";
	return 1;
  }
  const std::string inputFileName(argv[1]);
  const std::string outputFileName(argv[2]);
  ClusterAnalyzer theClusterAnalyzer;  
  theClusterAnalyzer.analyze(inputFileName, outputFileName);

  return 0;
}

