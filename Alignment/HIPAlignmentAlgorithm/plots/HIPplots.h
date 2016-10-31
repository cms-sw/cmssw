#include <Riostream.h>
#include <vector>
#include <string>
//#include <stdio.h>
#include "TFile.h"
#include "TH2F.h"
#include "TChain.h"
#include "TCanvas.h"
#include "TFileIter.h"
#include "TStyle.h"
#include "TMath.h"
#include "TCut.h"
#include "TGraph.h"

class HIPplots{
public:
	HIPplots(int IOV, char* path, char* outFile );
	void extractAlignParams( int i, int minHits = 0, int subDet = 0, int doubleSided = 0 );
	void extractAlignShifts( int i, int minHits = 0, int subDet = 0 );
	void plotAlignParams( string ShiftsOrParams, char* plotName = "test.png");
	void plotAlignParamsAtIter( int iter, string ShiftsOrParams, char* plotName = "test.png" );
	void plotHitMap( char *outpath,int subDet,int minHits=0  );
	void extractAlignableChiSquare( int minHits=0, int subDet =0 , int doubleSided = 0 );
	void plotAlignableChiSquare( char *plotName ="testchi2.png",float minChi2n=-1.0);
	void extractSurveyResiduals( int currentPar,int subDet =0);
	void dumpAlignedModules(int nhits=0);
private:
	
	char _path[256];
	char _outFile[256];
	char _inFile_params[256];
	char _inFile_uservars[256];	
	char _inFile_truepos[256];
	char _inFile_alipos[256];	
	char _inFile_mispos[256];
	char _inFile_HIPalign[256];
	char _inFile_surveys[256];
	enum TKdetector_id{unknown=0,TPBid=1,TPEid=2,TIBid=3,TIDid=4,TOBid=5,TECid=6,ALLid=99};	
        TLegend * MakeLegend (double x1=0.1,double y1=0.1, double x2=0.1,double y2=0.1);	
	int GetNIterations(TDirectory *f,char *tag,int startingcounter=0);
	int GetSubDet( unsigned int id );
        int GetBarrelLayer( unsigned int id );
	void SetMinMax( TH1* h );
	int FindPeaks(TH1F *h1,float *peaklist,const int maxNpeaks,int startbin=-1,int endbin=-1);
	void SetPeakThreshold(float newpeakthreshold);
	void CheckFiles(int &ierr);
	bool CheckFileExistence(char *filename);
	bool CheckHistoRising(TH1D *h);
	float peakthreshold;
	bool plotbadchi2;
};
