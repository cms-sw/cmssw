//#include "TMultiGraph.h"
#include "TFile.h"
//#include "TGraph.h"
#include "TCanvas.h"
//#include <vector.h>
//#include "TVectorD.h"
//#include "TVectorD.h"
//#include "TEllipse.h"
//#include "TH1F.h"
//#include "TVector2.h"
#include "TArrow.h"
#include "TPaveText.h"
#include "TFrame.h"
#include "TH1F.h"
//#include "TF1.h"
//#include "TGraph.h"
#include "TLegend.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include "TCut.h"

class comparisonPlots
	{
		
	public:
		comparisonPlots(std::string filename, std::string outputDir, std::string outputFilename = "OUTPUT_comparison.root");
		
		void readTree();
		void plot3x5(TCut Cut, char* dirName, bool savePlot = false, std::string plotName = "plot3x5.eps", bool autolimits = false,int ColorCode = 1);
		void plot3x3Rot(TCut Cut, char* dirName, bool savePlot = false, std::string plotName = "plot3x5.eps", bool autolimits = false,int ColorCode = 1);
		void plot3x5Profile(TCut Cut, char* dirName, int nBins, bool savePlot = false, std::string plotName = "plot3x5Profile.eps", bool autolimits = false,int ColorCode = 1);
		
                void plotTwist(TCut Cut, char* dirName, bool savePlot = false, std::string plotName = "plot3x5.eps", bool autolimits = false,int ColorCode = 1);
                
		float arrowSize;
		void Write();
		TFile* fin;
		TFile* output;
		TTree*  data;
		
		
		
	private:
		
		void getMaxMin();
		void getHistMaxMin( TH1* hist, double &max, double &min, int flag );
		
		std::string _outputDir;
		
		//reading tree
		int id_, level_, sublevel_, mid_, mlevel_, useDetId_, detDim_;
		float x_, y_, z_, r_, phi_, alpha_, beta_, gamma_,eta_;
		float dx_, dy_, dz_, dr_, dphi_, dalpha_, dbeta_, dgamma_;
		
		float maxR, minR;
		float maxZ, minZ;
		float maxPhi, minPhi;
		float maxDR, minDR;
		float maxDZ, minDZ;
		float maxRDPhi, minRDPhi;
		float maxDX, minDX;
		float maxDY, minDY;
		
		
		};
