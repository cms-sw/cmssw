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
		comparisonPlots(char* filename, char* outputfilename);
		
		void readTree();
		void plot3x5(TCut Cut);
		void plot3x5Profile(TCut Cut, int nBins);
		
		float arrowSize;
		void Write();
		TFile* fin;
		TFile* output;
		TTree*  data;
		
		
		
	private:
		
		void getMaxMin();
		void getHistMaxMin( TH1* hist, double &max, double &min );
		
		
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
