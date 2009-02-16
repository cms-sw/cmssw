//#include "compareGeometries.h"
#include <string>
#include <sstream>

#include "TProfile.h"
#include "TList.h"
#include "TNtuple.h"
#include "TString.h"

double arrowSize = 0.0095;
float y_,x_,z_,phi_,r_,dphi_,dr_,dx_,dz_,dy_;
int level_,sublevel_;
char outputDir_[192];

void Plot10Mu(char* text,float X, float Y, float size)
{
	TPaveText* atext = new TPaveText(X,Y,X+size,Y+size);
	atext->AddText(text);
	atext->SetLineColor(0);
	atext->SetFillColor(0);
	atext->SetTextFont(42);
	atext->SetTextSize(0.04);
	atext->Draw();
}


void normArrow(float x, float y, float norm)
{
	// draw 100 mu m arrow if norm = 1
	TArrow* normArrow = new TArrow(x,y,x+norm,y,arrowSize,">");
	// normArrow->SetLineWidth(2);
	normArrow->Draw();
}

void DrawRPhiLegend(double xLim, double yLim, double barrelRPhiRescale)
{
	float xTest = 0.9*xLim;
	float yTest = yLim/2;
	float testBlockSize = 0.2*xLim; //5cm in axis unit
	float disty = 0;
	float dYTest =0.1*xLim;
	float dZTest =2;
	float xLegObj = 20;
	float yLegObj = 18;
	
	Plot10Mu("#Delta xy:",xTest,yTest,testBlockSize);
	Plot10Mu("500 #mum",xTest,yTest-2*dYTest,testBlockSize);
	normArrow(xTest+dYTest,yTest-2*dYTest-disty,500./10000*barrelRPhiRescale);
}

void Write()
{
	output->Write();
}


int makeRPhiArrowPlot( TTree* data, char* name, double xLim, double yLim, double level, double sublevel, double zMin, double zMax, double rMin, double rMax, double barrelRPhiRescale){
	
	
	TCanvas* OBPCanvas = new TCanvas(name,name,1050,875);
	OBPCanvas->DrawFrame(-xLim, -yLim, 1.2*xLim, yLim, ";module position x [cm];module position y [cm]");
	OBPCanvas->SetFillColor(0);
	OBPCanvas->SetFrameBorderMode(0);
	//OBPCanvas ->SetLeftMargin(0.15);
	//OBPCanvas ->SetRightMargin(0.21);
	TFrame* aFrame = OBPCanvas->GetFrame();
	aFrame->SetFillColor(0);
	
	int passcut = 0;
	for(int entry = 0;entry<data->GetEntries(); entry++)
    {
		data->GetEntry(entry);
		if ((level_ == level)&&(sublevel_ == sublevel)){
			if ((z_ <= zMax)&&(z_ > zMin)&&(r_ <= rMax)&&(r_ > rMin)){
				TArrow* aArraw = new TArrow( x_, y_ , x_ + barrelRPhiRescale*dx_, y_+barrelRPhiRescale*dy_,0.0075,">");
				aArraw->Draw();
				passcut++;
			}
		}
	}
	DrawRPhiLegend( xLim, yLim, barrelRPhiRescale );
	
	char sliceLeg[192]; 
	sprintf( sliceLeg, "%s: %d < z <= %d", name, zMin, zMax );
	//Plot10Mu( name, xLim/2, yLim, 0.2*xLim );
	TPaveText* atext = new TPaveText(0.2*xLim,0.85*yLim,0.66*xLim,0.99*yLim);
	atext->AddText(sliceLeg);
	atext->SetLineColor(0);
	atext->SetFillColor(0);
	atext->SetTextFont(42);
	atext->SetTextSize(0.04);
	atext->Draw();
	
	
	
	char outfile[192];
	sprintf( outfile, "%s/%s.png", outputDir_, name );
	OBPCanvas->Print( outfile );

	return passcut;
}

void makeArrowPlots(char* filename, char* outputDir)
{
	
	fin = new TFile(filename);
	fin->cd();
	
	bool plotPXB = true;
	bool plotTIB = true;
	bool plotTOB = true;
	bool plotPXF = true;
	bool plotTID = true;
	bool plotTEC = true;
	
	TString outputfile("OUTPUT_");
	outputfile.Append(filename);
	TFile* output = new TFile(outputfile,"recreate");
	
	sprintf( outputDir_, "%s", outputDir );
	
	TTree* data = (TTree*)fin->Get("alignTree");
	data->SetBranchAddress("sublevel",&sublevel_);
	data->SetBranchAddress("level",&level_);
	data->SetBranchAddress("x",&x_);
	data->SetBranchAddress("y",&y_);
	data->SetBranchAddress("z",&z_);
	data->SetBranchAddress("dx",&dx_);
	data->SetBranchAddress("dy",&dy_);
	data->SetBranchAddress("dz",&dz_);
	data->SetBranchAddress("dphi",&dphi_);
	data->SetBranchAddress("dr",&dr_);
	data->SetBranchAddress("phi",&phi_);
	data->SetBranchAddress("r",&r_);
	
	// args are: tree, title, xLim, yLim
	// cuts are: level, sublevel, zMin, zMax, rMin, rMax, scaleFactor
	// PXB slices
	int totalPXB_modules = 0;
	if (plotPXB){
		double pxbScale = 30.0;
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel-4", 20, 20, 1, 1, -26, -20, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel-3", 20, 20, 1, 1, -19, -14, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel-2", 20, 20, 1, 1, -14, -6.5, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel-1", 20, 20, 1, 1, -6.5, 0, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel+1", 20, 20, 1, 1, 0, 6.5, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel+2", 20, 20, 1, 1, 6.5, 14, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel+3", 20, 20, 1, 1, 14, 19, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_Barrel+4", 20, 20, 1, 1, 19, 26, 0, 15, pxbScale);
	}
		
	// TIB slices
	int totalTIB_modules = 0;
	if (plotTIB){
		double tibScale = 30.0;
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel-6", 80, 80, 1, 3, -70, -56, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel-5", 80, 80, 1, 3, -56, -42, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel-4", 80, 80, 1, 3, -42, -32, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel-3", 80, 80, 1, 3, -32, -20, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel-2", 80, 80, 1, 3, -20, -10, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel-1", 80, 80, 1, 3, -10, 0, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel+1", 80, 80, 1, 3, 0, 10, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel+2", 80, 80, 1, 3, 10, 20, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel+3", 80, 80, 1, 3, 20, 32, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel+4", 80, 80, 1, 3, 32, 42, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel+5", 80, 80, 1, 3, 42, 56, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_Barrel+6", 80, 80, 1, 3, 56, 70, 0, 120, tibScale);
	}
		
	// TOB slices
	int totalTOB_modules = 0;
	if (plotTOB){
		double tobScale = 100.0;
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel-6", 150, 150, 1, 5, -108, -90, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel-5", 150, 150, 1, 5, -90, -72, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel-4", 150, 150, 1, 5, -72, -54, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel-3", 150, 150, 1, 5, -54, -36, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel-2", 150, 150, 1, 5, -36, -18, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel-1", 150, 150, 1, 5, -18, 0, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel+1", 150, 150, 1, 5, 0, 18, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel+2", 150, 150, 1, 5, 18, 36, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel+3", 150, 150, 1, 5, 36, 54, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel+4", 150, 150, 1, 5, 54, 72, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel+5", 150, 150, 1, 5, 72, 90, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_Barrel+6", 150, 150, 1, 5, 90, 108, 0, 120, tobScale);
	}
	
	// PXF slices
	int totalPXF_modules = 0;
	if (plotPXF){
		double pxfScale = 30.0;
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_Disk+1", 20, 20, 1, 2, 25, 40, 0, 120, pxfScale);
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_Disk+2", 20, 20, 1, 2, 40, 55, 0, 120, pxfScale);
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_Disk-1", 20, 20, 1, 2, -40, -25, 0, 120, pxfScale);
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_Disk-2", 20, 20, 1, 2, -55, -40, 0, 120, pxfScale);
	}
	
	// TID slices
	int totalTID_modules = 0;
	if (plotTID){
		double tidScale = 50.0;
		totalTID_modules += makeRPhiArrowPlot( data, "TID_Disk+1", 70, 70, 1, 4, 70, 80, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_Disk+2", 70, 70, 1, 4, 80, 95, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_Disk+3", 70, 70, 1, 4, 95, 110, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_Disk-1", 70, 70, 1, 4, -80, -70, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_Disk-2", 70, 70, 1, 4, -95, -80, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_Disk-3", 70, 70, 1, 4, -110, -95, 0, 120, tidScale);
	}
	
	
	// TEC slices
	int totalTEC_modules = 0;
	if (plotTEC){
		double tecScale = 100.0;
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+1", 150, 150, 1, 6, 120, 130, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+2", 150, 150, 1, 6, 130, 145, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+3", 150, 150, 1, 6, 145, 160, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+4", 150, 150, 1, 6, 160, 175, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+5", 150, 150, 1, 6, 175, 190, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+6", 150, 150, 1, 6, 190, 215, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+7", 150, 150, 1, 6, 215, 235, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+8", 150, 150, 1, 6, 235, 260, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk+9", 150, 150, 1, 6, 260, 280, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-1", 150, 150, 1, 6, -130, -120, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-2", 150, 150, 1, 6, -145, -130, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-3", 150, 150, 1, 6, -160, -145, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-4", 150, 150, 1, 6, -175, -160, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-5", 150, 150, 1, 6, -190, -175, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-6", 150, 150, 1, 6, -215, -190, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-7", 150, 150, 1, 6, -235, -215, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-8", 150, 150, 1, 6, -260, -235, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_Disk-9", 150, 150, 1, 6, -280, -260, 0, 120, tecScale);
	}

	std::cout << "Plotted PXB modules: " << totalPXB_modules << std::endl;
	std::cout << "Plotted TIB modules: " << totalTIB_modules << std::endl;
	std::cout << "Plotted TOB modules: " << totalTOB_modules << std::endl;
	std::cout << "Plotted PXF modules: " << totalPXF_modules << std::endl;
	std::cout << "Plotted TID modules: " << totalTID_modules << std::endl;
	std::cout << "Plotted TEC modules: " << totalTEC_modules << std::endl;
	



}
