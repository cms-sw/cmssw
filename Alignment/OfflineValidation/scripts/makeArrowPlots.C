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

void Plot10Mu(const char* text,float X, float Y, float size)
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
	
	Plot10Mu("#Delta r:",xTest,yTest,testBlockSize);
	Plot10Mu("500 #mum",xTest,yTest-3*dYTest,testBlockSize);
	normArrow(xTest+dYTest,yTest-4*dYTest-disty,500./10000*barrelRPhiRescale);
}


int makeRPhiArrowPlot( TTree* data, const char* name, double xLim, double yLim, double level, double sublevel, double zMin, double zMax, double rMin, double rMax, double barrelRPhiRescale){
	
	
	TCanvas* OBPCanvas = new TCanvas(name,name,1050,875);
	OBPCanvas->DrawFrame(-xLim, -yLim, 1.2*xLim, yLim, ";module position x [cm];module position y [cm]");
	OBPCanvas->SetFillColor(0);
	OBPCanvas->SetFrameBorderMode(0);
	
	TFrame* aFrame = OBPCanvas->GetFrame();
	aFrame->SetFillColor(0);
	
	int passcut = 0;
	for(int entry = 0;entry<data->GetEntries(); entry++)
    {
		data->GetEntry(entry);
		if ((level_ == level)&&(((sublevel_ == sublevel)&&(sublevel != 0))||(sublevel == 0))){
			if ((z_ <= zMax)&&(z_ > zMin)&&(r_ <= rMax)&&(r_ > rMin)){
				TArrow* aArraw = new TArrow( x_, y_ , x_ + barrelRPhiRescale*dx_, y_+barrelRPhiRescale*dy_,0.0075,">");
				aArraw->Draw();
				passcut++;
			}
		}
	}
	DrawRPhiLegend( xLim, yLim, barrelRPhiRescale );
	
	char sliceLeg[192]; 
	sprintf( sliceLeg, "%s: %f < z <= %f", name, zMin, zMax );
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

int makeZPhiArrowPlot( TTree* data, const char* name, double zLim, double phiLim, double level, double sublevel, double zMin, double zMax, double rMin, double rMax, double barrelRPhiRescale){
	
	
	TCanvas* OBPCanvas = new TCanvas(name,name,1050,875);
	OBPCanvas->DrawFrame(-zLim, -phiLim, 1.2*zLim, phiLim, ";module position z [cm];module position r*phi [cm]");
	OBPCanvas->SetFillColor(0);
	OBPCanvas->SetFrameBorderMode(0);
	
	TFrame* aFrame = OBPCanvas->GetFrame();
	aFrame->SetFillColor(0);
	
	int passcut = 0;
	for(int entry = 0;entry<data->GetEntries(); entry++)
    {
		data->GetEntry(entry);
			if ((level_ == level)&&(((sublevel_ == sublevel)&&(sublevel != 0))||(sublevel == 0))){
			if ((z_ <= zMax)&&(z_ > zMin)&&(r_ <= rMax)&&(r_ > rMin)){
				TArrow* aArraw = new TArrow( z_, r_*phi_ , z_ + barrelRPhiRescale*dz_, r_*phi_+barrelRPhiRescale*r_*dphi_,0.0075,">");
				aArraw->Draw();
				passcut++;
			}
		}
	}
	DrawRPhiLegend( zLim, phiLim, barrelRPhiRescale );
	
	char sliceLeg[192]; 
	sprintf( sliceLeg, "%s: %f < r <= %f", name, rMin, rMax );
	//Plot10Mu( name, xLim/2, yLim, 0.2*xLim );
	TPaveText* atext = new TPaveText(0.2*zLim,0.85*phiLim,0.66*zLim,0.99*phiLim);
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

/*
int makeRZArrowPlot( TTree* data, char* name, double zLim, double zLimMax, double rLim, double rLimMax, double level, double sublevel, double zMin, double zMax, double rMin, double rMax, double barrelRPhiRescale){
	
	
	TCanvas* OBPCanvas = new TCanvas(name,name,1050,875);
	OBPCanvas->DrawFrame(zLim, rLim, zLimMax, rLimMax, ";module position z [cm];module position r [cm]");
	OBPCanvas->SetFillColor(0);
	OBPCanvas->SetFrameBorderMode(0);
	
	TFrame* aFrame = OBPCanvas->GetFrame();
	aFrame->SetFillColor(0);
	
	int passcut = 0;
	for(int entry = 0;entry<data->GetEntries(); entry++)
    {
		data->GetEntry(entry);
		if ((level_ == level)&&(sublevel_ == sublevel)){
			if ((z_ <= zMax)&&(z_ > zMin)&&(r_ <= rMax)&&(r_ > rMin)){
				TArrow* aArraw = new TArrow( z_, r_ , z_ + barrelRPhiRescale*dz_, r_+barrelRPhiRescale*dr_,0.0075,">");
				aArraw->Draw();
				passcut++;
			}
		}
	}
	// legend
	double xpos = 0.8*(zLimMax-zLim) + zLim;
	double ypos = 0.7*(rLimMax-rLim) + rLim;	
	double sizer = 0.05*(zLimMax-zLim);
	Plot10Mu("#Delta r:",xpos,ypos,sizer);
	Plot10Mu("500 #mum",xpos,ypos-1.5*sizer,sizer);
	normArrow(xpos,ypos-2*sizer,500./10000*barrelRPhiRescale);	
	
	
	char sliceLeg[192]; 
	sprintf( sliceLeg, "%s: %d < z <= %d", name, zMin, zMax );
	//Plot10Mu( name, xLim/2, yLim, 0.2*xLim );
	TPaveText* atext = new TPaveText(0.4*(zLimMax-zLim) + zLim,0.85*(rLimMax-rLim)+rLim,0.86*(zLimMax-zLim) + zLim,0.99*(rLimMax-rLim)+rLim);
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
*/

void makeArrowPlots(const char* filename, const char* outputDir)
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
	int totalPXB_modules_zphi = 0;
	int dummy = 0;
	if (plotPXB){
		double pxbScale = 30.0;
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY-4", 20, 20, 1, 1, -26, -20, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY-3", 20, 20, 1, 1, -19, -14, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY-2", 20, 20, 1, 1, -14, -6.5, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY-1", 20, 20, 1, 1, -6.5, 0, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY+1", 20, 20, 1, 1, 0, 6.5, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY+2", 20, 20, 1, 1, 6.5, 14, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY+3", 20, 20, 1, 1, 14, 19, 0, 15, pxbScale);
		totalPXB_modules += makeRPhiArrowPlot( data, "PXB_BarrelXY+4", 20, 20, 1, 1, 19, 26, 0, 15, pxbScale);
		double pxbScale_zphi = 40.0;
		totalPXB_modules_zphi += makeZPhiArrowPlot( data, "PXB_BarrelZPhi_1", 35, 20, 1, 1, -300, 300, 0, 5, pxbScale_zphi);
		totalPXB_modules_zphi += makeZPhiArrowPlot( data, "PXB_BarrelZPhi_2", 35, 30, 1, 1, -300, 300, 5, 8, pxbScale_zphi);
		totalPXB_modules_zphi += makeZPhiArrowPlot( data, "PXB_BarrelZPhi_3", 35, 40, 1, 1, -300, 300, 8, 14, pxbScale_zphi);
	}
		
	// TIB slices
	int totalTIB_modules = 0;
	int totalTIB_modules_zphi = 0;
	if (plotTIB){
		double tibScale = 30.0;
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY-6", 80, 80, 1, 3, -70, -56, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY-5", 80, 80, 1, 3, -56, -42, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY-4", 80, 80, 1, 3, -42, -32, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY-3", 80, 80, 1, 3, -32, -20, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY-2", 80, 80, 1, 3, -20, -10, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY-1", 80, 80, 1, 3, -10, 0, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY+1", 80, 80, 1, 3, 0, 10, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY+2", 80, 80, 1, 3, 10, 20, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY+3", 80, 80, 1, 3, 20, 32, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY+4", 80, 80, 1, 3, 32, 42, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY+5", 80, 80, 1, 3, 42, 56, 0, 120, tibScale);
		totalTIB_modules += makeRPhiArrowPlot( data, "TIB_BarrelXY+6", 80, 80, 1, 3, 56, 70, 0, 120, tibScale);
		double tibScale_zphi = 40.0;
		totalTIB_modules_zphi += makeZPhiArrowPlot( data, "TIB_BarrelZPhi_1", 80, 120, 1, 3, -300, 300, 20.0, 29.0, tibScale_zphi);
		totalTIB_modules_zphi += makeZPhiArrowPlot( data, "TIB_BarrelZPhi_2", 80, 140, 1, 3, -300, 300, 29.0, 37.5, tibScale_zphi);
		totalTIB_modules_zphi += makeZPhiArrowPlot( data, "TIB_BarrelZPhi_3", 80, 170, 1, 3, -300, 300, 37.5, 47.5, tibScale_zphi);
		totalTIB_modules_zphi += makeZPhiArrowPlot( data, "TIB_BarrelZPhi_4", 80, 200, 1, 3, -300, 300, 47.5, 60.0, tibScale_zphi);

	}
		
	// TOB slices
	int totalTOB_modules = 0;
	int totalTOB_modules_zphi = 0;
	if (plotTOB){
		double tobScale = 100.0;
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY-6", 150, 150, 1, 5, -108, -90, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY-5", 150, 150, 1, 5, -90, -72, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY-4", 150, 150, 1, 5, -72, -54, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY-3", 150, 150, 1, 5, -54, -36, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY-2", 150, 150, 1, 5, -36, -18, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY-1", 150, 150, 1, 5, -18, 0, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY+1", 150, 150, 1, 5, 0, 18, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY+2", 150, 150, 1, 5, 18, 36, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY+3", 150, 150, 1, 5, 36, 54, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY+4", 150, 150, 1, 5, 54, 72, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY+5", 150, 150, 1, 5, 72, 90, 0, 120, tobScale);
		totalTOB_modules += makeRPhiArrowPlot( data, "TOB_BarrelXY+6", 150, 150, 1, 5, 90, 108, 0, 120, tobScale);
		double tobScale_zphi = 40.0;
		totalTOB_modules_zphi += makeZPhiArrowPlot( data, "TOB_BarrelZPhi_1", 130, 250, 1, 5, -300, 300, 55.0, 65.0, tobScale_zphi);
		totalTOB_modules_zphi += makeZPhiArrowPlot( data, "TOB_BarrelZPhi_2", 130, 280, 1, 5, -300, 300, 65.0, 75.0, tobScale_zphi);
		totalTOB_modules_zphi += makeZPhiArrowPlot( data, "TOB_BarrelZPhi_3", 130, 320, 1, 5, -300, 300, 75.0, 85.0, tobScale_zphi);
		totalTOB_modules_zphi += makeZPhiArrowPlot( data, "TOB_BarrelZPhi_4", 130, 350, 1, 5, -300, 300, 85.0, 93.0, tobScale_zphi);
		totalTOB_modules_zphi += makeZPhiArrowPlot( data, "TOB_BarrelZPhi_5", 130, 380, 1, 5, -300, 300, 93.0, 101.0, tobScale_zphi);
		totalTOB_modules_zphi += makeZPhiArrowPlot( data, "TOB_BarrelZPhi_6", 130, 410, 1, 5, -300, 300, 101.0, 110.0, tobScale_zphi);
	}
	
	// PXF slices
	int totalPXF_modules = 0;
	int totalPXF_modules_rz = 0;
	if (plotPXF){
		double pxfScale = 30.0;
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_DiskXY+1", 20, 20, 1, 2, 25, 40, 0, 120, pxfScale);
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_DiskXY+2", 20, 20, 1, 2, 40, 55, 0, 120, pxfScale);
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_DiskXY-1", 20, 20, 1, 2, -40, -25, 0, 120, pxfScale);
		totalPXF_modules += makeRPhiArrowPlot( data, "PXF_DiskXY-2", 20, 20, 1, 2, -55, -40, 0, 120, pxfScale);
		/*
		double pxfScale_zphi = 10.0;
		totalPXF_modules_rz += makeRZArrowPlot( data, "PXF_DiskRZ-1", -38, -30, 5, 17, 1, 2, -40, -25, 0, 120.0, pxfScale_zphi);
		totalPXF_modules_rz += makeRZArrowPlot( data, "PXF_DiskRZ-2", -52, -44, 5, 17, 1, 2, -55, -40, 0, 120.0, pxfScale_zphi);		
		totalPXF_modules_rz += makeRZArrowPlot( data, "PXF_DiskRZ+1", 32, 40, 5, 17, 1, 2, 25, 40, 0, 120.0, pxfScale_zphi);
		totalPXF_modules_rz += makeRZArrowPlot( data, "PXF_DiskRZ+2", 46, 54, 5, 17, 1, 2, 40, 55, 0, 120.0, pxfScale_zphi);		
		 */
	}
	
	// TID slices
	int totalTID_modules = 0;
	int totalTID_modules_rz = 0;
	if (plotTID){
		double tidScale = 50.0;
		totalTID_modules += makeRPhiArrowPlot( data, "TID_DiskXY+1", 70, 70, 1, 4, 70, 80, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_DiskXY+2", 70, 70, 1, 4, 80, 95, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_DiskXY+3", 70, 70, 1, 4, 95, 110, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_DiskXY-1", 70, 70, 1, 4, -80, -70, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_DiskXY-2", 70, 70, 1, 4, -95, -80, 0, 120, tidScale);
		totalTID_modules += makeRPhiArrowPlot( data, "TID_DiskXY-3", 70, 70, 1, 4, -110, -95, 0, 120, tidScale);
		/*
		double tidScale_zphi = 10.0;
		totalTID_modules_rz += makeRZArrowPlot( data, "TID_DiskRZ-1", -80, -70, -3, 55, 1, 4, -80, -70, 0, 120.0, tidScale_zphi);
		totalTID_modules_rz += makeRZArrowPlot( data, "TID_DiskRZ-2", -95, -80, 20, 55, 1, 4, -95, -80, 0, 120.0, tidScale_zphi);		
		totalTID_modules_rz += makeRZArrowPlot( data, "TID_DiskRZ-3", -110, -95, 20, 55, 1, 4, -110, -95, 0, 120.0, tidScale_zphi);		
		totalTID_modules_rz += makeRZArrowPlot( data, "TID_DiskRZ+1", 70, 80, 20, 55, 1, 4, 70, 80, 0, 120.0, tidScale_zphi);
		totalTID_modules_rz += makeRZArrowPlot( data, "TID_DiskRZ+2", 80, 95, 20, 55, 1, 4, 80, 95, 0, 120.0, tidScale_zphi);	
		totalTID_modules_rz += makeRZArrowPlot( data, "TID_DiskRZ+3", 95, 110, 20, 55, 1, 4, 95, 110, 0, 120.0, tidScale_zphi);	
		 */
	}
	
	
	// TEC slices
	int totalTEC_modules = 0;
	if (plotTEC){
		double tecScale = 100.0;
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+1", 150, 150, 1, 6, 120, 130, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+2", 150, 150, 1, 6, 130, 145, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+3", 150, 150, 1, 6, 145, 160, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+4", 150, 150, 1, 6, 160, 175, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+5", 150, 150, 1, 6, 175, 190, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+6", 150, 150, 1, 6, 190, 215, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+7", 150, 150, 1, 6, 215, 235, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+8", 150, 150, 1, 6, 235, 260, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY+9", 150, 150, 1, 6, 260, 280, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-1", 150, 150, 1, 6, -130, -120, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-2", 150, 150, 1, 6, -145, -130, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-3", 150, 150, 1, 6, -160, -145, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-4", 150, 150, 1, 6, -175, -160, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-5", 150, 150, 1, 6, -190, -175, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-6", 150, 150, 1, 6, -215, -190, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-7", 150, 150, 1, 6, -235, -215, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-8", 150, 150, 1, 6, -260, -235, 0, 120, tecScale);
		totalTEC_modules += makeRPhiArrowPlot( data, "TEC_DiskXY-9", 150, 150, 1, 6, -280, -260, 0, 120, tecScale);
	}

	std::cout << "Plotted PXB modules: " << totalPXB_modules << std::endl;
	std::cout << "Plotted PXB modules (zphi): " << totalPXB_modules_zphi << std::endl;
	std::cout << "Plotted TIB modules: " << totalTIB_modules << std::endl;
	std::cout << "Plotted TIB modules (zphi): " << totalTIB_modules_zphi << std::endl;
	std::cout << "Plotted TOB modules: " << totalTOB_modules << std::endl;
	std::cout << "Plotted TOB modules (zphi): " << totalTOB_modules_zphi << std::endl;
	std::cout << "Plotted PXF modules: " << totalPXF_modules << std::endl;
	//std::cout << "Plotted PXF modules (rz): " << totalPXF_modules_rz << std::endl;
	std::cout << "Plotted TID modules: " << totalTID_modules << std::endl;
	//std::cout << "Plotted TID modules (rz): " << totalTID_modules_rz << std::endl;
	std::cout << "Plotted TEC modules: " << totalTEC_modules << std::endl;
	



}
