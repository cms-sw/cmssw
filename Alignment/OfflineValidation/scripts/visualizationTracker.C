#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TH1F.h"
#include "TNtuple.h"
#include "TRotation.h"
#include "Riostream.h"
#include "TStyle.h"
#include "TLine.h" 
#include "TCanvas.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
//using namespace ROOT::Math;

// tree values
int _nEntries;
int _id, _level, _mid, _mlevel, _sublevel, _useid;
float _xVal, _yVal, _zVal, _rVal, _phiVal, _alphaVal, _betaVal, _gammaVal;
float _dxVal, _dyVal, _dzVal, _drVal, _dphiVal, _dalphaVal, _dbetaVal, _dgammaVal;
float _surWidth, _surLength;
int _identifiers[6];
double _surRot[9];

// global variables
std::vector< int > _moduleList;
TTree* _inTree;
std::string _moduleListName;
float _sclf;

bool readModuleList(){
	
	
	std::ifstream inFile;
	//inFile.open( "PXBdetids.txt" );
	inFile.open( _moduleListName.c_str() );
	int ctr = 0;
	while ( !inFile.eof() ){
		ctr++;
		int listId;
		inFile >> listId;
		inFile.ignore(256, '\n');
		
		_moduleList.push_back( listId );
		//if (listId == aliId) foundId = true;
	}
	
	
}


void getModule(TGeoManager* geom, TGeoVolume* top, TGeoVolume* mod){
	//--- define some materials
	TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
	//--- define some media
	TGeoMedium *Al = new TGeoMedium("Root Material",1, matAl);
	
	TGeoVolume *refMod = geom->MakeBox( "refMod", Al, 0.5*_surWidth, 0.5*_surLength, 0.30 );
	refMod->SetLineColor( 38 );
	refMod->SetFillColor( 13 );
	TGeoVolume *curMod = geom->MakeBox( "curMod", Al, 0.5*_surWidth, 0.5*_surLength, 0.30 );	
	if ((_xVal < 0)&&(_zVal>=0)) curMod->SetLineColor( kRed );
	if ((_xVal < 0)&&(_zVal<0)) curMod->SetLineColor( kGreen );
	if ((_xVal >= 0)&&(_zVal>=0)) curMod->SetLineColor( kBlue );
	if ((_xVal >= 0)&&(_zVal<0)) curMod->SetLineColor( kMagenta );
	
	const Double_t radc = 180./TMath::Pi();
	TGeoTranslation *tr1 = new TGeoTranslation( 0., 0., 0. );
	TGeoRotation *rt1 = new TGeoRotation();
	double rota[9];
	rota[0] = _surRot[0];	rota[1] = _surRot[3];	rota[2] = _surRot[6];
	rota[3] = _surRot[1];	rota[4] = _surRot[4];	rota[5] = _surRot[7];
	rota[6] = _surRot[2];	rota[7] = _surRot[5];	rota[8] = _surRot[8];
	rt1->SetMatrix( rota );
	
	TGeoTranslation *tr2 = new TGeoTranslation( _sclf*_dxVal, _sclf*_dyVal, _sclf*_dzVal );
	TGeoRotation *rt2 = new TGeoRotation( "rt2", _sclf*_dalphaVal*radc, _sclf*_dbetaVal*radc, _sclf*_dgammaVal*radc );
	rt2->MultiplyBy( rt1 );
	TGeoCombiTrans *combi1 = new TGeoCombiTrans( *tr1, *rt1 );
	TGeoCombiTrans *combi2 = new TGeoCombiTrans( *tr2, *rt2 );
	
	mod->AddNode( curMod, 1, combi2 );
	
	
	TGeoTranslation *trG = new TGeoTranslation( _xVal, _yVal, _zVal );
	TGeoRotation *rtG = new TGeoRotation( "rtG", 0., 0., 0. );
	TGeoCombiTrans *combi = new TGeoCombiTrans( *trG, *rtG );
	top->AddNode( mod, 1, combi );
}

void visualizationTracker(){
	
	gSystem->Load("libGeom");
	
	//------------------------------ONLY NEEDED INPUTS-------------------------------//
	//------Tree Read In--------
	TFile *fin = new TFile( "craftData/comparison_craft3900k_pxbDetswSCladders_special3.root" );
	int subdetector = 1; 
	_moduleListName = "PXBdetids.txt";
	_sclf = 25;
	//------------------------------End of ONLY NEEDED INPUTS-------------------------------//
	
	//++++++++++++++++++++ Set up stuff ++++++++++++++++++++//
	TGeoManager *geom = new TGeoManager("simple1", "Simple geometry");
	//--- define some materials and media
	TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
	TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
	//--- make the top container volume
	TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 500., 500., 500.);
	//TGeoVolume *toptop = geom->MakeBox("TOPTOP", Vacuum, 1000., 1000., 500.);
	geom->SetTopVolume(top);
	
	//++++++++++++++++++++ Read in tree ++++++++++++++++++++//
	_inTree = (TTree*) fin->Get("alignTree");
	_nEntries = _inTree->GetEntries();
	
	_inTree->SetBranchAddress( "id", &_id );
	_inTree->SetBranchAddress( "level", &_level);
	_inTree->SetBranchAddress("mid", &_mid);
	_inTree->SetBranchAddress("mlevel", &_mlevel);
	_inTree->SetBranchAddress("sublevel", &_sublevel);
	_inTree->SetBranchAddress("x", &_xVal);
	_inTree->SetBranchAddress("y", &_yVal);
	_inTree->SetBranchAddress("z", &_zVal);
	_inTree->SetBranchAddress("r", &_rVal);
	_inTree->SetBranchAddress("phi", &_phiVal);
	_inTree->SetBranchAddress("alpha", &_alphaVal);
	_inTree->SetBranchAddress("beta", &_betaVal);
	_inTree->SetBranchAddress("gamma", &_gammaVal);
	_inTree->SetBranchAddress("dx", &_dxVal);
	_inTree->SetBranchAddress("dy", &_dyVal);
	_inTree->SetBranchAddress("dz", &_dzVal);
	_inTree->SetBranchAddress("dr", &_drVal);
	_inTree->SetBranchAddress("dphi", &_dphiVal);
	_inTree->SetBranchAddress("dalpha", &_dalphaVal);
	_inTree->SetBranchAddress("dbeta", &_dbetaVal);
	_inTree->SetBranchAddress("dgamma", &_dgammaVal);
	_inTree->SetBranchAddress("surW", &_surWidth);
	_inTree->SetBranchAddress("surL", &_surLength);
	_inTree->SetBranchAddress("surRot", &_surRot);
	//_inTree->SetBranchAddress("useid", &_useid);
	_inTree->SetBranchAddress("identifiers", &_identifiers);
	
	
	//reading list ...
	readModuleList();


	//draw modules in the list
	for (int i = 0; i < _nEntries; ++i){
		_inTree->GetEntry(i);
		if ((_sublevel == subdetector)&&(_rVal <= 12)&&(_rVal >=8)){
			bool foundID = false;
			for (int j = 0; j < _moduleList.size(); ++j){
				if (_id == _moduleList[j]) foundID = true;
			}
			if (foundID){ 
				char modName[192];
				sprintf(modName, "testModule%i", i);
				TGeoVolume* testMod = geom->MakeBox( modName, Vacuum, 90., 90., 40. );
				getModule( geom, top, testMod );
			}
		}
	}
	
	TGeoTranslation *tr1 = new TGeoTranslation( 0., 0., 0. );
	TGeoRotation *rt1 = new TGeoRotation( "rt1", 90., 0., 0. );
	TGeoCombiTrans *combi1 = new TGeoCombiTrans( *tr1, *rt1 );
	
	// --- crate the top volume
	//toptop->AddNode(top, 1, combi1);
	//top->AddNode(disk, 1, 0);
	//--- close the geometry
	geom->CloseGeometry();
	// -- draw
	geom->SetVisLevel(4);
	top->Draw();
}
	