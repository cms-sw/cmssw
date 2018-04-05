#include "TText.h"
#include "TPaveText.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TH1F.h"
#include "TNtuple.h"
#include "TRotation.h"
#include "Riostream.h"
#include "TStyle.h"
#include "TLine.h" 
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "TSystem.h"
#include "TError.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>
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
int _i;

// global variables
TTree* _inTree;
int _sclftr;
int _sclfrt;
float _sclfmodulesizex;
float _sclfmodulesizey;
float _sclfmodulesizez;
int _subdetector1;
int _subdetector2;
string _outputFileName;
string _line1, _line2, _line3;
float _piperadius;
float _pipexcoord, _pipeycoord;
float _linexcoord, _lineycoord;

TTree *t;
float zval;
vector<int> vz;

bool iszilessthanzj(int i, int j)
{
    return vz[i] < vz[j];
}

void sortbyz(TString infile)
{
    TFile *f = TFile::Open(infile);
    if(f == 0) {
        cout << "***Exception Thrown: null input file***" << endl;
        assert(0);
    }
    t = (TTree*)f->Get("alignTree");
    if(t == 0){
        cout << "***Exception Thrown: tree is null***" << endl;
        assert(0);
    }
    if(t->GetEntries() == 0) {
        cout << "***Exception Thrown: tree has no entries***" << endl;
        assert(0);
    }

    t->SetBranchAddress("z", &zval);

    TString outfile = infile.ReplaceAll(".root","_sorted.root");
    TFile *newf = TFile::Open(outfile, "RECREATE");
    TTree *newt = t->CloneTree(0);

    vector<int> v;
    vz.clear();
    int length = t->GetEntries();
    for (int i = 0; i < length; i++)
    {
        v.push_back(i);

        t->GetEntry(i);
        vz.push_back(zval);
    }

    sort(v.begin(), v.end(), iszilessthanzj);

    for (int i = 0; i < length; i++)
    {
        t->GetEntry(v[i]);
        newt->Fill();
    }
    newt->Write();
    delete newf;
}

void getBeamVisuals(TGeoManager* geom, TGeoVolume* top, float minZ, float maxZ) {
    TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
    TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
    TGeoVolume *xyaxis = geom->MakeBox( "xyaxis", Vacuum, 90., 90., 40. );

    TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
    TGeoMedium *Al = new TGeoMedium("Root Material",1, matAl);
    //TGeoVolume *line = geom->MakeTube( "BeamLine", Al, 0, .3, (maxZ - minZ) / 2 + 5);
    TGeoVolume *xaxis = geom->MakeTube( "XAxis", Al, 0, .1, 30.);
    TGeoVolume *yaxis = geom->MakeTube( "YAxis", Al, 0, .1, 30.);
    //TGeoVolume *pipe = geom->MakeTube( "BeamPipe", Al, _piperadius-.05, _piperadius+.05, (maxZ - minZ) / 2 + 5);
    //line->SetLineColor(kRed);
    xaxis->SetLineColor(kBlue);
    yaxis->SetLineColor(kBlue);
    //pipe->SetLineColor(kBlack);

    xyaxis->AddNode(xaxis, 1, new TGeoRotation( "rtyz", 0, 90, 0));
    xyaxis->AddNode(yaxis, 1, new TGeoRotation( "rtxz", 90, 90, 0));
    
    TGeoCombiTrans * pipecenter = new TGeoCombiTrans( *new TGeoTranslation(_pipexcoord, _pipeycoord, 0), *new TGeoRotation());
    //TGeoCombiTrans * linecenter = new TGeoCombiTrans( *new TGeoTranslation(_linexcoord, _lineycoord, 0), *new TGeoRotation());
    //top->AddNode( pipe, 1, pipecenter);
    //top->AddNode( line, 1, linecenter);
    top->AddNode( xyaxis, 1, pipecenter);
}

void getModule(TGeoManager* geom, TGeoVolume* top, TGeoVolume* mod){
//--- define some materials
    TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
//--- define some media
    TGeoMedium *Al = new TGeoMedium("Root Material",1, matAl);
    TGeoVolume *refMod = geom->MakeBox( "refMod", Al, 0.5*_surWidth*_sclfmodulesizex, 0.5*_surLength*_sclfmodulesizey, 0.30*_sclfmodulesizez );
    refMod->SetLineColor( 38 );
    refMod->SetFillColor( 13 );
    TGeoVolume *curMod = geom->MakeBox( "curMod", Al, 0.5*_surWidth*_sclfmodulesizex, 0.5*_surLength*_sclfmodulesizey, 0.30*_sclfmodulesizez );

    if ((_yVal < 0)&&(_zVal>=0)) curMod->SetLineColor( kRed );
    if ((_yVal < 0)&&(_zVal<0)) curMod->SetLineColor( kGreen );
    if ((_yVal >= 0)&&(_zVal>=0)) curMod->SetLineColor( kBlue );
    if ((_yVal >= 0)&&(_zVal<0)) curMod->SetLineColor( kMagenta );
    refMod->SetLineColor( 14 );
    //curMod->SetLineColor(kBlue);
    //refMod->SetLineColor(kRed);

    const Double_t radc = 180./TMath::Pi();
    TGeoTranslation *tr1 = new TGeoTranslation( 0., 0., 0. );
    TGeoRotation *rt1 = new TGeoRotation();
    double rota[9];
    rota[0] = _surRot[0];   rota[1] = _surRot[3];   rota[2] = _surRot[6];
    rota[3] = _surRot[1];   rota[4] = _surRot[4];   rota[5] = _surRot[7];
    rota[6] = _surRot[2];   rota[7] = _surRot[5];   rota[8] = _surRot[8];
    rt1->SetMatrix( rota );
    TGeoTranslation *tr2 = new TGeoTranslation( _sclftr*_dxVal, _sclftr*_dyVal, _sclftr*_dzVal );
    TGeoRotation *rt2 = new TGeoRotation( "rt2", _sclfrt*_dalphaVal*radc, _sclfrt*_dbetaVal*radc, _sclfrt*_dgammaVal*radc );
    rt2->MultiplyBy( rt1 );
    TGeoCombiTrans *combi1 = new TGeoCombiTrans( *tr1, *rt1 );
    TGeoCombiTrans *combi2 = new TGeoCombiTrans( *tr2, *rt2 );
    mod->AddNode( curMod, 1, combi2 );
    mod->AddNode( refMod, 1, combi1 );
    TGeoTranslation *trG = new TGeoTranslation( _xVal - _dxVal, _yVal - _dyVal, _zVal - _dzVal);
    TGeoRotation *rtG = new TGeoRotation( "rtG", -1*_dalphaVal, -1*_dbetaVal, -1*_dgammaVal );
    TGeoCombiTrans *combi = new TGeoCombiTrans( *trG, *rtG );
    top->AddNode( mod, 1, combi );
}

bool isRightSubDet() {
    return (_sublevel == _subdetector1 || _sublevel == _subdetector2);
}

int visualizationTracker(float minZ, float maxZ, float minX, float maxX, float theta, float phi){
    gSystem->Load("libGeom");
//++++++++++++++++++++ Set up stuff ++++++++++++++++++++//
    TGeoManager *geom = new TGeoManager("simple1", "Simple geometry");
//--- define some materials and media
    TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
    TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
//--- make the top container volume
    TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 500., 500., 500.);
//TGeoVolume *toptop = geom->MakeBox("TOPTOP", Vacuum, 1000., 1000., 500.);
    geom->SetTopVolume(top);

    int count = 0;
    for (int i = 0; i < _nEntries; ++i){
        _inTree->GetEntry(i);
        if (isRightSubDet()&&(_zVal >= minZ && _zVal < maxZ)&&(_xVal >= minX && _xVal < maxX)/*&&(_rVal <= 12)&&(_rVal >=8)*/){
            char modName[192];
            sprintf(modName, "testModule%i", i);
            TGeoVolume* testMod = geom->MakeBox( modName, Vacuum, 90., 90., 40. );
            getModule( geom, top, testMod );
            count++;
        }
    }

    if(count == 0) return -1;

    getBeamVisuals(geom, top, minZ, maxZ);

//--- close the geometry
    geom->CloseGeometry();
// -- draw
    geom->SetVisLevel(4);

    TCanvas * c = new TCanvas();
    c->SetTheta(theta);
    c->SetPhi(phi);
    top->Draw();

//--- putting words on canvas...
    bool with0T = true;

    //can play with these numbers
    double widthofeach = 0.07;
    double textsize = 0.05;

    double xmax = 2*widthofeach;
    if (with0T) xmax = widthofeach;

    TPaveText* pt = new TPaveText(0,0,xmax,1,"brNDC");
    pt->SetBorderSize(0);
    pt->SetFillStyle(0);
    pt->SetTextAlign(22);
    pt->SetTextFont(42);
    pt->SetTextSize(0.1);
    TText *text = pt->AddText(0,0,TString("#font[42]{"+_line1+"}"));
    text->SetTextSize(textsize);
    text->SetTextAngle(90);
    pt->Draw();

    TPaveText *pt2 = new TPaveText(widthofeach, 0, 2*widthofeach, 1, "brNDC");
    pt2->SetBorderSize(0);
    pt2->SetFillStyle(0);
    pt2->SetTextAlign(22);
    pt2->SetTextFont(42);
    pt2->SetTextSize(0.1);
    TText *text2 = pt2->AddText(0,0,TString("#font[42]{"+_line2+"}"));
    text2->SetTextSize(textsize);
    text2->SetTextAngle(90);
    pt2->Draw();

    TPaveText *pt3 = new TPaveText(2*widthofeach, 0, 3*widthofeach, 1, "brNDC");
    pt3->SetBorderSize(0);
    pt3->SetFillStyle(0);
    pt3->SetTextAlign(22);
    pt3->SetTextFont(42);
    pt3->SetTextSize(0.1);
    TText *text3 = pt3->AddText(0,0,TString("#font[42]{"+_line3+"}"));
    text3->SetTextSize(textsize);
    text3->SetTextAngle(90);
    pt3->Draw();

    string str = string("i") + to_string(_i) + string(".gif");
    c->SaveAs(TString(str));
    gSystem->Exec(TString("mv "+str+" images/"+str));
    delete c;
    cout << "Created image " << str << endl;
    return 0;
}

//gets minimum and maximum values of Z and Y in the specified subdetectors
void getMinMax(float & minZ, float & maxZ, float & minX, float & maxX) {
    int i = 0;
    while(i < _nEntries){
        _inTree->GetEntry(i);
        if(isRightSubDet()) {
            _inTree->GetEntry(i);
            maxX = _xVal;
            minX = _xVal;
            maxZ = _zVal;
            minZ = _zVal;
            break;
        }
        ++i;
    }
    while ( i < _nEntries ) {
        _inTree->GetEntry(i);
        if (isRightSubDet()) {
            if( _xVal > maxX ) {
                maxX = _xVal;
            }
            if( _xVal < minX ) {
                minX = _xVal;
            }
            if( _zVal > maxZ ) {
                maxZ = _zVal;
            }
            if( _zVal < minZ ) {
                minZ = _zVal;
            }
        }
        ++i;
    }
    cout << minX << endl;
    cout << maxX << endl;
    cout << minZ << endl;
    cout << maxZ << endl;
}

//gets string that is a unix command that merges gifs using gifmerge (download at http://the-labs.com/GIFMerge/)
string getGifMergeCommand(int start, int breakspot1, int breakspot2, int end) {
    string str = "";
    str += "./gifmerge -192,192,192 -l0 -5 ";
    for (int i = start; i < breakspot1; i++) {
        str += "images/i"+to_string(i)+".gif ";
    }
    str += "-50 ";
    for (int i = breakspot1; i < breakspot2; i++) {
        str += "images/i"+to_string(i)+".gif ";
    }
    str += "-5 ";
    for (int i = breakspot2; i < end-1; i++) {
        str += "images/i"+to_string(i)+".gif ";
    }
    str += "-100 images/i"+to_string(end-1)+".gif > "+_outputFileName+".gif";
    return str;
}

//gets string that is a unix command that merges gifs using GraphicsMagick
string getConvertCommand(int start, int breakspot1, int breakspot2, int end) {
    string str = "";
    str += "gm convert -loop 0 -delay 5 ";
    for (int i = start; i < breakspot1; i++) {
        str += "images/i"+to_string(i)+".gif ";
    }
    str += "-delay 50 ";
    for (int i = breakspot1; i < breakspot2; i++) {
        str += "images/i"+to_string(i)+".gif ";
    }
    str += "-delay 5 ";
    for (int i = breakspot2; i < end-1; i++) {
        str += "images/i"+to_string(i)+".gif ";
    }
    str += "-delay 100 images/i"+to_string(end-1)+".gif   "+_outputFileName+".gif";
    return str;
}

void runVisualizer(TString input,
                    string output,
                    string line1,
                    string line2,
                    int subdetector1,
                    int subdetector2,
                    int sclftr,
                    int sclfrt,
                    float sclfmodulesizex,
                    float sclfmodulesizey,
                    float sclfmodulesizez,
                    float piperadius,
                    float pipexcoord,
                    float pipeycoord,
                    float linexcoord,
                    float lineycoord ) {
    gErrorIgnoreLevel = kError;

//------Tree Read In--------
    TString inputFileName = input;
    //output file name
    _outputFileName = output;
    //title
    _line1 = line1;
    _line2 = line2;
    //set subdetectors to see
    _subdetector1 = subdetector1;
    _subdetector2 = subdetector2;
    //translation scale factor
    _sclftr = sclftr;
    //rotation scale factor
    _sclfrt = sclfrt;
    //module size scale factor
    _sclfmodulesizex = sclfmodulesizex;
    _sclfmodulesizey = sclfmodulesizey;
    _sclfmodulesizez = sclfmodulesizez;
    //beam pipe radius
    _piperadius = piperadius;
    //beam pipe xy coordinates
    _pipexcoord = pipexcoord;
    _pipeycoord = pipeycoord;
    //beam line xy coordinates
    _linexcoord = linexcoord;
    _lineycoord = lineycoord;


    sortbyz( inputFileName );
    TFile *fin = TFile::Open( inputFileName.ReplaceAll(".root", "_sorted.root") );
    _line3 = Form("Translational Scale Factor: %i",_sclftr);
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

    float minZ, maxZ, minX, maxX;
    int zpos;
    int numincrements;
    getMinMax(minZ, maxZ, minX, maxX);
    
    gSystem->mkdir("images");

    _i = 0;
    for (int i = 0; i < 90; i+=1, _i++) {
        visualizationTracker(minZ, maxZ, minX, maxX, i, 90);
    }
    
    numincrements = 12;
    float length;
    int start1 = _i;
    length = (maxZ - minZ) / numincrements;
    for(int i = numincrements-1; i >= 0; i--) {
        zpos = minZ + i*length;
        if(visualizationTracker(zpos, zpos+length, minX, maxX, 90, 90) == 0) {
            _i++;
        }
    }
    
    int start2 = _i;
    for (int i = 90; i >= 0; i-=1, _i++){
        visualizationTracker(minZ, maxZ, (minX + maxX) / 2 - 3, (minX + maxX) / 2 + 1, i, 90);
    }
    delete fin;

    //gSystem->Exec(TString(getGifMergeCommand(0, start1, start2, _i)));
    gSystem->Exec(TString(getConvertCommand(0, start1, start2, _i)));
    cout << "images merged." << endl;
    gSystem->Exec(TString("gm convert "+_outputFileName+".gif -rotate 90 "+_outputFileName+"_rotated.gif"));
    cout << "images rotated." << endl;
}

void runVisualizer() {
        //------------------------------ONLY NEEDED INPUTS-------------------------------//
//------Tree Read In--------
    TString inputFileName = "~/normal_vs_test.Comparison_commonTracker.root";
    //output file name
    string outputFileName = "animation";
    //title
    string line1 = "";
    string line2 = "";
    //set subdetectors to see
    int subdetector1 = 1;
    int subdetector2 = 2;
    //translation scale factor
    int sclftr = 50;
    //rotation scale factor
    int sclfrt = 1;
    //module size scale factor
    float sclfmodulesizex = 1;
    float sclfmodulesizey = 1;
    float sclfmodulesizez = 1;
    //beam pipe radius
    float piperadius = 2.25;
    //beam pipe xy coordinates
    float pipexcoord = 0;
    float pipeycoord = 0;
    //beam line xy coordinates
    float linexcoord = 0;
    float lineycoord = 0;
//------------------------------End of ONLY NEEDED INPUTS-------------------------------//
    runVisualizer(inputFileName,
                    outputFileName,
                    line1,
                    line2,
                    subdetector1,
                    subdetector2,
                    sclftr,
                    sclfrt,
                    sclfmodulesizex,
                    sclfmodulesizey,
                    sclfmodulesizez,
                    piperadius,
                    pipexcoord,
                    pipeycoord,
                    linexcoord,
                    lineycoord );
}
