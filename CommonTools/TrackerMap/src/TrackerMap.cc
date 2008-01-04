#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CommonTools/TrackerMap/interface/TmApvPair.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "TCanvas.h"
#include "TPolyLine.h"
#include "TStyle.h"
#include "TColor.h"
#include "TROOT.h"

using namespace std;

/**********************************************************
Allocate all the modules in a map of TmModule
The filling of the values for each module is done later
when the user starts to fill it.
**********************************************************/

TrackerMap::TrackerMap(const edm::ParameterSet & tkmapPset,const edm::ESHandle<SiStripFedCabling> tkFed) {

 psetAvailable=true;
  xsize=340;ysize=200;
  title=" ";
  jsfilename="CommonTools/TrackerMap/data/trackermap.txt";
  infilename="CommonTools/TrackerMap/data/tracker.dat";
  saveAsSingleLayer=false;
  if(tkmapPset.exists("trackermaptxtPath")){
  jsfilename=tkmapPset.getUntrackedParameter<std::string>("trackermaptxtPath","")+"trackermap.txt";
  cout << jsfilename << endl;
  infilename=tkmapPset.getUntrackedParameter<std::string>("trackerdatPath","")+"tracker.dat";
  cout << infilename << endl;
  ncrates=0;
  enableFedProcessing=tkmapPset.getUntrackedParameter<bool>("loadFedCabling",false);
  } else cout << "no parameters found" << endl;

 init();
// Now load fed cabling information
 if(enableFedProcessing){
 const vector<unsigned short> feds = tkFed->feds();
  cout<<"SiStripFedCabling has "<< feds.size()<<" active FEDS"<<endl;
    int num_board=0;
    int num_crate;
  for(vector<unsigned short>::const_iterator ifed = feds.begin();ifed<feds.end();ifed++){
    const std::vector<FedChannelConnection> theconn = tkFed->connections( *ifed );
    int num_conn=0;
    for(std::vector<FedChannelConnection>::const_iterator iconn = theconn.begin();iconn<theconn.end();iconn++){

      if( iconn->fedId()== sistrip::invalid_    ||  
	  iconn->detId() == sistrip::invalid_   ||  
	  iconn->detId() == sistrip::invalid32_ ||  
	  iconn->apvPairNumber() == sistrip::invalid_  ||
	  iconn->nApvPairs() == sistrip::invalid_ ) {
	continue;
      }
	
      TmModule *imod = imoduleMap[iconn->detId()];
      int key = iconn->fedId()*1000+iconn->fedCh();
      TmApvPair* apvpair = apvMap[key];
      if(apvpair!=0)cout << "Fed "<< iconn->fedId() << " channel " << iconn->fedCh() << " seem to be already loaded!"<<endl;
      else
	{
	  num_conn++;
	  if(num_conn==1){
	    if(fedMap[iconn->fedId()]==0){num_crate=num_board/18+1;fedMap[iconn->fedId()]=num_crate;num_board++;}
          }
	  apvpair = new TmApvPair(key,num_crate);
	  apvpair->mod=imod;
          apvpair->mpos=iconn->apvPairNumber();
	  apvMap[key] = apvpair;	
          apvModuleMap.insert(make_pair(iconn->detId(),apvpair));
          stringstream s;
          iconn->print(s);  
          apvpair->text=s.str();
	}
    }
  }
  ncrates=num_crate;
  cout << num_crate << " crates used "<< endl;
//Now add APv information to module name
    std::map<int , TmModule *>::iterator i_mod;
    for( i_mod=imoduleMap.begin();i_mod !=imoduleMap.end(); i_mod++){
      TmModule *  mod= i_mod->second;
      if(mod!=0) {
       ostringstream outs,outs1;
       outs << " connected to ";
       outs1 << "(";

      int idmod=mod->idex;
       int nchan=0;
       multimap<const int, TmApvPair*>::iterator pos;
       for (pos = apvModuleMap.lower_bound(idmod);
         pos != apvModuleMap.upper_bound(idmod); ++pos) {
       TmApvPair* apvpair = pos->second;
       if(apvpair!=0){
       outs << apvpair->mpos << " " <<apvpair->getFedId() << "/"<<apvpair->getFedCh()<<" ";
       outs1 << apvpair->idex+apvpair->crate*1000000<<",";
      nchan++;
    }

  }
       outs<< "("<<nchan<<")";
      mod->name=mod->name + outs.str(); 
      string s = outs1.str(); s.erase(s.end()-1,s.end());
      mod->capvids=s+")";
  }
  }
}
}


TrackerMap::TrackerMap(const edm::ParameterSet & tkmapPset) {
 psetAvailable=true;
  xsize=340;ysize=200;
  title=" ";
  jsfilename="CommonTools/TrackerMap/data/trackermap.txt";
  infilename="CommonTools/TrackerMap/data/tracker.dat";
  enableFedProcessing=false;ncrates=0;
  saveAsSingleLayer=false;
  if(tkmapPset.exists("trackermaptxtPath")){
  jsfilename=tkmapPset.getUntrackedParameter<std::string>("trackermaptxtPath","")+"trackermap.txt";
  cout << jsfilename << endl;
  infilename=tkmapPset.getUntrackedParameter<std::string>("trackerdatPath","")+"tracker.dat";
  cout << infilename << endl;
  } else cout << "no parameters found" << endl;
 init();
}

TrackerMap::TrackerMap(string s,int xsize1,int ysize1) {
 psetAvailable=false;
  xsize=xsize1;ysize=ysize1;
  title=s;
  jsfilename="CommonTools/TrackerMap/data/trackermap.txt";
  infilename="CommonTools/TrackerMap/data/tracker.dat";
  enableFedProcessing=false; 
  saveAsSingleLayer=false;
 init();
}

void TrackerMap::init() {
  
  int ntotmod=0;
  ix=0;iy=0; //used to compute the place of each layer in the tracker map
  firstcall = true;
  minvalue=0.; maxvalue=minvalue;
  posrel=true;
  palette = 1;
  printflag=false;
  temporary_file=false;

  ndet = 3; // number of detectors: pixel, inner silicon, outer silicon
  npart = 3; // number of detector parts: endcap -z, barrel, endcap +z

  //allocate module map
  for (int subdet=1; subdet < ndet+1; subdet++){//loop on subdetectors
    for (int detpart=1; detpart < npart+1; detpart++){//loop on subdetectors parts
      int nlayers = getlayerCount(subdet,detpart); // compute number of layers
      for(int layer=1; layer < nlayers+1; layer++){//loop on layers
	int nrings = getringCount(subdet,detpart,layer);// compute number of rings
	//fill arrays used to do the loop on the rings	
        int layer_g = nlayer(subdet,detpart,layer);
	ntotRing[layer_g-1]=nrings;
	firstRing[layer_g-1]=1;
	if(subdet==3 && detpart!=2)  firstRing[layer_g-1]= 8-nrings; //special numbering for TEC 
	for (int ring=firstRing[layer_g-1]; ring < ntotRing[layer_g-1]+firstRing[layer_g-1];ring++){//loop on rings
	  int nmodules = getmoduleCount(subdet,detpart,layer,ring);// compute number of modules
	  int key;
	  TmModule *smodule; 
          for(int module=1; module < nmodules+1; module++){//loop on modules
            smodule = new TmModule(module,ring,layer_g);
	    key=layer_g*100000+ring*1000+module;//key identifying module
	    smoduleMap[key]=smodule;
	    ntotmod++;
	  }
	  if(isRingStereo(key))for(int module=1; module < nmodules+1; module++){//loop on stereo modules
            smodule = new TmModule(module+100,ring,layer_g);
	    int key=layer_g*100000+ring*1000+module+100;
	    smoduleMap[key]=smodule;
	    ntotmod++;
	  }
	}
      }
    }
  }
 build();
}

TrackerMap::~TrackerMap() {
}
void TrackerMap::drawModule(TmModule * mod, int key,int nlay, bool print_total, ofstream * svgfile){
  //int x,y;
  double phi,r,dx,dy, dy1;
  double xp[4],yp[4],xp1,yp1;
  double vhbot,vhtop,vhapo;
  double rmedio[]={0.041,0.0701,0.0988,0.255,0.340,0.430,0.520,0.610,0.696,0.782,0.868,0.965,1.080};
  double xt1,yt1,xs1=0.,ys1=0.,xt2,yt2,xs2,ys2,pv1,pv2;
  int green = 0;
  int red = 0;
  int blue = 0;
  double xd[4],yd[4];
  int np = 4;
  //int numrec=0;
  int numod=0;
  phi = phival(mod->posx,mod->posy);
  r = sqrt(mod->posx*mod->posx+mod->posy*mod->posy);
  vhbot = mod->width;
  vhtop=mod->width;
  vhapo=mod->length;
  if(nlay < 31){ //endcap
    vhbot = mod->widthAtHalfLength/2.-(mod->width/2.-mod->widthAtHalfLength/2.);
    vhtop=mod->width/2.;
    vhapo=mod->length/2.;
    if(nlay >12 && nlay <19){
      if(posrel)r = r+r;
      xp[0]=r-vhtop;yp[0]=-vhapo;
      xp[1]=r+vhtop;yp[1]=-vhapo;
      xp[2]=r+vhtop;yp[2]=vhapo;
      xp[3]=r-vhtop;yp[3]=vhapo;
    }else{
      if(posrel)r = r + r/3.;
      xp[0]=r-vhapo;yp[0]=-vhbot;
      xp[1]=r+vhapo;yp[1]=-vhtop;
      xp[2]=r+vhapo;yp[2]=vhtop;
      xp[3]=r-vhapo;yp[3]=vhbot;
    }
    for(int j=0;j<4;j++){
      xp1 = xp[j]*cos(phi)-yp[j]*sin(phi);
      yp1 = xp[j]*sin(phi)+yp[j]*cos(phi);
      xp[j] = xp1;yp[j]=yp1;
    }
  } else { //barrel
    numod=mod->idModule;if(numod>100)numod=numod-100;
    int vane = mod->ring;
    if(posrel){
      dx = vhapo;
      phi=M_PI;
      xt1=rmedio[nlay-31]; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=rmedio[nlay-31]; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      dy=phival(xs2,ys2)-phival(xs1,ys1);
	 dy1 = dy;
      if(nlay==31)dy1=0.39;
      if(nlay==32)dy1=0.23;
      if(nlay==33)dy1=0.16;
      xp[0]=vane*(dx+dx/8.);yp[0]=numod*(dy1);
      xp[1]=vane*(dx+dx/8.)+dx;yp[1]=numod*(dy1);
      xp[2]=vane*(dx+dx/8.)+dx;yp[2]=numod*(dy1)+dy;
      xp[3]=vane*(dx+dx/8.);yp[3]=numod*(dy1)+dy;
    }else{
      xt1=r; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=r; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      pv1=phival(xs1,ys1);
      pv2=phival(xs2,ys2);
      if(fabs(pv1-pv2)>M_PI && numod==1)pv1=pv1-2.*M_PI;
      if(fabs(pv1-pv2)>M_PI && numod!=1)pv2=pv2+2.*M_PI;
      xp[0]=mod->posz-vhapo/2.;yp[0]=4.2*pv1;
      xp[1]=mod->posz+vhapo/2.;yp[1]=4.2*pv1;
      xp[2]=mod->posz+vhapo/2. ;yp[2]=4.2*pv2;
          xp[3]=mod->posz-vhapo/2.;yp[3]=4.2*pv2;
    }
  }
  if(isRingStereo(key))
        {
	  np = 3;
	  if(mod->idModule>100 ){for(int j=0;j<3;j++){
	      xd[j]=xdpixel(xp[j]);yd[j]=ydpixel(yp[j]);
	    }
	  }else {
	    xd[0]=xdpixel(xp[2]);yd[0]=ydpixel(yp[2]);
	    xd[1]=xdpixel(xp[3]);yd[1]=ydpixel(yp[3]);
	    xd[2]=xdpixel(xp[0]);yd[2]=ydpixel(yp[0]);
	  }
        } else {
    for(int j=0;j<4;j++){
      xd[j]=xdpixel(xp[j]);yd[j]=ydpixel(yp[j]);
    }
  }
  char buffer [20];
   sprintf(buffer,"%X",mod->idex);

 if(mod->red < 0){ //use count to compute color
 int color = getcolor(mod->value,palette);
     red=(color>>16)&0xFF;
     green=(color>>8)&0xFF;
     blue=(color)&0xFF;
  
if(!print_total)mod->value=mod->value*mod->count;//restore mod->value
  
  if(mod->count > 0)
    if(temporary_file) *svgfile << red << " " << green << " " << blue << " "; else
 *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" capvids=\""<<mod->capvids<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
  else
    if(temporary_file) *svgfile << 255 << " " << 255 << " " << 255 << " "; else
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" capvids=\""<<mod->capvids<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" \" fill=\"white\" points=\"";
  if(temporary_file) *svgfile << np << " ";
  for(int k=0;k<np;k++){
    if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; else
    *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  if(temporary_file)*svgfile << endl; else *svgfile <<"\" />" <<endl;
 } else {//color defined with fillc
  if(mod->red>255)mod->red=255;
  if(mod->green>255)mod->green=255;
  if(mod->blue>255)mod->blue=255;
    if(temporary_file) *svgfile << mod->red << " " << mod->green << " " << mod->blue << " "; else
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" capvids=\""<<mod->capvids<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" \" fill=\"rgb("<<mod->red<<","<<mod->green<<","<<mod->blue<<")\" points=\"";
  if(temporary_file) *svgfile << np << " ";
  for(int k=0;k<np;k++){
    if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; else
    *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  if(temporary_file)*svgfile << endl; else *svgfile <<"\" />" <<endl;
 }
  
}

//export  tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void TrackerMap::save(bool print_total,float minval, float maxval,std::string s,int width, int height){
  std::string filetype=s,outputfilename=s;
  filetype.erase(0,filetype.find(".")+1);
  outputfilename.erase(outputfilename.begin()+outputfilename.find("."),outputfilename.end());
  temporary_file=true;
  if(filetype=="svg")temporary_file=false;

  ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".coor";
  savefile = new ofstream(outs.str().c_str(),ios::out);
 if(!print_total){
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          mod->value = mod->value / mod->count;
        }
      }
    }
  }
  }
  if(minvalue>=maxvalue){
  minvalue=9999999.;
  maxvalue=-9999999.;
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          if (minvalue > mod->value)minvalue=mod->value;
          if (maxvalue < mod->value)maxvalue=mod->value;
        }
      }
    }
  }
}
  if(!temporary_file){
  *savefile << "<?xml version=\"1.0\"  standalone=\"no\" ?>"<<endl;
  *savefile << "<svg  xmlns=\"http://www.w3.org/2000/svg\""<<endl;
  *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\" "<<endl;
  *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"<<endl;
  *savefile << "<svg:svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0 3000 1600"<<"\" width=\""<<width<<"\" height=\""<<height<<"\">"<<endl;
  *savefile << "<svg:rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"3000\" height=\"1600\" /> "<<endl;
  *savefile << "<svg:g id=\"tracker\" transform=\"translate(10,1500) rotate(270)\" style=\"fill:none;stroke:black;stroke-width:0;\"> "<<endl;
   }
  for (int layer=1; layer < 44; layer++){
    nlay=layer;
    defwindow(nlay);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
          drawModule(mod,key,layer,print_total,savefile);
        }
      }
    }
  }
  if(!temporary_file){
    *savefile << "</svg:g>"<<endl;
    *savefile << " <svg:text id=\"Title\" class=\"normalText\"  x=\"300\" y=\"0\">"<<title<<"</svg:text>"<<endl;
  }
  if(printflag)drawPalette(savefile);
  if(!temporary_file){
    *savefile << "</svg:svg>"<<endl;
    *savefile << "</svg>"<<endl;
  }
  savefile->close(); 

  const char * command1;
  string tempfilename = outputfilename + ".coor";
  if(filetype=="svg"){
    string command = "mv "+tempfilename +" " +outputfilename + ".svg";
    command1=command.c_str();
    cout << "Executing " << command1 << endl;
    system(command1);
  }
  if (temporary_file){ // create root trackermap image
    int red,green,blue,npoints,colindex,ncolor;
    double x[4],y[4];
    ifstream tempfile(tempfilename.c_str(),ios::in);
    TCanvas *MyC = new TCanvas("MyC", "TrackerMap",width,height);
    gPad->SetFillColor(38);
    
    gPad->Range(0,0,3000,1600);
    
    //First  build palette
    ncolor=0;
    typedef std::map<int,int> ColorList;
    ColorList colorList;
    ColorList::iterator pos;
    TColor *col;
    while(!tempfile.eof()) {
      tempfile  >> red >> green  >> blue >> npoints; 
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos == colorList.end()){ colorList[colindex]=ncolor+100; col =gROOT->GetColor(ncolor+100);
if(col) col->SetRGB((Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.)); else TColor *c = new TColor(ncolor+100,(Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));ncolor++;}
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
    }
    if(ncolor>0 && ncolor<10000){
      Int_t colors[10000];
      for(int i=0;i<ncolor;i++){colors[i]=i+100;}
      gStyle->SetPalette(ncolor,colors);
    }
    tempfile.clear();
    tempfile.seekg(0,ios::beg);
    cout << "created palette with " << ncolor << " colors" << endl;
    TPolyLine*  pline = new TPolyLine();
    while(!tempfile.eof()) {//create polylines
      tempfile  >> red >> green  >> blue >> npoints; 
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos != colorList.end()){
	pline->SetFillColor(colorList[colindex]);
	pline->SetLineWidth(0);
        pline->DrawPolyLine(npoints,y,x,"f");
      }
    }
    MyC->Update();
    if(filetype=="png"){
      std::cout << "printing " << std::endl;
      string filename = outputfilename + ".png";
      MyC->Print(filename.c_str());
    }
    if(filetype=="jpg"){
      string filename = outputfilename + ".jpg";
      MyC->Print(filename.c_str());
    }
    if(filetype=="pdf"){
      string filename = outputfilename + ".pdf";
      MyC->Print(filename.c_str());
    }
    string command = "rm "+tempfilename ;
    command1=command.c_str();
    cout << "Executing " << command1 << endl;
    system(command1);
    MyC->Clear();
    delete MyC;
    delete pline;
  }
  
  
}
void TrackerMap::drawApvPair(int crate, int numfed_incrate, bool print_total, TmApvPair* apvPair,ofstream * svgfile,bool useApvPairValue)
{
  double xp[4],yp[4];
  int color;
  int green = 0;
  int red = 0;
  int blue = 0;
  double xd[4],yd[4];
  int np = 4;
  double boxinitx=0., boxinity=0.; 
  double dx=.9,dy=.9;
  int numfedch_incolumn = 12;
  int numfedch_inrow = 8;
  int numfed_incolumn = 5;
  int numfed_inrow = 4;
  boxinitx=boxinitx+(numfed_incolumn-(numfed_incrate-1)/numfed_inrow)*14.;
  boxinity=boxinity+(numfed_inrow-(numfed_incrate-1)%numfed_inrow)*9.;
  boxinity=boxinity+numfedch_inrow-(apvPair->getFedCh()/numfedch_incolumn);
  boxinitx = boxinitx+numfedch_incolumn-(int)(apvPair->getFedCh()%numfedch_incolumn);
  //cout << crate << " " << numfed_incrate << " " << apvPair->getFedCh()<<" "<<boxinitx<< " " << boxinity << endl; ;
  xp[0]=boxinitx;yp[0]=boxinity;
  xp[1]=boxinitx+dx;yp[1]=boxinity;
  xp[2]=boxinitx+dx;yp[2]=boxinity + dy;
  xp[3]=boxinitx;yp[3]=boxinity + dy;
  for(int j=0;j<4;j++){
    xd[j]=xdpixelc(xp[j]);yd[j]=ydpixelc(yp[j]);
    //cout << boxinity << " "<< ymax << " "<< yp[j] << endl;
  }
  
  char buffer [20];
  sprintf(buffer,"%X",apvPair->mod->idex);
  string s = apvPair->mod->name;
  s.erase(s.begin()+s.find("connected"),s.end());

  if(useApvPairValue){ 
    if(apvPair->red < 0){ //use count to compute color
      if(apvPair->count > 0) {
	color = getcolor(apvPair->value,palette);
	red=(color>>16)&0xFF;
	green=(color>>8)&0xFF;
	blue=(color)&0xFF;
	if(!print_total)apvPair->value=apvPair->value*apvPair->count;//restore mod->value
	if(temporary_file)*svgfile << red << " " << green << " " << blue << " ";
           else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
      } else {
        if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\"  cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"white\" points=\"";
      }
    } else {//color defined with fillc
      if(apvPair->red>255)apvPair->red=255;
      if(apvPair->green>255)apvPair->green=255;
      if(apvPair->blue>255)apvPair->blue=255;
      if(temporary_file)*svgfile << apvPair->red << " " << apvPair->green << " " << apvPair->blue << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<apvPair->red<<","<<apvPair->green<<","<<apvPair->blue<<")\" points=\"";
    }
  }else{
    if(apvPair->mod->red < 0){ //use count to compute color
      if(apvPair->mod->count > 0) {
	color = getcolor(apvPair->mod->value,palette);
	red=(color>>16)&0xFF;
	green=(color>>8)&0xFF;
	blue=(color)&0xFF;
	if(temporary_file)*svgfile << red << " " << green << " " << blue << " ";
           else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
      } else {
        if(temporary_file)*svgfile << 255 << " " << 255 << " " << 255 << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\"  cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"white\" points=\"";
      }
    } else {//color defined with fillc
      if(apvPair->mod->red>255)apvPair->mod->red=255;
      if(apvPair->mod->green>255)apvPair->mod->green=255;
      if(apvPair->mod->blue>255)apvPair->mod->blue=255;
      if(temporary_file)*svgfile << apvPair->mod->red << " " << apvPair->mod->green << " " << apvPair->mod->blue << " ";
         else *svgfile <<"<svg:polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex+crate*1000000<<"\" cmodid=\""<<apvPair->mod->getKey()<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\"""\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<s<<" Id "<<buffer<<" \" fill=\"rgb("<<apvPair->mod->red<<","<<apvPair->mod->green<<","<<apvPair->mod->blue<<")\" points=\"";
    }
  }
  if(temporary_file)*svgfile << np << " ";
  for(int k=0;k<np;k++){
    if(temporary_file)*svgfile << xd[k] << " " << yd[k] << " " ; 
      else *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  if(temporary_file)*svgfile << endl;
     else *svgfile <<"\" />" <<endl;
}

void TrackerMap::save_as_fedtrackermap(bool print_total,float minval, float maxval,std::string s,int width, int height){
 if(enableFedProcessing){
  std::string filetype=s,outputfilename=s;
  filetype.erase(0,filetype.find(".")+1);
  outputfilename.erase(outputfilename.begin()+outputfilename.find("."),outputfilename.end());
  temporary_file=true;
  if(filetype=="xml")temporary_file=false;
  ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".coor";
  if(temporary_file)savefile = new ofstream(outs.str().c_str(),ios::out);
  std::map<int , TmApvPair *>::iterator i_apv;
  std::map<int , int>::iterator i_fed;
  //Decide if we must use Module or ApvPair value
  bool useApvPairValue=false;
  for( i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
    TmApvPair *  apvPair= i_apv->second;
    if(apvPair!=0) {
      TmModule * apv_mod = apvPair->mod;
      if(apv_mod !=0 && !apv_mod->notInUse()){
        if(apvPair->count > 0 || apvPair->red!=-1) { useApvPairValue=true; break;}
      }
    }
  }
  if(!print_total){
    for( i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
      TmApvPair *  apvPair= i_apv->second;
      if(apvPair!=0) {
	TmModule * apv_mod = apvPair->mod;
	if(apv_mod !=0 && !apv_mod->notInUse()){
	  if(useApvPairValue) apvPair->value = apvPair->value / apvPair->count;
	  else if(apvPair->mpos==0)apv_mod->value = apv_mod->value / apv_mod->count;
	}
      }
    }
  }
  if(minvalue>=maxvalue){
    
    minvalue=9999999.;
    maxvalue=-9999999.;
    for(i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
	TmApvPair *  apvPair= i_apv->second;
	if(apvPair!=0) {
	  TmModule * apv_mod = apvPair->mod;
	  if( apv_mod !=0 && !apv_mod->notInUse()){
	    if(useApvPairValue){
	      if (minvalue > apvPair->value)minvalue=apvPair->value;
	      if (maxvalue < apvPair->value)maxvalue=apvPair->value;
	    } else {
	      if (minvalue > apv_mod->value)minvalue=apv_mod->value;
	      if (maxvalue < apv_mod->value)maxvalue=apv_mod->value;
	    }
	  }
	}
    }
  }
  for (int crate=1; crate < (ncrates+1); crate++){
    if(!temporary_file){
      saveAsSingleLayer=true;
      ostringstream outs;
    outs << outputfilename<<"crate" <<crate<< ".xml";
    savefile = new ofstream(outs.str().c_str(),ios::out);
    *savefile << "<?xml version=\"1.0\" standalone=\"no\"?>"<<endl;
    *savefile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<<endl;
    *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\""<<endl;
    *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" >"<<endl;
    *savefile << "<script type=\"text/ecmascript\" xlink:href=\"crate.js\" />"<<endl;
    *savefile << "<svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0  500 500\" width=\"700\" height=\"700\" onload=\"TrackerCrate.init()\">"<<endl;
    *savefile << "<rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"700\" height=\"700\" />"<<endl;
    *savefile << "<g id=\"crate\" transform=\" translate(150,500) rotate(270) scale(1.,1.)\"  > "<<endl;
         }
    ncrate=crate;
    defcwindow(ncrate);
    int numfed_incrate=0;
    for (i_fed=fedMap.begin();i_fed != fedMap.end(); i_fed++){
      if(i_fed->second == crate){
	int fedId = i_fed->first;
	numfed_incrate++;
	for (int nconn=0;nconn<96;nconn++){
	  int key = fedId*1000+nconn; 
	  TmApvPair *  apvPair= apvMap[key];
	  if(apvPair !=0){
	    TmModule * apv_mod = apvPair->mod;
	    if(apv_mod !=0 && !apv_mod->notInUse()){
	      drawApvPair(crate,numfed_incrate,print_total,apvPair,savefile,useApvPairValue);
	    }
	  } 
	}
      }
    }
   if(!temporary_file){
    *savefile << "</g> </svg> <text id=\"currentElementText\" x=\"40\" y=\"30\"> - </text> </svg>" << endl;
    savefile->close();
     saveAsSingleLayer=false;
      }
    }
  if(!print_total && !useApvPairValue){
//Restore module value
    for( i_apv=apvMap.begin();i_apv !=apvMap.end(); i_apv++){
      TmApvPair *  apvPair= i_apv->second;
      if(apvPair!=0) {
	TmModule * apv_mod = apvPair->mod;
	if(apv_mod !=0 && apvPair->mpos==0 && !apv_mod->notInUse()){
	  apv_mod->value = apv_mod->value * apv_mod->count;
	}
      }
    }
}
  
  if(temporary_file){
    if(printflag)drawPalette(savefile);
  savefile->close(); 

  const char * command1;
  string tempfilename = outputfilename + ".coor";
    int red,green,blue,npoints,colindex,ncolor;
    double x[4],y[4];
    ifstream tempfile(tempfilename.c_str(),ios::in);
    TCanvas *MyC = new TCanvas("MyC", "TrackerMap",width,height);
    gPad->SetFillColor(38);
    
    gPad->Range(0,0,3000,1600);
    
    //First  build palette
    ncolor=0;
    typedef std::map<int,int> ColorList;
    ColorList colorList;
    ColorList::iterator pos;
    TColor *col;
    while(!tempfile.eof()) {
      tempfile  >> red >> green  >> blue >> npoints; 
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos == colorList.end()){ 
	colorList[colindex]=ncolor+100; 
	col =gROOT->GetColor(ncolor+100);
	if(col) 
	  col->SetRGB((Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.)); 
	else 
	  TColor *c = new TColor(ncolor+100,(Double_t)(red/255.),(Double_t)(green/255.),(Double_t)(blue/255.));
	ncolor++;
      }
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
    }
    if(ncolor>0 && ncolor<10000){
      Int_t colors[10000];
      for(int i=0;i<ncolor;i++){colors[i]=i+100;}
      gStyle->SetPalette(ncolor,colors);
    }
    tempfile.clear();
    tempfile.seekg(0,ios::beg);
    cout << "created palette with " << ncolor << " colors" << endl;
    TPolyLine*  pline = new TPolyLine();
    while(!tempfile.eof()) {//create polylines
      tempfile  >> red >> green  >> blue >> npoints; 
      for (int i=0;i<npoints;i++){
	tempfile >> x[i] >> y[i];  
      }
      colindex=red+green*1000+blue*1000000;
      pos=colorList.find(colindex); 
      if(pos != colorList.end()){
	pline->SetFillColor(colorList[colindex]);
	pline->SetLineWidth(0);
        pline->DrawPolyLine(npoints,y,x,"f");
      }
    }
    MyC->Update();
    std::cout << "Filetype " << filetype << std::endl;
    if(filetype=="png"){
      string filename = outputfilename + ".png";
      MyC->Print(filename.c_str());
    }
    if(filetype=="jpg"){
      string filename = outputfilename + ".jpg";
      MyC->Print(filename.c_str());
    }
    if(filetype=="pdf"){
      string filename = outputfilename + ".pdf";
      MyC->Print(filename.c_str());
    }
    string command = "rm "+tempfilename ;
    command1=command.c_str();
    cout << "Executing " << command1 << endl;
    system(command1);
    MyC->Clear();
    delete MyC;
    delete pline;
  
  
}//if(temporary_file)
}//if(enabledFedProcessing)
}

//print in svg format tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void TrackerMap::print(bool print_total, float minval, float maxval, string outputfilename){
  temporary_file=false;
  ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".xml";
  svgfile = new ofstream(outs.str().c_str(),ios::out);
  jsfile = new ifstream(edm::FileInPath(jsfilename).fullPath().c_str(),ios::in);

  //copy javascript interface from trackermap.txt file
  string line;
  while (getline( *jsfile, line ))
        {
            *svgfile << line << endl;
        }
  //
 if(!print_total){
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          mod->value = mod->value / mod->count;
        }
      }
    }
  }
  }
  if(minvalue>=maxvalue){
  minvalue=9999999.;
  maxvalue=-9999999.;
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          if (minvalue > mod->value)minvalue=mod->value;
          if (maxvalue < mod->value)maxvalue=mod->value;
        }
      }
    }
  }
}
  for (int layer=1; layer < 44; layer++){
    nlay=layer;
    defwindow(nlay);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
          drawModule(mod,key,layer,print_total,svgfile);
        }
      }
    }
  }
  *svgfile << "</svg:g></svg:svg>"<<endl;
  *svgfile << " <svg:text id=\"Title\" class=\"normalText\"  x=\"300\" y=\"0\">"<<title<<"</svg:text>"<<endl;
  if(printflag)drawPalette(svgfile);
  *svgfile << "</svg:svg>"<<endl;
  *svgfile << "</body></html>"<<endl;
   svgfile->close();

}

void TrackerMap::drawPalette(ofstream * svgfile){
  int color,red, green, blue;
  float val=minvalue;
  int paletteLength = 250;
  float dval = (maxvalue-minvalue)/(float)paletteLength;
  for(int i=0;i<paletteLength;i++){
  color = getcolor(val,palette);
     red=(color>>16)&0xFF;
     green=(color>>8)&0xFF;
     blue=(color)&0xFF;
    if(!temporary_file)*svgfile <<"<svg:rect  x=\""<<i<<"\" y=\"0\" width=\"1\" height=\"20\" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" />\n";
    if(i%50 == 0){
       if(!temporary_file)*svgfile <<"<svg:rect  x=\""<<i<<"\" y=\"10\" width=\"1\" height=\"10\" fill=\"black\" />\n";
      if(i%100==0&&!temporary_file)*svgfile << " <svg:text  class=\"normalText\"  x=\""<<i<<"\" y=\"30\">" <<val<<"</svg:text>"<<endl;
       }
    val = val + dval;
   }
} 
void TrackerMap::fillc_fed_channel(int fedId,int fedCh, int red, int green, int blue  )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  
  if(apvpair!=0){
    apvpair->red=red; apvpair->green=green; apvpair->blue=blue;
    return;
  }
  cout << "*** error in FedTrackerMap fillc method ***";
}

void TrackerMap::fill_fed_channel(int idmod, float qty  )
{
  multimap<const int, TmApvPair*>::iterator pos;
  for (pos = apvModuleMap.lower_bound(idmod);
         pos != apvModuleMap.upper_bound(idmod); ++pos) {
  TmApvPair* apvpair = pos->second;
  if(apvpair!=0){
    apvpair->value=apvpair->value+qty;
    apvpair->count++;
  }
  }
    return;
  cout << "*** error in FedTrackerMap fill by module method ***";
  }

void TrackerMap::fill_current_val_fed_channel(int fedId, int fedCh, float current_val )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  
  if(apvpair!=0)  apvpair->value=current_val;
  else 
    cout << "*** error in FedTrackerMap fill_current_val method ***";
}

int TrackerMap::module(int fedId, int fedCh)
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  if(apvpair!=0){
    return(apvpair->mod->idex);
  }
  return(0);
  cout << "*** error in FedTrackerMap module method ***";
}
void TrackerMap::fill_fed_channel(int fedId, int fedCh, float qty )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = apvMap[key];
  if(apvpair!=0){
    apvpair->value=apvpair->value+qty;
    apvpair->count++;
    return;
  }
  cout << "*** error inFedTrackerMap fill method ***";
}
void TrackerMap::fillc(int idmod, int red, int green, int blue  ){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0){
     mod->red=red; mod->green=green; mod->blue=blue;
    return;
  }
  cout << "**************************error in fill method **************";
}
void TrackerMap::fillc(int layer, int ring, int nmod, int red, int green, int blue  ){
  
  int key = layer*10000+ring*1000+nmod;
  TmModule * mod = smoduleMap[key];

  if(mod!=0){
     mod->red=red; mod->green=green; mod->blue=blue;
    return;
  }
  cout << "**************************error in fill method **************";
}

void TrackerMap::fill_current_val(int idmod, float current_val ){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0)  mod->value=current_val;
  else cout << "**error in fill_current_val method ***";
}

void TrackerMap::fill(int idmod, float qty ){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0){
    mod->value=mod->value+qty;
    mod->count++;
    return;
  }
  cout << "**************************error in fill method **************";
}

void TrackerMap::fill(int layer, int ring, int nmod,  float qty){

  int key = layer*100000+ring*1000+nmod;
  TmModule * mod = smoduleMap[key];
  if(mod!=0){
     mod->value=mod->value+qty;
     mod->count++;
  }
  else cout << "**************************error in SvgModuleMap **************";
} 

void TrackerMap::setText(int idmod, string s){

  TmModule * mod = imoduleMap[idmod];
  if(mod!=0){
     mod->text=s;
  }
  else cout << "**************************error in IdModuleMap **************";
}


void TrackerMap::setText(int layer, int ring, int nmod, string s){

  int key = layer*100000+ring*1000+nmod;
  TmModule * mod = smoduleMap[key];
  if(mod!=0){
     mod->text=s;
  }
  else cout << "**************************error in SvgModuleMap **************";
} 


void TrackerMap::build(){
  //  ifstream* infile;

  int nmods, pix_sil, fow_bar, ring, nmod, layer;
  unsigned int idex;
  float posx, posy, posz, length, width, thickness, widthAtHalfLength;
  int iModule=0,old_layer=0, ntotMod =0;
  string name,dummys;
  ifstream infile(edm::FileInPath(infilename).fullPath().c_str(),ios::in);
  while(!infile.eof()) {
    infile >> nmods >> pix_sil >> fow_bar >> layer >> ring >> nmod >> posx >> posy
	   >> posz>> length >> width >> thickness
	   >> widthAtHalfLength >> idex ;
    getline(infile,dummys); //necessary to reach end of record
    getline(infile,name); 
    if(old_layer!=layer){old_layer=layer;iModule=0;}
    iModule++;
    ntotMod++;
    int key=layer*100000+ring*1000+nmod;
    TmModule * mod = smoduleMap[key];
    
    imoduleMap[idex]=mod;

    if(mod==0) cout << "error in module "<<key <<endl;
    else
      {
          mod->posx = posx;
          mod->posy = posy;
          mod->setUsed();
          mod->value=0;
          mod->count=0;
          mod->posz = posz;
          mod->length = length;
          mod->width = width;
          mod->thickness = thickness;
          mod->widthAtHalfLength = widthAtHalfLength;
          mod->idex = idex;
          mod->name = name;
      }
  }
  infile.close();
  number_modules = ntotMod-1;
}
int TrackerMap::getcolor(float value,int palette){
   int red,green,blue;
   if(palette==1){//palette1 1 - raibow
   float delta=(maxvalue-minvalue);
   float x =(value-minvalue);
   red = (int) ( x<(delta/2) ? 0 : ( x > ((3./4.)*delta) ?  255 : 255/(delta/4) * (x-(2./4.)*delta)  ) );
   green= (int) ( x<delta/4 ? (x*255/(delta/4)) : ( x > ((3./4.)*delta) ?  255-255/(delta/4) *(x-(3./4.)*delta) : 255 ) );
   blue = (int) ( x<delta/4 ? 255 : ( x > ((1./2.)*delta) ?  0 : 255-255/(delta/4) * (x-(1./4.)*delta) ) );
     }
     if (palette==2){//palette 2 yellow-green
     green = (int)((value-minvalue)/(maxvalue-minvalue)*256.);
         if (green > 255) green=255;
         red = 255; blue=0;green=255-green;  
        } 
   return(blue|(green<<8)|(red<<16));
}
void TrackerMap::printall(bool print_total, float minval, float maxval, string outputfilename){
//Copy interface
  std::ofstream * ofilename;
  std::ifstream * ifilename;
  std::ostringstream ofname;
  std::string ifname;
  string line;
  ifname="CommonTools/TrackerMap/data/viewer.xhtml";
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),ios::in);
  ofname << outputfilename << "viewer.xhtml";
  ofilename = new ofstream(ofname.str().c_str(),ios::out);
*ofilename <<"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\""<<endl;
*ofilename <<"    \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">"<<endl;
*ofilename <<"<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"en\" xml:lang=\"en\">"<<endl;
*ofilename <<"  <head>"<<endl;
*ofilename <<"    <meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\" />"<<endl;
*ofilename <<"    <title>TrackerMap Viewer</title>"<<endl;
*ofilename <<"    <link rel=\"stylesheet\" type=\"text/css\" href=\""<<outputfilename<<"viewer.css\" />"<<endl;
*ofilename <<"    <script type=\"text/javascript\" src=\""<<outputfilename<<"viewer.js\">"<<endl;
*ofilename <<"    </script>"<<endl;
*ofilename <<"    <script type=\"text/javascript\">"<<endl;
*ofilename <<"    //<![CDATA["<<endl;
*ofilename <<"    var tmapname=\"" <<outputfilename << "\""<<endl;
*ofilename <<"    var ncrates=" <<ncrates << ";"<<endl;
  while (getline( *ifilename, line )) { *ofilename << line << endl; }
  ofname.str("");
  ifname="CommonTools/TrackerMap/data/viewer.css";
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),ios::in);
  ofname << outputfilename << "viewer.css";
  ofilename = new ofstream(ofname.str().c_str(),ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << endl; }
  ofname.str("");
  ifname="CommonTools/TrackerMap/data/viewer.js";
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),ios::in);
  ofname << outputfilename << "viewer.js";
  ofilename = new ofstream(ofname.str().c_str(),ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << endl; }
  ofname.str("");
  ifname="CommonTools/TrackerMap/data/crate.js";
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),ios::in);
  ofname <<  "crate.js";
  ofilename = new ofstream(ofname.str().c_str(),ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << endl; }
  ofname.str("");
  ifname="CommonTools/TrackerMap/data/layer.js";
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),ios::in);
  ofname <<  "layer.js";
  ofilename = new ofstream(ofname.str().c_str(),ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << endl; }
  ofname.str("");
  ifname="CommonTools/TrackerMap/data/null.png";
  ifilename = new ifstream(edm::FileInPath(ifname).fullPath().c_str(),ios::in);
  ofname <<  "null.png";
  ofilename = new ofstream(ofname.str().c_str(),ios::out);
  while (getline( *ifilename, line )) { *ofilename << line << endl; }
  
    ostringstream outs,outs1,outs2;
    outs << outputfilename<<".png";
save(true,0.,0.,outs.str(),3000,1600);
temporary_file=false;
printlayers(true,0.,0.,outputfilename);

//Now print a text file for each layer 
  ofstream * txtfile;
for (int layer=1; layer < 44; layer++){
    ostringstream outs;
    outs << outputfilename <<"layer"<<layer<< ".html";
    txtfile = new ofstream(outs.str().c_str(),ios::out);
    *txtfile << "<html><head></head> <body>" << endl;
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
            int idmod=mod->idex;
            int nchan=0;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<endl;
             multimap<const int, TmApvPair*>::iterator pos;
             for (pos = apvModuleMap.lower_bound(idmod);
                pos != apvModuleMap.upper_bound(idmod); ++pos) {
               TmApvPair* apvpair = pos->second;
               if(apvpair!=0){
                   nchan++;
                   *txtfile  <<  apvpair->text << endl;
                    }

                    }
                   *txtfile  << "</pre><h3>"<< mod->name<<"</h3>"<<endl;
                  }
                }
                }
    *txtfile << "</body></html>" << endl;
    txtfile->close();
                }
if(enableFedProcessing){
    outs1 << outputfilename<<"fed.png";
save_as_fedtrackermap(true,0.,0.,outs1.str(),3000,1600);
    outs2 << outputfilename<<".xml";
save_as_fedtrackermap(true,0.,0.,outs2.str(),3000,1600);
//And a text file for each crate 
  std::map<int , int>::iterator i_fed;
  ofstream * txtfile;
  for (int crate=1; crate < (ncrates+1); crate++){
    ostringstream outs;
    outs << outputfilename <<"crate"<<crate<< ".html";
    txtfile = new ofstream(outs.str().c_str(),ios::out);
    *txtfile << "<html><head></head> <body>" << endl;
    for (i_fed=fedMap.begin();i_fed != fedMap.end(); i_fed++){
      if(i_fed->second == crate){
	int fedId = i_fed->first;
	for (int nconn=0;nconn<96;nconn++){
	  int key = fedId*1000+nconn; 
	  TmApvPair *  apvPair= apvMap[key];
	  if(apvPair !=0){
            int idmod=apvPair->idex;
            *txtfile  << "<a name="<<idmod<<"><pre>"<<endl;
            *txtfile  <<  apvPair->text << endl;
            ostringstream outs;
            outs << "fedchannel "  <<apvPair->getFedId() << "/"<<apvPair->getFedCh()<<" connects to module  " << apvPair->mod->idex ;
            *txtfile  << "</pre><h3>"<< outs.str()<<"</h3>"<<endl;
             }
          }
      }
      }
    *txtfile << "</body></html>" << endl;
    txtfile->close();
                }
  }
}
void TrackerMap::printlayers(bool print_total, float minval, float maxval, string outputfilename){
  ofstream * xmlfile;
saveAsSingleLayer=true;
if(!print_total){
    for (int layer=1; layer < 44; layer++){
      for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
	for (int module=1;module<200;module++) {
	  int key=layer*100000+ring*1000+module;
	  TmModule * mod = smoduleMap[key];
	  if(mod !=0 && !mod->notInUse()){
	    mod->value = mod->value / mod->count;
	  }
	}
      }
    }
  }
  if(minvalue>=maxvalue){
    minvalue=9999999.;
    maxvalue=-9999999.;
    for (int layer=1; layer < 44; layer++){
      for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
	for (int module=1;module<200;module++) {
	  int key=layer*100000+ring*1000+module;
	  TmModule * mod = smoduleMap[key];
	  if(mod !=0 && !mod->notInUse()){
	    if (minvalue > mod->value)minvalue=mod->value;
	    if (maxvalue < mod->value)maxvalue=mod->value;
	  }
	}
      }
    }
  }
for (int layer=1; layer < 44; layer++){
    ostringstream outs;
    outs << outputfilename <<"layer"<<layer<< ".xml";
    xmlfile = new ofstream(outs.str().c_str(),ios::out);
    *xmlfile << "<?xml version=\"1.0\" standalone=\"no\"?>"<<endl;
    *xmlfile << "<svg xmlns=\"http://www.w3.org/2000/svg\""<<endl;
    *xmlfile << "xmlns:svg=\"http://www.w3.org/2000/svg\""<<endl;
    *xmlfile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" >"<<endl;
    *xmlfile << "<script type=\"text/ecmascript\" xlink:href=\"layer.js\" />"<<endl;
    *xmlfile << "<svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0  500 500\" width=\"700\" height=\"700\" onload=\"TrackerLayer.init()\">"<<endl;
    if(layer<31)*xmlfile << "<g id=\"layer\" transform=\" translate(0,400) rotate(270) scale(1.,2.)\"  > "<<endl;
    else *xmlfile << "<g id=\"layer\" transform=\" translate(0,400) rotate(270) scale(1.,1.)\"  > "<<endl;
    *xmlfile << "<rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"700\" height=\"700\" />"<<endl;
    nlay=layer;
    defwindow(nlay);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
          drawModule(mod,key,layer,print_total,xmlfile);
        }
      }
    }
    *xmlfile << "</g> </svg> <text id=\"currentElementText\" x=\"40\" y=\"30\"> - </text> </svg>" << endl;
    xmlfile->close();
  }
saveAsSingleLayer=false;
}
