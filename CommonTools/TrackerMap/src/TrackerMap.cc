#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

/**********************************************************
Allocate all the modules in a map of TmModule
The filling of the values for each module is done later
when the user starts to fill it.
**********************************************************/

TrackerMap::TrackerMap(string s,int xsize1,int ysize1) {
  
  int ntotmod=0;
  xsize=xsize1;ysize=ysize1;ix=0;iy=0; //used to compute the place of each layer in the tracker map
  firstcall = true;
  minvalue=0.; maxvalue=minvalue;
  title=s;
  posrel=true;
  palette = 1;
  printflag=false;

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
   if(palette==1){//palette1 1 - raibow
   float delta=(maxvalue-minvalue);
   float x =(mod->value-minvalue);
   red = (int) ( x<(delta/2) ? 0 : ( x > ((3./4.)*delta) ?  255 : 255/(delta/4) * (x-(2./4.)*delta)  ) );
   green= (int) ( x<delta/4 ? (x*255/(delta/4)) : ( x > ((3./4.)*delta) ?  255-255/(delta/4) *(x-(3./4.)*delta) : 255 ) );
   blue = (int) ( x<delta/4 ? 255 : ( x > ((1./2.)*delta) ?  0 : 255-255/(delta/4) * (x-(1./4.)*delta) ) );
     }
     if (palette==2){//palette 2 yellow-green
     green = (int)((mod->value-minvalue)/(maxvalue-minvalue)*256.);
         if (green > 255) green=255;
         red = 255; blue=0;green=255-green;  
        } 

if(!print_total)mod->value=mod->value*mod->count;//restore mod->value
  
  if(mod->count > 0)
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" Id "<<mod->idex<<" \" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" points=\"";
  else
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\"  onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" Id "<<mod->idex<<" \" fill=\"white\" points=\"";
  for(int k=0;k<np;k++){
    *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  *svgfile <<"\" />" <<endl;
 } else {//color defined with fillc
  if(mod->red>255)mod->red=255;
  if(mod->green>255)mod->green=255;
  if(mod->blue>255)mod->blue=255;
    *svgfile <<"<svg:polygon detid=\""<<mod->idex<<"\" count=\""<<mod->count <<"\" value=\""<<mod->value<<"\" id=\""<<key<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<mod->text<<"\" POS=\""<<mod->name<<" Id "<<mod->idex<<" \" fill=\"rgb("<<mod->red<<","<<mod->green<<","<<mod->blue<<")\" points=\"";
  for(int k=0;k<np;k++){
    *svgfile << xd[k] << "," << yd[k] << " " ;
  }
  *svgfile <<"\" />" <<endl;
 }
  
}


//export  tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void TrackerMap::save(bool print_total, float minval, float maxval, string outputfilename,int width, int height, string filetype){
  ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".svg";
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
  *savefile << "<?xml version=\"1.0\"  standalone=\"no\" ?>"<<endl;
  *savefile << "<svg  xmlns=\"http://www.w3.org/2000/svg\""<<endl;
  *savefile << "xmlns:svg=\"http://www.w3.org/2000/svg\" "<<endl;
  *savefile << "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"<<endl;
  *savefile << "<svg:svg id=\"mainMap\" x=\"0\" y=\"0\" viewBox=\"0 0 3000 1600"<<"\" width=\""<<width<<"\" height=\""<<height<<"\">"<<endl;
  *savefile << "<svg:rect fill=\"lightblue\" stroke=\"none\" x=\"0\" y=\"0\" width=\"3000\" height=\"1600\" /> "<<endl;
  *savefile << "<svg:g id=\"tracker\" transform=\"translate(10,1500) rotate(270)\" style=\"fill:none;stroke:black;stroke-width:0;\"> "<<endl;
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
  *savefile << "</svg:g>"<<endl;
  *savefile << " <svg:text id=\"Title\" class=\"normalText\"  x=\"300\" y=\"0\">"<<title<<"</svg:text>"<<endl;
  if(printflag)drawPalette(savefile);
  *savefile << "</svg:svg>"<<endl;
  *savefile << "</svg>"<<endl;
  savefile->close(); 

 string tempfilename = outputfilename + ".svg";
 const char * command1;
 if(filetype=="png"){
 ostringstream commands;
 commands << "java -Xmx256m  -jar batik/batik-rasterizer.jar -w "<<width<<" -h "<< height << " " <<tempfilename; 
 command1=commands.str().c_str();
 cout << "Executing " << command1 << endl;
 system(command1);
  }
 if(filetype=="jpg"){
 ostringstream commands;
 commands << "java -Xmx256m  -jar batik/batik-rasterizer.jar -w "<<width<<" -h "<< height << " " <<tempfilename <<" -m image/jpeg -q 0.8 "; 
 command1=commands.str().c_str();
 cout << "Executing " << command1 << endl;
 system(command1);
  }
 if(filetype=="pdf"){
 ostringstream commands;
 commands << "java -Xmx256m  -jar batik/batik-rasterizer.jar -w "<<width<<" -h "<< height << " " <<tempfilename<<" -m application/pdf"; 
 command1=commands.str().c_str();
 cout << "Executing " << command1 << endl;
 system(command1);
  }
 if(filetype!="svg"){
 string command = "rm "+tempfilename;
 command1=command.c_str();
 cout << "Executing " << command1 << endl;
 system(command1);
  }


}


//print in svg format tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void TrackerMap::print(bool print_total, float minval, float maxval, string outputfilename){
  ostringstream outs;
  minvalue=minval; maxvalue=maxval;
  outs << outputfilename << ".xml";
  svgfile = new ofstream(outs.str().c_str(),ios::out);
  jsfile = new ifstream("trackermap.txt",ios::in);

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
  int red, green, blue;
  float val=minvalue;
  int paletteLength = 250;
  float dval = (maxvalue-minvalue)/(float)paletteLength;
  for(int i=0;i<paletteLength;i++){
 if(palette==1){//palette1 1 - raibow
   float delta=(maxvalue-minvalue);
   float x =(val-minvalue);
   red = (int) ( x<(delta/2) ? 0 : ( x > ((3./4.)*delta) ?  255 : 255/(delta/4) * (x-(2./4.)*delta)  ) );
   green= (int) ( x<delta/4 ? (x*255/(delta/4)) : ( x > ((3./4.)*delta) ?  255-255/(delta/4) *(x-(3./4.)*delta) : 255 ) );
   blue = (int) ( x<delta/4 ? 255 : ( x > ((1./2.)*delta) ?  0 : 255-255/(delta/4) * (x-(1./4.)*delta) ) );
     }
     if (palette==2){//palette 2 yellow-green
     green = (int)((val-minvalue)/(maxvalue-minvalue)*256.);
         if (green > 255) green=255;
         red = 255; blue=0;green=255-green;
        }
    *svgfile <<"<svg:rect  x=\""<<i<<"\" y=\"0\" width=\"1\" height=\"20\" fill=\"rgb("<<red<<","<<green<<","<<blue<<")\" />\n";
    if(i%50 == 0){
       *svgfile <<"<svg:rect  x=\""<<i<<"\" y=\"10\" width=\"1\" height=\"10\" fill=\"black\" />\n";
      if(i%100==0)*svgfile << " <svg:text  class=\"normalText\"  x=\""<<i<<"\" y=\"30\">" <<val<<"</svg:text>"<<endl;
       }
    val = val + dval;
   }
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

  ifstream infile("tracker.dat",ios::in);
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

