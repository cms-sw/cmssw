#include "CommonTools/TrackerMap/interface/TmModule.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMap.h"
SiStripTrackerMap::SiStripTrackerMap(string s,int xsize1, int ysize1) : TrackerMap(s,xsize1,ysize1) 
{
  title = s ;
  firsttime=true;
}
//print trackermap in 36 different xml files:one for each layer
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void SiStripTrackerMap::printlayers(bool print_total, float minval, float maxval, string outputfilename){
ofstream * xmlfile;
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
  outs << outputfilename <<layer<< ".xml";
  xmlfile = new ofstream(outs.str().c_str(),ios::out);
  *xmlfile << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?><layer>" << endl;
    nlay=layer;
    defwindow(nlay);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
        int red,green,blue;
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
        if(mod->count==0)red=green=blue=255;
        if(!print_total)mod->value=mod->value*mod->count;//restore mod->value
        } else {//color defined with fillc
         red=mod->red;green=mod->green;blue=mod->blue;
         }
      *xmlfile << "<mod id=\""<<key<<"\" color=\"("<<red<<","<<green<<","<<blue<<")\" />"<<endl;
        }
      }
    }
  *xmlfile << "</layer>" << endl;
  }
}
void SiStripTrackerMap::printonline()
{
 bool print_total=true;
 ofstream * svgfile;
 ifstream * jsfile;
   if(firsttime)
{
minvalue=0.; maxvalue=0.;
 svgfile = new ofstream("svgmap.xml",ios::out);
 jsfile = new ifstream("TrackerMapHeader.txt",ios::in);
 
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
*svgfile << "<svg:defs>" << endl;
for (int layer=1; layer < 44; layer++){
      *svgfile<<"<svg:g id=\"layer"<<layer<<"\">"<<endl;
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
   *svgfile << "</svg:g>" << endl;
  }
*svgfile << "</svg:defs>" << endl;
delete jsfile ;					 
 jsfile = new ifstream("TrackerMapTrailer.txt",ios::in); 
 while (getline( *jsfile, line ))			 
       {						 
 	   *svgfile << line << endl;			 
       }						 
 delete jsfile ;
 
 svgfile->close() ;
  firsttime=false;
}

   save(true,0.,0.,"svgmap.png",3000,1600);
   printlayers(true,0.,0.,"Layer");
}

