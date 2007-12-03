#include "CommonTools/TrackerMap/interface/TmModule.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMap.h"
SiStripTrackerMap::SiStripTrackerMap(const edm::ParameterSet & tkmapPset,const edm::ESHandle<SiStripFedCabling> tkFed) : TrackerMap(tkmapPset, tkFed) 
{
  firsttime=true;
  fedfirsttime=true;  
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
	  int red,green,blue, color;
	  if(mod->red < 0){ //use count to compute color
            color = getcolor(mod->value,palette);
	    red=(color>>16)&0xFF;
	    green=(color>>8)&0xFF;
	    blue=(color)&0xFF;
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
      jsfile = new ifstream(edm::FileInPath("DQM/SiStripMonitorClient/data/TrackerMapHeader.txt").fullPath().c_str(),ios::in);
      
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
      jsfile = new ifstream(edm::FileInPath("DQM/SiStripMonitorClient/data/TrackerMapTrailer.txt").fullPath().c_str(),ios::in);
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

void SiStripTrackerMap::fedprintonline() {
  bool print_total=true;
  ofstream * svgfile;
  ifstream * jsfile;
  if(fedfirsttime)
    {
      svgfile = new ofstream("fedtrackermap.html",ios::out);
      jsfile = new ifstream(edm::FileInPath("DQM/SiStripMonitorClient/data/FedTrackerMapHeader.txt").fullPath().c_str(),ios::in);
      
      //copy javascript interface 
      string line;
      while (getline( *jsfile, line ))
	{
	  *svgfile << line << endl;
	}
      //
      *svgfile << "<img src=fedtrackermap.png>" << endl;
      delete jsfile ;					 
      jsfile = new ifstream(edm::FileInPath("DQM/SiStripMonitorClient/data/FedTrackerMapTrailer.txt").fullPath().c_str(),ios::in);
      while (getline( *jsfile, line ))			 
	{						 
	  *svgfile << line << endl;			 
	}						 
      delete jsfile ;
      
      svgfile->close() ;
      fedfirsttime=false;
    }
  
  save_as_fedtrackermap(true,0.,0.,"fedtrackermap.png",3000,1600);
}
 
