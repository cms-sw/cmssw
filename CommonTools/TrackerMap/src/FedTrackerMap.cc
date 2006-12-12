#include "CommonTools/TrackerMap/interface/FedTrackerMap.h"
#include "CommonTools/TrackerMap/interface/TmApvPair.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

#include <fstream>
#include <iostream>

FedTrackerMap::FedTrackerMap( edm::ESHandle<SiStripFedCabling> tkFed) 
{
  const vector<unsigned short> feds = tkFed->feds();
  cout<<"SiStripFedCabling has "<< feds.size()<<" active FEDS"<<endl;
  
  for(vector<unsigned short>::const_iterator ifed = feds.begin();ifed<feds.end();ifed++){
    const std::vector<FedChannelConnection> theconn = tkFed->connections( *ifed );
    int num_conn=0;
    for(std::vector<FedChannelConnection>::const_iterator iconn = theconn.begin();iconn<theconn.end();iconn++){
      TmModule *imod = IdModuleMap::imoduleMap[iconn->detId()];
      int key = iconn->fedId()*1000+iconn->fedCh();
      TmApvPair* apvpair = SvgApvPair::apvMap[key];
      if(apvpair!=0)cout << "Fed "<< iconn->fedId() << " channel " << iconn->fedCh() << " seem to be already loaded!"<<endl;
      else
	{
	  num_conn++;
	  if(num_conn==1){
	    SvgFed::fedMap[iconn->fedId()]=imod->layer;
          }
	  apvpair = new TmApvPair(key);
	  apvpair->mod=imod;
	  
	  SvgApvPair::apvMap[key] = apvpair;	
	}
    }
  }
}

FedTrackerMap::~FedTrackerMap()
{
}

void FedTrackerMap::drawApvPair(int nlay, int numfed_inlayer, bool print_total, TmApvPair* apvPair)
{
  double xp[4],yp[4];
  int green = 0;
  double xd[4],yd[4];
  int np = 4;
  double boxinitx=0., boxinity=0.; 
  double dx=.9,dy=.9;
  int numfed_incolumn;
  numfed_incolumn = 10;
  if(nlay < 31) //endcap
    numfed_incolumn=5;
  if(numfed_inlayer>numfed_incolumn)boxinitx=boxinitx+25.;
  boxinity=boxinity+((numfed_inlayer-1)%numfed_incolumn)*5.;
  boxinity=boxinity+(int)(apvPair->getFedCh()/24);
  boxinitx = boxinitx+(apvPair->getFedCh()%24);
  xp[0]=boxinitx;yp[0]=boxinity;
  xp[1]=boxinitx+dx;yp[1]=boxinity;
  xp[2]=boxinitx+dx;yp[2]=boxinity + dy;
  xp[3]=boxinitx;yp[3]=boxinity + dy;
  for(int j=0;j<4;j++){
    xd[j]=xdpixel(xp[j]);yd[j]=ydpixel(ymax-yp[j]);
    //cout << boxinity << " "<< ymax << " "<< yp[j] << endl;
  }
  
  char buffer [20];
  sprintf(buffer,"%X",apvPair->mod->idex);
  
  if(apvPair->red < 0){ //use count to compute color
    green = (int)((apvPair->value-minvalue)/(maxvalue-minvalue)*256.); 
    
    if (green > 255) green=255;
    if(!print_total)apvPair->value=apvPair->value*apvPair->count;//restore mod->value
    
    if(apvPair->count > 0)
      *svgfile <<"<polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<apvPair->text<<"\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<apvPair->mod->name<<" Id "<<buffer<<" \" fill=\"rgb(255,"<<255-green<<",0)\" points=\"";
    else
      *svgfile <<"<polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex<<"\"  onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<apvPair->text<<"\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<apvPair->mod->name<<" Id "<<buffer<<" \" fill=\"white\" points=\"";
    for(int k=0;k<np;k++){
      *svgfile << xd[k] << "," << yd[k] << " " ;
    }
    *svgfile <<"\" />" <<endl;
  } else {//color defined with fillc
    if(apvPair->red>255)apvPair->red=255;
    if(apvPair->green>255)apvPair->green=255;
    if(apvPair->blue>255)apvPair->blue=255;
    *svgfile <<"<polygon detid=\""<<apvPair->idex<<"\" count=\""<<apvPair->count <<"\" value=\""<<apvPair->value<<"\" id=\""<<apvPair->idex<<"\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""<<apvPair->text<<"\" POS=\"Fed/Ch "<<apvPair->getFedId()<<"/"<<apvPair->getFedCh()<<" connected to "<<apvPair->mod->name<<" Id "<<buffer<<" \" fill=\"rgb("<<apvPair->red<<","<<apvPair->green<<","<<apvPair->blue<<")\" points=\"";
    for(int k=0;k<np;k++){
      *svgfile << xd[k] << "," << yd[k] << " " ;
    }
    *svgfile <<"\" />" <<endl;
  }
}

//print in svg format fed tracker map
void FedTrackerMap::print(bool print_total, float minval, float maxval)
{
  minvalue=minval; maxvalue=maxval;
  svgfile = new ofstream("svgmap_fed.svg",ios::out);
  jsfile = new ifstream("trackermap.txt",ios::in);
  
  //copy javascript interface from trackermap.txt file
  string line;
  while (getline( *jsfile, line ))
    {
      *svgfile << line << endl;
    }
  
  std::map<int , TmApvPair *>::iterator i_apv;
  std::map<int , int>::iterator i_fed;
  if(!print_total){
    for( i_apv=SvgApvPair::apvMap.begin();i_apv !=SvgApvPair::apvMap.end(); i_apv++){
      TmApvPair *  apvPair= i_apv->second;
      TmModule * apv_mod = apvPair->mod;
      if(apv_mod !=0 && !apv_mod->notInUse()){
        apvPair->value = apvPair->value / apvPair->count;
      }
    }
  }
  
  if(minvalue>=maxvalue){
    minvalue=9999999.;
    maxvalue=-9999999.;
    for(i_apv=SvgApvPair::apvMap.begin();i_apv !=SvgApvPair::apvMap.end(); i_apv++){
      TmApvPair *  apvPair= i_apv->second;
      TmModule * apv_mod = apvPair->mod;
      if(apv_mod !=0 && !apv_mod->notInUse()){
	if (minvalue > apvPair->value)minvalue=apvPair->value;
	if (maxvalue < apvPair->value)maxvalue=apvPair->value;
      }
    }
  }
  
  for (int layer=1; layer < 44; layer++){
    nlay=layer;
    defwindow(nlay);
    int numfed_inlayer=0;
    for (i_fed=SvgFed::fedMap.begin();i_fed != SvgFed::fedMap.end(); i_fed++){
      if(i_fed->second == layer){
	int fedId = i_fed->first;
	numfed_inlayer++;
	for (int nconn=0;nconn<96;nconn++){
	  int key = fedId*1000+nconn; 
	  TmApvPair *  apvPair= SvgApvPair::apvMap[key];
	  if(apvPair !=0){
	    TmModule * apv_mod = apvPair->mod;
	    if(apv_mod !=0 && !apv_mod->notInUse()){
	      drawApvPair(layer,numfed_inlayer,print_total,apvPair);
	    }
	  } 
	}
      }
    }
  }
  *svgfile << "</g></svg>"<<endl;
  *svgfile << " <text id=\"Title\" class=\"normalText\"  x=\"100\" y=\"0\">"<<title<<"</text>"<<endl;
  *svgfile << "</svg>"<<endl;
}

void FedTrackerMap::fillc(int fedId,int fedCh, int red, int green, int blue  )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = SvgApvPair::apvMap[key];
  
  if(apvpair!=0){
    apvpair->red=red; apvpair->green=green; apvpair->blue=blue;
    return;
  }
  cout << "*** error in FedTrackerMap fillc method ***";
}

void FedTrackerMap::fill_current_val(int fedId, int fedCh, float current_val )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = SvgApvPair::apvMap[key];
  
  if(apvpair!=0)  apvpair->value=current_val;
  else 
    cout << "*** error in FedTrackerMap fill_current_val method ***";
}

void FedTrackerMap::fill(int fedId, int fedCh, float qty )
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = SvgApvPair::apvMap[key];
  if(apvpair!=0){
    apvpair->value=apvpair->value+qty;
    apvpair->count++;
    return;
  }
  cout << "*** error inFedTrackerMap fill method ***";
}

void FedTrackerMap::setText(int fedId, int fedCh, string s)
{
  int key = fedId*1000+fedCh;
  TmApvPair* apvpair = SvgApvPair::apvMap[key];
  if(apvpair!=0){
    apvpair->text=s;
  }
  else 
    cout << "*** error in FedTrackerMap setText method ***";
}


