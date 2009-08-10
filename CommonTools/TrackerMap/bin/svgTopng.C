#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "TCanvas.h"
#include "TPad.h"
#include "TLatex.h"
#include "TArrow.h"
#include "TGaxis.h"
#include "TPolyLine.h"
#include "TColor.h"
#include <map>
#include "TStyle.h"
#include "TROOT.h"

TCanvas *canvas;
float minvalue;
float maxvalue;
std::vector<TPolyLine*> vp;
TStyle style;

static int width=3200;
static int height=1600;
float hmin,hmax;

struct Data{
  std::vector<short> rgb;
  std::vector<float> points;
  float value;

  int getColorIndex() const {
    return rgb[0]+rgb[1]*1000+rgb[2]*1000000+10000000;
  }

  void print() const{
    std::cout << "rgb ";
    for(size_t i=0;i<rgb.size();++i)
      std::cout << rgb[i] << " ";
    std::cout << std::endl;

    std::cout << "point ";
    for(size_t i=0;i<points.size();++i)
      std::cout << points[i] << " ";
    std::cout << std::endl;

    std::cout << "value " << value << std::endl; 
  }
};
std::vector<Data> DataVect;

//------------------------------------------------------------------

struct ColorList{

  size_t getIndex(const Data& data){
    int idx=data.getColorIndex();
    std::vector<int>::iterator p=std::lower_bound(ColIndex.begin(),ColIndex.end(),idx);
    return index[p-ColIndex.begin()];
  }

  void put(const Data& data){
    int idx=data.getColorIndex();
    std::vector<int>::iterator p=std::lower_bound(ColIndex.begin(),ColIndex.end(),idx);   
    if(p!=ColIndex.end() && *p==idx)
      return;
    int id=1000+ColIndex.size()+1;
    TColor *c=new TColor(id,
			 data.rgb[0]/255.,
			 data.rgb[1]/255.,
			 data.rgb[2]/255.
			 );
    
    index.insert(index.begin()+(p-ColIndex.begin()),id);
    pool.insert(pool.begin()+(p-ColIndex.begin()),c);
    ColIndex.insert(p,idx);

  }

  void setPalette(){
    int palette[index.size()];
    for(size_t i=0;i<index.size();++i){
      palette[i]=index[i];
    }
    
    gStyle->SetPalette(index.size(),palette);
  }

  std::vector<int> index;
  std::vector<int> ColIndex;
  std::vector<TColor*> pool;
};

ColorList colorList;

//---------------------------------------------------------------------

int checkLine(std::string & line){
  if(line.find("svg:polygon detid")!=std::string::npos)
    return 1;
  else if(line.find("svg:rect  x")!=std::string::npos)
    return 2;

  return 0;
}

//---------------------------------------------------------------------

std::vector<short> getRGB(std::string& line){
  std::vector<short> col(3,255);
  size_t first=line.find("fill=\"");
  size_t fillLenght=6;
  size_t second=line.find(" ",first);
  std::string scolor=line.substr(first+fillLenght,second-first-fillLenght-1);
  if (scolor.find("rgb")!=std::string::npos){
    size_t r=scolor.find("(")+1;
    size_t g=scolor.find(",",r+1)+1;
    size_t b=scolor.find(",",g+1)+1;
    size_t e=scolor.find(")");
    
    col[0]=atoi(scolor.substr(r,g-r-1).c_str());
    col[1]=atoi(scolor.substr(g,b-g-1).c_str());
    col[2]=atoi(scolor.substr(b,e-b).c_str());
  
  }
  return col;
}

//---------------------------------------------------------------------

void getPair(std::string line, std::vector<float>& out){
  size_t a=line.find(",");
  if(a==std::string::npos)
    return;
  out.push_back(atof(line.substr(0,a).c_str()));
  out.push_back(atof(line.substr(a+1).c_str()));
}

//---------------------------------------------------------------------

std::vector<float> getPoints(std::string& line){
  std::vector<float> points;

  size_t first=line.find("points=\"");
  size_t fillLenght=8;
  size_t second=line.find("\"",first);
  std::string spoints=line.substr(first+fillLenght,second-first-fillLenght-1);
  size_t p1=0;
  size_t p2=spoints.find(" ",0)+1;
  size_t p3=spoints.find(" ",p2)+1;
  size_t p4=spoints.find(" ",p3)+1;
  
  getPair(spoints.substr(p1,p2-p1-1),points);
  getPair(spoints.substr(p2,p3-p2-1),points);
  getPair(spoints.substr(p3,p4-p3-1),points);  
  getPair(spoints.substr(p4),points);

  return points;
}

//---------------------------------------------------------------------

std::vector<float> getRect(std::string& line){
  std::vector<float> points;

  size_t fillLenght;
  size_t first=line.find("x=\"");
  fillLenght=3;
  size_t second=line.find("\"",first);
  float x= atof(line.substr(first+fillLenght,second-first-fillLenght-1).c_str());

  first=line.find("y=\"");
  fillLenght=3;
  second=line.find("\"",first);
  float y= atof(line.substr(first+fillLenght,second-first-fillLenght-1).c_str());
  y=height-y;

  first=line.find("width=\"");
  fillLenght=7;
  second=line.find("\"",first);
  float w= atof(line.substr(first+fillLenght,second-first-fillLenght-1).c_str());

  first=line.find("height=\"");
  fillLenght=8;
  second=line.find("\"",first);
  float h= atof(line.substr(first+fillLenght,second-first-fillLenght-1).c_str());

  if(hmin>y-h)
    hmin=y-h;
  if(hmax<y)
    hmax=y;

  points.push_back(y);
  points.push_back(x);

  points.push_back(y-h);
  points.push_back(x);

  points.push_back(y-h);
  points.push_back(x+w);

  points.push_back(y);
  points.push_back(x+w);

  return points;
}

//---------------------------------------------------------------------

float getValue(std::string& line){

  size_t first=line.find("value=\"");
  size_t fillLenght=7;
  size_t second=line.find("\"",first);
  return atof(line.substr(first+fillLenght,second-first-fillLenght-1).c_str());
}

//---------------------------------------------------------------------

void parseSVG(char* inputFile){

  Data data;  
  std::ifstream inFile(inputFile, std::ios::in);
  std::string line;
  int check=0;
  while(std::getline(inFile,line)){
    check=checkLine(line);
    if(check>0){
      
      data.rgb=getRGB(line);
      if(check==1)
	data.points=getPoints(line);
      else
	data.points=getRect(line);

      data.value=getValue(line);
      DataVect.push_back(data);
      if(check==2)
	data.print();
    }
  }
}

//---------------------------------------------------------------------

void createPolyline(const Data& data){

  double x[data.points.size()],y[data.points.size()];
  for(size_t i=0;i<data.points.size()/2;i++){
    x[i]=data.points[2*i];
    y[i]=data.points[2*i+1];
  }
  TPolyLine*  pline = new TPolyLine(data.points.size()/2,y,x);
  vp.push_back(pline);
  pline->SetFillColor(colorList.getIndex(data));
  pline->SetLineWidth(0);
  pline->Draw("f");
}

//---------------------------------------------------------------------

void drawTkMap(){

  std::vector<Data>::const_iterator iter=DataVect.begin();
  std::vector<Data>::const_iterator iterE=DataVect.end();
  for(;iter!=iterE;++iter)
    createPolyline(*iter);

}

//---------------------------------------------------------------------

void setMinMax(){
  //FIXME : when tkmaps with taxis in color palette, introduce this
  minvalue=0;
  maxvalue=50;
};

//---------------------------------------------------------------------

void getColorScale(){

  std::vector<Data>::const_iterator iter=DataVect.begin();
  std::vector<Data>::const_iterator iterE=DataVect.end();

  for(;iter!=iterE;++iter)
    colorList.put(*iter);
}

//---------------------------------------------------------------------

void drawLabels(){
  TLatex l;
  l.SetTextSize(0.04);
  l.DrawLatex(500,50,"-z");
  l.DrawLatex(500,1430,"+z");
  l.DrawLatex(900,330,"TIB L1");
  l.DrawLatex(900,1000,"TIB L2");
  l.DrawLatex(1300,330,"TIB L3");
  l.DrawLatex(1300,1000,"TIB L4");
  l.DrawLatex(1700,330,"TOB L1");
  l.DrawLatex(1700,1000,"TOB L2");
  l.DrawLatex(2100,330,"TOB L3");
  l.DrawLatex(2100,1000,"TOB L4");
  l.DrawLatex(2500,330,"TOB L5");
  l.DrawLatex(2500,1000,"TOB L6");
  TArrow arx(2900,1190,2900,1350,0.01,"|>");
  l.DrawLatex(2915,1350,"x");
  TArrow ary(2900,1190,2790,1190,0.01,"|>");
  l.DrawLatex(2790,1210,"y");
  TArrow arz(2790,373,2790,672,0.01,"|>");
  l.DrawLatex(2820,667,"z");
  TArrow arphi(2790,511,2447,511,0.01,"|>");
  l.DrawLatex(2433,520,"#Phi");
  arx.SetLineWidth(3);
  ary.SetLineWidth(3);
  arz.SetLineWidth(3);
  arphi.SetLineWidth(3);
  arx.Draw();
  ary.Draw();
  arz.Draw();
  arphi.Draw();

  //FIXME : when tkmaps with taxis in color palette, introduce this
  TGaxis* axis = new TGaxis(3060,hmin,3060,hmax,0,100,510,"+L");
  axis->SetLabelSize(0.02);
  axis->Draw();

  canvas->Update();
}

//---------------------------------------------------------------------

void save(char* outputFile){
    canvas->Print(outputFile);
}

//---------------------------------------------------------------------

void createCanvas(){
  canvas= new TCanvas("canvas", "TrackerMap",width,height);
  gPad->SetFillColor(38);
  gPad->Range(0,0,width,height);

  getColorScale();
  drawTkMap();
  drawLabels();
 
}

//---------------------------------------------------------------------

int main(int argc, char *argv[]){

  char* inputFile;
  char* outputFile;

  hmin=99999999; hmax=0;

  if(argc>1)
    inputFile=argv[1];
  if(argc>2)
    outputFile=argv[2];

  parseSVG(inputFile);

  createCanvas();  

  save(outputFile);


  return 0;
}
