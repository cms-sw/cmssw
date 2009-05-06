#include "TRandom.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TGraph.h"
////////////////////////////////////////////////////////////
double FindMaximum(TGraph *g)
{
  int i;
  double x[1000],y[1000];
  for(i=0;i<g->GetN();i++)
    g->GetPoint(i,x[i],y[i]);
  return TMath::MaxElement(i,y);
}
////////////////////////////////////////////////////////////
double FindMaximum(TGraphErrors *g)
{
  int i;
  double x[1000],y[1000];
  for(i=0;i<g->GetN();i++)
    g->GetPoint(i,x[i],y[i]);
  return TMath::MaxElement(i,y);
}
////////////////////////////////////////////////////////////
double FindMaximum(TGraphAsymmErrors *g)
{
  int i;
  double x[1000],y[1000];
  for(i=0;i<g->GetN();i++)
    g->GetPoint(i,x[i],y[i]);
  return TMath::MaxElement(i,y);
}
////////////////////////////////////////////////////////////
TGraph* getError(TGraphErrors *g)
{
  int i,N;
  double x[1000],y[1000],e[1000],er[1000];
  N = g->GetN();
  for(i=0;i<N;i++)
    {
      g->GetPoint(i,x[i],y[i]);   
      e[i] = g->GetErrorY(i);
      er[i] = e[i]/y[i];
    } 
  TGraph *g1 = new TGraph(N,x,er);
  return g1;
}
////////////////////////////////////////////////////////////
TGraph* getError(TGraphAsymmErrors *g,string option)
{
  int i,N;
  double x[1000],y[1000],e[1000],er[1000];
  N = g->GetN();
  for(i=0;i<N;i++)
    {
      g->GetPoint(i,x[i],y[i]); 
      if (option=="HIGH")  
        e[i] = g->GetErrorYhigh(i);
      else if (option=="LOW")
        e[i] = g->GetErrorYlow(i);
      else
        e[i] = 0.; 
      er[i] = e[i]/y[i];
    } 
  TGraph *g1 = new TGraph(N,x,er);
  return g1;
}
////////////////////////////////////////////////////////////
void scale(TGraph *g, double a)
{
  int i,N;
  double x,y;
  N = g->GetN();
  for(i=0;i<N;i++)
    {
      g->GetPoint(i,x,y); 
      g->SetPoint(i,x,a*y);
    }
}
////////////////////////////////////////////////////////////
void setStyle(TGraphErrors *g, int color, int fillStyle, int lineStyle, int markerStyle, TString title, TString titleX, TString titleY)
{
  g->SetTitle(title);
  g->GetXaxis()->SetTitle(titleX);
  g->GetYaxis()->SetTitle(titleY);
  g->SetMarkerColor(color);
  g->SetFillColor(color);
  g->SetLineColor(color);
  g->SetLineStyle(lineStyle);
  g->SetLineWidth(2);
  g->SetFillStyle(fillStyle);
  g->SetMarkerStyle(markerStyle);
}
////////////////////////////////////////////////////////////
void setStyle(TGraphAsymmErrors *g, int color, int fillStyle, int lineStyle, int markerStyle, TString title, TString titleX, TString titleY)
{
  g->SetTitle(title);
  g->GetXaxis()->SetTitle(titleX);
  g->GetYaxis()->SetTitle(titleY);
  g->SetMarkerColor(color);
  g->SetFillColor(color);
  g->SetLineColor(color);
  g->SetLineStyle(lineStyle);
  g->SetLineWidth(2);
  g->SetFillStyle(fillStyle);
  g->SetMarkerStyle(markerStyle);
}
////////////////////////////////////////////////////////////
void setStyle(TGraph *g, int color, int fillStyle, int lineStyle, int markerStyle, TString title, TString titleX, TString titleY)
{
  g->SetTitle(title);
  g->GetXaxis()->SetTitle(titleX);
  g->GetYaxis()->SetTitle(titleY);
  g->SetMarkerColor(color);
  g->SetFillColor(color);
  g->SetLineColor(color);
  g->SetLineStyle(lineStyle);
  g->SetLineWidth(2);
  g->SetFillStyle(fillStyle);
  g->SetMarkerStyle(markerStyle);
}
////////////////////////////////////////////////////////////
vector<string> parseLevels(string ss)
{
  vector<string> result;
  unsigned int pos(0),j,newPos;
  int i;
  string tmp;
  //---- The ss string must be of the form: "LX:LY:...:LZ"
  while (pos<ss.length())
    {
      tmp = "";
      i = ss.find(":" , pos);
      if (i<0 && pos==0)
        {
          result.push_back(ss);
          pos = ss.length();
        }
      else if (i<0 && pos>0)
        {
          for(j=pos;j<ss.length();j++)
            tmp+=ss[j];
          result.push_back(tmp);
          pos = ss.length();
        }  
      else
        {
          newPos = i;
          for(j=pos;j<newPos;j++)
            tmp+=ss[j];
          result.push_back(tmp);
          pos = newPos+1;     
        }
    }
  return result;
}
