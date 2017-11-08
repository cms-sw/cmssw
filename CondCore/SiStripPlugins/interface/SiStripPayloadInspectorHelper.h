#ifndef CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H
#define CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>
#include <string>
#include "TH1.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"   
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 

namespace SiStripPI {
  
  enum estimator {
    min,
    max,
    mean,
    rms
  };
  
  /*--------------------------------------------------------------------*/
  std::string estimatorType(SiStripPI::estimator e)
  /*--------------------------------------------------------------------*/
  {
    switch(e){
    case SiStripPI::min : return "minimum";
    case SiStripPI::max : return "maximum";
    case SiStripPI::mean : return "mean";
    case SiStripPI::rms  : return "RMS";
    default: return "should never be here";
    }
  }
   
  /*--------------------------------------------------------------------*/
  std::string getStringFromSubdet(StripSubdetector::SubDetector sub)
  /*-------------------------------------------------------------------*/
  {
    switch(sub){
    case StripSubdetector::TIB : return "TIB";
    case StripSubdetector::TOB : return "TOB";
    case StripSubdetector::TID : return "TID";
    case StripSubdetector::TEC : return "TEC";
    default : return "should never be here"; 
    }
  }
 
  enum TrackerRegion { 
    TIB1r = 1010, TIB1s = 1011, 
    TIB2r = 1020, TIB2s = 1021,
    TIB3r = 1030,
    TIB4r = 1040,
    TOB1r = 2010, TOB1s = 2011,
    TOB2r = 2020, TOB2s = 2021,
    TOB3r = 2030,
    TOB4r = 2040,
    TOB5r = 2050,
    TOB6r = 2060,
    TEC1r = 3010, TEC1s = 3011,
    TEC2r = 3020, TEC2s = 3021,
    TEC3r = 3030, TEC3s = 3031,
    TEC4r = 3040, TEC4s = 3041,
    TEC5r = 3050, TEC5s = 3051,
    TEC6r = 3060, TEC6s = 3061,
    TEC7r = 3070, TEC7s = 3071,
    TEC8r = 3080, TEC8s = 3081,
    TEC9r = 3090, TEC9s = 3091,
    TID1r = 4010, TID1s = 4011,
    TID2r = 4020, TID2s = 4021,
    TID3r = 4030, TID3s = 4031,
    END_OF_REGIONS	
  };

  // mapping to get the bin number
  std::map<SiStripPI::TrackerRegion,unsigned int> EnumToBinMap{
    {TrackerRegion::TIB1r,1 }, 
    {TrackerRegion::TIB1s,2 },
    {TrackerRegion::TIB2r,3 }, 
    {TrackerRegion::TIB2s,4 },
    {TrackerRegion::TIB3r,5 },
    {TrackerRegion::TIB4r,6 },
    {TrackerRegion::TOB1r,7 }, 
    {TrackerRegion::TOB1s,8 },
    {TrackerRegion::TOB2r,9 }, 
    {TrackerRegion::TOB2s,10},
    {TrackerRegion::TOB3r,11},
    {TrackerRegion::TOB4r,12},
    {TrackerRegion::TOB5r,13},
    {TrackerRegion::TOB6r,14},
    {TrackerRegion::TEC1r,15}, 
    {TrackerRegion::TEC1s,16},
    {TrackerRegion::TEC2r,17}, 
    {TrackerRegion::TEC2s,18},
    {TrackerRegion::TEC3r,19}, 
    {TrackerRegion::TEC3s,20},
    {TrackerRegion::TEC4r,21}, 
    {TrackerRegion::TEC4s,22},
    {TrackerRegion::TEC5r,23}, 
    {TrackerRegion::TEC5s,24},
    {TrackerRegion::TEC6r,25}, 
    {TrackerRegion::TEC6s,26},
    {TrackerRegion::TEC7r,27}, 
    {TrackerRegion::TEC7s,28},
    {TrackerRegion::TEC8r,29}, 
    {TrackerRegion::TEC8s,30},
    {TrackerRegion::TEC9r,31}, 
    {TrackerRegion::TEC9s,32},
    {TrackerRegion::TID1r,33}, 
    {TrackerRegion::TID1s,34},
    {TrackerRegion::TID2r,35}, 
    {TrackerRegion::TID2s,36},
    {TrackerRegion::TID3r,37}, 
    {TrackerRegion::TID3s,38}
  };

  /*--------------------------------------------------------------------*/
  const char * regionType(int index)
  /*--------------------------------------------------------------------*/
  {
    
    auto region = static_cast<std::underlying_type_t<SiStripPI::TrackerRegion> >(index);

    switch(region){
    case SiStripPI::TIB1r: return "TIB L1 r-#varphi";
    case SiStripPI::TIB1s: return "TIB L1 stereo";
    case SiStripPI::TIB2r: return "TIB L2 r-#varphi";
    case SiStripPI::TIB2s: return "TIB L2 stereo";
    case SiStripPI::TIB3r: return "TIB L3";
    case SiStripPI::TIB4r: return "TIB L4";
    case SiStripPI::TOB1r: return "TOB L1 r-#varphi";
    case SiStripPI::TOB1s: return "TOB L1 stereo";
    case SiStripPI::TOB2r: return "TOB L2 r-#varphi";
    case SiStripPI::TOB2s: return "TOB L2 stereo";
    case SiStripPI::TOB3r: return "TOB L3 r-#varphi";
    case SiStripPI::TOB4r: return "TOB L4";
    case SiStripPI::TOB5r: return "TOB L5";
    case SiStripPI::TOB6r: return "TOB L6";
    case SiStripPI::TEC1r: return "TEC D1 r-#varphi";
    case SiStripPI::TEC1s: return "TEC D1 stereo";
    case SiStripPI::TEC2r: return "TEC D2 r-#varphi";
    case SiStripPI::TEC2s: return "TEC D2 stereo";
    case SiStripPI::TEC3r: return "TEC D3 r-#varphi";
    case SiStripPI::TEC3s: return "TEC D3 stereo";
    case SiStripPI::TEC4r: return "TEC D4 r-#varphi";
    case SiStripPI::TEC4s: return "TEC D4 stereo";
    case SiStripPI::TEC5r: return "TEC D5 r-#varphi";
    case SiStripPI::TEC5s: return "TEC D5 stereo";
    case SiStripPI::TEC6r: return "TEC D6 r-#varphi";
    case SiStripPI::TEC6s: return "TEC D6 stereo";
    case SiStripPI::TEC7r: return "TEC D7 r-#varphi";
    case SiStripPI::TEC7s: return "TEC D7 stereo";
    case SiStripPI::TEC8r: return "TEC D8 r-#varphi";
    case SiStripPI::TEC8s: return "TEC D8 stereo";
    case SiStripPI::TEC9r: return "TEC D9 r-#varphi";
    case SiStripPI::TEC9s: return "TEC D9 stereo";
    case SiStripPI::TID1r: return "TID D1 r-#varphi";
    case SiStripPI::TID1s: return "TID D1 stereo";
    case SiStripPI::TID2r: return "TID D2 r-#varphi";
    case SiStripPI::TID2s: return "TID D2 stereo";
    case SiStripPI::TID3r: return "TID D3 r-#varphi"; 
    case SiStripPI::TID3s: return "TID D3 stereo";
    case SiStripPI::END_OF_REGIONS : return "undefined";
    default : return "should never be here";  
    }
  }

  /*--------------------------------------------------------------------*/
  std::pair<float,float> getTheRange(std::map<uint32_t,float> values,const int nsigma)
  /*--------------------------------------------------------------------*/
  {
    float sum = std::accumulate(std::begin(values), 
				std::end(values), 
				0.0,
				[] (float value, const std::map<uint32_t,float>::value_type& p)
				{ return value + p.second; }
				);
    
    float m =  sum / values.size();
    
    float accum = 0.0;
    std::for_each (std::begin(values), 
		   std::end(values), 
		   [&](const std::map<uint32_t,float>::value_type& p) 
		   {accum += (p.second - m) * (p.second - m);}
		   );
    
    float stdev = sqrt(accum / (values.size()-1)); 
    
    return std::make_pair(m-nsigma*stdev,m+nsigma*stdev);
    
  }
  
  /*--------------------------------------------------------------------*/
  void drawStatBox(std::map<std::string,std::shared_ptr<TH1F>> histos, std::map<std::string,int> colormap, std::vector<std::string> legend, double X=0.15, double Y=0.93, double W=0.15, double H=0.10)
  /*--------------------------------------------------------------------*/
  {  
    char   buffer[255];
    
    int i=0;
    for ( const auto &element : legend ){
      TPaveText* stat = new TPaveText(X,Y-(i*H), X+W, Y-(i+1)*H, "NDC");
      i++;
      auto Histo = histos[element];
      sprintf(buffer,"Entries : %i\n",(int)Histo->GetEntries());
      stat->AddText(buffer);
      
      sprintf(buffer,"Mean    : %6.2f\n",Histo->GetMean());
      stat->AddText(buffer);
      
      sprintf(buffer,"RMS     : %6.2f\n",Histo->GetRMS());
      stat->AddText(buffer);
      
      stat->SetFillColor(0);
      stat->SetLineColor(colormap[element]);
      stat->SetTextColor(colormap[element]);
      stat->SetTextSize(0.03);
      stat->SetBorderSize(0);
      stat->SetMargin(0.05);
      stat->SetTextAlign(12);
      stat->Draw();
    }
  }
  
  /*--------------------------------------------------------------------*/
  std::pair<float,float> getExtrema(TH1 *h1,TH1 *h2)
  /*--------------------------------------------------------------------*/
  {
    float theMax(-9999.);
    float theMin(9999.);
    theMax =  h1->GetMaximum() > h2->GetMaximum() ?  h1->GetMaximum() :  h2->GetMaximum();
    theMin =  h1->GetMinimum() < h2->GetMaximum() ?  h1->GetMinimum() :  h2->GetMinimum();
    
    float add_min = theMin>0. ? -0.05 :  0.05;
    float add_max = theMax>0. ?  0.05 : -0.05;

    auto result = std::make_pair(theMin*(1+add_min),theMax*(1+add_max));
    return result;
    
  }
  
  
  /*--------------------------------------------------------------------*/
  void makeNicePlotStyle(TH1 *hist)
  /*--------------------------------------------------------------------*/
  { 
    hist->SetStats(kFALSE);  
    hist->SetLineWidth(2);
    hist->GetXaxis()->CenterTitle(true);
    hist->GetYaxis()->CenterTitle(true);
    hist->GetXaxis()->SetTitleFont(42); 
    hist->GetYaxis()->SetTitleFont(42);  
    hist->GetXaxis()->SetTitleSize(0.05);
    hist->GetYaxis()->SetTitleSize(0.05);
    hist->GetXaxis()->SetTitleOffset(0.9);
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetLabelSize(.05);
  }
  
  
  /*--------------------------------------------------------------------*/
  void printSummary(const std::map<unsigned int, SiStripDetSummary::Values>& map)
  /*--------------------------------------------------------------------*/
  {
    for (const auto &element : map){
      int count   = element.second.count;
      double mean = count>0 ? (element.second.mean)/count : 0. ;
      double rms  = count>0 ? (element.second.rms)/count - mean*mean : 0.;
      if(rms <= 0)
	rms = 0;
      else
	rms = sqrt(rms);
      
      std::string detector;
      
      switch ((element.first)/1000) 
	{
	case 1:
	  detector = "TIB ";
	  break;
	case 2:
	  detector = "TOB ";
	  break;
	case 3:
	  detector = "TEC ";
	  break;
	case 4:
	  detector = "TID ";
	  break;
	}
      
      int layer  = (element.first)/10 - (element.first)/1000*100;
      int stereo = (element.first) - (layer*10) -(element.first)/1000*1000;
      
      std::cout<<"key of the map:"<<element.first <<" ( region: "<<regionType(element.first) <<" ) "  
	       << detector<<" layer: "<<layer<<" stereo:"<<stereo
	       <<"| count:"<<count<<" mean: "<<mean<<" rms: "<<rms<<std::endl;
      
    }
  }

  // code is mutuated from CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics

  /*--------------------------------------------------------------------*/
  void setBadComponents(int i, int component, SiStripQuality::BadComponent& BC,int NBadComponent[4][19][4])
  /*--------------------------------------------------------------------*/
  {
   
    if (BC.BadApvs){
      NBadComponent[i][0][2]+= std::bitset<16>(BC.BadApvs&0x3f).count(); 
      NBadComponent[i][component][2]+= std::bitset<16>(BC.BadApvs&0x3f).count(); 
    }

    if (BC.BadFibers){ 
      NBadComponent[i][0][1]+= std::bitset<4>(BC.BadFibers&0x7).count();
      NBadComponent[i][component][1]+= std::bitset<4>(BC.BadFibers&0x7).count();
    }   

    if (BC.BadModule){
      NBadComponent[i][0][0]++;
      NBadComponent[i][component][0]++;
    }
  }
  
  enum palette {HALFGRAY,GRAY,BLUES,REDS,ANTIGRAY,FIRE,ANTIFIRE,LOGREDBLUE,LOGBLUERED,DEFAULT};

  /*--------------------------------------------------------------------*/
  void setPaletteStyle(SiStripPI::palette palette) 
  /*--------------------------------------------------------------------*/
  {
  
    TStyle *palettestyle = new TStyle("palettestyle","Style for P-TDR");
  
    const int NRGBs = 5;
    const int NCont = 255;
    
    switch(palette){

    case HALFGRAY:
      {
	double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
	double red[NRGBs]   = {1.00, 0.91, 0.80, 0.67, 1.00};
	double green[NRGBs] = {1.00, 0.91, 0.80, 0.67, 1.00};
	double blue[NRGBs]  = {1.00, 0.91, 0.80, 0.67, 1.00};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      }
      break;

    case GRAY:
      {
	double stops[NRGBs] = {0.00, 0.01, 0.05, 0.09, 0.1};
	double red[NRGBs]   = {1.00, 0.84, 0.61, 0.34, 0.00};
	double green[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
	double blue[NRGBs]  = {1.00, 0.84, 0.61, 0.34, 0.00};
      	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      }
      break;

    case BLUES:
      {
	double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
	double red[NRGBs]   = {1.00, 0.84, 0.61, 0.34, 0.00};
	double green[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
	double blue[NRGBs]  = {1.00, 1.00, 1.00, 1.00, 1.00};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
	
      }
      break;
      
    case REDS:
	{
	  double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
	  double red[NRGBs]   = {1.00, 1.00, 1.00, 1.00, 1.00};
	  double green[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
	  double blue[NRGBs]  = {1.00, 0.84, 0.61, 0.34, 0.00};
	  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);	
	}
	break;
      
    case ANTIGRAY:
      {
	double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
	double red[NRGBs]   = {0.00, 0.34, 0.61, 0.84, 1.00};
	double green[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
	double blue[NRGBs]  = {0.00, 0.34, 0.61, 0.84, 1.00};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);		
      }      
      break;
      
    case FIRE:
      {
	double stops[NRGBs] = {0.00, 0.20, 0.80, 1.00};
	double red[NRGBs]   = {1.00, 1.00, 1.00, 0.50};
	double green[NRGBs] = {1.00, 1.00, 0.00, 0.00};
	double blue[NRGBs]  = {0.20, 0.00, 0.00, 0.00};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);	
      }
      break;

    case ANTIFIRE:
      {
	double stops[NRGBs] = {0.00, 0.20, 0.80, 1.00};
	double red[NRGBs]   = {0.50, 1.00, 1.00, 1.00};
	double green[NRGBs] = {0.00, 0.00, 1.00, 1.00};
	double blue[NRGBs]  = {0.00, 0.00, 0.00, 0.20};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);		
      }
      break;

    case LOGREDBLUE:
      {
	double stops[NRGBs] = {0.0001, 0.0010, 0.0100, 0.1000,  1.0000};
	double red[NRGBs]   = {1.00,   0.75,   0.50,   0.25,    0.00};
	double green[NRGBs] = {0.00,   0.00,   0.00,   0.00,    0.00};
	double blue[NRGBs]  = {0.00,   0.25,   0.50,   0.75,    1.00};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);		
      }
      break;
      
    case LOGBLUERED:
      {
	double stops[NRGBs] = {0.0001, 0.0010, 0.0100, 0.1000,  1.0000};
	double red[NRGBs]   = {0.00,   0.25,   0.50,   0.75,    1.00};
	double green[NRGBs] = {0.00,   0.00,   0.00,   0.00,    0.00};
	double blue[NRGBs]  = {1.00,   0.75,   0.50,   0.25,    0.00};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);			
      } 
      break;
    
    case DEFAULT:
      {
	double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
	double red[NRGBs]   = {0.00, 0.00, 0.87, 1.00, 0.51};
	double green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
	double blue[NRGBs]  = {0.51, 1.00, 0.12, 0.00, 0.00};
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);	
      }
      break;
    default:
      std::cout<<"should nevere be here" << std::endl;
      break;
    }
    
    palettestyle->SetNumberContours(NCont);
  }

};
#endif
