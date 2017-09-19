#ifndef CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H
#define CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>
#include <string>
#include "TH1.h"
#include "TPaveStats.h"
#include "TStyle.h"
#include "TList.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace AlignmentPI {

  // size of the phase-I Tracker APE payload (including both SS + DS modules)
  static const unsigned int phase0size=19876;
  static const float cmToUm = 10000;      

  // M.M. 2017/09/12
  // As the matrix is symmetric, we map only 6/9 terms
  // More terms for the extended APE can be added to the following methods

  enum index {
    XX=1,
    XY=2,
    XZ=3,
    YZ=4,
    YY=5,
    ZZ=6
  };

  enum partitions {
    BPix=1,   // Barrel Pixel
    FPix=2,   // Forward Pixel
    TIB=3,    // Tracker Inner Barrel
    TID=4,    // Tracker Inner Disks
    TOB=5,    // Tracker Outer Barrel
    TEC=6     // Tracker Endcaps
  };

  enum regions {
    BPixL1o,   //1  Barrel Pixel Layer 1 outer
    BPixL1i,   //2  Barrel Pixel Layer 1 inner
    BPixL2o,   //3  Barrel Pixel Layer 2 outer
    BPixL2i,   //4  Barrel Pixel Layer 2 inner
    BPixL3o,   //5  Barrel Pixel Layer 3 outer
    BPixL3i,   //6  Barrel Pixel Layer 3 inner
    BPixL4o,   //7  Barrel Pixel Layer 4 outer
    BPixL4i,   //8  Barrel Pixel Layer 4 inner
    FPixmL1,   //9  Forward Pixel Minus side Disk 1
    FPixmL2,   //10 Forward Pixel Minus side Disk 2
    FPixmL3,   //11 Forward Pixel Minus side Disk 3
    FPixpL1,   //12 Forward Pixel Plus side Disk 1
    FPixpL2,   //13 Forward Pixel Plus side Disk 2
    FPixpL3,   //14 Forward Pixel Plus side Disk 3
    TIBL1Ro,   //15 Inner Barrel Layer 1 Rphi outer
    TIBL1Ri,   //16 Inner Barrel Layer 1 Rphi inner
    TIBL1So,   //17 Inner Barrel Layer 1 Stereo outer
    TIBL1Si,   //18 Inner Barrel Layer 1 Stereo inner
    TIBL2Ro,   //19 Inner Barrel Layer 2 Rphi outer  
    TIBL2Ri,   //20 Inner Barrel Layer 2 Rphi inner  
    TIBL2So,   //21 Inner Barrel Layer 2 Stereo outer
    TIBL2Si,   //22 Inner Barrel Layer 2 Stereo inner
    TIBL3o,    //23 Inner Barrel Layer 3 outer  
    TIBL3i,    //24 Inner Barrel Layer 3 inner  
    TIBL4o,    //25 Inner Barrel Layer 4 outer
    TIBL4i,    //26 Inner Barrel Layer 4 inner
    TOBL1Ro,   //27 Outer Barrel Layer 1 Rphi outer  
    TOBL1Ri,   //28 Outer Barrel Layer 1 Rphi inner  
    TOBL1So,   //29 Outer Barrel Layer 1 Stereo outer
    TOBL1Si,   //30 Outer Barrel Layer 1 Stereo inner
    TOBL2Ro,   //31 Outer Barrel Layer 2 Rphi outer  
    TOBL2Ri,   //32 Outer Barrel Layer 2 Rphi inner  
    TOBL2So,   //33 Outer Barrel Layer 2 Stereo outer
    TOBL2Si,   //34 Outer Barrel Layer 2 Stereo inner
    TOBL3o,    //35 Outer Barrel Layer 3 outer
    TOBL3i,    //36 Outer Barrel Layer 3 inner
    TOBL4o,    //37 Outer Barrel Layer 4 outer
    TOBL4i,    //38 Outer Barrel Layer 4 inner
    TOBL5o,    //39 Outer Barrel Layer 5 outer
    TOBL5i,    //40 Outer Barrel Layer 5 inner
    TOBL6o,    //41 Outer Barrel Layer 6 outer
    TOBL6i,    //42 Outer Barrel Layer 6 inner
    TIDmR1R,   //43 Inner Disk Minus side Ring 1 Rphi
    TIDmR1S,   //44 Inner Disk Minus side Ring 1 Stereo
    TIDmR2R,   //45 Inner Disk Minus side Ring 2 Rphi  
    TIDmR2S,   //46 Inner Disk Minus side Ring 2 Stereo
    TIDmR3,    //47 Inner Disk Minus side Ring 3 
    TIDpR1R,   //48 Inner Disk Plus side Ring 1 Rphi  
    TIDpR1S,   //49 Inner Disk Plus side Ring 1 Stereo
    TIDpR2R,   //50 Inner Disk Plus side Ring 2 Rphi  
    TIDpR2S,   //51 Inner Disk Plus side Ring 2 Stereo
    TIDpR3,    //52 Inner Disk Plus side Ring 3
    NUM_OF_REGIONS
  };
  
  std::string getStringFromRegionEnum(AlignmentPI::regions e){
  switch(e)
    {
    case AlignmentPI::BPixL1o : return "BPixL1o";            
    case AlignmentPI::BPixL1i : return "BPixL1i"; 
    case AlignmentPI::BPixL2o : return "BPixL2o"; 
    case AlignmentPI::BPixL2i : return "BPixL2i"; 
    case AlignmentPI::BPixL3o : return "BPixL3o"; 
    case AlignmentPI::BPixL3i : return "BPixL3i"; 
    case AlignmentPI::BPixL4o : return "BPixL4o"; 
    case AlignmentPI::BPixL4i : return "BPixL4i"; 
    case AlignmentPI::FPixmL1 : return "FPixmL1"; 
    case AlignmentPI::FPixmL2 : return "FPixmL2"; 
    case AlignmentPI::FPixmL3 : return "FPixmL3"; 
    case AlignmentPI::FPixpL1 : return "FPixpL1"; 
    case AlignmentPI::FPixpL2 : return "FPixpL2"; 
    case AlignmentPI::FPixpL3 : return "FPixpL3"; 
    case AlignmentPI::TIBL1Ro : return "TIBL1Ro"; 
    case AlignmentPI::TIBL1Ri : return "TIBL1Ri"; 
    case AlignmentPI::TIBL1So : return "TIBL1So"; 
    case AlignmentPI::TIBL1Si : return "TIBL1Si"; 
    case AlignmentPI::TIBL2Ro : return "TIBL2Ro"; 
    case AlignmentPI::TIBL2Ri : return "TIBL2Ri"; 
    case AlignmentPI::TIBL2So : return "TIBL2So"; 
    case AlignmentPI::TIBL2Si : return "TIBL2Si"; 
    case AlignmentPI::TIBL3o  : return "TIBL3o";  
    case AlignmentPI::TIBL3i  : return "TIBL3i";  
    case AlignmentPI::TIBL4o  : return "TIBL4o";  
    case AlignmentPI::TIBL4i  : return "TIBL4i";  
    case AlignmentPI::TOBL1Ro : return "TOBL1Ro"; 
    case AlignmentPI::TOBL1Ri : return "TOBL1Ri"; 
    case AlignmentPI::TOBL1So : return "TOBL1So"; 
    case AlignmentPI::TOBL1Si : return "TOBL1Si"; 
    case AlignmentPI::TOBL2Ro : return "TOBL2Ro"; 
    case AlignmentPI::TOBL2Ri : return "TOBL2Ri"; 
    case AlignmentPI::TOBL2So : return "TOBL2So"; 
    case AlignmentPI::TOBL2Si : return "TOBL2Si"; 
    case AlignmentPI::TOBL3o  : return "TOBL3o";  
    case AlignmentPI::TOBL3i  : return "TOBL3i";  
    case AlignmentPI::TOBL4o  : return "TOBL4o";  
    case AlignmentPI::TOBL4i  : return "TOBL4i";  
    case AlignmentPI::TOBL5o  : return "TOBL5o";  
    case AlignmentPI::TOBL5i  : return "TOBL5i";  
    case AlignmentPI::TOBL6o  : return "TOBL6o";  
    case AlignmentPI::TOBL6i  : return "TOBL6i";  
    case AlignmentPI::TIDmR1R : return "TIDmR1R"; 
    case AlignmentPI::TIDmR1S : return "TIDmR1S"; 
    case AlignmentPI::TIDmR2R : return "TIDmR2R"; 
    case AlignmentPI::TIDmR2S : return "TIDmR2S"; 
    case AlignmentPI::TIDmR3  : return "TIDmR3";  
    case AlignmentPI::TIDpR1R : return "TIDpR1R"; 
    case AlignmentPI::TIDpR1S : return "TIDpR1S"; 
    case AlignmentPI::TIDpR2R : return "TIDpR2R"; 
    case AlignmentPI::TIDpR2S : return "TIDpR2S"; 
    case AlignmentPI::TIDpR3  : return "TIDpR3";  
    default: 
      edm::LogWarning("LogicError") << "Unknown partition: " <<  e;
      return "";
    }
  }

  struct topolInfo
  {
    uint32_t m_rawid;  
    int      m_subdetid;
    int      m_layer;
    int      m_side;
    int      m_ring;
    bool     m_isRphi; 
    bool     m_isDoubleSide;
    bool     m_isInternal;
    void init();
    void fillGeometryInfo(const DetId& detId,const TrackerTopology& tTopo);
    AlignmentPI::regions filterThePartition();
    void printAll();
  }; 
    

  void
  topolInfo::printAll(){
    
    std::cout<<" detId:"       <<   m_rawid     
	     <<" subdetid: "   <<   m_subdetid  
	     <<" layer: "      <<   m_layer   
	     <<" side: "       <<   m_side      
	     <<" ring: "       <<   m_ring      
	     <<" isRphi:"      <<   m_isRphi    
	     <<" isDoubleSide:"<<   m_isDoubleSide
	     <<" isInternal:"  <<   m_isInternal
	     << std::endl;
  }

  void 
  topolInfo::init(){
    m_rawid         = 0;  
    m_subdetid      = -1;
    m_layer         = -1;
    m_side          = -1;
    m_ring          = -1;
    m_isRphi        = false;
    m_isDoubleSide  = false;
    m_isInternal    = false;
  };

  void 
  topolInfo::fillGeometryInfo(const DetId& detId, 
			      const TrackerTopology& tTopo){
    
    unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());
    
    if ( subdetId == StripSubdetector::TIB) { 	   
      m_layer  = tTopo.tibLayer(detId.rawId());
      m_side   = tTopo.tibSide(detId.rawId());
      m_isRphi = tTopo.isRPhi(detId.rawId());
      m_isDoubleSide = tTopo.tibIsDoubleSide(detId.rawId());
      m_isInternal = tTopo.tibStringInfo(detId.rawId())[1];
    }
    else if ( subdetId ==  StripSubdetector::TOB ){
      m_layer  = tTopo.tobLayer(detId.rawId());
      m_side   = tTopo.tobSide(detId.rawId());
      m_isRphi = tTopo.isRPhi(detId.rawId()); 
      m_isDoubleSide = tTopo.tobIsDoubleSide(detId.rawId()); 
 
    }
    else if ( subdetId ==  StripSubdetector::TID) { 
      m_layer  = tTopo.tidWheel(detId.rawId());
      m_side   = tTopo.tidSide(detId.rawId());
      m_isRphi = tTopo.isRPhi(detId.rawId());
      m_ring   = tTopo.tidRing(detId.rawId());
      m_isDoubleSide = tTopo.tidIsDoubleSide(detId.rawId());
      m_isInternal = tTopo.tidModuleInfo(detId.rawId())[0];     
    }
    else if ( subdetId ==  StripSubdetector::TEC ){ 
      m_layer  = tTopo.tecWheel(detId.rawId());
      m_side   = tTopo.tecSide(detId.rawId()); 
      m_isRphi = tTopo.isRPhi(detId.rawId());
      m_ring   = tTopo.tecRing(detId.rawId());
      m_isDoubleSide = tTopo.tecIsDoubleSide(detId.rawId());
      m_isInternal = tTopo.tecPetalInfo(detId.rawId())[0];
    }
    else if ( subdetId ==  PixelSubdetector::PixelBarrel ) { 
      m_layer = tTopo.pxbLayer(detId.rawId());  
    }
    else if ( subdetId ==  PixelSubdetector::PixelEndcap ) { 
      m_layer = tTopo.pxfDisk(detId.rawId()); 
      m_side  = tTopo.pxfSide(detId.rawId());
    }
    else
      edm::LogWarning("LogicError") << "Unknown subdetid: " <<  subdetId;
  }

  // ------------ method to assign a partition based on the topology struct info ---------------
  AlignmentPI::regions
  topolInfo::filterThePartition(){
  
    AlignmentPI::regions ret = AlignmentPI::NUM_OF_REGIONS;

    if(m_isDoubleSide){
      return ret;
    }

    // BPix
    if(m_subdetid==1){
      switch(m_layer)
	{
	case 1:
	  m_isInternal > 0 ? ret = AlignmentPI::BPixL1o  : ret = AlignmentPI::BPixL1i;
	  break;
	case 2:
	  m_isInternal > 0 ? ret = AlignmentPI::BPixL2o  : ret = AlignmentPI::BPixL2i;
	  break;
	case 3:
	  m_isInternal > 0 ? ret = AlignmentPI::BPixL3o  : ret = AlignmentPI::BPixL3i;
	  break;
	case 4:
	  m_isInternal > 0 ? ret = AlignmentPI::BPixL4o  : ret = AlignmentPI::BPixL4i;
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknow BPix layer: " <<  m_layer;
	  break;
	}
      // FPix
  } else if (m_subdetid==2) {
      switch(m_layer)
	{
	case 1:
	  m_side > 1 ? ret = AlignmentPI::FPixpL1 : ret = AlignmentPI::FPixmL1; 
	  break;
	case 2:
	  m_side > 1 ? ret = AlignmentPI::FPixpL2 : ret = AlignmentPI::FPixmL2; 
	  break;
	case 3:
	  m_side > 1 ? ret = AlignmentPI::FPixpL3 : ret = AlignmentPI::FPixmL3; 
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknow FPix disk: " <<  m_layer;
	  break;
	}
      // TIB
    } else if (m_subdetid==3) {
      switch(m_layer)
	{
	case 1:
	  if(m_isRphi){
	    m_isInternal > 0 ? ret = AlignmentPI::TIBL1Ro : ret = AlignmentPI::TIBL1Ri;
	  } else {
	    m_isInternal > 0 ? ret = AlignmentPI::TIBL1So : ret = AlignmentPI::TIBL1Si;
	  }
	  break;
	case 2:
	  if(m_isRphi){
	    m_isInternal > 0 ? ret = AlignmentPI::TIBL2Ro  : ret = AlignmentPI::TIBL2Ri;
	  } else {
	    m_isInternal > 0 ? ret = AlignmentPI::TIBL2So  : ret = AlignmentPI::TIBL2Si;
	  }
	  break;
	case 3:
	  m_isInternal > 0 ? ret = AlignmentPI::TIBL3o  : ret = AlignmentPI::TIBL3i;
	  break;
	case 4:
	  m_isInternal > 0 ? ret = AlignmentPI::TIBL4o  : ret = AlignmentPI::TIBL4i;
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknow TIB layer: " <<  m_layer;
	  break;
	}
      // TID
    } else if (m_subdetid==4) {
      switch(m_ring)
	{
	case 1:
	  if(m_isRphi){
	    m_side > 1 ? ret = AlignmentPI::TIDpR1R  : ret = AlignmentPI::TIDmR1R;
	  } else {
	    m_side > 1 ? ret = AlignmentPI::TIDpR1S  : ret = AlignmentPI::TIDmR1S;
	  }
	  break;
	case 2:
	  if(m_isRphi){
	    m_side > 1 ? ret = AlignmentPI::TIDpR2R  : ret = AlignmentPI::TIDmR2R;
	  } else {
	    m_side > 1 ? ret = AlignmentPI::TIDpR2S  : ret = AlignmentPI::TIDmR2S;
	  }
	  break;
	case 3:
	  m_side > 1 ? ret = AlignmentPI::TIDpR3 : ret = AlignmentPI::TIDmR3; 
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknow TID wheel: " <<  m_layer;
	  break;
	}
      // TOB
    } else if (m_subdetid==5) {
      switch(m_layer)
	{
	case 1:
	  if(m_isRphi){
	    m_isInternal > 0 ? ret = AlignmentPI::TOBL1Ro  : ret = AlignmentPI::TOBL1Ri;
	  } else {
	    m_isInternal > 0 ? ret = AlignmentPI::TOBL1So  : ret = AlignmentPI::TOBL1Si;
	  }
	  break;
	case 2:
	  if(m_isRphi){
	    m_isInternal > 0 ? ret = AlignmentPI::TOBL2Ro  : ret = AlignmentPI::TOBL2Ri;
	  } else {
	    m_isInternal > 0 ? ret = AlignmentPI::TOBL2So  : ret = AlignmentPI::TOBL2Si;
	  }
	  break;
	case 3:
	  m_isInternal > 0 ? ret = AlignmentPI::TOBL3o  : ret = AlignmentPI::TOBL3i;
	  break;
	case 4:
	  m_isInternal > 0 ? ret = AlignmentPI::TOBL4o  : ret = AlignmentPI::TOBL4i;
	  break;
	case 5:
	  m_isInternal > 0 ? ret = AlignmentPI::TOBL5o  : ret = AlignmentPI::TOBL5i;
	  break;
	case 6:
	  m_isInternal > 0 ? ret = AlignmentPI::TOBL6o  : ret = AlignmentPI::TOBL6i;
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknow TOB layer: " <<  m_layer;
	  break;
	}
      // TEC
    } else if (m_subdetid==6) {
      ret = AlignmentPI::NUM_OF_REGIONS;
    }
    /*
      switch(m_ring)
      {
      case 1:
      break;
      case 2:
      break;
      case 3:
      break;
      case 4:
      break;
      case 5:
      break;
      case 6:
      break;
      case 7:
      break;
      case 8:
      break;
      case 9:
      break
      default
      break;
      }
      }
    */
  return ret;

}


  /*--------------------------------------------------------------------*/
  std::string getStringFromIndex (AlignmentPI::index i)
  /*--------------------------------------------------------------------*/
  {
    switch(i){
    case XX : return "XX";
    case XY : return "XY";
    case XZ : return "XZ";
    case YZ : return "YX";
    case YY : return "YY";
    case ZZ : return "ZZ";
    default : return "should never be here!";
    }
  }
  
  /*--------------------------------------------------------------------*/
  std::string getStringFromPart (AlignmentPI::partitions i)
  /*--------------------------------------------------------------------*/
  {
    switch(i){
    case BPix : return "BPix";
    case FPix : return "FPix";
    case TIB  : return "TIB";
    case TID  : return "TID";
    case TOB  : return "TOB";
    case TEC  : return "TEC";
    default : return "should never be here!";
    }
  }


  /*--------------------------------------------------------------------*/
  std::pair<int,int> getIndices(AlignmentPI::index i)
  /*--------------------------------------------------------------------*/    
  {
    switch(i){
    case XX : return std::make_pair(0,0);
    case XY : return std::make_pair(0,1);
    case XZ : return std::make_pair(0,2);
    case YZ : return std::make_pair(1,0);
    case YY : return std::make_pair(1,1);
    case ZZ : return std::make_pair(2,2);
    default : return std::make_pair(-1,-1);
    }
  }
  
  /*--------------------------------------------------------------------*/
  void makeNicePlotStyle(TH1 *hist,int color)
  /*--------------------------------------------------------------------*/
  { 

    hist->SetStats(kFALSE);

    hist->GetXaxis()->SetTitleColor(color);
    hist->SetLineColor(color);
    hist->SetTitleSize(0.08);
    hist->SetLineWidth(2);
    hist->GetXaxis()->CenterTitle(true);
    hist->GetYaxis()->CenterTitle(true);
    hist->GetXaxis()->SetTitleFont(42); 
    hist->GetYaxis()->SetTitleFont(42);  
    hist->GetXaxis()->SetNdivisions(505);
    hist->GetXaxis()->SetTitleSize(0.06);
    hist->GetYaxis()->SetTitleSize(0.06);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetLabelSize(.05);

  }
  
  /*--------------------------------------------------------------------*/
  void makeNiceStats(TH1F* hist,AlignmentPI::partitions part,int color)
  /*--------------------------------------------------------------------*/
  {
    char   buffer[255]; 
    TPaveText* stat = new TPaveText(0.60,0.75,0.95,0.97,"NDC");
    sprintf(buffer,"%s \n",AlignmentPI::getStringFromPart(part).c_str());
    stat->AddText(buffer);

    sprintf(buffer,"Entries : %i\n",(int)hist->GetEntries());
    stat->AddText(buffer);
    
    sprintf(buffer,"Mean    : %6.2f\n",hist->GetMean());
    stat->AddText(buffer);
    
    sprintf(buffer,"RMS     : %6.2f\n",hist->GetRMS());
    stat->AddText(buffer);
    
    stat->SetLineColor(color);
    stat->SetTextColor(color);
    stat->SetFillColor(10);
    stat->SetShadowColor(10);
    stat->Draw(); 
  }
}

#endif
