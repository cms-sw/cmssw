/*
 *  Routine to calculate CSC efficiencies 
 *   (no ME1/1 yet)
 *  Comments about the program logic are denoted by //----
 * 
 *  Stoyan Stoynev, Northwestern University.
 */
  
#include "RecoLocalMuon/CSCEfficiency/interface/CSCEfficiency.h"

//#define DATA 1 // 0 - MC; 1 - data 
#define SQR(x) ((x)*(x))
//---- Histogram limits
#define XMIN  -70.
#define XMAX  70.
#define YMIN -165.
#define YMAX 165.
#define LAYER_MIN -0.5
#define LAYER_MAX 9.5

template <class T>
inline std::string to_string (const T& t)
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}

void Rotate(double Xinit, double Yinit, double angle, double & Xrot, double & Yrot);

using namespace std;
using namespace edm;

// Constructor
//CSCEfficiency::CSCEfficiency(const ParameterSet& pset){
//---- this allows access to MC (if needed)
CSCEfficiency::CSCEfficiency(const ParameterSet& pset) : theSimHitMap("MuonCSCHits"){
  const float Xmin = XMIN;
  const float Xmax = XMAX;
  const int nXbins = int(4.*(Xmax - Xmin));
  const float Ymin = YMIN;
  const float Ymax = YMAX;
  const int nYbins = int(2.*(Ymax - Ymin));
  const float Layer_min = LAYER_MIN;
  const float Layer_max = LAYER_MAX;
  const int nLayer_bins = int(Layer_max - Layer_min);
  //

  //---- Get the input parameters
  rootFileName     = pset.getUntrackedParameter<string>("rootFileName");
  WorkInEndcap     = pset.getUntrackedParameter<int>("WorkInEndcap");
  ExtrapolateFromStation      = pset.getUntrackedParameter<int>("ExtrapolateFromStation");
  ExtrapolateToStation     = pset.getUntrackedParameter<int>("ExtrapolateToStation");
  ExtrapolateToRing     = pset.getUntrackedParameter<int>("ExtrapolateToRing");
  DATA  = pset.getUntrackedParameter<bool>("runOnData");// // 0 - MC; 1 - data
  update  = pset.getUntrackedParameter<bool>("update");
  //
  //if(!update){

    if(!DATA){
      mycscunpacker     = pset.getUntrackedParameter<string>("mycscunpacker");
    }
    //---- set counter to zero
    nEventsAnalyzed = 0;
    std::string Path = "AllChambers/";
    std::string FullName;
    if(update){
      //---- File with input histograms 
      theFile = new TFile(rootFileName.c_str(), "UPDATE");
    }
    else{
      //---- File with output histograms 
      theFile = new TFile(rootFileName.c_str(), "RECREATE");
    }
    theFile->cd();
    //---- Book histograms for the analysis
    char SpecName[50];
    sprintf(SpecName,"DataFlow");
    
    if(!update){
      DataFlow = 
	new TH1F(SpecName,"Data flow;condition number;entries",30,-0.5,29.5);
    }
    else{
      FullName = Path + to_string(SpecName);
      strcpy(SpecName, FullName.c_str());
      DataFlow = (TH1F*)(theFile)->Get(SpecName);
    }
    //
    sprintf(SpecName,"XY_ALCTmissing");
    if(!update){
      XY_ALCTmissing =
        new TH2F(SpecName,"XY - ALCT missing;cm;cm",nXbins,XMIN,XMAX,nYbins,YMIN,YMAX);
    }
    else{
      FullName = Path + to_string(SpecName);
      strcpy(SpecName, FullName.c_str());
      XY_ALCTmissing = (TH2F*)(theFile)->Get(SpecName);
    }
    //
    sprintf(SpecName,"dydz_Eff_ALCT");
    if(!update){
      dydz_Eff_ALCT =
        new TH1F(SpecName,"ALCT efficient events vs. dy/dz of the segment in ref. station;dydz;entries",30,-1.5,1.5);
    }
    else{
      FullName = Path + to_string(SpecName);
      strcpy(SpecName, FullName.c_str());
      dydz_Eff_ALCT = (TH1F*)(theFile)->Get(SpecName);
    }
    //
    sprintf(SpecName,"dydz_All_ALCT");
    if(!update){
      dydz_All_ALCT =
        new TH1F(SpecName,"ALCT events vs. dy/dz of the segment in ref. station ;dydz;entries",30,-1.5,1.5);
    }
    else{
      FullName = Path + to_string(SpecName);
      strcpy(SpecName, FullName.c_str());
      dydz_All_ALCT = (TH1F*)(theFile)->Get(SpecName);
    }    
    //
    sprintf(SpecName,"EfficientSegments");
    if(!update){
      EfficientSegments = 
	new TH1F(SpecName,"Efficient segments;chamber number;entries",NumCh, FirstCh-0.5, LastCh+0.5);
    }
    else{
      FullName = Path + to_string(SpecName);
      strcpy(SpecName, FullName.c_str());
      EfficientSegments = (TH1F*)(theFile)->Get(SpecName);
    }
    //
    sprintf(SpecName,"AllSegments");
    if(!update){
      AllSegments = 
	new TH1F(SpecName,"All segments;chamber number;entries",NumCh, FirstCh-0.5, LastCh+0.5);
    }
    else{
      FullName = Path + to_string(SpecName);
      strcpy(SpecName, FullName.c_str());
      AllSegments = (TH1F*)(theFile)->Get(SpecName);
    }

    //
    sprintf(SpecName,"EfficientRechits_inSegment");
    if(!update){
      EfficientRechits_inSegment = 
	new TH1F(SpecName,"Existing RecHit given a segment;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      EfficientRechits_inSegment = (TH1F*)((theFile))->Get(SpecName);
    }
    sprintf(SpecName,"InefficientSingleHits");
    if(!update){
      InefficientSingleHits = 
      new TH1F(SpecName,"Single RecHits not in the segment;layers (1-6);entries ",nLayer_bins,Layer_min,Layer_max);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      InefficientSingleHits =  (TH1F*)((theFile))->Get(SpecName);
    }
    sprintf(SpecName,"AllSingleHits");
    if(!update){
      AllSingleHits = 
	new TH1F(SpecName,"Single RecHits given a segment; layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      AllSingleHits = (TH1F*)((theFile))->Get(SpecName);
    }
    
    sprintf(SpecName,"XvsY_InefficientRecHits");
    if(!update){
      XvsY_InefficientRecHits =  
	new TH2F(SpecName,"Rechits if one or more layers have no any (local system);X, cm; Y, cm",
		 nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      XvsY_InefficientRecHits = (TH2F*)((theFile))->Get(SpecName);
    }
    sprintf(SpecName,"XvsY_InefficientRecHits_good");
    if(!update){
      XvsY_InefficientRecHits_good =  
	new TH2F(SpecName,"Rechits if one or more layers have no any (local system) - sensitive area only;X, cm; Y, cm",
		 nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
     XvsY_InefficientRecHits_good = (TH2F*)((theFile))->Get(SpecName);
    }

    sprintf(SpecName,"XvsY_InefficientSegments");
    if(!update){
      XvsY_InefficientSegments =  
	new TH2F(SpecName,"Segments with less than 6 hits;X, cm; Y, cm",nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      XvsY_InefficientSegments = (TH2F*)((theFile))->Get(SpecName);
    }

    sprintf(SpecName,"XvsY_InefficientSegments_good");
    if(!update){
      XvsY_InefficientSegments_good =  
	new TH2F(SpecName,"Segments with less than 6 hits - sensitive area only;X, cm; Y, cm",
		 nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      XvsY_InefficientSegments_good = (TH2F*)((theFile))->Get(SpecName);
    }
    //
    sprintf(SpecName,"EfficientRechits");
    if(!update){
      EfficientRechits = 
	new TH1F(SpecName,"Existing RecHit;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      EfficientRechits = (TH1F*)((theFile))->Get(SpecName);
    }

    sprintf(SpecName,"EfficientRechits_good");
    if(!update){
      EfficientRechits_good = 
	new TH1F(SpecName,"Existing RecHit - sensitive area only;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      EfficientRechits_good = (TH1F*)((theFile))->Get(SpecName);
    }
    sprintf(SpecName,"EfficientLCTs");
    if(!update){
      EfficientLCTs =  
	new TH1F(SpecName,"Existing LCTs (1-a, 2-c, 3-corr);3 sets + normalization;entries",30,0.5,30.5);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      EfficientLCTs = (TH1F*)((theFile))->Get(SpecName);
    }

    sprintf(SpecName,"EfficientStrips");
    if(!update){
      EfficientStrips = 
	new TH1F(SpecName,"Existing strip;layer (1-6); entries",nLayer_bins,Layer_min,Layer_max);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      EfficientStrips = (TH1F*)((theFile))->Get(SpecName);
    }
    sprintf(SpecName,"EfficientWireGroups");
    if(!update){
      EfficientWireGroups = 
	new TH1F(SpecName,"Existing WireGroups;layer (1-6); entries ",nLayer_bins,Layer_min,Layer_max);
    }
    else{
      FullName = Path + to_string(SpecName)+"_AllCh";
      strcpy(SpecName, FullName.c_str());
      EfficientWireGroups = (TH1F*)((theFile))->Get(SpecName);
    }

    for(int iLayer=0; iLayer<6;iLayer++){
      sprintf(SpecName,"XvsY_InefficientRecHits_inSegment_L%d",iLayer);
      if(!update){
	XvsY_InefficientRecHits_inSegment.push_back
	  (new TH2F(SpecName,"Missing RecHit/layer in a segment (local system, good region);X, cm; Y, cm",
		    nXbins,Xmin,Xmax,nYbins,Ymin, Ymax));
      }
      else{
	FullName = Path + to_string(SpecName)+"_AllCh";
	strcpy(SpecName, FullName.c_str());
	XvsY_InefficientRecHits_inSegment.push_back( (TH2F*)((theFile))->Get(SpecName));
      }
      //
      sprintf(SpecName,"Y_InefficientRecHits_inSegment_L%d",iLayer);
      if(!update){
	Y_InefficientRecHits_inSegment.push_back
	  (new TH1F(SpecName,"Missing RecHit/layer in a segment (local system, whole chamber);Y, cm; entries",
		    nYbins,Ymin, Ymax));
      }
      else{
	FullName = Path + to_string(SpecName)+"_AllCh";
	strcpy(SpecName, FullName.c_str());
	Y_InefficientRecHits_inSegment.push_back( (TH1F*)((theFile))->Get(SpecName));
      }    
        //
      sprintf(SpecName,"Y_AllRecHits_inSegment_L%d",iLayer);
      if(!update){
	Y_AllRecHits_inSegment.push_back
	  (new TH1F(SpecName,"All (extrapolated from the segment) RecHit/layer in a segment (local system, whole chamber);Y, cm; entries",
		    nYbins,Ymin, Ymax));
      }
      else{
	FullName = Path + to_string(SpecName)+"_AllCh";
	strcpy(SpecName, FullName.c_str());
	Y_AllRecHits_inSegment.push_back( (TH1F*)((theFile))->Get(SpecName));
      }
    }
    //---- Book groups of histograms (for any chamber)
    for(int iChamber=FirstCh;iChamber<FirstCh+NumCh;iChamber++){
      sprintf(SpecName,"Chamber_%d",iChamber);
      if(!update){
	theFile->mkdir(SpecName);
      }
      theFile->cd(SpecName);
      std::string Path = to_string(SpecName)+"/";
      sprintf(SpecName,"EfficientRechits_inSegment_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].EfficientRechits_inSegment = 
	  new TH1F(SpecName,"Existing RecHit given a segment;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].EfficientRechits_inSegment = (TH1F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"InefficientSingleHits_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].InefficientSingleHits = 
	  new TH1F(SpecName,"Single RecHits not in the segment;layers (1-6);entries ",nLayer_bins,Layer_min,Layer_max);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].InefficientSingleHits = (TH1F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"AllSingleHits_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].AllSingleHits = 
	  new TH1F(SpecName,"Single RecHits given a segment; layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].AllSingleHits = (TH1F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"XvsY_InefficientRecHits_Ch%d ",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].XvsY_InefficientRecHits =  
	  new TH2F(SpecName,"Rechits if one or more layers have no any (local system);X, cm; Y, cm",
		 nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].XvsY_InefficientRecHits = (TH2F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"XvsY_InefficientRecHits_good_Ch%d ",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_good =  
	  new TH2F(SpecName,"Rechits if one or more layers have no any (local system) - sensitive area only;X, cm; Y, cm",
		 nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_good = (TH2F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"XvsY_InefficientSegments_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].XvsY_InefficientSegments =  
	  new TH2F(SpecName,"Segments with less than 6 hits;X, cm; Y, cm",nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].XvsY_InefficientSegments = (TH2F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"XvsY_InefficientSegments_good_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].XvsY_InefficientSegments_good =  
	  new TH2F(SpecName,"Segments with less than 6 hits - sensitive area only;X, cm; Y, cm",
		   nXbins,Xmin,Xmax,nYbins,Ymin,Ymax);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].XvsY_InefficientSegments_good = (TH2F*)((theFile))->Get(SpecName);
      }

      //
      sprintf(SpecName,"EfficientRechits_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].EfficientRechits = 
	  new TH1F(SpecName,"Existing RecHit;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].EfficientRechits = (TH1F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"EfficientRechits_good_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].EfficientRechits_good = 
	  new TH1F(SpecName,"Existing RecHit - sensitive area only;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].EfficientRechits_good = (TH1F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"EfficientLCTs_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].EfficientLCTs =  
	  new TH1F(SpecName,"Existing LCTs (1-a, 2-c, 3-corr);3 sets + normalization;entries",30,0.5,30.5);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].EfficientLCTs = (TH1F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"EfficientStrips_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].EfficientStrips = 
	  new TH1F(SpecName,"Existing strip;layer (1-6); entries",nLayer_bins,Layer_min,Layer_max);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].EfficientStrips = (TH1F*)((theFile))->Get(SpecName);
      }

      sprintf(SpecName,"EfficientWireGroups_Ch%d",iChamber);
      if(!update){
	ChHist[iChamber-FirstCh].EfficientWireGroups = 
	  new TH1F(SpecName,"Existing WireGroups;layer (1-6); entries ",nLayer_bins,Layer_min,Layer_max);
      }
      else{
	FullName = Path + to_string(SpecName);
	strcpy(SpecName, FullName.c_str());
	ChHist[iChamber-FirstCh].EfficientWireGroups = (TH1F*)((theFile))->Get(SpecName);
      }

      for(int iLayer=0; iLayer<6;iLayer++){
	sprintf(SpecName,"XvsY_InefficientRecHits_inSegment_Ch%d_L%d",iChamber,iLayer);
	if(!update){
	  ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_inSegment.push_back
	    (new TH2F(SpecName,"Missing RecHit/layer in a segment (local system, good region);X, cm; Y, cm",
		      nXbins,Xmin,Xmax,nYbins,Ymin, Ymax));
	}
	else{
	  FullName = Path + to_string(SpecName);
	  strcpy(SpecName, FullName.c_str());
	  ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_inSegment.push_back((TH2F*)((theFile))->Get(SpecName));
	}
	//
	sprintf(SpecName,"Y_InefficientRecHits_inSegment_Ch%d_L%d",iChamber,iLayer);
	if(!update){
	  ChHist[iChamber-FirstCh].Y_InefficientRecHits_inSegment.push_back
	    (new TH1F(SpecName,"Missing RecHit/layer in a segment (local system, whole chamber);Y, cm; entries",
		      nYbins,Ymin, Ymax));
	}
	else{
	  FullName = Path + to_string(SpecName);
	  strcpy(SpecName, FullName.c_str());
	  ChHist[iChamber-FirstCh].Y_InefficientRecHits_inSegment.push_back((TH1F*)((theFile))->Get(SpecName));
	}
	//
	sprintf(SpecName,"Y_AllRecHits_inSegment_Ch%d_L%d",iChamber,iLayer);
	if(!update){
	  ChHist[iChamber-FirstCh].Y_AllRecHits_inSegment.push_back
	    (new TH1F(SpecName,"All (extrapolated from the segment) RecHit/layer in a segment (local system, whole chamber);Y, cm; entries",
		      nYbins,Ymin, Ymax));
	}
	else{
	  FullName = Path + to_string(SpecName);
	  strcpy(SpecName, FullName.c_str());
	  ChHist[iChamber-FirstCh].Y_AllRecHits_inSegment.push_back((TH1F*)((theFile))->Get(SpecName));
	}
      }
      //Auto_ptr... ? better but it doesn't work... (with root?...) 
      //    sprintf(SpecName,"IneffperLayerRecHit_st3_Ch%d",iChamber);
      //ChHist[iChamber-FirstCh].perLayerIneffRecHit = new TH1F(SpecName,"Ineff per Layer Rec Hit",10,-0.5,9.5);
      //std::auto_ptr<TH1F> q2(new TH1F(SpecName,"Ineff per Layer Rec Hit",10,-0.5,9.5));
      //ChHist[iChamber-FirstCh].perLayerIneffRecHit =q2;
      theFile->cd();
    }
}

// Destructor
CSCEfficiency::~CSCEfficiency(){
  // Write the histos to a file
  theFile->cd();
  //
  char SpecName[20];
  int Nbins;
  std::vector<float> bins, Efficiency, EffError;
  TH1F * readHisto;
  TH1F * writeHisto;
  std::vector<float> eff(2);
  //

  const float Ymin = YMIN;
  const float Ymax = YMAX;
  const int nYbins = int(2.*(Ymax - Ymin));
  const float Layer_min = LAYER_MIN;
  const float Layer_max = LAYER_MAX-2.;
  const int nLayer_bins = int(Layer_max - Layer_min);


  //---- loop over chambers
  for(int iChamber=FirstCh;iChamber<FirstCh+NumCh;iChamber++){
    sprintf(SpecName,"Chamber_%d",iChamber);
    //---- Histograms are added chamber by chamber (all data summed up)
    if(!update){
      if(iChamber==FirstCh){
	const char *current_title;
	const char *changed_title;
	//
	AllSingleHits = (TH1F*)ChHist[iChamber-FirstCh].AllSingleHits->Clone();
	current_title = AllSingleHits->GetName();
	changed_title = ChangeTitle(current_title);
	AllSingleHits->SetName(changed_title);
	//
	EfficientRechits_inSegment = (TH1F*)ChHist[iChamber-FirstCh].EfficientRechits_inSegment->Clone();
	current_title = EfficientRechits_inSegment->GetName();
	changed_title = ChangeTitle(current_title);
	EfficientRechits_inSegment->SetName(changed_title);
	//
	InefficientSingleHits = (TH1F*)ChHist[iChamber-FirstCh].InefficientSingleHits->Clone();
	current_title = InefficientSingleHits->GetName();
	changed_title = ChangeTitle(current_title);
	InefficientSingleHits->SetName(changed_title);
	//
	XvsY_InefficientRecHits = (TH2F*)ChHist[iChamber-FirstCh].XvsY_InefficientRecHits->Clone();
	current_title = XvsY_InefficientRecHits->GetName();
	changed_title = ChangeTitle(current_title);
	XvsY_InefficientRecHits->SetName(changed_title);
	//
	XvsY_InefficientRecHits_good = (TH2F*)ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_good->Clone();
	current_title = XvsY_InefficientRecHits_good->GetName();
	changed_title = ChangeTitle(current_title);
	XvsY_InefficientRecHits_good->SetName(changed_title);
	//
	XvsY_InefficientSegments  = (TH2F*)ChHist[iChamber-FirstCh].XvsY_InefficientSegments->Clone();
	current_title = XvsY_InefficientSegments->GetName();
	changed_title = ChangeTitle(current_title);
	XvsY_InefficientSegments->SetName(changed_title);
	//
	XvsY_InefficientSegments_good = (TH2F*)ChHist[iChamber-FirstCh].XvsY_InefficientSegments_good->Clone();
	current_title = XvsY_InefficientSegments_good->GetName();
	changed_title = ChangeTitle(current_title);
	XvsY_InefficientSegments_good->SetName(changed_title);
	//
	EfficientRechits = (TH1F*)ChHist[iChamber-FirstCh].EfficientRechits->Clone();
	current_title = EfficientRechits->GetName();
	changed_title = ChangeTitle(current_title);
	EfficientRechits->SetName(changed_title);
	//
	EfficientRechits_good = (TH1F*)ChHist[iChamber-FirstCh].EfficientRechits_good->Clone();
	current_title = EfficientRechits_good->GetName();
	changed_title = ChangeTitle(current_title);
	EfficientRechits_good->SetName(changed_title);
	//
	EfficientLCTs = (TH1F*)ChHist[iChamber-FirstCh].EfficientLCTs->Clone();
	current_title = EfficientLCTs->GetName();
	changed_title = ChangeTitle(current_title);
	EfficientLCTs->SetName(changed_title);
	//
	EfficientStrips = (TH1F*)ChHist[iChamber-FirstCh].EfficientStrips->Clone();
	current_title = EfficientStrips->GetName();
	changed_title = ChangeTitle(current_title);
	EfficientStrips->SetName(changed_title);
	//
	EfficientWireGroups = (TH1F*)ChHist[iChamber-FirstCh].EfficientWireGroups->Clone();
	current_title = EfficientWireGroups->GetName();
	changed_title = ChangeTitle(current_title);
	EfficientWireGroups->SetName(changed_title);
	for(int iLayer=0; iLayer<6;iLayer++){
	  XvsY_InefficientRecHits_inSegment[iLayer] = 
	    (TH2F*)ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_inSegment[iLayer]->Clone();
	  current_title = XvsY_InefficientRecHits_inSegment[iLayer]->GetName();
	  changed_title = ChangeTitle(current_title);
	  XvsY_InefficientRecHits_inSegment[iLayer]->SetName(changed_title);
	  //
	  Y_InefficientRecHits_inSegment[iLayer] = 
	    (TH1F*)ChHist[iChamber-FirstCh].Y_InefficientRecHits_inSegment[iLayer]->Clone();
	  current_title = Y_InefficientRecHits_inSegment[iLayer]->GetName();
	  changed_title = ChangeTitle(current_title);
	  Y_InefficientRecHits_inSegment[iLayer]->SetName(changed_title);
	  //
	  Y_AllRecHits_inSegment[iLayer] = 
	    (TH1F*)ChHist[iChamber-FirstCh].Y_AllRecHits_inSegment[iLayer]->Clone();
	  current_title = Y_AllRecHits_inSegment[iLayer]->GetName();
	  changed_title = ChangeTitle(current_title);
	  Y_AllRecHits_inSegment[iLayer]->SetName(changed_title);
	}
      }
      else{
	AllSingleHits->Add(ChHist[iChamber-FirstCh].AllSingleHits);
	EfficientRechits_inSegment->Add(ChHist[iChamber-FirstCh].EfficientRechits_inSegment);
	InefficientSingleHits->Add(ChHist[iChamber-FirstCh].InefficientSingleHits);
	XvsY_InefficientRecHits->Add(ChHist[iChamber-FirstCh].XvsY_InefficientRecHits);
	XvsY_InefficientRecHits_good->Add(ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_good);
	XvsY_InefficientSegments->Add(ChHist[iChamber-FirstCh].XvsY_InefficientSegments);
	XvsY_InefficientSegments_good->Add(ChHist[iChamber-FirstCh].XvsY_InefficientSegments_good);
	EfficientRechits->Add(ChHist[iChamber-FirstCh].EfficientRechits);
	EfficientRechits_good->Add(ChHist[iChamber-FirstCh].EfficientRechits_good);
	EfficientLCTs->Add(ChHist[iChamber-FirstCh].EfficientLCTs);
	EfficientStrips->Add(ChHist[iChamber-FirstCh].EfficientStrips);
	EfficientWireGroups->Add(ChHist[iChamber-FirstCh].EfficientWireGroups);
	for(int iLayer=0; iLayer<6;iLayer++){
	  XvsY_InefficientRecHits_inSegment[iLayer]->
	    Add(ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_inSegment[iLayer]);
	  Y_InefficientRecHits_inSegment[iLayer]->
	    Add(ChHist[iChamber-FirstCh].Y_InefficientRecHits_inSegment[iLayer]);
	  Y_AllRecHits_inSegment[iLayer]->Add(ChHist[iChamber-FirstCh].Y_AllRecHits_inSegment[iLayer]);
	}
      }
      //---- Write histograms chamber by chamber 
      theFile->cd(SpecName);
      
      ChHist[iChamber-FirstCh].EfficientRechits_inSegment->Write();
      ChHist[iChamber-FirstCh].AllSingleHits->Write();
      ChHist[iChamber-FirstCh].InefficientSingleHits->Write();
      ChHist[iChamber-FirstCh].XvsY_InefficientSegments->Write();
      ChHist[iChamber-FirstCh].XvsY_InefficientSegments_good->Write();
      ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_good->Write();
      ChHist[iChamber-FirstCh].XvsY_InefficientRecHits->Write();
      ChHist[iChamber-FirstCh].EfficientRechits->Write();
      ChHist[iChamber-FirstCh].EfficientRechits_good->Write();
      ChHist[iChamber-FirstCh].EfficientLCTs->Write();
      ChHist[iChamber-FirstCh].EfficientStrips->Write();
      ChHist[iChamber-FirstCh].EfficientWireGroups->Write();
      for(unsigned int iLayer = 0; iLayer< 6; iLayer++){
	ChHist[iChamber-FirstCh].XvsY_InefficientRecHits_inSegment[iLayer]->Write();
	ChHist[iChamber-FirstCh].Y_InefficientRecHits_inSegment[iLayer]->Write();
	ChHist[iChamber-FirstCh].Y_AllRecHits_inSegment[iLayer]->Write();
      }
    }
    theFile->cd(SpecName);
    //---- Calculate the efficiencies, write the result in histograms
    sprintf(SpecName,"FINAL_Rechit_inSegment_Efficiency_Ch%d",iChamber);
    ChHist[iChamber-FirstCh].FINAL_Rechit_inSegment_Efficiency =  
      new TH1F(SpecName,"Rechit in segment Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
    readHisto = ChHist[iChamber-FirstCh].EfficientRechits_inSegment;
    writeHisto = ChHist[iChamber-FirstCh].FINAL_Rechit_inSegment_Efficiency;
    histoEfficiency(readHisto, writeHisto,10);
    ChHist[iChamber-FirstCh].FINAL_Rechit_inSegment_Efficiency->Write("",TObject::kOverwrite); 

    //
    sprintf(SpecName,"FINAL_Attachment_Efficiency_Ch%d",iChamber);
    ChHist[iChamber-FirstCh].FINAL_Attachment_Efficiency =
      new TH1F(SpecName,"Attachment Efficiency (rechit to segment);layer (1-6);efficiency",nLayer_bins+2,Layer_min,Layer_max+2.);
    ChHist[iChamber-FirstCh].FINAL_Attachment_Efficiency->Sumw2();
    sprintf(SpecName,"efficientSegments_Ch%d",iChamber);
    TH1F * efficientSegments = new TH1F(SpecName,"Attachment Efficiency (rechit to segment);layer (1-6);efficiency",nLayer_bins+2,Layer_min,Layer_max+2.);
    efficientSegments = (TH1F*)ChHist[iChamber-FirstCh].AllSingleHits->Clone();
    efficientSegments->Add(ChHist[iChamber-FirstCh].InefficientSingleHits,-1.);
    ChHist[iChamber-FirstCh].FINAL_Attachment_Efficiency->
      Divide(efficientSegments,
              ChHist[iChamber-FirstCh].AllSingleHits,
              1.,1.,"B");
    delete efficientSegments;
    ChHist[iChamber-FirstCh].FINAL_Attachment_Efficiency->Write(); 

//
    sprintf(SpecName,"FINAL_Rechit_Efficiency_Ch%d",iChamber);
    ChHist[iChamber-FirstCh].FINAL_Rechit_Efficiency =  
      new TH1F(SpecName,"Rechit Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
    readHisto = ChHist[iChamber-FirstCh].EfficientRechits;
    writeHisto = ChHist[iChamber-FirstCh].FINAL_Rechit_Efficiency;
    histoEfficiency(readHisto, writeHisto,9);
    ChHist[iChamber-FirstCh].FINAL_Rechit_Efficiency->Write(); 

//
    sprintf(SpecName,"FINAL_Rechit_Efficiency_good_Ch%d",iChamber);
    ChHist[iChamber-FirstCh].FINAL_Rechit_Efficiency_good =  
      new TH1F(SpecName,"Rechit Efficiency - sensitive area only;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
    readHisto = ChHist[iChamber-FirstCh].EfficientRechits_good;
    writeHisto = ChHist[iChamber-FirstCh].FINAL_Rechit_Efficiency_good;
    histoEfficiency(readHisto, writeHisto,9);
    ChHist[iChamber-FirstCh].FINAL_Rechit_Efficiency_good->Write(); 

//
    sprintf(SpecName,"FINAL_LCTs_Efficiency_Ch%d",iChamber);
    ChHist[iChamber-FirstCh].FINAL_LCTs_Efficiency =  new TH1F(SpecName,"LCTs Efficiency;1-a, 2-c, 3-corr (3 sets);efficiency",30,0.5,30.5);
    Nbins =  ChHist[iChamber-FirstCh].EfficientLCTs->GetSize()-2;//without underflows and overflows
    bins.clear();
    bins.resize(Nbins);
    Efficiency.clear();
    Efficiency.resize(Nbins);
    EffError.clear();
    EffError.resize(Nbins);
    bins[Nbins-1] = ChHist[iChamber-FirstCh].EfficientLCTs->GetBinContent(Nbins);
    bins[Nbins-2] = ChHist[iChamber-FirstCh].EfficientLCTs->GetBinContent(Nbins-1);
    bins[Nbins-3] = ChHist[iChamber-FirstCh].EfficientLCTs->GetBinContent(Nbins-2);
    for (int i=0;i<Nbins;i++){
      bins[i] = ChHist[iChamber-FirstCh].EfficientLCTs->GetBinContent(i+1);
      float Norm = bins[Nbins-1];
      //---- special logic
      if(i>19){
	 Norm = bins[Nbins-3];
      }
      getEfficiency(bins[i], Norm, eff);
      Efficiency[i] = eff[0];
      EffError[i] = eff[1];
      ChHist[iChamber-FirstCh].FINAL_LCTs_Efficiency->SetBinContent(i+1, Efficiency[i]);
      ChHist[iChamber-FirstCh].FINAL_LCTs_Efficiency->SetBinError(i+1, EffError[i]);
    }
    ChHist[iChamber-FirstCh].FINAL_LCTs_Efficiency->Write();

//
    sprintf(SpecName,"FINAL_Strip_Efficiency_Ch%d",iChamber);
    ChHist[iChamber-FirstCh].FINAL_Strip_Efficiency =  
      new TH1F(SpecName,"Strip Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
    readHisto = ChHist[iChamber-FirstCh].EfficientStrips;
    writeHisto = ChHist[iChamber-FirstCh].FINAL_Strip_Efficiency;
    histoEfficiency(readHisto, writeHisto,9);
    ChHist[iChamber-FirstCh].FINAL_Strip_Efficiency->Write(); 

//
    sprintf(SpecName,"FINAL_WireGroup_Efficiency_Ch%d",iChamber);
    ChHist[iChamber-FirstCh].FINAL_WireGroup_Efficiency =  
      new TH1F(SpecName,"WireGroup Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
    readHisto = ChHist[iChamber-FirstCh].EfficientWireGroups;
    writeHisto = ChHist[iChamber-FirstCh].FINAL_WireGroup_Efficiency;
    histoEfficiency(readHisto, writeHisto,9);
    ChHist[iChamber-FirstCh].FINAL_WireGroup_Efficiency->Write(); 
    //
    for(int iLayer=0; iLayer<6;iLayer++){
      sprintf(SpecName,"FINAL_Y_RecHit_InSegment_Efficiency_Ch%d_L%d",iChamber,iLayer);
      ChHist[iChamber-FirstCh].FINAL_Y_RecHit_InSegment_Efficiency.push_back
	(new TH1F(SpecName,"RecHit/layer in a segment efficiency (local system, whole chamber);Y, cm;entries",
		  nYbins,Ymin, Ymax));
      ChHist[iChamber-FirstCh].FINAL_Y_RecHit_InSegment_Efficiency.back()->Sumw2();
      sprintf(SpecName,"efficientRecHits_Ch%d_L%d",iChamber,iLayer);
      TH1F *efficientRecHits_Y  = new TH1F(SpecName,"RecHit/layer in a segment efficiency (local system, whole chamber);Y, cm;entries",
					   nYbins,Ymin, Ymax);
      efficientRecHits_Y = (TH1F*)ChHist[iChamber-FirstCh].Y_AllRecHits_inSegment.back()->Clone();
      efficientRecHits_Y->Add(ChHist[iChamber-FirstCh].Y_InefficientRecHits_inSegment.back(),-1.);
      ChHist[iChamber-FirstCh].FINAL_Y_RecHit_InSegment_Efficiency.back()->
      Divide(efficientRecHits_Y,
             ChHist[iChamber-FirstCh].Y_AllRecHits_inSegment[iLayer],
             1.,1.,"B");
      delete efficientRecHits_Y;
      ChHist[iChamber-FirstCh].FINAL_Y_RecHit_InSegment_Efficiency.back()->Write();
    }
    //
    theFile->cd();
    //
  }
  sprintf(SpecName,"AllChambers");
  if(!update){
    theFile->mkdir(SpecName);
    theFile->cd(SpecName);
    DataFlow->Write();
    XY_ALCTmissing->Write();
    EfficientSegments->Write();
    AllSegments->Write();
    //---- Write "summed" histograms 
    EfficientRechits_inSegment->Write();
    AllSingleHits->Write();
    InefficientSingleHits->Write();
    XvsY_InefficientRecHits->Write();
    XvsY_InefficientRecHits_good->Write();
    XvsY_InefficientSegments->Write();
    XvsY_InefficientSegments_good->Write();
    EfficientRechits->Write();
    EfficientRechits_good->Write();
    EfficientLCTs->Write();
    EfficientStrips->Write();
    EfficientWireGroups->Write();
    for(unsigned int iLayer = 0; iLayer< 6; iLayer++){
      XvsY_InefficientRecHits_inSegment[iLayer]->Write();
      Y_InefficientRecHits_inSegment[iLayer]->Write();
      Y_AllRecHits_inSegment[iLayer]->Write();
    }
  }
  theFile->cd(SpecName);
  //
  sprintf(SpecName,"FINAL_dydz_Efficiency_ALCT");
  FINAL_dydz_Efficiency_ALCT=
    new TH1F(SpecName,"ALCT efficiency vs dy/dz of the segment in ref. system;dydz;efficiency", 30, -1.5, 1.5);
  FINAL_dydz_Efficiency_ALCT->Sumw2();

  FINAL_dydz_Efficiency_ALCT->Divide(dydz_Eff_ALCT, dydz_All_ALCT, 1.,1.,"B");
  FINAL_dydz_Efficiency_ALCT ->Write();
  dydz_Eff_ALCT->Write();// skip?
  dydz_All_ALCT->Write();// skip?

  //Calculate the efficiency, write the result in a histogram
  sprintf(SpecName,"FINAL_Segment_Efficiency");
  FINAL_Segment_Efficiency =
    new TH1F(SpecName,"Segment Efficiency;chamber number;efficiency", NumCh, FirstCh-0.5, LastCh+0.5);
  FINAL_Segment_Efficiency->Sumw2();
  FINAL_Segment_Efficiency->
    Divide(EfficientSegments,
            AllSegments,
            1.,1.,"B");
  FINAL_Segment_Efficiency->Write(); 

//
  sprintf(SpecName,"FINAL_Rechit_inSegment_Efficiency");
  FINAL_Rechit_inSegment_Efficiency =  
    new TH1F(SpecName,"Rechit in segment Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
  readHisto = EfficientRechits_inSegment;
  writeHisto = FINAL_Rechit_inSegment_Efficiency;
  histoEfficiency(readHisto, writeHisto,10);
  FINAL_Rechit_inSegment_Efficiency->Write(); 

  //
  sprintf(SpecName,"FINAL_Attachment_Efficiency");
  FINAL_Attachment_Efficiency =
    new TH1F(SpecName,"Attachment Efficiency (rechit to segment);layer (1-6);efficiency",nLayer_bins+2,Layer_min,Layer_max+2.);
  FINAL_Attachment_Efficiency->Sumw2();
  sprintf(SpecName,"efficientSegments");
  TH1F * efficientSegments = new TH1F(SpecName,"Attachment Efficiency (rechit to segment);layer (1-6);efficiency",nLayer_bins+2,Layer_min,Layer_max+2.);
  efficientSegments = (TH1F*)AllSingleHits->Clone();
  efficientSegments->Add(InefficientSingleHits,-1.);
  FINAL_Attachment_Efficiency->
    Divide(efficientSegments,
            AllSingleHits,
            1.,1.,"B");
  delete efficientSegments;
  FINAL_Attachment_Efficiency->Write(); 

  //
  sprintf(SpecName,"FINAL_Rechit_Efficiency");
  FINAL_Rechit_Efficiency =  
    new TH1F(SpecName,"Rechit Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
  readHisto = EfficientRechits;
  writeHisto = FINAL_Rechit_Efficiency;
  histoEfficiency(readHisto, writeHisto,9);
  FINAL_Rechit_Efficiency->Write(); 
  
//
  sprintf(SpecName,"FINAL_Rechit_Efficiency_good");
  FINAL_Rechit_Efficiency_good =  
    new TH1F(SpecName,"Rechit Efficiency - sensitive area only;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
  readHisto = EfficientRechits_good;
  writeHisto = FINAL_Rechit_Efficiency_good;
  histoEfficiency(readHisto, writeHisto,9);
  FINAL_Rechit_Efficiency_good->Write(); 
  
//
  sprintf(SpecName,"FINAL_LCTs_Efficiency");
  FINAL_LCTs_Efficiency =  new TH1F(SpecName,"LCTs Efficiency;1-a, 2-c, 3-corr (3 sets);efficiency",30,0.5,30.5);
  Nbins =  EfficientLCTs->GetSize()-2;//without underflows and overflows
  bins.clear();
  bins.resize(Nbins);
  Efficiency.clear();
  Efficiency.resize(Nbins);
  EffError.clear();
  EffError.resize(Nbins);
  bins[Nbins-1] = EfficientLCTs->GetBinContent(Nbins);
  bins[Nbins-2] = EfficientLCTs->GetBinContent(Nbins-1);
  bins[Nbins-3] = EfficientLCTs->GetBinContent(Nbins-2);
  for (int i=0;i<Nbins;i++){
    bins[i] = EfficientLCTs->GetBinContent(i+1);
    float Norm = bins[Nbins-1];
    //---- special logic
    if(i>19){
      Norm = bins[Nbins-3];
    }
    getEfficiency(bins[i], Norm, eff);
    Efficiency[i] = eff[0];
    EffError[i] = eff[1];
    FINAL_LCTs_Efficiency->SetBinContent(i+1, Efficiency[i]);
    FINAL_LCTs_Efficiency->SetBinError(i+1, EffError[i]);
  }
  FINAL_LCTs_Efficiency->Write();
  
//
  sprintf(SpecName,"FINAL_Strip_Efficiency");
  FINAL_Strip_Efficiency =  
    new TH1F(SpecName,"Strip Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
  readHisto = EfficientStrips;
  writeHisto = FINAL_Strip_Efficiency;
  histoEfficiency(readHisto, writeHisto,9);
  FINAL_Strip_Efficiency->Write(); 
  
//
  sprintf(SpecName,"FINAL_WireGroup_Efficiency");
  FINAL_WireGroup_Efficiency =  
    new TH1F(SpecName,"WireGroup Efficiency;layer (1-6);efficiency",nLayer_bins,Layer_min,Layer_max);
  readHisto = EfficientWireGroups;
  writeHisto = FINAL_WireGroup_Efficiency;
  histoEfficiency(readHisto, writeHisto,9);
  FINAL_WireGroup_Efficiency->Write(); 
  //
  for(int iLayer=0; iLayer<6;iLayer++){
    sprintf(SpecName,"FINAL_Y_RecHit_InSegment_Efficiency_L%d",iLayer);
    FINAL_Y_RecHit_InSegment_Efficiency.push_back
      (new TH1F(SpecName,"RecHit/layer in a segment efficiency (local system);Y, cm;entries",
		nYbins,Ymin, Ymax));
    FINAL_Y_RecHit_InSegment_Efficiency[iLayer]->Sumw2();
    sprintf(SpecName,"efficientRecHits_L%d",iLayer);
    TH1F *efficientRecHits_Y  = new TH1F(SpecName,"RecHit/layer in a segment efficiency (local system, whole chamber);Y, cm;entries",
					 nYbins,Ymin, Ymax);
    efficientRecHits_Y = (TH1F*)Y_AllRecHits_inSegment[iLayer]->Clone();
    efficientRecHits_Y->Add(Y_InefficientRecHits_inSegment[iLayer],-1.);
    FINAL_Y_RecHit_InSegment_Efficiency[iLayer]->
      Divide(efficientRecHits_Y,
	     Y_AllRecHits_inSegment[iLayer],
	     1.,1.,"B");
    delete efficientRecHits_Y;
    FINAL_Y_RecHit_InSegment_Efficiency[iLayer]->Write(); 
  }
  
  //---- Close the file
  theFile->Close();
}

//---- The Analysis  (main)
void CSCEfficiency::analyze(const Event & event, const EventSetup& eventSetup){
  DataFlow->Fill(0.);  
  //---- increment counter
  nEventsAnalyzed++;

  //----IBL - test to read simhits a la digi/rechit validation
  //---- MC treatment is reserved in case 
  if(!DATA){
    //theSimHitMap.reset();
    theSimHitMap.fill(event);
  }

  // printalot debug output
  printalot = (nEventsAnalyzed < 100);
  int iRun   = event.id().run();
  int iEvent = event.id().event();
  if(0==fmod(double (nEventsAnalyzed) ,double(100) )){
    printf("\n==enter==CSCEfficiency===== run %i\tevent %i\tn Analyzed %i\n",iRun,iEvent,nEventsAnalyzed);
  }
  
  //---- These declarations create handles to the types of records that you want
  //---- to retrieve from event "e".
  if (printalot) printf("\tget handles for digi collections\n");
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  
  //---- Pass the handle to the method "getByType", which is used to retrieve
  //---- one and only one instance of the type in question out of event "e". If
  //---- zero or more than one instance exists in the event an exception is thrown.
  if (printalot) printf("\tpass handles\n");
  if(DATA){
    event.getByLabel("cscunpacker","MuonCSCWireDigi",wires);
    event.getByLabel("cscunpacker","MuonCSCStripDigi",strips);    
  }
  else{
    event.getByLabel(mycscunpacker,"MuonCSCWireDigi",wires);
    event.getByLabel(mycscunpacker,"MuonCSCStripDigi",strips);
  }

  //---- Get the CSC Geometry :
  if (printalot) printf("\tget the CSC geometry.\n");
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  //
  //---- ==============================================
  //----
  //---- look at DIGIs
  //----
  //---- ===============================================
  //

  //---- WIRE GROUPS
  for(int iE=0;iE<2;iE++){
    for(int iS=0;iS<4;iS++){
      for(int iR=0;iR<3;iR++){
	for(int iC=0;iC<NumCh;iC++){
	  for(int iL=0;iL<6;iL++){
	    AllWG[iE][iS][iR][iC][iL].clear();
            AllStrips[iE][iS][iR][iC][iL].clear(); 
	  }
	}
      }
    }
  }

  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    const CSCLayer *layer_p = cscGeom->layer (id);
    const CSCLayerGeometry *layerGeom = layer_p->geometry ();
    const std::vector<float> LayerBounds = layerGeom->parameters ();
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    //
    for( ; digiItr != last; ++digiItr) {
      std::pair < int, float > WG_pos(digiItr->getWireGroup(), layerGeom->yOfWireGroup(digiItr->getWireGroup())); 
      std::pair <std::pair < int, float >, int >  LayerSignal(WG_pos, digiItr->getTimeBin()); 
      
      //---- AllWG contains basic information about WG (WG number and Y-position, time bin)
      AllWG[id.endcap()-1][id.station()-1][id.ring()-1][id.chamber()-FirstCh]
	[id.layer()-1].push_back(LayerSignal);
    }
  }  

  //---- STRIPS
  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    const CSCLayer *layer_p = cscGeom->layer (id);
    const CSCLayerGeometry *layerGeom = layer_p->geometry ();
    const std::vector<float> LayerBounds = layerGeom->parameters ();
    int largestADCValue = -1;
    int largestStrip = -1;
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int maxADC=largestADCValue;
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float threshold = 13.3 ;
      float diff = 0.;
      float peakADC  = -1000.;
      int peakTime = -1;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
	diff = (float)myADCVals[iCount]-thisPedestal;
	if (diff > threshold) { 
	  thisStripFired = true; 
	  if (myADCVals[iCount] > largestADCValue) {
	    largestADCValue = myADCVals[iCount];
	    largestStrip = myStrip;
	  }
	}
	if (diff > threshold && diff > peakADC) {
	  peakADC  = diff;
	  peakTime = iCount;
	}
      }
      if(largestADCValue>maxADC){
	maxADC = largestADCValue;
	std::pair <int, float> LayerSignal (myStrip, peakADC);
	
        //---- AllStrips contains basic information about strips 
        //---- (strip number and peak signal for most significant strip in the layer) 
	AllStrips[id.endcap()-1][id.station()-1][id.ring()-1][id.chamber()-1][id.layer()-1].clear();
	AllStrips[id.endcap()-1][id.station()-1][id.ring()-1][id.chamber()-1][id.layer()-1].push_back(LayerSignal);
      }
    }
  }

  //
  //---- ==============================================
  //----
  //---- look at RECHITs
  //----
  //---- ===============================================

  if (printalot) printf("\tGet the recHits collection.\t");
  Handle<CSCRecHit2DCollection> recHits; 
  event.getByLabel("csc2DRecHits",recHits);  
  int nRecHits = recHits->size();
  if (printalot) printf("  The size is %i\n",nRecHits);
  //
  SetOfRecHits AllRecHits[2][4][3][ NumCh];
  std::vector<bool> InitVectBool6(6);
  std::vector<double> InitVectD6(6);
  map<int,  std::vector <bool> > MyRecHits;// 6 chambers, 6 layers
  map<int,  std::vector <double> > MyRecHitsPosX;
  map<int,  std::vector <double> > MyRecHitsPosY;
  map<int,  std::vector <double> > MyRecHitsPosZ;
  for(int iSt=0;iSt<NumCh;iSt++){
    MyRecHits[iSt] = InitVectBool6;
    MyRecHitsPosX[iSt] = MyRecHitsPosY[iSt] = MyRecHitsPosZ[iSt] = InitVectD6;
  }


  //---- Loop over rechits 
  if (printalot) printf("\t...start loop over rechits...\n");
  //---- Build iterator for rechits and loop :
  CSCRecHit2DCollection::const_iterator recIt;

  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    //---- Find chamber with rechits in CSC 
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();
    if (printalot) printf("\t\t\tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",kEndcap,kStation,kRing,kChamber,kLayer);
    //---- Store reco hit as a Local Point:
    LocalPoint rhitlocal = (*recIt).localPosition();  
    double xreco = rhitlocal.x();
    double yreco = rhitlocal.y();
    double zreco = rhitlocal.z();
    LocalError rerrlocal = (*recIt).localPositionError();  
    double xxerr = rerrlocal.xx();
    double yyerr = rerrlocal.yy();
    double xyerr = rerrlocal.xy();

    //---- Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );

    //---- Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);

    double grecx = rhitglobal.x();
    double grecy = rhitglobal.y();
    double grecz = rhitglobal.z();

    // Fill RecHit information in the arrays
    if(WorkInEndcap==kEndcap && 
       ExtrapolateToStation==kStation && 
       ExtrapolateToRing==kRing){
      if(kChamber>=FirstCh && kChamber<=LastCh){
	MyRecHits[kChamber-FirstCh][kLayer-1] = true;
	MyRecHitsPosX[kChamber-FirstCh][kLayer-1] = rhitglobal.x();
	MyRecHitsPosY[kChamber-FirstCh][kLayer-1] = rhitglobal.y();
	MyRecHitsPosZ[kChamber-FirstCh][kLayer-1] = rhitglobal.z();
      }
    }

    //---- Fill RecHit information in a structure (contain basic info about rechits)
    ChamberRecHits *ThisChamber = &AllRecHits[kEndcap-1][kStation-1][kRing-1][kChamber-FirstCh].sChamber;
    std::vector <double> *vec_p = &ThisChamber->RecHitsPosX[kLayer-1];// possibly many RecHits in a layer 
    vec_p->push_back(grecx);
    vec_p = &ThisChamber->RecHitsPosY[kLayer-1];
    vec_p->push_back(grecy);
    vec_p = &ThisChamber->RecHitsPosZ[kLayer-1];
    vec_p->push_back(grecz);
    vec_p = &ThisChamber->RecHitsPosXlocal[kLayer-1];
    vec_p->push_back(xreco);
    vec_p = &ThisChamber->RecHitsPosYlocal[kLayer-1];
    vec_p->push_back(yreco);

    //---- obsolete...
    AllRecHits[kEndcap-1][kStation-1][kRing-1][kChamber-FirstCh].nEndcap=kEndcap;
    AllRecHits[kEndcap-1][kStation-1][kRing-1][kChamber-FirstCh].nStation=kStation;
    AllRecHits[kEndcap-1][kStation-1][kRing-1][kChamber-FirstCh].nRing=kRing;
    AllRecHits[kEndcap-1][kStation-1][kRing-1][kChamber-FirstCh].Nchamber=kChamber;
    //
    if (printalot) printf("\t\t\tx,y,z: %f, %f, %f\texx,eey,exy: %f, %f, %f\tglobal x,y,z: %f, %f, %f \n",xreco,yreco,zreco,xxerr,yyerr,xyerr,grecx,grecy,grecz);
  }
  
  //---- loop over all layers, chambers, etc.
  for(int ii=0;ii<2;ii++){ // endcaps
    for(int jj=0;jj<4;jj++){ // stations
      for(int kk=0;kk<3;kk++){ // rings
  	for(int ll=0;ll<LastCh-FirstCh+1;ll++){ // chambers
	  for(int mm = 0;mm<6;mm++){ // layers
	    int new_size = 
	      AllRecHits[ii][jj][kk][ll].sChamber.RecHitsPosX[mm].size();
	    //---- number of rechits in the layer
	    AllRecHits[ii][jj][kk][ll].sChamber.NRecHits[mm]=new_size;
	    //---- if this is the right one
	    if((WorkInEndcap-1) == ii && 
	       (ExtrapolateToStation-1) == jj &&
	       (ExtrapolateToRing-1)== kk){
	      //---- if the number of RecHits in the layer is NOT 1!
              //---- ...used later for getting clean samples
	      if( 1!=new_size){
		MyRecHits[ll][mm] = false;
	      }
	    }
	  }
	}
      }
    }
  }

  //---- ==============================================
  //----
  //---- look at SEGMENTs
  //----
  //---- ===============================================

  //---- get CSC segment collection
  if (printalot) printf("\tGet CSC segment collection...\n");
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel("cscSegments", cscSegments);
  int nSegments = cscSegments->size();
  if (printalot) printf("  The size is %i\n",nSegments);

  //---- Initializations...
  int iSegment = 0;
  //---- A couple of segments is looked for in a sertain part of the program. 
  //---- They are required to be in different stations and rings
  int Couple = 2;
  std::vector<int> InitVect6(6);
  std::vector<int> InitVectCh(NumCh);
  std::vector<double> InitVect3(3);
  //---- A way to create 2-dim array...
  map<int, std::vector<int> > ChambersWithSegments;//
  //
  map<int, std::vector<double> > PosLocalCouple;
  //
  map<int, std::vector<double> > DirLocalCouple;
  //
  map<int, std::vector<double> > PosCouple;
  //
  map<int, std::vector<double> > DirCouple;
  //
  map<int, std::vector<double> > ChamberBoundsCouple;
  //--- 2 map elements created (explicit correspondence needed below) 
  for(int iCop=0;iCop<2;iCop++){
    PosLocalCouple[iCop] =
      DirLocalCouple[iCop] =
      PosCouple[iCop] =
      DirCouple[iCop] =  
      ChamberBoundsCouple[iCop] = InitVect3;
    ChambersWithSegments[iCop] = InitVectCh;
  }  
  std::vector<double> Chi2Couple(Couple);
  std::vector<double> NDFCouple(Couple);
  std::vector<double> NhitsCouple(Couple);
  std::vector<double> XchamberCouple(Couple);
  std::vector<double> YchamberCouple(Couple);
  std::vector<double> RotPhiCouple(Couple);
  std::vector<int> goodSegment(Couple);
  std::vector<int> NChamberCouple(Couple);
  std::vector<int> LayersInSecondCouple(6);

  std::vector<int> SegmentInChamber;
  std::vector<int> SegmentInRing;
  std::vector<int> SegmentInStation;
  double rotationPhi = 0.;
  std::vector <double> DirGlobal_ThirdSegment;

  //---- Fill utility info in RecHit structure
  int thisEndcap = -99; 
  int thisRing = -99;
  int thisStation = - 99;
  int thisChamber = -99;
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    if(thisEndcap==id.endcap() && thisRing==id.ring() && 
       thisStation == id.station() && thisChamber == id.chamber()){
      AllRecHits[thisEndcap-1][thisStation-1][thisRing-1][thisChamber-FirstCh].sChamber.nSegments++;
      
    }
    else{
      AllRecHits[id.endcap()-1][id.station()-1][id.ring()-1][id.chamber()-FirstCh].sChamber.nSegments = 1;
      thisEndcap=id.endcap();
      thisRing=id.ring();
      thisStation = id.station();
      thisChamber = id.chamber();
    }
  }
  //---- all_RecHits is actually passed to other functions (below) 
  all_RecHits = &AllRecHits;

  //---- Loop over segments
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    iSegment++;
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    SegmentInChamber.push_back(id.chamber());
    SegmentInRing.push_back(id.ring());
    SegmentInStation.push_back(id.station());
    //
    std::vector <int> LayersInChamber(6);
    printf("\t iSegment = %i",iSegment);
    double chisq    = (*it).chi2();
    int DOF = (*it).degreesOfFreedom();
    int nhits      = (*it).nRecHits();
    LocalPoint localPos = (*it).localPosition();
    LocalVector localDir = (*it).localDirection();
    if (printalot){ 
      printf("\tendcap/station/ring/chamber: %i %i %i %i\n",
	     id.endcap(),id.station(),id.ring(),id.chamber());
    }
    //---- try to get the CSC recHits that contribute to this segment.
    if (printalot) printf("\tGet the recHits for this segment.\t");
    std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
    int nRH = (*it).nRecHits();
    if (printalot) printf("    nRH = %i\n",nRH);
    int jRH = 0;
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      jRH++;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      int kEndcap  = idRH.endcap();
      int kRing    = idRH.ring();
      int kStation = idRH.station();
      int kChamber = idRH.chamber();
      int kLayer   = idRH.layer();
      LayersInChamber[kLayer-1] = 1;

      //---- Find which of the rechits (number) in the chamber is in the segment
      int iterations = 0;
      int RecHitCoincidence = 0;
      ChamberRecHits *sChamber_p=&AllRecHits[kEndcap-1][kStation-1][kRing-1][kChamber-FirstCh].sChamber;
      for(unsigned int hitsIn =0; hitsIn < (*sChamber_p).RecHitsPosXlocal[kLayer-1].size();hitsIn++){

      //---- OK but find another condition to check (int, bool)!
	if( (*sChamber_p).RecHitsPosXlocal[kLayer-1][hitsIn] == (*iRH).localPosition().x() &&
	    (*sChamber_p).RecHitsPosYlocal[kLayer-1][hitsIn] == (*iRH).localPosition().y() ){
	  (*sChamber_p).TheRightRecHit[kLayer-1] = iterations;
          RecHitCoincidence++;
	}
	iterations++;
      }
      if(!RecHitCoincidence){
	(*sChamber_p).TheRightRecHit[kLayer-1] = -1;
      }
      if (printalot) printf("\t%i RH\tendcap/station/ring/chamber/layer: %i %i %i %i %i\n",jRH,kEndcap,kStation,kRing,kChamber,kLayer);
      if(printalot) std::cout<<"recHit number from the layer in the segment = "<< (*sChamber_p).TheRightRecHit[kLayer-1]<<std::endl;
    }

    //---- global transformation: from Ingo Bloch
    double globX = 0.;
    double globY = 0.;
    double globZ = 0.;
    double globDirX = 0.;
    double globDirY = 0.;
    double globDirZ = 0.;
    //
    double globchamberPhi = 0.;
    double globchamberX =0.;
    double globchamberY =0.;
    double globchamberZ =0.;
    //
    const CSCChamber *cscchamber = cscGeom->chamber(id);
    if (cscchamber) {
      LocalPoint localCenter(0.,0.,0);
      GlobalPoint cscchamberCenter =  cscchamber->toGlobal(localCenter);
      rotationPhi = globchamberPhi = cscchamberCenter.phi();
      globchamberX = cscchamberCenter.x();
      globchamberY = cscchamberCenter.y();
      globchamberZ = cscchamberCenter.z();
      //
      GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      globX = globalPosition.x();
      globY = globalPosition.y();
      globZ = globalPosition.z();
      //
      GlobalVector globalDirection = cscchamber->toGlobal(localDir);
      globDirX   = globalDirection.x();
      globDirY   = globalDirection.y();
      globDirZ   = globalDirection.z();
      if(printalot) std::cout<<"SEGMENT: globDirX/globDirZ = "<<globDirX/globDirZ<<" globDirY/globDirZ = "<<globDirY/globDirZ<<std::endl;
    } else {
      if (printalot) printf("\tFailed to get a local->global segment tranformation.\n");
    }

    //---- Group a couple of segments in the two stations (later require only 1 in each station)  
    if (WorkInEndcap==id.endcap()) {
      if((ExtrapolateToStation==id.station()&& ExtrapolateToRing==id.ring() && 
	  id.station()>=FirstCh &&id.station()<=LastCh) || 
	 ExtrapolateFromStation==id.station()){
        int CoupleNum = id.station()-2;
	//
   	if(id.station()==ExtrapolateToStation){
	  CoupleNum = 1;
	}
	else if(id.station()==ExtrapolateFromStation){
	  CoupleNum = 0;
	}
	else{
	  cout<<"Wrong reference station!!! (shouldn't be)"<<endl;
	}
	//
	ChambersWithSegments[CoupleNum][LastCh-id.chamber()]++;
        PosLocalCouple[CoupleNum][0] = localPos.x();
        PosLocalCouple[CoupleNum][1] = localPos.y();
        PosLocalCouple[CoupleNum][2] = localPos.z();
	//
        DirLocalCouple[CoupleNum][0] = localDir.x();
        DirLocalCouple[CoupleNum][1] = localDir.y();
        DirLocalCouple[CoupleNum][2] = localDir.z();
	//
        PosCouple[CoupleNum][0] = globX;
        PosCouple[CoupleNum][1] = globY;
        PosCouple[CoupleNum][2] = globZ;
	//
        DirCouple[CoupleNum][0] = globDirX;
        DirCouple[CoupleNum][1] = globDirY;
        DirCouple[CoupleNum][2] = globDirZ;
	//

	//
	RotPhiCouple[CoupleNum] = rotationPhi;
        Chi2Couple[CoupleNum] = chisq;
	NDFCouple[CoupleNum] = DOF;
	NhitsCouple[CoupleNum] = nhits;
	//
	XchamberCouple[CoupleNum] = globchamberX;
	YchamberCouple[CoupleNum] = globchamberY;
	NChamberCouple[CoupleNum] = id.chamber();
	//
	const CSCLayer *layer_p = cscchamber->layer(1);//layer 1
	const CSCLayerGeometry *layerGeom = layer_p->geometry ();
	const std::vector<float> LayerBounds = layerGeom->parameters ();
	ChamberBoundsCouple[CoupleNum][0] = LayerBounds[0]; // (+-)x1 shorter
	ChamberBoundsCouple[CoupleNum][1] = LayerBounds[1]; // (+-)x2 longer 
	ChamberBoundsCouple[CoupleNum][2] = LayerBounds[3]; // (+-)y1=y2
	if(ExtrapolateToStation==id.station()){
	  LayersInSecondCouple = LayersInChamber;
	}
      }
      if(2 == ExtrapolateToStation && 
	 (1==ExtrapolateFromStation || 3==ExtrapolateFromStation)){
	if(ExtrapolateFromStation != id.station() && ExtrapolateToStation != id.station() && 6==nhits && chisq/DOF<3){
	  DirGlobal_ThirdSegment.push_back(globDirX);
	  DirGlobal_ThirdSegment.push_back(globDirY);
	  DirGlobal_ThirdSegment.push_back(globDirZ);
	} 
      }
    }
  }
  printf("My nSegments: %i\n",nSegments); 

  //---- Are there segments at all?
  if(nSegments){
    DataFlow->Fill(2.);  
    std::vector<int> SegmentsInStation(2);
    for(int iCh=0;iCh<NumCh;iCh++){
      SegmentsInStation[0]+=ChambersWithSegments[0][iCh];// refernce station
      SegmentsInStation[1]+=ChambersWithSegments[1][iCh];// investigated station
    }

    //---- One (only) segment in the reference station... 
    if(1==SegmentsInStation[0] ){ 
      DataFlow->Fill(4.);  

      //---- ...with a good quality
      if(6==NhitsCouple[0] && (Chi2Couple[0]/NDFCouple[0])<3.){
	//if(6==NhitsCouple[0]){
	DataFlow->Fill(6.);  
        flag = false;
	// For calculations of LCT efficiencies (ask for 2 segments in St1 and St3 if St2 is investigated)
	if(DirGlobal_ThirdSegment.size()){
	  double XDirprime, YDirprime;
	  Rotate(DirCouple[0][0], DirCouple[0][1], -(RotPhiCouple[1]+M_PI/2), XDirprime, YDirprime);
	  double ZDirprime = DirCouple[0][2];
	  double XDirsecond, YDirsecond;
	  Rotate(DirGlobal_ThirdSegment[0], DirGlobal_ThirdSegment[1], -(RotPhiCouple[1]+M_PI/2), XDirsecond, YDirsecond);
	  double ZDirsecond  = DirGlobal_ThirdSegment[2];
	  float diff_dxdz = XDirprime/ZDirprime - XDirsecond/ZDirsecond;
	  if(fabs(diff_dxdz)<0.4){// && XDirprime/ZDirprime<0.4){
	    flag = true;
	    seg_dydz = YDirprime/ZDirprime;
	  }
	}
	int NSegFound = 0;
	for(int iSeg=0;iSeg<nSegments;iSeg++){
	  if( NChamberCouple[1]==SegmentInChamber[iSeg] // is there a segment in the chamber required
	      && ExtrapolateToRing==SegmentInRing[iSeg] // and in the extrapolated ring
	      && ExtrapolateToStation==SegmentInStation[iSeg]){ // and in the extrapolated station  
	    NSegFound++;
	  }
	}

	//---- Various efficiency calcultions
	CalculateEfficiencies( event, eventSetup, PosCouple[0], DirCouple[0],NSegFound);

	//---- One (only) segment in the station/ring to which we extrapolate 
	if(1==NSegFound){
	  DataFlow->Fill(25.);
	  //
  
	  for (int iLayer=0; iLayer<6; iLayer++) {
	    //---- Exactly 1 rechit in the layer  
	    if(MyRecHits[NChamberCouple[1]-FirstCh][iLayer]){
	      //---- Is it present in the segment?
	      if(!LayersInSecondCouple[iLayer]){
		ChHist[NChamberCouple[1]-FirstCh].InefficientSingleHits->Fill(iLayer+1);
	      }
	      ChHist[NChamberCouple[1]-FirstCh].AllSingleHits->Fill(iLayer+1);
	    }
	  }
	  
	  //---- rotation at an angle (rotationPhi+PI/2.) positions the
          //---- station "rotationPhi" at -pi (global co-ordinates)
	  double Xsecond, Ysecond;
	  Rotate(PosCouple[1][0], PosCouple[1][1], -(RotPhiCouple[1]+M_PI/2), Xsecond, Ysecond);
	  double XDirprime, YDirprime;
	  Rotate(DirCouple[0][0], DirCouple[0][1], -(RotPhiCouple[1]+M_PI/2), XDirprime, YDirprime);
	  double ZDirprime = DirCouple[0][2];
	  double XDirsecond, YDirsecond;
	  Rotate(DirCouple[1][0], DirCouple[1][1], -(RotPhiCouple[1]+M_PI/2), XDirsecond, YDirsecond);
	  double ZDirsecond  = DirCouple[1][2];
	  double dxdz_diff = XDirprime/ZDirprime - XDirsecond/ZDirsecond;
	  //---- Let the segments (in the two different stations) have "close" directions
	  if(abs(dxdz_diff)<0.15){
	    DataFlow->Fill(26.);  
	    if(6!=NhitsCouple[1]){
	      ChHist[NChamberCouple[1]-FirstCh].XvsY_InefficientSegments->Fill(PosLocalCouple[1][0],PosLocalCouple[1][1]);
	    }
	    double Xchambersecond, Ychambersecond;
	    Rotate(XchamberCouple[1], YchamberCouple[1], -(RotPhiCouple[1]+M_PI/2.), Xchambersecond, Ychambersecond);
	    //
	    double Yup, Ydown, LineSlope , LineConst, Yright;
	    Yup = Ychambersecond + ChamberBoundsCouple[1][2];
	    Ydown = Ychambersecond - ChamberBoundsCouple[1][2];
	    LineSlope = (Yup - Ydown)/(ChamberBoundsCouple[1][0]-ChamberBoundsCouple[1][1]);
	    LineConst = Yup - LineSlope*ChamberBoundsCouple[1][0];
	    Yright =  LineSlope*abs(Xsecond) + LineConst;
	    double XBound1Shifted = ChamberBoundsCouple[1][0]-20.;//
	    double XBound2Shifted = ChamberBoundsCouple[1][1]-20.;//
	    LineSlope = (Yup - Ydown)/(XBound1Shifted-XBound2Shifted);
	    LineConst = Yup - LineSlope*XBound1Shifted;
	    Yright =  LineSlope*abs(Xsecond) + LineConst;
	    
	    //---- "Good region" checks
	    if(GoodRegion(Ysecond, Yright, ExtrapolateToStation, ExtrapolateToRing, 0)){
	      DataFlow->Fill(27.);  
	      if(6!=NhitsCouple[1]){
		ChHist[NChamberCouple[1]-FirstCh].XvsY_InefficientSegments_good->Fill(PosLocalCouple[1][0],PosLocalCouple[1][1]);
	      }
	    }
	    //
	    ChamberRecHits *sChamber_p =
	      &(*all_RecHits)[WorkInEndcap-1][ExtrapolateToStation-1][ExtrapolateToRing-1][NChamberCouple[1]-FirstCh].sChamber; 
	    
	    double Zlayer = 0.;
	    int ChosenLayer = -1;
	    if(sChamber_p->RecHitsPosZ[3-1].size()){
	      ChosenLayer = 2;
	      Zlayer = sChamber_p->RecHitsPosZ[3-1][0];
	    }
	    else if(sChamber_p->RecHitsPosZ[4-1].size()){
	      ChosenLayer = 3;
	      Zlayer = sChamber_p->RecHitsPosZ[4-1][0]; 
	    }
	    else if(sChamber_p->RecHitsPosZ[5-1].size()){
	      ChosenLayer = 4;
	      Zlayer = sChamber_p->RecHitsPosZ[5-1][0]; 
	    }
	    else if(sChamber_p->RecHitsPosZ[2-1].size()){
	      ChosenLayer = 1;
	      Zlayer = sChamber_p->RecHitsPosZ[2-1][0]; 
	    }
	    
	    if(-1!=ChosenLayer){
	      
	      //---- hardcoded values... noot good
	      double dist = 2.54; // distance between two layers is 2.54 cm except for ME1/1 chambers!
	      for (int iLayer=0; iLayer<6; iLayer++) {
		
		//---- two steps because Zsegment is not (exactly) at the middle...
		double z1Position = PosCouple[1][2]; // segment Z position (between layers 2 and 3)
		double z2Position = Zlayer;// rechit in the chosen layer ;  Z position
		double z1Direction = DirLocalCouple[1][2]; // segment Z direction
		double initPosition = PosLocalCouple[1][1]; // segment Y position
		double initDirection = DirLocalCouple[1][1]; // segment Y direction
		double ParamLine = LineParam(z1Position, z2Position, z1Direction);
		
		//---- find extrapolated position of a segment at a given layer
		double y = Extrapolate1D(initPosition, initDirection, ParamLine); // this is still the position 
		//in the chosen layer!
		
		initPosition = PosLocalCouple[1][0];
		initDirection = DirLocalCouple[1][0];
		double x = Extrapolate1D(initPosition, initDirection, ParamLine);//  this is still the position 
		// in the chosen layer!
		int sign;
		
		if( z1Position>z2Position){
		  sign = -1;
		}
		else{
		  sign = 1;
		}
		z1Position = z2Position;// start from the chosen layer and go to layer iLayer (z2Position below)
		int diffLayer = abs(ChosenLayer - iLayer);
		z2Position = z1Position + float(sign)*float(diffLayer)*dist;
		
		ParamLine = LineParam(z1Position, z2Position, z1Direction);
		initPosition = y;
		initDirection = DirLocalCouple[1][1];
		y = Extrapolate1D(initPosition, initDirection, ParamLine); // this is the extrapolated position in layer iLayer
		
		initPosition = x;
		initDirection = DirLocalCouple[1][0];
		x = Extrapolate1D(initPosition, initDirection, ParamLine); // this is the extrapolated position in layer iLayer
		
		if(GoodRegion(Ysecond, Yright, ExtrapolateToStation, ExtrapolateToRing, 0)){
		  if(sChamber_p->NRecHits[iLayer]>0){
		    ChHist[NChamberCouple[1]-FirstCh].EfficientRechits_inSegment->Fill(iLayer+1);
		  }
		  else{
		    ChHist[NChamberCouple[1]-FirstCh].XvsY_InefficientRecHits_inSegment[iLayer]->Fill(x,y); 
		  }
		}
		if(sChamber_p->NRecHits[iLayer]<1){
		  ChHist[NChamberCouple[1]-FirstCh].Y_InefficientRecHits_inSegment[iLayer]->Fill(y); 
		}
		ChHist[NChamberCouple[1]-FirstCh].Y_AllRecHits_inSegment[iLayer]->Fill(y);
	      }
              //---- Normalization 
	      if(GoodRegion(Ysecond, Yright, ExtrapolateToStation, ExtrapolateToRing, 0)){
		ChHist[NChamberCouple[1]-FirstCh].EfficientRechits_inSegment->Fill(9);
	      }
	    }
	  }
	}
      }
    }
  }
  //---- End
  if (printalot) printf("==exit===CSCEfficiency===== run %i\tevent %i\n\n",iRun,iEvent);
}

void Rotate(double Xinit, double Yinit, double angle, double & Xrot, double & Yrot){
  // rotation is counterclockwise (if angle>0)
  Xrot = Xinit*cos(angle) - Yinit*sin(angle);
  Yrot = Xinit*sin(angle) + Yinit*cos(angle);
}
bool CSCEfficiency::GoodRegion(double Y, double Yborder, int Station, int Ring, int Chamber){
//---- Good region means sensitive area of a chamber (i.e. geometrical and HV boundaries excluded)
//---- hardcoded... not good
  bool pass = false;
  double Ycenter = 99999.; 
  float y_center = 99999.;
  if(Station>1 && Station<5){
    if(2==Ring){  
      y_center = -0.95;
      Ycenter = 338.0/2+360.99-3.49+y_center;
    }
    else if(1==Ring){ 
    }   
  }
  else{
    if(3==Ring){
      y_center = -1.075;
      Ycenter = 179.30/2+508.99-3.49+y_center; 
    }
    else if(2==Ring){ 
      y_center = -0.96;
       Ycenter = 189.41/2+278.49-3.49+y_center;
    }
    else if(1==Ring){
    }
  }
  Ycenter = -Ycenter;
  double Y_local = -(Y - Ycenter);
  double Yborder_local = -(Yborder - Ycenter);
  bool  withinChamberOnly = false;
  pass = CheckLocal(Y_local, Yborder_local, Station, Ring, withinChamberOnly); 
  return pass;
}
bool CSCEfficiency::GoodLocalRegion(double X, double Y, int Station, int Ring, int Chamber){
  //---- Good region means sensitive area of a chamber. "Local" stands for the local system 
  bool pass = false;
  std::vector <double> ChamberBoundsCouple(3);
  float y_center = 99999.;
  //---- hardcoded... not good
  if(Station>1 && Station<5){
   ChamberBoundsCouple[0] = 66.46/2; // (+-)x1 shorter
   ChamberBoundsCouple[1] = 127.15/2; // (+-)x2 longer 
   ChamberBoundsCouple[2] = 323.06/2;
   y_center = -0.95;
  }
  else if(1==Station){
    if(3==Ring){
      ChamberBoundsCouple[0] = 63.40/2; // (+-)x1 shorter
      ChamberBoundsCouple[1] = 92.10/2; // (+-)x2 longer 
      ChamberBoundsCouple[2] = 164.16/2;
      y_center = -1.075;
    }
    else if(2==Ring){
      ChamberBoundsCouple[0] = 51.00/2; // (+-)x1 shorter
      ChamberBoundsCouple[1] = 83.74/2; // (+-)x2 longer 
      ChamberBoundsCouple[2] = 174.49/2;
      y_center = -0.96;
    }
  }
  double Yup = ChamberBoundsCouple[2] + y_center;
  double Ydown = - ChamberBoundsCouple[2] + y_center;
  double XBound1Shifted = ChamberBoundsCouple[0]-20.;//
  double XBound2Shifted = ChamberBoundsCouple[1]-20.;//
  double LineSlope = (Yup - Ydown)/(XBound2Shifted-XBound1Shifted);
  double LineConst = Yup - LineSlope*XBound2Shifted;
  double Yborder =  LineSlope*abs(X) + LineConst;
  bool  withinChamberOnly = false;
  pass = CheckLocal(Y, Yborder, Station, Ring, withinChamberOnly);
  return pass;
}

bool CSCEfficiency::CheckLocal(double Y, double Yborder, int Station, int Ring, bool withinChamberOnly){
//---- check if it is in a good local region (sensitive area - geometrical and HV boundaries excluded) 
  bool pass = false;
  //bool withinChamberOnly = false;// false = "good region"; true - boundaries only
  std::vector <float> DeadZoneCenter(6);
  float CutZone = 10.;//cm
  //---- hardcoded... not good
  if(!withinChamberOnly){
    if(Station>1 && Station<5){
      if(2==Ring){
	DeadZoneCenter[0]= -162.48 ;
	DeadZoneCenter[1] = -81.8744;
	DeadZoneCenter[2] = -21.18165;
	DeadZoneCenter[3] = 39.51105;
	DeadZoneCenter[4] = 100.2939;
	DeadZoneCenter[5] = 160.58;

	if(Y >Yborder &&
	   ((Y> DeadZoneCenter[0] + CutZone && Y< DeadZoneCenter[1] - CutZone) ||
	    (Y> DeadZoneCenter[1] + CutZone && Y< DeadZoneCenter[2] - CutZone) ||
	    (Y> DeadZoneCenter[2] + CutZone && Y< DeadZoneCenter[3] - CutZone) ||
	    (Y> DeadZoneCenter[3] + CutZone && Y< DeadZoneCenter[4] - CutZone) ||
	    (Y> DeadZoneCenter[4] + CutZone && Y< DeadZoneCenter[5] - CutZone))){
	  pass = true;
	}
      }
      else if(1==Ring){
	//pass = true;
      }
    }
    else if(1==Station){
      if(3==Ring){
	DeadZoneCenter[0]= -83.155 ;
	DeadZoneCenter[1] = -22.7401;
	DeadZoneCenter[2] = 27.86665;
	DeadZoneCenter[3] = 81.005;
	if(Y > Yborder &&
	   ((Y> DeadZoneCenter[0] + CutZone && Y< DeadZoneCenter[1] - CutZone) ||
	    (Y> DeadZoneCenter[1] + CutZone && Y< DeadZoneCenter[2] - CutZone) ||
	    (Y> DeadZoneCenter[2] + CutZone && Y< DeadZoneCenter[3] - CutZone))){
	  pass = true;
	}
      }
      else if(2==Ring){
	DeadZoneCenter[0]= -86.285 ;
	DeadZoneCenter[1] = -32.88305;
	DeadZoneCenter[2] = 32.867423;
	DeadZoneCenter[3] = 88.205;
	if(Y > (Yborder) &&
	   ((Y> DeadZoneCenter[0] + CutZone && Y< DeadZoneCenter[1] - CutZone) ||
	    (Y> DeadZoneCenter[1] + CutZone && Y< DeadZoneCenter[2] - CutZone) ||
	    (Y> DeadZoneCenter[2] + CutZone && Y< DeadZoneCenter[3] - CutZone))){
	  pass = true;
	}
      }
      else{
      }
    }
  }
  else{
    if(Station>1 && Station<5){
      if(2==Ring){
	if(Y >Yborder && fabs(Y+0.95)<151.53){
	  pass = true;
	}
      }
      else if(1==Ring){
	//pass = true;
      }
    }
    else if(1==Station){
      if(3==Ring){
	if(Y > Yborder &&  fabs(Y+1.075)<72.08){
	  pass = true;
	}
      }
      else if(2==Ring){
	if(Y > (Yborder) && fabs(Y+0.96)<77.245){
	  pass = true;
	}
      }
      else{
      }
    }
  }
  return pass;
}

void CSCEfficiency::CalculateEfficiencies(const Event & event, const EventSetup& eventSetup,
				    std::vector<double> &Pos , std::vector<double> &Dir, int NSegFound){
  DataFlow->Fill(7.);  
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  //edm::Handle<CSCRPCDigiCollection> rpcs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts;
  if(DATA){
    event.getByLabel("cscunpacker","MuonCSCALCTDigi",alcts);
    event.getByLabel("cscunpacker","MuonCSCCLCTDigi",clcts);
    //event.getByLabel("cscunpacker","MuonCSCRPCDigi",rpcs);
    event.getByLabel("cscunpacker","MuonCSCCorrelatedLCTDigi",correlatedlcts); 
  }
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);
  //
  const std::vector<CSCChamber*> ChamberContainer = cscGeom->chambers();

  //---- Find a chamber fulfilling conditions 
  for(unsigned int nCh=0;nCh<ChamberContainer.size();nCh++){
    const CSCChamber *cscchamber = ChamberContainer[nCh];
    CSCDetId id  = cscchamber->id();
    if(id.chamber() > (FirstCh-1) && id.chamber() < (LastCh+1) &&
       id.station() == ExtrapolateToStation && 
       id.ring() == ExtrapolateToRing && id.endcap() == WorkInEndcap){
      LocalPoint localCenter(0.,0.,0);
      GlobalPoint cscchamberCenter =  cscchamber->toGlobal(localCenter);
      float ZStation = cscchamberCenter.z();
      double ParLine = LineParam(Pos[2], ZStation, Dir[2]);
      double Xextrapolated = Extrapolate1D(Pos[0],Dir[0], ParLine );
      double Yextrapolated = Extrapolate1D(Pos[1],Dir[1], ParLine );
      
      GlobalPoint ExtrapolatedSegment(Xextrapolated, Yextrapolated, ZStation);
      LocalPoint ExtrapolatedSegmentLocal = cscchamber->toLocal(ExtrapolatedSegment);
      const CSCLayer *layer_p = cscchamber->layer(1);//layer 1
      const CSCLayerGeometry *layerGeom = layer_p->geometry ();
      const std::vector<float> LayerBounds = layerGeom->parameters ();
      float y_center = 0.;
      float ShiftFromEdge = 20.; //cm from the edge
      double Yup = LayerBounds[3] + y_center;
      double Ydown = - LayerBounds[3] + y_center;
      double XBound1Shifted = LayerBounds[0] - ShiftFromEdge;//
      double XBound2Shifted = LayerBounds[1] - ShiftFromEdge;//
      double LineSlope = (Yup - Ydown)/(XBound2Shifted-XBound1Shifted);
      double LineConst = Yup - LineSlope*XBound2Shifted;
      double Yborder =  LineSlope*abs(ExtrapolatedSegmentLocal.x()) + LineConst;
      CSCDetId id  = cscchamber->id();
      bool  withinChamberOnly = false;
      if( CheckLocal(ExtrapolatedSegmentLocal.y(), Yborder, id.station(), id.ring(), withinChamberOnly)){
        DataFlow->Fill(9.);
        int cond = 0;
        if(flag){
          cond = 1;
        }

        //---- So at this point a segments from the reference station points
        //---- to a chamber ("good region") in the investigated station and ring
        //---- Calculate efficiencies
        bool LCTflag = false;
        if(DATA){
          LCTflag = LCT_Efficiencies(alcts, clcts, correlatedlcts, id.chamber(), cond);
        }
        if(!LCTflag){
          XY_ALCTmissing->Fill(ExtrapolatedSegmentLocal.x(),ExtrapolatedSegmentLocal.y());
          std::cout<<"NO ALCT when ME1 and ME3"<<std::endl;
        }
        StripWire_Efficiencies(id.chamber());
        Segment_Efficiency(id.chamber(), NSegFound);
      }
      //---- Good regions are checked separately within;
      // here just check chamber (this is a quick fix to avoid noise hits)
      withinChamberOnly = true;
      ShiftFromEdge = 10.; //cm from the edge
      XBound1Shifted = LayerBounds[0] - ShiftFromEdge;//
      XBound2Shifted = LayerBounds[1] - ShiftFromEdge;//
      LineSlope = (Yup - Ydown)/(XBound2Shifted-XBound1Shifted);
      LineConst = Yup - LineSlope*XBound2Shifted;
      Yborder =  LineSlope*abs(ExtrapolatedSegmentLocal.x()) + LineConst;
      if( CheckLocal(ExtrapolatedSegmentLocal.y(), Yborder, id.station(), id.ring(), withinChamberOnly)){
        RecHitEfficiency(ExtrapolatedSegmentLocal.x() , ExtrapolatedSegmentLocal.y(), id.chamber());
      }
    }
  }
}
//
double CSCEfficiency::Extrapolate1D(double initPosition, double initDirection, double ParameterOfTheLine){
  double ExtrapolatedPosition = initPosition + initDirection*ParameterOfTheLine;
  return ExtrapolatedPosition; 
}
//
double CSCEfficiency::LineParam(double z1Position, double z2Position, double z1Direction){
  double ParamLine = (z2Position-z1Position)/z1Direction;
  return ParamLine;
}
//
void CSCEfficiency::RecHitEfficiency(double X, double Y, int iCh){
  ChamberRecHits *sChamber_p =
    &(*all_RecHits)[WorkInEndcap-1][ExtrapolateToStation-1][ExtrapolateToRing-1][iCh-FirstCh].sChamber;
  ChamberRecHits *sChamberLeft_p = sChamber_p;
  ChamberRecHits *sChamberRight_p = sChamber_p;
  // neighbouring chamber (-1)

  if(iCh-FirstCh-1>=0){
    sChamberLeft_p =
      &(*all_RecHits)[WorkInEndcap-1][ExtrapolateToStation-1][ExtrapolateToRing-1][iCh-FirstCh-1].sChamber;
  }
  // neighbouring chamber (+1)
  if(iCh-FirstCh+1<NumCh){
    sChamberRight_p =
      &(*all_RecHits)[WorkInEndcap-1][ExtrapolateToStation-1][ExtrapolateToRing-1][iCh-FirstCh+1].sChamber;
  }
  int missingLayersLeft = 0;
  int missingLayersRight = 0;
  int missingLayers = 0;
  for(int iLayer=0;iLayer<6;iLayer++){
    if(!sChamber_p->RecHitsPosX[iLayer].size()){
      missingLayers++;
    }
    if(!sChamberLeft_p->RecHitsPosX[iLayer].size()){
      missingLayersLeft++;
    }
    if(!sChamberRight_p->RecHitsPosX[iLayer].size()){
      missingLayersRight++;
    }
  }
  if(missingLayers>missingLayersLeft || missingLayers>missingLayersRight){
    // Skip that chamber - the signal is noise most probably
  }
  else{
    //---- The segments points to "good region"  
    if(GoodLocalRegion(X, Y, ExtrapolateToStation, ExtrapolateToRing, iCh)){
      if(6==missingLayers){
	if(printalot) std::cout<<"missing all layers"<<std::endl;
      }
      for(int iLayer=0;iLayer<6;iLayer++){
	if(missingLayers){
	  for(unsigned int iRH=0; iRH<sChamber_p->RecHitsPosX[iLayer].size();iRH++){
	    ChHist[iCh-FirstCh].XvsY_InefficientRecHits->Fill(sChamber_p->RecHitsPosXlocal[iLayer][iRH],
							      sChamber_p->RecHitsPosYlocal[iLayer][iRH]);
	  }
	}
	//---- Are there rechits in the layer
	if(sChamber_p->NRecHits[iLayer]>0){
	  ChHist[iCh-FirstCh].EfficientRechits->Fill(iLayer+1);
	}
      }
      if(6!=missingLayers){
	ChHist[iCh-FirstCh].EfficientRechits->Fill(8);
      }
      ChHist[iCh-FirstCh].EfficientRechits->Fill(9);
    }
    //
    int badhits = 0;
    int realhit = 0; 
    for(int iLayer=0;iLayer<6;iLayer++){
      for(unsigned int iRH=0; iRH<sChamber_p->RecHitsPosX[iLayer].size();iRH++){
	realhit++;
	//---- A rechit in "good region"
	if(!GoodLocalRegion(sChamber_p->RecHitsPosXlocal[iLayer][iRH], 
			    sChamber_p->RecHitsPosYlocal[iLayer][iRH], 
			    ExtrapolateToStation, ExtrapolateToRing, iCh)){
	  badhits++;
	}
      }
    }
    
    //---- All rechits are in "good region" (no segment is required in the chamber)
    if(0!=realhit && 0==badhits ){
      if(printalot) std::cout<<"good rechits"<<std::endl;
      for(int iLayer=0;iLayer<6;iLayer++){
	if(missingLayers){
	  for(unsigned int iRH=0; iRH<sChamber_p->RecHitsPosX[iLayer].size();iRH++){
	    ChHist[iCh-FirstCh].XvsY_InefficientRecHits_good->Fill(sChamber_p->RecHitsPosXlocal[iLayer][iRH],
								   sChamber_p->RecHitsPosYlocal[iLayer][iRH]);
	  }
	}
	if(sChamber_p->NRecHits[iLayer]>0){
	  ChHist[iCh-FirstCh].EfficientRechits_good->Fill(iLayer+1);
	}
      }
      if(6!=missingLayers){
	ChHist[iCh-FirstCh].EfficientRechits_good->Fill(8);
      }
      ChHist[iCh-FirstCh].EfficientRechits_good->Fill(9);
    }
  }
}
//
bool CSCEfficiency::LCT_Efficiencies(edm::Handle<CSCALCTDigiCollection> alcts, 
				  edm::Handle<CSCCLCTDigiCollection> clcts, 
				  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts, int iCh, int cond){
  bool result = true;
  int Nalcts = 0;
  int Nclcts = 0;
  int Ncorr = 0;
  int Nalcts_1ch = 0;
  int Nclcts_1ch = 0;
  int Ncorr_1ch = 0;

  //---- ALCTDigis
  for (CSCALCTDigiCollection::DigiRangeIterator j=alcts->begin(); j!=alcts->end(); j++) {
    const CSCDetId& id = (*j).first;
    const CSCALCTDigiCollection::Range& range =(*j).second;
    for (CSCALCTDigiCollection::const_iterator digiIt =
	   range.first; digiIt!=range.second;
	 ++digiIt){
      //digiIt->print();
      // Valid digi in the chamber (or in neighbouring chamber) 
      if(0!=(*digiIt).isValid() && 
	 id.station()==ExtrapolateToStation && id.ring()==ExtrapolateToRing){
	if(id.chamber()==iCh){
	  //std::cout<<"iCh = "<<iCh<<std::endl;
          //digiIt->print();
	  Nalcts++;
	  Nalcts_1ch++;
	}
	else if(1==abs(id.chamber()-iCh)){
	  Nalcts++;
	}
      }
    }// for digis in layer
  }// end of for (j=...

  //---- CLCTDigis
  for (CSCCLCTDigiCollection::DigiRangeIterator j=clcts->begin(); j!=clcts->end(); j++) {
    const CSCDetId& id = (*j).first;
    std::vector<CSCCLCTDigi>::const_iterator digiIt = (*j).second.first;
    std::vector<CSCCLCTDigi>::const_iterator last = (*j).second.second;
    for( ; digiIt != last; ++digiIt) {
      //digiIt->print();
      // Valid digi in the chamber (or in neighbouring chamber) 
      if(0!=(*digiIt).isValid() && 
	 id.station()==ExtrapolateToStation && id.ring()==ExtrapolateToRing){
	if(id.chamber()==iCh){
          //digiIt->print();
	  //std::cout<<"iCh = "<<iCh<<std::endl;
	  Nclcts++;
	  Nclcts_1ch++;
	}
	else if(1==abs(id.chamber()-iCh)){
	  Nclcts++;
	}
      }
    }
  }

  //---- CorrLCTDigis
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j=correlatedlcts->begin(); j!=correlatedlcts->end(); j++) {
    const CSCDetId& id = (*j).first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator digiIt = (*j).second.first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator last = (*j).second.second;
    for( ; digiIt != last; ++digiIt) {
      //digiIt->print();
      // Valid digi in the chamber (or in neighbouring chamber) 
      if(0!=(*digiIt).isValid() && 
	 id.station()==ExtrapolateToStation && id.ring()==ExtrapolateToRing){
	if(id.chamber()==iCh){
          //digiIt->print();
	  //std::cout<<"iCh = "<<iCh<<std::endl;
	  Ncorr++;
	  Ncorr_1ch++;
	}
	else if(1==abs(id.chamber()-iCh)){
	  Ncorr++;
	}
      }
    }
  }
  //
  if(Nalcts){
    if(!Nclcts){
      if(printalot) std::cout<<"No alct-clct coincidence!"<<std::endl;
    }
    //---- Special logic
    ChHist[iCh-FirstCh].EfficientLCTs->Fill(1);
    if(Nalcts_1ch){
      ChHist[iCh-FirstCh].EfficientLCTs->Fill(11);
      if(cond){
	ChHist[iCh-FirstCh].EfficientLCTs->Fill(21);
	dydz_Eff_ALCT->Fill(seg_dydz);
      }
    }
  }
  else{
    if(cond){
      result = false;
    }
  //std::cout<<"no ALCT!!!"<<std::endl;
  }

  if(Nclcts){
    ChHist[iCh-FirstCh].EfficientLCTs->Fill(3);
    if(Nclcts_1ch){
      ChHist[iCh-FirstCh].EfficientLCTs->Fill(13);
      if(cond){
	ChHist[iCh-FirstCh].EfficientLCTs->Fill(23);
      }
    }
  }

  if(Ncorr){
    ChHist[iCh-FirstCh].EfficientLCTs->Fill(5);
    if(Ncorr_1ch){
      ChHist[iCh-FirstCh].EfficientLCTs->Fill(15);
      if(cond){
	ChHist[iCh-FirstCh].EfficientLCTs->Fill(25);
      }
    }
  }
  ChHist[iCh-FirstCh].EfficientLCTs->Fill(30);
  if(cond){
    if(!Nalcts) if(printalot) std::cout<<"NO ALCT!"<<std::endl;
    ChHist[iCh-FirstCh].EfficientLCTs->Fill(28);
    dydz_All_ALCT->Fill(seg_dydz);
  }
  return result;
}
//
void CSCEfficiency::StripWire_Efficiencies( int iCh){
  int EndCap = WorkInEndcap -1;
  int Ring = ExtrapolateToRing - 1;
  int Station = ExtrapolateToStation - 1;

  int missingLayers_s = 0;
  int missingLayers_wg = 0;
  int badhits_s = 0;
  int badhits_wg = 0;

 //----Strips
  for(int iLayer=0;iLayer<6;iLayer++){
    if(!AllStrips[EndCap][Station][Ring][iCh-FirstCh][iLayer].size()){
      missingLayers_s++;
    }
    for(unsigned int iStrip=0; 
	iStrip<AllStrips[EndCap][Station][Ring][iCh-FirstCh][iLayer].size();
	iStrip++){
      //---- better?
      int Nstrips =80;
      int Step = 10;
      if(1==ExtrapolateToStation && 3==ExtrapolateToRing){
	Nstrips =64;
      }
      //---- away from boundaries
      if(AllStrips[EndCap][Station][Ring][iCh-FirstCh][iLayer][iStrip].first<Step ||
	 AllStrips[EndCap][Station][Ring][iCh-FirstCh][iLayer][iStrip].first>(Nstrips-Step) ){
	badhits_s++;
      }
    }
  }

  //---- Wire groups
  for(int iLayer=0;iLayer<6;iLayer++){
    if(!AllWG[EndCap][Station][Ring][iCh-FirstCh][iLayer].size()){
      missingLayers_wg++;
    }
    for(unsigned int iWG=0; 
	iWG<AllWG[EndCap][Station][Ring][iCh-FirstCh][iLayer].size();
	iWG++){
      if(!GoodLocalRegion(0.,
			  AllWG[EndCap][Station][Ring][iCh-FirstCh][iLayer][iWG].first.second, 
			  ExtrapolateToStation, ExtrapolateToRing, iCh)){
	badhits_wg++;
      }
    }
  }
  //


  //
 //----Strips
  if(0==badhits_s && 0==badhits_wg){
    if(printalot){
      if(6!=missingLayers_wg){
	if(printalot) std::cout<<"good strips"<<std::endl;
      }
    }
    for(int iLayer=0;iLayer<6;iLayer++){
      if(AllStrips[EndCap][Station][Ring][iCh-FirstCh][iLayer].size()>0){
	ChHist[iCh-FirstCh].EfficientStrips->Fill(iLayer+1);
      }
      else{
	if(6!=missingLayers_s){
	  if(printalot) std::cout<<"missing strips iLayer+1 = "<<iLayer+1<<std::endl;
	}
      }
    }
    if(6!=missingLayers_s){
      ChHist[iCh-FirstCh].EfficientStrips->Fill(8);
    }
    ChHist[iCh-FirstCh].EfficientStrips->Fill(9);
  }

  //---- Wire groups

  if(0==badhits_wg && 0==badhits_s){
    if(printalot){
      if(6!=missingLayers_wg){
	if(printalot) std::cout<<"good WG"<<std::endl;
      }
    }
    for(int iLayer=0;iLayer<6;iLayer++){
      if(AllWG[EndCap][Station][Ring][iCh-FirstCh][iLayer].size()>0){
	ChHist[iCh-FirstCh].EfficientWireGroups->Fill(iLayer+1);
      }
      else{
	if(6!=missingLayers_wg){
	  if(printalot) std::cout<<"missing WG iLayer+1 = "<<iLayer+1<<std::endl;
	}
      }
    }
    if(6!=missingLayers_wg){
      ChHist[iCh-FirstCh].EfficientWireGroups->Fill(8);
    }
    ChHist[iCh-FirstCh].EfficientWireGroups->Fill(9);
  }
}
//
void CSCEfficiency::Segment_Efficiency(int iCh,  int NSegmentsFound){
  ChamberRecHits *sChamber_p =
    &(*all_RecHits)[WorkInEndcap-1][ExtrapolateToStation-1][ExtrapolateToRing-1][iCh-FirstCh].sChamber;
  int missingLayers = 0;
  for(int iLayer=0;iLayer<6;iLayer++){
    if(!sChamber_p->RecHitsPosX[iLayer].size()){
      missingLayers++;
    }
  }
  if(missingLayers<5){
    if(NSegmentsFound){
      EfficientSegments->Fill(iCh);
    }
    AllSegments->Fill(iCh);
  }
}
//
void CSCEfficiency::getEfficiency(float bin, float Norm, std::vector<float> &eff){
  //---- Efficiency with binomial error
  float Efficiency = 0.;
  float EffError = 0.;
  if(fabs(Norm)>0.000000001){
    Efficiency = bin/Norm;
    if(bin<Norm){
      EffError = sqrt( (1.-Efficiency)*Efficiency/Norm );
    }
  }
  eff[0] = Efficiency;
  eff[1] = EffError;
}
//
void CSCEfficiency::histoEfficiency(TH1F *readHisto, TH1F *writeHisto, int flag){
  std::vector<float> eff(2);
  int Nbins =  readHisto->GetSize()-2;//without underflows and overflows
  std::vector<float> bins(Nbins);
  std::vector<float> Efficiency(Nbins);
  std::vector<float> EffError(Nbins);
  float Norm = 1;
  if(flag<=Nbins){
    Norm = readHisto->GetBinContent(flag);;
  }
  for (int i=0;i<Nbins;i++){
    bins[i] = readHisto->GetBinContent(i+1);
    getEfficiency(bins[i], Norm, eff);
    Efficiency[i] = eff[0];
    EffError[i] = eff[1];
    writeHisto->SetBinContent(i+1, Efficiency[i]);
    writeHisto->SetBinError(i+1, EffError[i]);
  }  
}
//
const char* CSCEfficiency::ChangeTitle(const char * name){
  std::string str = to_string(name);
  std::string searchString( "Ch1" ); 
  std::string replaceString( "AllCh" );

  assert( searchString != replaceString );

  std::string::size_type pos = 0;
  while ( (pos = str.find(searchString, pos)) != string::npos ) {
    str.replace( pos, searchString.size(), replaceString );
    pos++;
  }
  const char* NewName = str.c_str();
  return NewName;
}
DEFINE_FWK_MODULE(CSCEfficiency);


