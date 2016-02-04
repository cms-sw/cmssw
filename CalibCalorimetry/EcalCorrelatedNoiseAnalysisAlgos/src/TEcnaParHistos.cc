//---------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified:17/03/2010
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParHistos.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"

ClassImp(TEcnaParHistos)
//______________________________________________________________________________
//
// TEcnaParHistos.
//
//    Values of different parameters for plots in the framework of TEcnaHistos
//    (see description of this class)
//
//    Examples of parameters:  ymin and ymax values for histos, title sizes,
//                             margins for plots, etc...
//                             
//
//-------------------------------------------------------------------------
//
//        For more details on other classes of the CNA package:
//
//                 http://www.cern.ch/cms-fabbro/cna
//
//-------------------------------------------------------------------------
//

//---------------------- TEcnaParHistos.cc -------------------------------
//  
//   Creation (first version): 19 May 2005
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------

  TEcnaParHistos::~TEcnaParHistos()
{
  //destructor
  
  //if (fEcalNumbering != 0){delete  fEcalNumbering; fCdelete++;}
  //if (fEcal          != 0){delete  fEcal;          fCdelete++;}
  
  //cout << "[Info Management] CLASS: TEcnaParHistos.     DESTROY OBJECT: this = " << this << endl;
}

//===================================================================
//
//                   Constructors
//
//===================================================================
TEcnaParHistos::TEcnaParHistos()
{
// Constructor without argument

  //cout << "[Info Management] CLASS: TEcnaParHistos.     CREATE OBJECT: this = " << this << endl;

  Init();
}

TEcnaParHistos::TEcnaParHistos(const TString SubDet,
			     const TEcnaParEcal* pEcal,
			     const TEcnaNumbering*  pEcalNumbering)
{
// Constructor with argument

  //cout << "[Info Management] CLASS: TEcnaParHistos.     CREATE OBJECT: this = " << this << endl;

  Init();
  SetEcalSubDetector(SubDet.Data(), pEcal, pEcalNumbering);
}

void  TEcnaParHistos::Init()
{
  fgMaxCar = (Int_t)512;              // max number of characters in TStrings

  fTTBELL = '\007';

  //................ Init CNA Command and error numbering
  fCnaCommand = 0;
  fCnaError   = 0;

  //........................................................................................

  fMaxColorNumber     = (Int_t)4;       // number_of_colors - 1 for histos in option SAME

  fMarginAutoMinMax   = (Double_t)0.05; // factor for automatic margin plots:
                                        // plot min = min - (max-min)*fMarginAutoMinMax
                                        // plot max = max + (max-min)*fMarginAutoMinMax

  fNbOfRunsDisplayed  = (Int_t)20;      // Nb of run numbers displayed on the top of the plot
                                        // in stability/evolution plots, option ONLYONE

  fMaxNbOfRunsInLists = (Int_t)5000;    // Maximum number of runs in the ListOfRun files
                                        // used for the stability/evolution plots.

}// end of Init()

void TEcnaParHistos::SetEcalSubDetector(const TString SubDet,
				       const TEcnaParEcal* pEcal,
				       const TEcnaNumbering*  pEcalNumbering)
{
 // Set Subdetector (EB or EE)

  fEcal = 0;
  if( pEcal == 0 )
    {fEcal = new TEcnaParEcal(SubDet.Data());     /*fCnew++*/ ;}
  else
    {fEcal = (TEcnaParEcal*)pEcal;}

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();

  fEcalNumbering = 0;
  if( pEcalNumbering == 0 )
    {fEcalNumbering = new TEcnaNumbering(fFlagSubDet.Data(), fEcal);     /*fCnew++*/ ;}
  else
    {fEcalNumbering = (TEcnaNumbering*)pEcalNumbering;}
}

Int_t TEcnaParHistos::MaxNbOfRunsInLists(){return fMaxNbOfRunsInLists;}

//===========================================================================
//
//    SetCanvasWidth, SetCanvasHeight,
//    CanvasFormatW, CanvasFormatH,
//
//===========================================================================
// *  1 H1NbOfEvtsGlobal     D_NOE_ChNb    NOE = Number Of Events
// *  2 H1NbOfEvtsProj       D_NOE_ChDs    
// *  3 H1EvEvGlobal         D_Ped_ChNb    Ped = Pedestal
// *  4 H1EvEvProj           D_Ped_ChDs
// *  5 H1EvSigGlobal        D_TNo_ChNb    TNo = Total Noise
// *  6 H1EvSigProj          D_TNo_ChDs
// *  7 H1SigEvGlobal        D_LFN_ChNb    LFN = Low Frequency noise
// *  8 H1SigEvProj          D_LFN_ChDs
// *  9 H1SigSigGlobal       D_HFN_ChNb    HFN = High Frequency noise
// * 10 H1SigSigProj         D_HFN_ChDs
// * 11 H1EvCorssGlobal      D_MCs_ChNb    MCs = Mean of the correlations between samples
// * 12 H1EvCorssProj        D_MCs_ChDs
// * 13 H1SigCorssGlobal     D_SCs_ChNb    Scs = Sigma of the correlations between samples
// * 14 H1SigCorssProj       D_SCs_ChDs
// * 15 Ev                   D_MSp_Samp    MSp = Mean of the samples
// * 16 Sigma                D_SSp_Samp    SSp = Sigma of the samples
// * 17 SampTime             D_Adc_EvNb    Adc = ADC count as a function of Event number
// * 18 AdcProj              D_Adc_EvDs    EvDs = Event distribution
// * 19 EvolEvEv             H_Ped_Date    Date = date in format YYMMJJ hhmmss
// * 20 EvolEvEvProj         H_Ped_RuDs    RuDs = Run distribution
// * 21 EvolEvSig            H_TNo_Date
// * 22 EvolEvSigProj        H_TNo_RuDs   
// * 23 EvolSigEv            H_LFN_Date 
// * 24 EvolSigEvProj        H_LFN_RuDs 
// * 25 EvolSigSig           H_HFN_Date 
// * 26 EvolSigSigProj       H_HFN_RuDs 
// * 27 EvolEvCorss          H_MCs_Date   
// * 28 EvolEvCorssProj      H_MCs_RuDs   
// * 29 EvolSigCorss         H_SCs_Date   
// * 30 EvolSigCorssProj     H_SCs_RuDs   

UInt_t TEcnaParHistos::SetCanvasWidth(const TString HistoCode, const TString opt_plot)
{
//........................................ Taille/format canvas
  
  UInt_t canv_w = CanvasFormatW("petit");

  if( HistoCode == "D_Adc_EvNb" || HistoCode == "D_NOE_ChNb" ||
      HistoCode == "D_Ped_ChNb" || HistoCode == "D_LFN_ChNb" ||
      HistoCode == "D_TNo_ChNb" || HistoCode == "D_HFN_ChNb" ||
      HistoCode == "D_MCs_ChNb" || HistoCode == "D_SCs_ChNb" ||
      HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" ||
      HistoCode == "H_MCs_Date" || HistoCode == "H_LFN_Date" ||
      HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" ||
      (HistoCode == "D_MSp_Samp" && opt_plot != "ONLYONE")   || 
      (HistoCode == "D_SSp_Samp" && opt_plot != "ONLYONE") )
    {	     
      canv_w = CanvasFormatH("moyen");    // format 29.7*21 ( = 21*29.7 en paysage)
    }
  
  if( (HistoCode == "D_Adc_EvDs" || HistoCode == "D_NOE_ChDs" ||
       HistoCode == "D_Ped_ChDs" || HistoCode == "D_LFN_ChDs" ||
       HistoCode == "D_TNo_ChDs" || HistoCode == "D_HFN_ChDs" ||
       HistoCode == "D_MCs_ChDs" || HistoCode == "D_SCs_ChDs" ||
       HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" ||
       HistoCode == "H_MCs_RuDs" || HistoCode == "H_LFN_RuDs" ||
       HistoCode == "H_HFN_RuDs" || HistoCode == "H_SCs_RuDs") && opt_plot != "ONLYONE" )
    {	     
      canv_w = CanvasFormatH("moyen");    // format 29.7*21 ( = 21*29.7 en paysage)
    }

  return canv_w;
}

UInt_t TEcnaParHistos::SetCanvasHeight(const TString HistoCode, const TString opt_plot)
{
//........................................ Taille/format canvas
  
  UInt_t canv_h = CanvasFormatH("petit");

  if( HistoCode == "D_Adc_EvNb" || HistoCode == "D_NOE_ChNb" ||
      HistoCode == "D_Ped_ChNb" || HistoCode == "D_LFN_ChNb" ||
      HistoCode == "D_TNo_ChNb" || HistoCode == "D_HFN_ChNb" ||
      HistoCode == "D_MCs_ChNb" || HistoCode == "D_SCs_ChNb" ||
      HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" ||
      HistoCode == "H_MCs_Date" || HistoCode == "H_LFN_Date" ||
      HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" ||
      (HistoCode == "D_MSp_Samp" && opt_plot != "ONLYONE")   ||
      (HistoCode == "D_SSp_Samp" && opt_plot != "ONLYONE") )
    {
      canv_h = CanvasFormatW("moyen");    // format 29.7*21 ( = 21*29.7 en paysage)
    }

  if( (HistoCode == "D_Adc_EvDs" || HistoCode == "D_NOE_ChDs" ||
       HistoCode == "D_Ped_ChDs" || HistoCode == "D_LFN_ChDs" ||
       HistoCode == "D_TNo_ChDs" || HistoCode == "D_HFN_ChDs" ||
       HistoCode == "D_MCs_ChDs" || HistoCode == "D_SCs_ChDs" ||
       HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" ||
       HistoCode == "H_MCs_RuDs" || HistoCode == "H_LFN_RuDs" ||
       HistoCode == "H_HFN_RuDs" || HistoCode == "H_SCs_RuDs") && opt_plot != "ONLYONE" )
    {	     
      canv_h = CanvasFormatW("moyen");    // format 29.7*21 ( = 21*29.7 en paysage)
    }

  return canv_h;
}

UInt_t TEcnaParHistos::CanvasFormatW(const TString chformat)
{
//Set Canvas width

  UInt_t canv_w = 375;         // default = "petit"

  if ( chformat == "petit"    ){canv_w = 375;} // 21x29.7 portrait
  if ( chformat == "moyen"    ){canv_w = 450;} // 21x29.7 portrait
  if ( chformat == "grand"    ){canv_w = 572;} // 21x29.7 portrait

  if ( chformat == "etaphiSM" ){canv_w = 760;} // landscape etroit en hauteur pour plot SM
  if ( chformat == "phietaEB" ){canv_w = 636;} // landscape pour plot EB (32 SMs)

  if ( chformat == "IXIYDee"  ){canv_w = 320;} // portrait etroit en largeur pour plot Dee
  if ( chformat == "IYIXEE"   ){canv_w = 636;} // landscape pour plot EE (4 dees)
  return canv_w;
}
//......................................................................
UInt_t TEcnaParHistos::CanvasFormatH(const TString chformat)
{
//Set Canvas height

  UInt_t canv_h = 530;         // default = "petit"

  if ( chformat == "petit"    ){canv_h = 530;} // 21x29.7 portrait
  if ( chformat == "moyen"    ){canv_h = 636;} // 21x29.7 portrait
  if ( chformat == "grand"    ){canv_h = 810;} // 21x29.7 portrait

  if ( chformat == "etaphiSM" ){canv_h = 306;} // landscape etroit en hauteur pour plot SM
  if ( chformat == "phietaEB" ){canv_h = 450;} // landscape pour plot EB (32 SMs)

  if ( chformat == "IXIYDee"  ){canv_h = 720;} // portrait etroit en largeur pour plot Dee
  if ( chformat == "IYIXEE"   ){canv_h = 450;} // landscape pour plot EE (4 dees)
  return canv_h;
}

//===========================================================================
//
//          BoxLeftX, BoxRightX, BoxBottomY, BoxTopY
//
//===========================================================================
Double_t TEcnaParHistos::BoxLeftX(const TString chtype)
{
//Set the x left coordinate of the box

  Double_t value = 0.4;

  if ( chtype == "general_comment"      ) {value = 0.015;}

  if ( chtype == "top_left_box_SM"      ) {value = 0.015;}
  if ( chtype == "top_left_box_EB"      ) {value = 0.015;}
  if ( chtype == "top_mid_box_EB"       ) {value = 0.350;}
  if ( chtype == "top_right_box_EB"     ) {value = 0.630;}

  if ( chtype == "top_left_box_Dee"     ) {value = 0.015;}
  if ( chtype == "top_left_box_EE"      ) {value = 0.015;}
  if ( chtype == "top_mid_box_EE"       ) {value = 0.240;}
  if ( chtype == "top_right_box_EE"     ) {value = 0.610;}

  if ( chtype == "bottom_left_box"      ) {value = 0.015;}
  if ( chtype == "bottom_left_box_evol" ) {value = 0.015;}
  if ( chtype == "bottom_mid_box"       ) {value = 0.250;}
  if ( chtype == "bottom_right_box"     ) {value = 0.455;}
  if ( chtype == "bottom_right_box_evol") {value = 0.620;}
  if ( chtype == "bottom_right_box_evpr") {value = 0.580;}

  if ( chtype == "several_plots_box"    ) {value = 0.015;}
  if ( chtype == "several_evol_box"     ) {value = 0.015;}
  if ( chtype == "several_evpr_box"     ) {value = 0.015;}

 return value;
}
//.................................................................
Double_t TEcnaParHistos::BoxRightX(const TString chtype)
{
//Set the x right coordinate of the box
  
  Double_t value = 0.6;
  
  if ( chtype == "general_comment"      ) {value = 0.680;}

  if ( chtype == "top_left_box_SM"      ) {value = 0.125;}
  if ( chtype == "top_left_box_EB"      ) {value = 0.349;}
  if ( chtype == "top_mid_box_EB"       ) {value = 0.629;}
  if ( chtype == "top_right_box_EB"     ) {value = 0.985;}

  if ( chtype == "top_left_box_Dee"     ) {value = 0.425;}
  if ( chtype == "top_left_box_EE"      ) {value = 0.260;}
  if ( chtype == "top_mid_box_EE"       ) {value = 0.609;}
  if ( chtype == "top_right_box_EE"     ) {value = 0.985;}

  if ( chtype == "bottom_left_box"      ) {value = 0.445;}
  if ( chtype == "bottom_left_box_evol" ) {value = 0.450;} // ex 370
  if ( chtype == "bottom_mid_box"       ) {value = 0.750;}
  if ( chtype == "bottom_right_box"     ) {value = 0.985;}
  if ( chtype == "bottom_right_box_evol") {value = 0.985;}
  if ( chtype == "bottom_right_box_evpr") {value = 0.985;}

  if ( chtype == "several_plots_box"    ) {value = 0.985;}
  if ( chtype == "several_evol_box"     ) {value = 0.615;}
  if ( chtype == "several_evpr_box"     ) {value = 0.615;}
  
 return value;
}
//.................................................................
Double_t TEcnaParHistos::BoxBottomY(const TString chtype)
{
//Set the y bottom coordinate of the box

  Double_t value = 0.8;
  
  if ( chtype == "general_comment"      ) {value = 0.960;}

  if ( chtype == "top_left_box_SM"      ) {value = 0.880;}
  if ( chtype == "top_left_box_EB"      ) {value = 0.880;}
  if ( chtype == "top_mid_box_EB"       ) {value = 0.880;}
  if ( chtype == "top_right_box_EB"     ) {value = 0.880;}

  if ( chtype == "top_left_box_Dee"     ) {value = 0.925;}
  if ( chtype == "top_left_box_EE"      ) {value = 0.880;}
  if ( chtype == "top_mid_box_EE"       ) {value = 0.880;}
  if ( chtype == "top_right_box_EE"     ) {value = 0.880;}

  if ( chtype == "bottom_left_box"      ) {value = 0.010;}
  if ( chtype == "bottom_left_box_evol" ) {value = 0.010;}
  if ( chtype == "bottom_right_box"     ) {value = 0.010;}
  if ( chtype == "bottom_right_box_evol") {value = 0.010;}
  if ( chtype == "bottom_right_box_evpr") {value = 0.010;}
  if ( chtype == "several_plots_box"    ) {value = 0.010;}
  if ( chtype == "several_evol_box"     ) {value = 0.010;}
  if ( chtype == "several_evpr_box"     ) {value = 0.010;}

 return value;
}
//.................................................................
Double_t TEcnaParHistos::BoxTopY(const TString chtype)
{
//Set the y top coordinate of the box

  Double_t value = 0.9;
  
  if ( chtype == "general_comment"      ) {value = 0.999;}

  if ( chtype == "top_left_box_SM"      ) {value = 0.955;}
  if ( chtype == "top_left_box_EB"      ) {value = 0.955;}
  if ( chtype == "top_mid_box_EB"       ) {value = 0.955;}
  if ( chtype == "top_right_box_EB"     ) {value = 0.955;}

  if ( chtype == "top_left_box_Dee"     ) {value = 0.969;}
  if ( chtype == "top_left_box_EE"      ) {value = 0.955;}
  if ( chtype == "top_mid_box_EE"       ) {value = 0.955;}
  if ( chtype == "top_right_box_EE"     ) {value = 0.955;}
  
  if ( chtype == "bottom_left_box"      ) {value = 0.120;}  
  if ( chtype == "bottom_left_box_evol" ) {value = 0.120;}
  if ( chtype == "bottom_right_box"     ) {value = 0.120;}
  if ( chtype == "bottom_right_box_Dee" ) {value = 0.100;}

  if ( chtype == "bottom_right_box_evol") {value = 0.120;}
  if ( chtype == "bottom_right_box_evpr") {value = 0.120;}
  if ( chtype == "several_plots_box"    ) {value = 0.200;}
  if ( chtype == "several_evol_box"     ) {value = 0.200;}
  if ( chtype == "several_evpr_box"     ) {value = 0.200;}

 return value;
}

//===========================================================================
//
//       SetColorPalette, ColorTab, ColorDefinition 
//
//===========================================================================

void TEcnaParHistos::SetColorPalette(const TString user_color_palette)
{
  //Set color number for palette

  fMaxColorNumber = 4;         // default: 4 ==> Black/Red/Blue/Green/Braun

  if (!(user_color_palette == "Rainbow" || user_color_palette == "rainbow") ){fMaxColorNumber = 5;}
  if (  user_color_palette == "Rainbow" || user_color_palette == "rainbow"  ){fMaxColorNumber = 6;}
}
//...........................................................................
Color_t TEcnaParHistos::ColorTab(const Int_t& user_color_number)
{
  //Set color from user color number

  //=========> Color definition: see ROOT User's guide p.151

  TColor* my_color = new TColor();

  Color_t couleur = ColorDefinition("noir");        // default = "noir"

  if ( fMaxColorNumber  != 6 )
    {
      //................... Standard: 6 colors (fMaxColorNumber = 5)
      if(user_color_number == 0){couleur = ColorDefinition("noir");}                  //    noir        black    1
      if(user_color_number == 1){couleur = ColorDefinition("rouge");}                 //    rouge       red      2
      if(user_color_number == 2){couleur = ColorDefinition("bleu");}                  //    bleu        blue     9
      if(user_color_number == 3){couleur = (Color_t)my_color->GetColor("#009900");}   //    vert        green
      if(user_color_number == 4){couleur = ColorDefinition("marron28");}              //    marron 28   braun   28
      if(user_color_number == 5){couleur = (Color_t)my_color->GetColor("#9900BB");}   //    violet      purple

    }
  if ( fMaxColorNumber  == 6 )
    {
      //................... Rainbow:7  colors (fMaxColorNumber = 6)
      if(user_color_number == 0){couleur = (Color_t)my_color->GetColor("#EE0000");}  //   rouge
      if(user_color_number == 1){couleur = (Color_t)my_color->GetColor("#FF6611");}  //   orange
      if(user_color_number == 2){couleur = (Color_t)my_color->GetColor("#FFCC00");}  //   jaune
      if(user_color_number == 3){couleur = (Color_t)my_color->GetColor("#009900");}  //   vert
      if(user_color_number == 4){couleur = (Color_t)my_color->GetColor("#0044EE");}  //   bleu
      if(user_color_number == 5){couleur = (Color_t)my_color->GetColor("#6633BB");}  //   indigo
      if(user_color_number == 6){couleur = (Color_t)my_color->GetColor("#9900BB");}  //   violet
    }

  if( user_color_number < 0 || user_color_number > fMaxColorNumber ){couleur = 0;}

  return couleur;
}
// end of ColorTab

Color_t TEcnaParHistos::ColorDefinition(const TString chcolor)
{
  //Set color from color name

  //=========> Color definition: see ROOT User's guide p.151

  Color_t couleur = 1;        // default = "noir"

  if ( chcolor == "noir"       ) {couleur =  1;}
  if ( chcolor == "rouge"      ) {couleur =  2;}
  if ( chcolor == "vert_fonce" ) {couleur =  3;}
  if ( chcolor == "bleu_fonce" ) {couleur =  4;}
  if ( chcolor == "jaune"      ) {couleur =  5;}
  if ( chcolor == "rose"       ) {couleur =  6;}
  if ( chcolor == "bleu_clair" ) {couleur =  7;}
  if ( chcolor == "vert"       ) {couleur =  8;}
  if ( chcolor == "bleu"       ) {couleur =  9;}
  if ( chcolor == "blanc"      ) {couleur = 10;}

  if ( chcolor == "marron23"   ) {couleur = 23;}
  if ( chcolor == "marron24"   ) {couleur = 24;}
  if ( chcolor == "marron25"   ) {couleur = 24;}
  if ( chcolor == "marron27"   ) {couleur = 27;}
  if ( chcolor == "marron28"   ) {couleur = 28;}

  if ( chcolor == "bleu33"     ) {couleur = 33;}
  if ( chcolor == "bleu36"     ) {couleur = 36;}
  if ( chcolor == "bleu38"     ) {couleur = 38;}
  if ( chcolor == "bleu39"     ) {couleur = 39;}

  if ( chcolor == "orange41"   ) {couleur = 41;}
  if ( chcolor == "orange42"   ) {couleur = 42;}

  if ( chcolor == "rouge44"    ) {couleur = 44;}
  if ( chcolor == "rouge46"    ) {couleur = 46;}
  if ( chcolor == "rouge47"    ) {couleur = 47;}
  if ( chcolor == "rouge48"    ) {couleur = 48;}
  if ( chcolor == "rouge49"    ) {couleur = 49;}
  if ( chcolor == "rouge50"    ) {couleur = 50;}

  if ( chcolor == "vert31"     ) {couleur = 31;}
  if ( chcolor == "vert32"     ) {couleur = 32;}
  if ( chcolor == "vert36"     ) {couleur = 36;}
  if ( chcolor == "vert37"     ) {couleur = 37;}

  if ( chcolor == "violet"     ) {couleur = 49;}

  if ( chcolor == "turquoise29") {couleur = 29;}

  if ( chcolor == "gris12"     ) {couleur = 12;}
  if ( chcolor == "gris13"     ) {couleur = 13;}
  if ( chcolor == "gris14"     ) {couleur = 14;}
  if ( chcolor == "gris15"     ) {couleur = 15;}
  if ( chcolor == "gris16"     ) {couleur = 16;}
  if ( chcolor == "gris17"     ) {couleur = 17;}
  if ( chcolor == "gris18"     ) {couleur = 18;}

  return couleur;
}
// end of ColorDefinition

//===========================================================================
//       GetMaxNbOfColors, GetMarginAutoMinMax, GetNbOfRunsDisplayed
//===========================================================================
Int_t TEcnaParHistos::GetMaxNbOfColors(){return fMaxColorNumber;}
Int_t TEcnaParHistos::GetNbOfRunsDisplayed(){return fNbOfRunsDisplayed;}
Double_t TEcnaParHistos::GetMarginAutoMinMax(){return fMarginAutoMinMax;}

//===========================================================================
//
//      SetViewHistoStyle, SetViewHistoPadMargins
//
//===========================================================================
void TEcnaParHistos::SetViewHistoStyle(const TString HistoType)
{
// Set style parameters for histo view

  //......................... Palette
  gStyle->SetPalette(1,0);  // default: rainbow spectrum

  //............................... Date
  gStyle->SetOptDate(0);

  //............................... Histo title size
  gStyle->SetTitleW(0.5); gStyle->SetTitleH(0.075);     // default

  if( HistoType == "colz"  || HistoType == "lego"  ||
      HistoType == "surf"  || HistoType == "Stin" )
    {gStyle->SetTitleW(0.8); gStyle->SetTitleH(0.075);}

  if( (HistoType == "H1Basic") || (HistoType == "Evol") )
    {gStyle->SetTitleW(0.5); gStyle->SetTitleH(0.075);}

  if( HistoType == "Stex2DEB" )
    {gStyle->SetTitleW(0.2); gStyle->SetTitleH(0.095);}

  if( HistoType == "Stex2DEE" )
    {gStyle->SetTitleW(0.45); gStyle->SetTitleH(0.08);}

  if( HistoType == "Stex2DEENb" )
    {gStyle->SetTitleW(0.85); gStyle->SetTitleH(0.08);}

  if( HistoType == "Stas2DEB" || HistoType == "Stas2DEE" )
    {gStyle->SetTitleW(0.225); gStyle->SetTitleH(0.095);}

  if(HistoType == "Global" || HistoType == "Proj" || HistoType == "SampProj" || HistoType == "EvolProj")
    {gStyle->SetTitleW(0.5); gStyle->SetTitleH(0.08);}

  //............................. Statistics box style
     gStyle->SetOptStat(1110); 

  //............................. Statistics box size
  if( HistoType == "colz"  || HistoType == "lego"  ||
      HistoType == "surf"  || HistoType == "Stin" )
    {
      gStyle->SetStatW(0.55);  gStyle->SetStatH(0.2);
      gStyle->SetStatY(1);
    }

  if( HistoType == "Global" )
    {
      gStyle->SetStatW(0.375);  gStyle->SetStatH(0.180);
      gStyle->SetStatY(0.9875);
    }
  if( HistoType == "Proj" || HistoType == "SampProj" || HistoType == "EvolProj" )
    {
      gStyle->SetStatW(0.375);  gStyle->SetStatH(0.180);
      gStyle->SetStatY(0.9875);
    }

  if( (HistoType == "H1Basic") || (HistoType == "Evol") )
    {
      gStyle->SetStatW(0.375);   gStyle->SetStatH(0.180);
      gStyle->SetStatY(0.9875);
    }

  //....... Marker and line style

  gStyle->SetMarkerStyle(1);
  if( HistoType == "Global" ){gStyle->SetMarkerStyle(20);}


}
//------- (end of SetViewHistoStyle) ----------------

void TEcnaParHistos::SetViewHistoPadMargins(const TString HistoType, const TString opt_plot)
{
// Set active pad margins for histo view

//.......... default
  gStyle->SetPadBottomMargin(0.125);	  	  
  gStyle->SetPadTopMargin(0.125);
  gStyle->SetPadLeftMargin(0.125);
  gStyle->SetPadRightMargin(0.125);
  
  if(HistoType == "colz"  || HistoType == "lego"  ||
     HistoType == "surf"  || HistoType == "Stin")  
    {
      gStyle->SetPadBottomMargin(0.125);	  	  
      gStyle->SetPadTopMargin(0.135);
      gStyle->SetPadLeftMargin(0.135);
      gStyle->SetPadRightMargin(0.135); 
    }
  
  if( HistoType == "Stex2DEB" )
    {
      gStyle->SetPadBottomMargin(0.145);	  	  
      gStyle->SetPadTopMargin(0.135);
      gStyle->SetPadLeftMargin(0.125);
      gStyle->SetPadRightMargin(0.125); 
    }

  if( HistoType == "Stex2DEE" || HistoType == "Stex2DEENb")
    {
      gStyle->SetPadBottomMargin(0.120);	  	  
      gStyle->SetPadTopMargin(0.110);
      gStyle->SetPadLeftMargin(0.130);
      gStyle->SetPadRightMargin(0.130); 
    }
  if( HistoType == "Stas2DEB" )
    {
      gStyle->SetPadBottomMargin(0.145);	  	  
      gStyle->SetPadTopMargin(0.135);
      gStyle->SetPadLeftMargin(0.125);
      gStyle->SetPadRightMargin(0.125); 
    }

  if( HistoType == "Stas2DEE")
    {
      gStyle->SetPadBottomMargin(0.145);	  	  
      gStyle->SetPadTopMargin(0.135);
      gStyle->SetPadLeftMargin(0.105);
      gStyle->SetPadRightMargin(0.115); 
    }
  if(HistoType == "Global")
    {
      gStyle->SetPadBottomMargin((Float_t)0.125);	  	  
      gStyle->SetPadTopMargin((Float_t)0.175);
      gStyle->SetPadLeftMargin((Float_t)0.115);
      gStyle->SetPadRightMargin((Float_t)0.05);
    }

  if(HistoType == "Proj" || HistoType == "SampProj")
    {
      gStyle->SetPadBottomMargin(0.115);
      gStyle->SetPadTopMargin(0.155);
      gStyle->SetPadLeftMargin(0.15);
      gStyle->SetPadRightMargin(0.05);
    }

  if( HistoType == "H1Basic" )
    {
      gStyle->SetPadBottomMargin(0.1275);	  	  
      gStyle->SetPadTopMargin(0.165);
      gStyle->SetPadLeftMargin(0.15);
      gStyle->SetPadRightMargin(0.05);
    }

  if( HistoType == "Evol" && opt_plot == "ONLYONE")
    {
      gStyle->SetPadBottomMargin(0.110);	  	  
      gStyle->SetPadTopMargin(0.2);
      gStyle->SetPadLeftMargin(0.115);
      gStyle->SetPadRightMargin(0.035);
    }
  if( HistoType == "Evol" && opt_plot == "SAME")
    {
      gStyle->SetPadBottomMargin(0.110);	  	  
      gStyle->SetPadTopMargin(0.185);
      gStyle->SetPadLeftMargin(0.115);
      gStyle->SetPadRightMargin(0.035);   
    }
  if( HistoType == "Evol" && opt_plot == "SAME n")
    {
      gStyle->SetPadBottomMargin(0.110);	  	  
      gStyle->SetPadTopMargin(0.1);
      gStyle->SetPadLeftMargin(0.115);
      gStyle->SetPadRightMargin(0.035);   
    }

  if( HistoType == "EvolProj" && opt_plot == "ONLYONE")
    {
      gStyle->SetPadBottomMargin(0.1275);	  	  
      gStyle->SetPadTopMargin(0.165);
      gStyle->SetPadLeftMargin(0.15);
      gStyle->SetPadRightMargin(0.05);
    }
  if( HistoType == "EvolProj" && opt_plot == "SAME")
    {
      gStyle->SetPadBottomMargin(0.110);	  	  
      gStyle->SetPadTopMargin(0.185);
      gStyle->SetPadLeftMargin(0.115);
      gStyle->SetPadRightMargin(0.035);   
    }
  if( HistoType == "EvolProj" && opt_plot == "SAME n")
    {
      gStyle->SetPadBottomMargin(0.110);	  	  
      gStyle->SetPadTopMargin(0.1);
      gStyle->SetPadLeftMargin(0.115);
      gStyle->SetPadRightMargin(0.035);   
    }
}
//------- (end of SetViewHistoPadMargins) ----------------

//===========================================================================
//
//      SetViewHistoStats, SetViewHistoOffsets,
//      AxisTitleOffset, AxisTitleSize, AxisLabelOffset,
//      AxisLabelSize, AxisTickSize, DeeOffsetX, DeeNameOffsetX,
//      SetViewGraphOffsets
//      
//===========================================================================
void TEcnaParHistos::SetViewHistoStats(TH1D* histo, const TString HistoType)
{
  // Set stats box for histo view

  Bool_t b_true  = kTRUE;
  Bool_t b_false = kFALSE;

  histo->SetStats(b_false);
	      
  if(HistoType == "Global"){histo->SetStats(b_false);}
  if(HistoType == "Proj" || HistoType == "SampProj" ){histo->SetStats(b_true);}
}

void TEcnaParHistos::SetViewHistoOffsets(TH1D* histo, const TString HistoType, const TString opt_plot)
{
// Set offsets of labels, title axis, etc... for histo view
  
  if(HistoType == "Global")
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.05);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.006);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.5);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }
  
  if(HistoType == "Proj" || HistoType == "SampGlobal" || HistoType == "SampProj" )
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.05);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.006);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.75);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }
  
  if( HistoType == "H1Basic" || HistoType == "Evol" ||
      ( HistoType == "EvolProj" && !(opt_plot == "SAME" || opt_plot == "SAME n") ) )
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.25);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.005);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.75);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);
      
      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);
      
      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }
  
  if( HistoType == "EvolProj" && (opt_plot == "SAME" || opt_plot == "SAME n") )
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.25);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);
      
      histo->GetXaxis()->SetLabelOffset((Float_t)0.005);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.25);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }
  
  if(HistoType == "colz")
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.25);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.005);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.45);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }

  if(HistoType == "lego" || HistoType == "surf" )
    {
      //....................... x axis
      histo->GetXaxis()->SetTitleOffset((Float_t)1.7);
      histo->GetXaxis()->SetTitleSize((Float_t)0.04);

      histo->GetXaxis()->SetLabelOffset((Float_t)0.005);
      histo->GetXaxis()->SetLabelSize((Float_t)0.04);

      histo->GetXaxis()->SetTickLength((Float_t)0.03);
      histo->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      histo->GetYaxis()->SetTitleOffset((Float_t)1.85);
      histo->GetYaxis()->SetTitleSize((Float_t)0.04);

      histo->GetYaxis()->SetLabelOffset((Float_t)0.01);
      histo->GetYaxis()->SetLabelSize((Float_t)0.04);

      histo->GetYaxis()->SetTickLength((Float_t)0.03);
      histo->GetYaxis()->SetNdivisions((Int_t)510);
    }

  if(HistoType == "Stin")
    {
      //.......... x axis (remove labels and ticks)
      histo->GetXaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetXaxis()->SetTitleSize((Float_t)0.05); 
 
      histo->GetXaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetXaxis()->SetLabelSize((Float_t)0.);

      histo->GetXaxis()->SetNdivisions((Int_t)1);
      histo->GetXaxis()->SetTickLength((Float_t)0.);

      //.......... y axis (remove labels and ticks)
      histo->GetYaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetYaxis()->SetTitleSize((Float_t)0.05);
     
      histo->GetYaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetYaxis()->SetLabelSize((Float_t)0.);

      histo->GetYaxis()->SetNdivisions((Int_t)1);
      histo->GetYaxis()->SetTickLength((Float_t)0.);
    }

  if( HistoType == "Stex2DEB" || HistoType == "Stex2DEE" || HistoType == "Stex2DEENb" ||
      HistoType == "Stas2DEB" || HistoType == "Stas2DEE" )
    {
      //.......... x axis (remove labels and ticks)
      histo->GetXaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetXaxis()->SetTitleSize((Float_t)0.075); 
 
      histo->GetXaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetXaxis()->SetLabelSize((Float_t)0.);

      histo->GetXaxis()->SetNdivisions((Int_t)1);
      histo->GetXaxis()->SetTickLength((Float_t)0.);

      //.......... y axis (remove labels and ticks)
      histo->GetYaxis()->SetTitleOffset((Float_t)9999.);
      histo->GetYaxis()->SetTitleSize((Float_t)0.075);
     
      histo->GetYaxis()->SetLabelOffset((Float_t)9999.);
      histo->GetYaxis()->SetLabelSize((Float_t)0.);

      histo->GetYaxis()->SetNdivisions((Int_t)1);
      histo->GetYaxis()->SetTickLength((Float_t)0.);
    }
}

//----------------------------------------------
Float_t TEcnaParHistos::AxisTitleOffset()
{
 return AxisTitleOffset("");
}
Float_t TEcnaParHistos::AxisTitleOffset(const TString opt)
{
  Float_t tit_off = 1.;

  if( opt == ""        ){tit_off = 0.055;}
  if( opt == "Towx"    ){tit_off = 1.15;}
  if( opt == "Towy"    ){tit_off = 1.15;}
  if( opt == "SMx"     ){tit_off = 1.15;}
  if( opt == "SMy"     ){tit_off = 0.5;}
  if( opt == "SMyInEB" ){tit_off = 0.7;}
  if( opt == "EBy"     ){tit_off = 1.;}
  if( opt == "EBx"     ){tit_off = 1.3;}

  if( opt == "SCx" ){tit_off = 1.15;}
  if( opt == "SCy" ){tit_off = 1.15;}
  if( opt == "Deex"){tit_off = 0.75;}
  if( opt == "Deey"){tit_off = 1.2;}
  if( opt == "EEx" ){tit_off = 1.2;}
  if( opt == "EEy" ){tit_off = 1.;}

  return tit_off;
}
//............................................
Float_t TEcnaParHistos::AxisTitleSize()
{
 return AxisTitleSize("");
}
Float_t  TEcnaParHistos::AxisTitleSize(const TString opt)
{
  Float_t tit_siz = 0.045;
  if( opt == ""  ){tit_siz = 0.045;}
  return tit_siz;
}
//............................................
Float_t  TEcnaParHistos::AxisLabelOffset()
{
 return AxisLabelOffset("");
}
Float_t TEcnaParHistos::AxisLabelOffset(const TString opt)
{
  Float_t lab_off = 0.02;
  if( opt == ""    ){lab_off = 0.02;}

  if( opt == "Towx"){lab_off = 0.01;}
  if( opt == "Towy"){lab_off = 0.03;}
  if( opt == "SMx" ){lab_off = 0.01;}
  if( opt == "SMy" ){lab_off = 0.015;}
  if( opt == "EBx" ){lab_off = 0.03;}
  if( opt == "EBy" ){lab_off = 0.015;}

  if( opt == "SCx" ){lab_off = 0.01;}
  if( opt == "SCy" ){lab_off = 0.03;}
  if( opt == "Deex"){lab_off = 0.002;}
  if( opt == "Deey"){lab_off = 0.03;}
  if( opt == "EEx" ){lab_off = 0.015;}
  if( opt == "EEy" ){lab_off = 0.02;}

  return lab_off;
}
//............................................
Float_t TEcnaParHistos::AxisLabelSize()
{
 return AxisLabelSize("");
}
Float_t  TEcnaParHistos::AxisLabelSize(const TString opt)
{
  Float_t lab_siz = 0.035;
  if( opt == "SMx"  ){lab_siz = 0.05;}
  if( opt == "SMy"  ){lab_siz = 0.05;}
  if( opt == "EBx"  ){lab_siz = 0.04;}
  if( opt == "EBy"  ){lab_siz = 0.04;}
  return lab_siz;
}
//............................................
Float_t TEcnaParHistos::AxisTickSize()
{
 return AxisTickSize("");
}
Float_t TEcnaParHistos::AxisTickSize(const TString opt)
{
//.......... works only with option "S" in TGaxis
  Float_t tic_siz = 0.;  
  if( opt == "Dee"  ){tic_siz = 0.;}
  if( opt == "Deex" ){tic_siz = 0.01;}
  if( opt == "Deey" ){tic_siz = 0.02;}
  if( opt == "SMx"  ){tic_siz = 0.03;}
  if( opt == "SMy"  ){tic_siz = 0.0075;}
  if( opt == "EEx"  ){tic_siz = 0.06;}
  if( opt == "EEy"  ){tic_siz = 0.005;}

  return tic_siz;
}
//-----------------------------------------------

void TEcnaParHistos::SetViewGraphOffsets(TGraph* graph, const TString HistoType)
{
// Set offsets of labels, title axis, etc... for histo view

  if( (HistoType == "H1Basic") || (HistoType == "Evol") )
    {
      //....................... x axis
      graph->GetXaxis()->SetTitleOffset((Float_t)1.10);
      graph->GetXaxis()->SetTitleSize((Float_t)0.04);

      graph->GetXaxis()->SetLabelOffset((Float_t)0.015);
      graph->GetXaxis()->SetLabelSize((Float_t)0.04);

      graph->GetXaxis()->SetTickLength((Float_t)0.02);
      graph->GetXaxis()->SetNdivisions((Int_t)510);

      //....................... y axis
      graph->GetYaxis()->SetTitleOffset((Float_t)1.25);
      graph->GetYaxis()->SetTitleSize((Float_t)0.04);

      graph->GetYaxis()->SetLabelOffset((Float_t)0.01);
      graph->GetYaxis()->SetLabelSize((Float_t)0.04);

      graph->GetYaxis()->SetTickLength((Float_t)0.03);
      graph->GetYaxis()->SetNdivisions((Int_t)510);
    }
}
//-----------------------------------------------
Float_t TEcnaParHistos::DeeOffsetX(const TString opt_plot, const Int_t& DeeNumber)
{
  Float_t offset = 0;
  if( opt_plot == "Dee" ){offset = 0;}
  if( opt_plot == "EE"  )
    {
      Double_t vertic_empty_strip_1 = 1.;
      if( DeeNumber == 1 ){offset = vertic_empty_strip_1
			     + fEcal->MaxStinHocoInStex()+ 2*fEcal->MaxStinHocoInStex() + 1.;}
      if( DeeNumber == 2 ){offset = vertic_empty_strip_1 + 2*fEcal->MaxStinHocoInStex() + 1.;}
      if( DeeNumber == 3 ){offset = vertic_empty_strip_1 + fEcal->MaxStinHocoInStex();}
      if( DeeNumber == 4 ){offset = vertic_empty_strip_1 + 0.;}
    }
  return offset;
}

//-----------------------------------------------
Float_t TEcnaParHistos::DeeNameOffsetX(const Int_t& DeeNumber)
{
  Float_t offset = 0;

  if( DeeNumber == 1 ){offset = 6.9;}
  if( DeeNumber == 2 ){offset = 0.;}
  if( DeeNumber == 3 ){offset = 7.6;}
  if( DeeNumber == 4 ){offset = 0.5;}
  
  return offset;
}
//-----------------------------------------------
Float_t TEcnaParHistos::DeeNumberOffsetX(const TString opt_plot, const Int_t& DeeNumber)
{
  Float_t offset = 0;

  if( opt_plot == "Dee"  )
    {
      if( DeeNumber == 1 ){offset = 2.5;}
      if( DeeNumber == 2 ){offset = 41.5;}
      if( DeeNumber == 3 ){offset = 2.5;}
      if( DeeNumber == 4 ){offset = 41.5;}
    }
  if( opt_plot == "EE"  )
    {
      if( DeeNumber == 1 ){offset = 0.42;}
      if( DeeNumber == 2 ){offset = 8.15;}
      if( DeeNumber == 3 ){offset = 0.42;}
      if( DeeNumber == 4 ){offset = 8.15;}
    }
  return offset;
}
//===========================================================================
//
//     SetPaveGeneralComment,
//     SetPaveAnalysisRun, SetPaveNbOfEvts,
//     SetPaveEvolNbOfEvtsAna, SetPaveEvolRuns,
//     SetOptionSamePaveBorder
//
//
//                           W A R N I N G
//
//     ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
//
//     WARNING => Paves for plot option "SEVERAL" and "SAMEONE" are treated
//                in TEcnaHistos
//
//     ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
//
//===========================================================================

TPaveText* TEcnaParHistos::SetPaveGeneralComment(const TString comment)
{
// General comment
 
  TPaveText* title_g1;
 
  if( comment != "" )
    {
      char* f_in = new char[fgMaxCar];                           fCnew++;
      
      Double_t pav_gen_xgauche = BoxLeftX("general_comment");
      Double_t pav_gen_xdroite = BoxRightX("general_comment");
      Double_t pav_gen_ybas    = BoxBottomY("general_comment");
      Double_t pav_gen_yhaut   = BoxTopY("general_comment");

      title_g1 =
	new TPaveText(pav_gen_xgauche, pav_gen_ybas,
		      pav_gen_xdroite, pav_gen_yhaut);       fCnewRoot++;

      Option_t* opt_pav = "arc";  title_g1->SetOption(opt_pav);  // <= NE MARCHE PAS
      
      TString tit_gen = comment.Data();
      
      sprintf( f_in, tit_gen);
      title_g1->AddText(f_in);

      //   1 = left adjusted, 2 = vertically centered      
      Int_t    cTextPaveAlign   = 12;    title_g1->SetTextAlign(cTextPaveAlign);
      // 10*10 = 10*(ID10 = Courier New) // 10*4  = 10*(ID4  = Arial)      
      Int_t    cTextPaveFont    = 40;    title_g1->SetTextFont(cTextPaveFont);
      // 0.95 = 95% of the pave size      
      Float_t  cTextPaveSize    = 0.04;  title_g1->SetTextSize(cTextPaveSize);
      Int_t    cTextBorderSize  = 0;     title_g1->SetBorderSize(cTextBorderSize);
      Double_t cTextPaveRadius  = 5.;    title_g1->SetCornerRadius(cTextPaveRadius); // <= NE MARCHE PAS
      // Int_t cTextPaveShadCol = 0; title_g1->SetShadowColor(cTextPaveShadCol);  // <= methode non reconnue   

      delete [] f_in;                                                 fCdelete++;
    }
  else
    {
      title_g1 = new TPaveText( 0, 0, 0, 0);  title_g1=0;  fCnewRoot++;
    }
  return title_g1;
}

TPaveText* TEcnaParHistos::SetPaveAnalysisRun(const TString ana_type,  const Int_t&  nb_of_samples,
					     const Int_t& run_number, const TString run_type,
					     const Int_t& first_evt,  const Int_t&  last_evt, const TString nb_col)
{
// Analysis name + Nb of samples + run number comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Pave Analysis name/run number (bottom_left_box)
  Double_t pav_bot_left_xgauche = BoxLeftX("bottom_left_box");
  Double_t pav_bot_left_xdroite = BoxRightX("bottom_left_box");
  Double_t pav_bot_left_ybas    = BoxBottomY("bottom_left_box");
  Double_t pav_bot_left_yhaut   = BoxTopY("bottom_left_box");

  TPaveText *com_bot_left =
    new TPaveText(pav_bot_left_xgauche, pav_bot_left_ybas,
		  pav_bot_left_xdroite, pav_bot_left_yhaut);  fCnewRoot++;

  // (1 = left adjusted, 2 = centered adjusted, 3 = left adjusted), 2 = vertically centered
  Int_t   cTextPaveAlign   = 12;    com_bot_left->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New) // 10*4  = 10*(ID4  = Arial)
  Int_t   cTextPaveFont    = 100;    com_bot_left->SetTextFont(cTextPaveFont); 
  // 0.95 = 95% of the pave size         
  Float_t cTextPaveSize    = 0.025;  com_bot_left->SetTextSize(cTextPaveSize);
  // Int_t   cTextPaveShadCol = 0;     com_bot_left->SetShadowColor(cTextPaveShadCol);
  Int_t   cTextBorderSize = 1;      com_bot_left->SetBorderSize(cTextBorderSize);

  if( nb_col == "TwoCol")
    {
      cTextPaveSize = 0.035; com_bot_left->SetTextSize(cTextPaveSize);
      sprintf(f_in, "Run:  %d                  Samples:   1 - %d", run_number, nb_of_samples);   
      com_bot_left->AddText(f_in);
      cTextPaveSize = 0.035; com_bot_left->SetTextSize(cTextPaveSize);
      sprintf(f_in, "Type: %-20s", run_type.Data());  
      com_bot_left->AddText(f_in);
      sprintf(f_in, "Analysis: %-10s         Evts range: %d - %d ", ana_type.Data(), first_evt, last_evt);
      com_bot_left->AddText(f_in);   
    }
  else
    {
      cTextPaveSize    = 0.0225;  com_bot_left->SetTextSize(cTextPaveSize);
      sprintf(f_in, "Run:  %d ", run_number);  
      com_bot_left->AddText(f_in);
      sprintf(f_in, "Type: %-20s", run_type.Data());  
      com_bot_left->AddText(f_in);
      sprintf(f_in, "Analysis: %-10s ", ana_type.Data());
      com_bot_left->AddText(f_in);
      Int_t un = 1; 
      sprintf(f_in, "Samples:    %d - %d", un, nb_of_samples);
      com_bot_left->AddText(f_in);
      sprintf(f_in, "Evts range: %d - %d", first_evt, last_evt);
      com_bot_left->AddText(f_in);
    }

  delete [] f_in;                                           fCdelete++;
  
  return com_bot_left;
}

TPaveText* TEcnaParHistos::SetPaveNbOfEvts(const Int_t& nb_of_evts,
					  const TString start_date, const TString stop_date,
					  const TString nb_col)
{
// Number of events box

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Pave number of events (bottom_right_box)

  Double_t pav_bot_right_xgauche = BoxLeftX("bottom_right_box");
  Double_t pav_bot_right_xdroite = BoxRightX("bottom_right_box");
  Double_t pav_bot_right_ybas    = BoxBottomY("bottom_right_box");
  Double_t pav_bot_right_yhaut   = BoxTopY("bottom_right_box");
  TPaveText *com_bot_right =
    new TPaveText(pav_bot_right_xgauche, pav_bot_right_ybas,
		  pav_bot_right_xdroite, pav_bot_right_yhaut);      fCnewRoot++;

  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign   = 12;     com_bot_right->SetTextAlign(cTextPaveAlign);
  // 10*4 = 10*(ID4 = Arial)   // 10*10 = 10*(ID10 = Courier New)          
  Int_t   cTextPaveFont    = 100;    com_bot_right->SetTextFont(cTextPaveFont);         
  // 0.05 = 5% of the pave size        
  Float_t cTextPaveSize    = 0.025;  com_bot_right->SetTextSize(cTextPaveSize);
  //Int_t   cTextPaveShadCol = 0;     com_bot_right->SetShadowColor(cTextPaveShadCol);
  Int_t   cTextBorderSize = 1;       com_bot_right->SetBorderSize(cTextBorderSize);

  if( nb_col == "TwoCol")
    {
      cTextPaveSize = 0.0325; com_bot_right->SetTextSize(cTextPaveSize);
      sprintf(f_in, "First evt: %s              %8d events ", start_date.Data(), nb_of_evts);
      com_bot_right->AddText(f_in);  
      sprintf(f_in, "Last  evt: %s ",  stop_date.Data());
      com_bot_right->AddText(f_in);
    }
  else
    {
      cTextPaveSize = 0.0225; com_bot_right->SetTextSize(cTextPaveSize);
      sprintf(f_in, "%d events", nb_of_evts);
      com_bot_right->AddText(f_in);     
      sprintf(f_in, "First evt: %s ", start_date.Data());
      com_bot_right->AddText(f_in);      
      sprintf(f_in, "Last  evt: %s ",  stop_date.Data());
      com_bot_right->AddText(f_in);
    }
  
  delete [] f_in;                                           fCdelete++;
  
  return com_bot_right;
}

TPaveText* TEcnaParHistos::SetPaveEvolNbOfEvtsAna(const TString ana_type,      const Int_t& nb_of_samples,
						 const Int_t&  first_req_evt, const Int_t& last_req_evt,
						 const TString HistoType)
{
// Analysis name + run number comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  
  //...................... Pave Analysis name/run number (bottom_left_box)
  Double_t pav_bot_left_xgauche = BoxLeftX("bottom_left_box");
  Double_t pav_bot_left_xdroite = BoxRightX("bottom_left_box");
  Double_t pav_bot_left_ybas    = BoxBottomY("bottom_left_box");
  Double_t pav_bot_left_yhaut   = BoxTopY("bottom_left_box");

  if( HistoType == "EvolProj" )
    {
      pav_bot_left_xgauche = BoxLeftX("bottom_left_box_evol");
      pav_bot_left_xdroite = BoxRightX("bottom_left_box_evol");
      pav_bot_left_ybas    = BoxBottomY("bottom_left_box_evol");
      pav_bot_left_yhaut   = BoxTopY("bottom_left_box_evol");
    }

  TPaveText *com_bot_left =
    new TPaveText(pav_bot_left_xgauche, pav_bot_left_ybas,
		  pav_bot_left_xdroite, pav_bot_left_yhaut);  fCnewRoot++;
  
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;    com_bot_left->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)    // 10*4 = 10*(ID4 = Arial)        
  Int_t   cTextPaveFont  = 100;   com_bot_left->SetTextFont(cTextPaveFont);      
  // 0.95 = 95% of the pave size   
  Float_t cTextPaveSize  = 0.03;  com_bot_left->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;    com_bot_left->SetBorderSize(cTextBorderSize);

  TString analysis_name    = ana_type.Data();

  sprintf(f_in, "Analysis:   %s", analysis_name.Data());
  com_bot_left->AddText(f_in);
  sprintf(f_in, "Samples:    1 - %d", nb_of_samples);
  com_bot_left->AddText(f_in);
  sprintf(f_in, "Evts range: %d - %d ", first_req_evt, last_req_evt);
  com_bot_left->AddText(f_in);

  delete [] f_in;                                           fCdelete++;
  
  return com_bot_left;
}

TPaveText* TEcnaParHistos::SetPaveEvolRuns(const Int_t&  start_evol_run, const TString start_evol_date,
					  const Int_t&  stop_evol_run,  const TString stop_evol_date,
					  const TString opt_plot,       const TString HistoType)
{
// First and last run of the list of runs

  char* f_in = new char[fgMaxCar];                           fCnew++;

  //...................... Pave first and last runs (bottom_right_box)
  Double_t pav_border_xgauche = BoxLeftX("bottom_right_box_evol");
  Double_t pav_border_xdroite = BoxRightX("bottom_right_box_evol");
  Double_t pav_border_ybas    = BoxBottomY("bottom_right_box_evol");
  Double_t pav_border_yhaut   = BoxTopY("bottom_right_box_evol");

  if( opt_plot == "SAME n" )
    {
      pav_border_xgauche = BoxLeftX("bottom_right_box_evpr");
      pav_border_xdroite = BoxRightX("bottom_right_box_evpr");
      pav_border_ybas    = BoxBottomY("top_right_box_EB");
      pav_border_yhaut   = BoxTopY("top_right_box_EB");
    }

  TPaveText *pav_evol_runs =
    new TPaveText(pav_border_xgauche, pav_border_ybas,
		  pav_border_xdroite, pav_border_yhaut);      fCnewRoot++;

  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12; pav_evol_runs->SetTextAlign(cTextPaveAlign);
  // 10*4 = 10*(ID4 = Courier helvetica-medium-r-normal = Arial)
  // 10*10 = 10*(ID10 = Courier bold-r-normal = Courier New)          
  Int_t   cTextPaveFont  = 100; pav_evol_runs->SetTextFont(cTextPaveFont);
  // 0.03 = 3% of the pave height           
  Float_t cTextPaveSize  = 0.025;
  if( HistoType == "Evol" || HistoType == "EvolProj"){cTextPaveSize  = 0.0225;}

  pav_evol_runs->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1; pav_evol_runs->SetBorderSize(cTextBorderSize);

  if( !( (HistoType == "Evol"     && opt_plot == "SAME n") ||
	 (HistoType == "EvolProj" && opt_plot == "SAME n") ) )
    {
      sprintf(f_in, "First run: %d", start_evol_run);
      pav_evol_runs->AddText(f_in);
      sprintf(f_in, "(%s) ", start_evol_date.Data());
      pav_evol_runs->AddText(f_in);
      sprintf(f_in, "Last run:  %d", stop_evol_run);
      pav_evol_runs->AddText(f_in);
      sprintf(f_in, "(%s) ", stop_evol_date.Data());
      pav_evol_runs->AddText(f_in);
    }

  if( (HistoType == "Evol"     && opt_plot == "SAME n") ||
      (HistoType == "EvolProj" && opt_plot == "SAME n") )
    {
      sprintf(f_in, "First run: %d (%s) ", start_evol_run, start_evol_date.Data());
      pav_evol_runs->AddText(f_in);
      sprintf(f_in, "Last run:  %d (%s)", stop_evol_run, stop_evol_date.Data());
      pav_evol_runs->AddText(f_in);
    }

  delete [] f_in;                                           fCdelete++;
  
  return pav_evol_runs;
}

TPaveText* TEcnaParHistos::SetOptionSamePaveBorder(const TString chopt, const TString HistoType)
{
// Pave for plots with option SAME and SAME n.
// Gives just the size/border and return the pointer to the Pave. No text.
// The Pave text is drawn in TEcnaHistos.cc
  
  //.................................. DEFAULT OPTION: "several"
  
  Double_t pav_left_xgauche = BoxLeftX("several_plots_box");
  Double_t pav_left_xdroite = BoxRightX("several_plots_box");
  Double_t pav_left_ybas    = BoxBottomY("several_plots_box");
  Double_t pav_left_yhaut   = BoxTopY("several_plots_box");

  if(chopt == "sevevol")
    {    
      pav_left_xgauche = BoxLeftX("several_evol_box");
      pav_left_xdroite = BoxRightX("several_evol_box");
      pav_left_ybas    = BoxBottomY("several_evol_box");
      pav_left_yhaut   = BoxTopY("several_evol_box");

      if( HistoType == "EvolProj" )
	{
	  pav_left_xgauche = BoxLeftX("several_evpr_box");
	  pav_left_xdroite = BoxRightX("several_evpr_box");
	  pav_left_ybas    = BoxBottomY("several_evpr_box");
	  pav_left_yhaut   = BoxTopY("several_evpr_box");
	}
    }

  TPaveText *com_several =
    new TPaveText(pav_left_xgauche, pav_left_ybas,
		  pav_left_xdroite, pav_left_yhaut);  fCnewRoot++;

  return com_several;
}
//.... end of SetOptionSamePaveBorder
//===========================================================================
//
//     SetPaveSM, SetPaveTower, SetPaveTowersXY,
//     SetPaveEBCrystal, SetPaveEBCrystalSample,
//     SetPaveLVRB, SetColorsForNumbers
//
//===========================================================================
TPaveText* TEcnaParHistos::SetPaveStas()
{
  char* f_in = new char[fgMaxCar];                           fCnew++;
  //.................................. DEFAULT OPTION: "standard"   
  Double_t pav_top_left_xgauche = BoxLeftX("top_left_box_SM");
  Double_t pav_top_left_xdroite = BoxRightX("top_left_box_SM");
  Double_t pav_top_left_ybas    = BoxBottomY("top_left_box_SM");
  Double_t pav_top_left_yhaut   = BoxTopY("top_left_box_SM");

  TPaveText *com_top_left =
    new TPaveText(pav_top_left_xgauche, pav_top_left_ybas,
		  pav_top_left_xdroite, pav_top_left_yhaut);  fCnewRoot++;
 
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign  = 12;   com_top_left->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)          
  Int_t   cTextPaveFont   = 100;  com_top_left->SetTextFont(cTextPaveFont);
  // 0.95 = 95% of the pave size         
  Float_t cTextPaveSize   = 0.04; com_top_left->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;    com_top_left->SetBorderSize(cTextBorderSize);

  if ( fFlagSubDet == "EB"){sprintf(f_in, "EB");}
  if ( fFlagSubDet == "EE"){sprintf(f_in, "EE");}

  com_top_left->AddText(f_in);
  
  delete [] f_in;                                           fCdelete++;
  
  return com_top_left;
}

TPaveText* TEcnaParHistos::SetPaveSM(const TString chopt, const Int_t& SM_number,
				    const TString EB_type)
{
// Pave for Super-module plots
  
  char* f_in = new char[fgMaxCar];                           fCnew++;

  //.................................. DEFAULT OPTION: "standard"   
  Double_t pav_top_left_xgauche = BoxLeftX("top_left_box_SM");
  Double_t pav_top_left_xdroite = BoxRightX("top_left_box_SM");
  Double_t pav_top_left_ybas    = BoxBottomY("top_left_box_SM");
  Double_t pav_top_left_yhaut   = BoxTopY("top_left_box_SM");

  if( chopt == "standard" || (chopt == "standGH" && EB_type == "EB-") )
    {  
      pav_top_left_xgauche = BoxLeftX("top_left_box_EB");
      pav_top_left_xdroite = BoxRightX("top_left_box_EB");
      pav_top_left_ybas    = BoxBottomY("top_left_box_EB");
      pav_top_left_yhaut   = BoxTopY("top_left_box_EB");
    }

  if( chopt == "standSM" || chopt == "standStex" || (chopt == "standGH" && EB_type == "EB+") )
    {  
      pav_top_left_xgauche = BoxLeftX("top_left_box_SM");
      pav_top_left_xdroite = BoxRightX("top_left_box_SM");
      pav_top_left_ybas    = BoxBottomY("top_left_box_SM");
      pav_top_left_yhaut   = BoxTopY("top_left_box_SM");
    }

  if( chopt == "several" )
    {    
      pav_top_left_xgauche = BoxLeftX("several_plots_box");
      pav_top_left_xdroite = BoxRightX("several_plots_box");
      pav_top_left_ybas    = BoxBottomY("several_plots_box");
      pav_top_left_yhaut   = BoxTopY("several_plots_box");
    }

  if( chopt == "sevevol" )
    {    
      pav_top_left_xgauche = BoxLeftX("several_evol_box");
      pav_top_left_xdroite = BoxRightX("several_evol_box");
      pav_top_left_ybas    = BoxBottomY("several_evol_box");
      pav_top_left_yhaut   = BoxTopY("several_evol_box");
    }

  TPaveText *com_top_left =
    new TPaveText(pav_top_left_xgauche, pav_top_left_ybas,
		  pav_top_left_xdroite, pav_top_left_yhaut);  fCnewRoot++;
 
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign  = 12;   com_top_left->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)          
  Int_t   cTextPaveFont   = 100;  com_top_left->SetTextFont(cTextPaveFont);
  // 0.95 = 95% of the pave size         
  Float_t cTextPaveSize   = 0.04; com_top_left->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;    com_top_left->SetBorderSize(cTextBorderSize);
 
  if( chopt == "standard" || chopt == "standSM" || chopt == "standStex" || chopt == "standGH")
    {
      Int_t sm_nb = SM_number;
      if( EB_type == "EB+" ){sprintf(f_in, "EB+%d", sm_nb);}
      if( EB_type == "EB-" )
	{sm_nb = -SM_number+fEcal->MaxSMInEB()/2;
	sprintf(f_in, "EB%d (SM%d)", sm_nb, SM_number);}
      com_top_left->AddText(f_in);
    }

  delete [] f_in;                                           fCdelete++;
  
  return com_top_left;
}

TPaveText* TEcnaParHistos::SetPaveTower(const Int_t& SMtower_X)
{
// Tower comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/crystal(channel)/sample (top_right_box)
  Double_t pav_top_mid_xgauche = BoxLeftX("top_mid_box_EB");
  Double_t pav_top_mid_xdroite = BoxRightX("top_mid_box_EB");
  Double_t pav_top_mid_ybas    = BoxBottomY("top_mid_box_EB");
  Double_t pav_top_mid_yhaut   = BoxTopY("top_mid_box_EB");
  TPaveText *com_top_mid =
    new TPaveText(pav_top_mid_xgauche, pav_top_mid_ybas,
		  pav_top_mid_xdroite, pav_top_mid_yhaut);  fCnewRoot++;

  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;    com_top_mid->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)          
  Int_t   cTextPaveFont  = 100;   com_top_mid->SetTextFont(cTextPaveFont);   
  // 0.95 = 95% of the pave size     
  Float_t cTextPaveSize  = 0.03;  com_top_mid->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;    com_top_mid->SetBorderSize(cTextBorderSize);
	  
  sprintf(f_in, " Tower: %d ", SMtower_X);
  com_top_mid->AddText(f_in);
  
  delete [] f_in;                                           fCdelete++;

  return com_top_mid;
}

TPaveText* TEcnaParHistos::SetPaveTowersXY(const Int_t& SMtower_X, const Int_t& SMtower_Y)
{
// Towers X and Y for (TowEcha,TowEcha) cov or cor matrix

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/TowEcha(channel)/sample (top_right_box)
  Double_t pav_top_mid_xgauche = BoxLeftX("top_mid_box_EB");
  Double_t pav_top_mid_xdroite = BoxRightX("top_mid_box_EB");
  Double_t pav_top_mid_ybas    = BoxBottomY("top_mid_box_EB");
  Double_t pav_top_mid_yhaut   = BoxTopY("top_mid_box_EB");
  TPaveText *com_top_mid =
    new TPaveText(pav_top_mid_xgauche, pav_top_mid_ybas,
		  pav_top_mid_xdroite, pav_top_mid_yhaut);  fCnewRoot++;
  
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;   com_top_mid->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)          
  Int_t   cTextPaveFont  = 100;  com_top_mid->SetTextFont(cTextPaveFont);
  // 0.95 = 95% of the pave size            
  Float_t cTextPaveSize  = 0.03; com_top_mid->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;   com_top_mid->SetBorderSize(cTextBorderSize);
		  
  sprintf(f_in, " Tower X: %d ", SMtower_X);
  com_top_mid->AddText(f_in);
  sprintf(f_in, " Tower Y: %d ", SMtower_Y);
  com_top_mid->AddText(f_in);  

  delete [] f_in;                                           fCdelete++;

  return com_top_mid;
}

TPaveText* TEcnaParHistos::SetPaveCrystal(const Int_t& StexCrys, const Int_t& StexStinA, const Int_t& StinEcha)
{
// Tower + StinEcha comment. StexCrys range: [1,max]

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/StinEcha(channel)/sample (top_right_box)

  Double_t pav_top_right_xgauche = BoxLeftX("top_right_box_EB");
  Double_t pav_top_right_xdroite = BoxRightX("top_right_box_EB");
  Double_t pav_top_right_ybas    = BoxBottomY("top_right_box_EB");
  Double_t pav_top_right_yhaut   = BoxTopY("top_right_box_EB");
  if( fFlagSubDet == "EE" )
    {
      pav_top_right_xgauche = BoxLeftX("top_right_box_EE");
      pav_top_right_xdroite = BoxRightX("top_right_box_EE");
      pav_top_right_ybas    = BoxBottomY("top_right_box_EE");
      pav_top_right_yhaut   = BoxTopY("top_right_box_EE");
    }

  TPaveText *com_top_right =
    new TPaveText(pav_top_right_xgauche, pav_top_right_ybas,
		  pav_top_right_xdroite, pav_top_right_yhaut);  fCnewRoot++;

  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;   com_top_right->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)           
  Int_t   cTextPaveFont  = 100;  com_top_right->SetTextFont(cTextPaveFont);   
  // 0.95 = 95% of the pave size      
  Float_t cTextPaveSize  = 0.03; com_top_right->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;   com_top_right->SetBorderSize(cTextBorderSize);

  if( fFlagSubDet == "EB" )
    {
      sprintf(f_in, "Channel: %d ", StinEcha);
      com_top_right->AddText(f_in); 
      sprintf(f_in, "Crystal in SM: %d ", StexCrys);
      com_top_right->AddText(f_in);
    }

  if( fFlagSubDet == "EE" )
    {
      sprintf(f_in, "Crystal in SC: %d ", StinEcha+1);     // EE => range = [1,25]
      com_top_right->AddText(f_in);
      Int_t IX_Dee_crys = StexCrys/fEcal->MaxCrysIYInDee() + 1;
      Int_t IY_Dee_crys = StexCrys%fEcal->MaxCrysIYInDee();
      if( IY_Dee_crys == 0 ){IX_Dee_crys--; IY_Dee_crys = fEcal->MaxCrysIYInDee();}     
      sprintf(f_in, "(IX,IY)[Xtal]=(%d,%d)", IX_Dee_crys, IY_Dee_crys);
      com_top_right->AddText(f_in);
    }

  delete [] f_in;                                           fCdelete++;

  return com_top_right;	
}

TPaveText* TEcnaParHistos::SetPaveCrystalSample(const Int_t& StexCrys, const Int_t& StexStinA,
					       const Int_t& StinEcha, const Int_t& iSample)
{
// Tower + StinEcha + sample comment

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave tower/StinEcha(channel)/sample (top_right_box)
  Double_t pav_top_right_xgauche = BoxLeftX("top_right_box_EB");
  Double_t pav_top_right_xdroite = BoxRightX("top_right_box_EB");
  Double_t pav_top_right_ybas    = BoxBottomY("top_right_box_EB");
  Double_t pav_top_right_yhaut   = BoxTopY("top_right_box_EB");
  if( fFlagSubDet == "EE" )
    {
      pav_top_right_xgauche = BoxLeftX("top_right_box_EE");
      pav_top_right_xdroite = BoxRightX("top_right_box_EE");
      pav_top_right_ybas    = BoxBottomY("top_right_box_EE");
      pav_top_right_yhaut   = BoxTopY("top_right_box_EE");
    }

  TPaveText *com_top_right =
    new TPaveText(pav_top_right_xgauche, pav_top_right_ybas,
		  pav_top_right_xdroite, pav_top_right_yhaut);  fCnewRoot++;
  
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;   com_top_right->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)            
  Int_t   cTextPaveFont  = 100;  com_top_right->SetTextFont(cTextPaveFont);
  // 0.95 = 95% of the pave size    
  Float_t cTextPaveSize  = 0.03; com_top_right->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;   com_top_right->SetBorderSize(cTextBorderSize);


  if( fFlagSubDet == "EB" )
    {
      sprintf(f_in, " Channel: %d ", StinEcha);
      com_top_right->AddText(f_in); 
      sprintf(f_in, " Crystal in SM: %d ", StexCrys);
      com_top_right->AddText(f_in);
    }

  if( fFlagSubDet == "EE" )
    {
      sprintf(f_in, " Channel: %d ", StinEcha+1);    // EE => range = [1,25]
      com_top_right->AddText(f_in); 
      Int_t IX_Dee_crys = StexCrys/fEcal->MaxCrysIYInDee() + 1;
      Int_t IY_Dee_crys = StexCrys%fEcal->MaxCrysIYInDee();
      if( IY_Dee_crys == 0 ){IX_Dee_crys--; IY_Dee_crys = fEcal->MaxCrysIYInDee();}
      sprintf(f_in, "(IX,IY)[Xtal]=(%d,%d)", IX_Dee_crys, IY_Dee_crys);
      com_top_right->AddText(f_in);
    }

  sprintf(f_in, " Sample: %d ", iSample+1);
  com_top_right->AddText(f_in);

  delete [] f_in;                                           fCdelete++;

  return com_top_right;
}

TPaveText* TEcnaParHistos::SetPaveLVRB(const Int_t& SMNumber, const Int_t& SMtower)
{
// LVRB at the top or at the bottom comment

  //....................... GRAND pave "LVRB"
  Double_t pav_bot_xgauche = BoxLeftX("bottom_left_box");
  Double_t pav_bot_xdroite = BoxRightX("bottom_right_box");
  Double_t pav_bot_ybas    = BoxBottomY("bottom_left_box");
  Double_t pav_bot_yhaut   = BoxTopY("bottom_left_box");
  TPaveText *com_bot_mid =
    new TPaveText(pav_bot_xgauche, pav_bot_ybas,
		  pav_bot_xdroite, pav_bot_yhaut);    fCnewRoot++;

  Color_t couleur_noir       = ColorDefinition("noir");
  Color_t couleur_rouge      = SetColorsForNumbers("lvrb_top");
  Color_t couleur_bleu_fonce = SetColorsForNumbers("lvrb_bottom");

  if(fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+")
    {
      TText *t1 = com_bot_mid->AddText("   <= IP (#eta = 0)       (#eta = +85) =>   ");
      t1->SetTextColor(couleur_noir);
    }

  if(fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-")
    {
      TText *t2 = com_bot_mid->AddText("   <= (#eta = -85)       IP (#eta = 0) =>   ");
      t2->SetTextColor(couleur_noir);
    }

  if(fEcalNumbering->GetTowerLvrbType(SMtower) == "top")
    {
      TText *t3 = 0;
      if(fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+")
	{t3 =  com_bot_mid->AddText("       <=== LVRB       ");}
      if(fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-")
	{t3 =  com_bot_mid->AddText("       LVRB ===>       ");}
      t3->SetTextColor(couleur_rouge);
    }
  
  if(fEcalNumbering->GetTowerLvrbType(SMtower) == "bottom")
    {
      TText *t4 = 0;
      if(fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB+")
	{t4 = com_bot_mid->AddText("        LVRB ===>       ");}
      if(fEcalNumbering->GetSMHalfBarrel(SMNumber) == "EB-")
	{t4 = com_bot_mid->AddText("        <=== LVRB       ");}
      t4->SetTextColor(couleur_bleu_fonce);
    }
  return com_bot_mid;

}//  end of SetPaveLVRB

Color_t TEcnaParHistos::SetColorsForNumbers(const TString chtype_number)
{
 //Set color of the numbers for SuperModule- or Tower-  numbering plots

  Color_t couleur = ColorDefinition("noir");        // default = "noir"

  if ( chtype_number == "crystal"     ){couleur = ColorDefinition("noir");}
  if ( chtype_number == "lvrb_top"    ){couleur = ColorDefinition("rouge");}
  if ( chtype_number == "lvrb_bottom" ){couleur = ColorDefinition("bleu_fonce");}

  return couleur;
}

//===========================================================================
//
//     SetPaveDee, SetPaveSC, SetPaveSCsXY,
//     SetPaveEECrystal
//
//===========================================================================

TPaveText* TEcnaParHistos::SetPaveDee(const TString chopt,   const Int_t&  DeeNumber,
				     const TString DeeType)
{
// Dee pav. Called only once.
  
  char* f_in = new char[fgMaxCar];                           fCnew++;

  //.................................. DEFAULT OPTION: "standard"   
  Double_t pav_top_left_xgauche = BoxLeftX("top_left_box_EE");
  Double_t pav_top_left_xdroite = BoxRightX("top_left_box_EE");
  Double_t pav_top_left_ybas    = BoxBottomY("top_left_box_EE");
  Double_t pav_top_left_yhaut   = BoxTopY("top_left_box_EE");
  
  if(chopt == "standard" || chopt == "standGH")
    {  
      pav_top_left_xgauche = BoxLeftX("top_left_box_EE");
      pav_top_left_xdroite = BoxRightX("top_left_box_EE");
      pav_top_left_ybas    = BoxBottomY("top_left_box_EE");
      pav_top_left_yhaut   = BoxTopY("top_left_box_EE");
    }

  if( chopt == "standDee" || chopt == "standStex" )
    {  
      pav_top_left_xgauche = BoxLeftX("top_left_box_Dee");
      pav_top_left_xdroite = BoxRightX("top_left_box_Dee");
      pav_top_left_ybas    = BoxBottomY("top_left_box_Dee");
      pav_top_left_yhaut   = BoxTopY("top_left_box_Dee");
    }

  if(chopt == "several")
    {    
      pav_top_left_xgauche = BoxLeftX("several_plots_box");
      pav_top_left_xdroite = BoxRightX("several_plots_box");
      pav_top_left_ybas    = BoxBottomY("several_plots_box");
      pav_top_left_yhaut   = BoxTopY("several_plots_box");
    }
  if(chopt == "sevevol")
    {    
      pav_top_left_xgauche = BoxLeftX("several_evol_box");
      pav_top_left_xdroite = BoxRightX("several_evol_box");
      pav_top_left_ybas    = BoxBottomY("several_evol_box");
      pav_top_left_yhaut   = BoxTopY("several_evol_box");
    }
  
  TPaveText *com_top_left =
    new TPaveText(pav_top_left_xgauche, pav_top_left_ybas,
		  pav_top_left_xdroite, pav_top_left_yhaut);  fCnewRoot++;
  
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;    com_top_left->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)            
  Int_t   cTextPaveFont  = 100;   com_top_left->SetTextFont(cTextPaveFont);
  // 0.95 = 95% of the pave size            
  Float_t cTextPaveSize  = 0.04;  com_top_left->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;    com_top_left->SetBorderSize(cTextBorderSize);
  
  if( chopt == "standard" )
    {
      sprintf(f_in, " Dee: %d", DeeNumber);
      com_top_left->AddText(f_in);
      sprintf(f_in, " (%s)", DeeType.Data());
      com_top_left->AddText(f_in);
    }
  
  if( chopt == "standDee" || chopt == "standStex" || chopt == "standGH" )
    {
      sprintf(f_in, " Dee: %d  (%s) ", DeeNumber, DeeType.Data());
      com_top_left->AddText(f_in);
    }
  
  delete [] f_in;                                           fCdelete++;
  
  return com_top_left;
}

TPaveText* TEcnaParHistos::SetPaveSC(const Int_t& DeeSC_X, const Int_t& DeeNumber)
{
// SC comment
  
  char* f_in = new char[fgMaxCar];                     fCnew++;
  //...................... Pave SC/crystal(channel)/sample (top_right_box)
  Double_t pav_top_mid_xgauche = BoxLeftX("top_mid_box_EE");
  Double_t pav_top_mid_xdroite = BoxRightX("top_mid_box_EE");
  Double_t pav_top_mid_ybas    = BoxBottomY("top_mid_box_EE");
  Double_t pav_top_mid_yhaut   = BoxTopY("top_mid_box_EE");
  TPaveText *com_top_mid =
    new TPaveText(pav_top_mid_xgauche, pav_top_mid_ybas,
		  pav_top_mid_xdroite, pav_top_mid_yhaut);  fCnewRoot++;
  
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;    com_top_mid->SetTextAlign(cTextPaveAlign);        
  // 10*10 = 10*(ID10 = Courier New)
  Int_t   cTextPaveFont  = 100;   com_top_mid->SetTextFont(cTextPaveFont);     
  // 0.95 = 95% of the pave size
  Float_t cTextPaveSize  = 0.03;  com_top_mid->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;    com_top_mid->SetBorderSize(cTextBorderSize);

  sprintf(f_in, "Sector: S%d, SC: %d",
	  fEcalNumbering->GetDSFrom1DeeSCEcna(DeeNumber, DeeSC_X),
	  fEcalNumbering->GetDSSCFrom1DeeSCEcna(DeeNumber, DeeSC_X));
  com_top_mid->AddText(f_in);
  sprintf(f_in, "SC for const.: %d",
	  fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(DeeNumber,DeeSC_X));
  com_top_mid->AddText(f_in); 
  sprintf(f_in, "Quadrant: %s",
    fEcalNumbering->GetSCQuadFrom1DeeSCEcna(DeeSC_X).Data());
   com_top_mid->AddText(f_in); 
  delete [] f_in;                                     fCdelete++;

  return com_top_mid;
}

TPaveText* TEcnaParHistos::SetPaveSCsXY(const Int_t& DeeSC_X, const Int_t& DeeSC_Y)
{
// SCs X and Y for (SCEcha,SCEcha) cov or cor matrix

  char* f_in = new char[fgMaxCar];                           fCnew++;
  //...................... Pave SC/SCEcha(channel)/sample (top_right_box)
  Double_t pav_top_mid_xgauche = BoxLeftX("top_mid_box_EE");
  Double_t pav_top_mid_xdroite = BoxRightX("top_mid_box_EE");
  Double_t pav_top_mid_ybas    = BoxBottomY("top_mid_box_EE");
  Double_t pav_top_mid_yhaut   = BoxTopY("top_mid_box_EE");
  TPaveText *com_top_mid =
    new TPaveText(pav_top_mid_xgauche, pav_top_mid_ybas,
		  pav_top_mid_xdroite, pav_top_mid_yhaut);  fCnewRoot++;
  
  // 1 = left adjusted, 2 = vertically centered
  Int_t   cTextPaveAlign = 12;   com_top_mid->SetTextAlign(cTextPaveAlign);
  // 10*10 = 10*(ID10 = Courier New)
  Int_t   cTextPaveFont  = 100;  com_top_mid->SetTextFont(cTextPaveFont);
  // 0.95 = 95% of the pave size
  Float_t cTextPaveSize  = 0.03; com_top_mid->SetTextSize(cTextPaveSize);
  Int_t   cTextBorderSize = 1;   com_top_mid->SetBorderSize(cTextBorderSize);

  sprintf(f_in, "SC X: %d", DeeSC_X);
  com_top_mid->AddText(f_in);
  sprintf(f_in, "SC Y: %d", DeeSC_Y);
  com_top_mid->AddText(f_in);  

  delete [] f_in;                                           fCdelete++;

  return com_top_mid;
}

TPaveText* TEcnaParHistos::SetPaveCxyz(const Int_t& DeeNumber)
{
// Cxyz at the top or at the bottom comment

  //....................... GRAND pave "Cxyz"
  Double_t pav_bot_xgauche = BoxRightX("bottom_mid_box");
  Double_t pav_bot_xdroite = BoxLeftX("bottom_mid_box");
  Double_t pav_bot_ybas    = BoxBottomY("bottom_left_box");
  Double_t pav_bot_yhaut   = BoxTopY("bottom_left_box");
  TPaveText *com_bot_mid =
    new TPaveText(pav_bot_xgauche, pav_bot_ybas,
		  pav_bot_xdroite, pav_bot_yhaut);    fCnewRoot++;

  //com_bot_mid->SetOption("arc");
  //com_bot_mid->SetCornerRadius((Double_t)0.5);
  //Int_t   cTextPaveFont  = 100;  com_bot_mid->SetTextFont(cTextPaveFont);

  com_bot_mid->SetTextSize(0.0325);
  //........................................... (SetPaveCxyz)
  if( DeeNumber == 1 )
    {
      com_bot_mid->AddText("        y ");
      com_bot_mid->AddText("        #uparrow ");
      com_bot_mid->AddText(" (Near)    x #leftarrow #otimes z     (Far) " );
    }
  if( DeeNumber == 2 )
    {
      com_bot_mid->AddText("        y ");
      com_bot_mid->AddText("        #uparrow ");
      com_bot_mid->AddText(" (Near)    x #leftarrow #otimes z     (Far) " );
    }
  if( DeeNumber == 3 )
    {
      com_bot_mid->AddText(" y           ");
      com_bot_mid->AddText(" #uparrow          ");
      com_bot_mid->AddText(" (Far)      z (#bullet) #rightarrow x     (Near) " );
    }
  if( DeeNumber == 4 )
    {
      com_bot_mid->AddText(" y           ");
      com_bot_mid->AddText(" #uparrow          ");
      com_bot_mid->AddText(" (Far)      z (#bullet) #rightarrow x     (Near) " );
    }

  // com_bot_mid->AddText(" (from IP) " );
  return com_bot_mid;
} 
//===========================================================================
//
//            SetPaveStex,  SetPaveStin
//
//===========================================================================

TPaveText* TEcnaParHistos::SetPaveStex(const TString chopt, const Int_t& StexNumber)
{
  TPaveText* pav_text = 0;

  if( StexNumber > 0 )
    {
      if ( fFlagSubDet == "EB" )
	{
	  TString EB_type = fEcalNumbering->GetSMHalfBarrel(StexNumber);
	  pav_text = SetPaveSM(chopt, StexNumber, EB_type);
	}
      if ( fFlagSubDet == "EE" )
	{
	  TString dee_type = fEcalNumbering->GetEEDeeType(StexNumber);
	  pav_text = SetPaveDee(chopt, StexNumber, dee_type);
	}
    }
  if( StexNumber == 0 ){pav_text = SetPaveStas();}

  return pav_text;
}

TPaveText* TEcnaParHistos::SetPaveStin(const Int_t& StinNumber, const Int_t& StexNumber)
{
  TPaveText* pav_text = 0;
  if ( fFlagSubDet == "EB"){pav_text = SetPaveTower(StinNumber);}
  if ( fFlagSubDet == "EE"){pav_text = SetPaveSC(StinNumber, StexNumber);}
  return pav_text;
}

TPaveText* TEcnaParHistos::SetPaveStinsXY(const Int_t& StexStin_X, const Int_t& StexStin_Y) 
{
  TPaveText* pav_text = 0;
  if ( fFlagSubDet == "EB"){pav_text = SetPaveTowersXY(StexStin_X, StexStin_Y);}
  if ( fFlagSubDet == "EE"){pav_text = SetPaveSCsXY(StexStin_X, StexStin_Y);}
  return pav_text;
}

//===========================================================================
//
//               GetHistoType, GetQuantityName
//
//===========================================================================

TString TEcnaParHistos::GetHistoType(const TString HistoCode)
{
// Type of the quantity as a function of the quantity code
// Histo type <=> Variable in X coordinate

  TString HistoType;
  Int_t MaxCar = fgMaxCar;
  HistoType.Resize(MaxCar);
  HistoType = "(no quantity type info)";

  //.(1a)............................ Global
  if ( HistoCode == "D_NOE_ChNb" || HistoCode == "D_Ped_ChNb" ||
       HistoCode == "D_TNo_ChNb" || HistoCode == "D_MCs_ChNb" ||
       HistoCode == "D_LFN_ChNb" || HistoCode == "D_HFN_ChNb" || 
       HistoCode == "D_SCs_ChNb" )
    {HistoType = "Global";}   // X coordinate = Xtals for SM or Dee
                              //                Tower for EB
                              //                SC    for EE
  //.(1b)............................ Proj  
  if ( HistoCode == "D_NOE_ChDs" || HistoCode == "D_Ped_ChDs" ||
       HistoCode == "D_TNo_ChDs" || HistoCode == "D_MCs_ChDs" ||
       HistoCode == "D_LFN_ChDs" || HistoCode == "D_HFN_ChDs" || 
       HistoCode == "D_SCs_ChDs" )
    {HistoType = "Proj";}    // X coordinate = bin number
  
  //.(2)............................ H1Basic
  if ( HistoCode == "D_MSp_Samp" || HistoCode == "D_SSp_Samp" )
    {HistoType = "H1Basic";}  // X coordinate = sample number

  //.(3a)............................ SampGlobal
  if ( HistoCode == "D_Adc_EvNb" ){HistoType = "SampGlobal";}  // X coordinate = event number
  //.(3b)............................ SampProj
  if ( HistoCode == "D_Adc_EvDs"  ){HistoType = "SampProj";}    // X coordinate = bin number

  //.(4a)............................ Evol
  if ( HistoCode == "H_Ped_Date" || HistoCode == "H_TNo_Date" || HistoCode == "H_MCs_Date" ||
       HistoCode == "H_LFN_Date" || HistoCode == "H_HFN_Date" || HistoCode == "H_SCs_Date" )
    {HistoType = "Evol";}   // X coordinate = date

  //.(4b)............................ EvolProj
  if ( HistoCode == "H_Ped_RuDs" || HistoCode == "H_TNo_RuDs" || HistoCode == "H_MCs_RuDs" ||
       HistoCode == "H_LFN_RuDs" || HistoCode == "H_HFN_RuDs" || HistoCode == "H_SCs_RuDs" )
    {HistoType = "EvolProj";}   // X coordinate = bin number

  return HistoType;
}// end of GetHistoType(...)

TString TEcnaParHistos::GetXVarHisto(const TString HistoCode, const TString SubDet, const Int_t& StexNumber)
{
// Type of the quantity as a function of the quantity code
// Histo type <=> Variable in X coordinate

  TString HistoType = GetHistoType(HistoCode.Data());

  TString XVarHisto;
  Int_t MaxCar = fgMaxCar;
  XVarHisto.Resize(MaxCar);
  XVarHisto = "(no X variable info)";

  if( HistoType == "Global" )
    {
      if( StexNumber == 0 && SubDet == "EB" ){XVarHisto = "Tower number";}
      if( StexNumber == 0 && SubDet == "EE" ){XVarHisto = "SC number";}
      if( StexNumber != 0){XVarHisto = "Xtal number";}
    }
  if( HistoType == "Proj" ){XVarHisto = "Bin number";}

  if( HistoType == "H1Basic"){XVarHisto = "Sample number";}

  if( HistoType == "SampGlobal" ){XVarHisto = "Event number";}

  if( HistoType == "SampProj" ){XVarHisto = "ADC count";}

  if( HistoType == "Evol" ){XVarHisto = "Date";}

  if( HistoType == "EvolProj" ){XVarHisto = "Bin number";}

  return XVarHisto;

}// end of GetXVarHisto(...)

TString TEcnaParHistos::GetYVarHisto(const TString HistoCode, const TString SubDet, const Int_t& StexNumber)
{
// Type of the quantity as a function of the quantity code
// Histo type <=> Variable in Y coordinate

  TString HistoType = GetHistoType(HistoCode.Data());

  TString YVarHisto;
  Int_t MaxCar = fgMaxCar;
  YVarHisto.Resize(MaxCar);
  YVarHisto = "(no Y variable info)";

  if( HistoType == "Global" )
    {
      if( HistoCode == "D_NOE_ChNb" ){YVarHisto = "Nb of events";}
      if( HistoCode == "D_Ped_ChNb" ){YVarHisto = "Pedestal";}
      if( HistoCode == "D_TNo_ChNb" ){YVarHisto = "Total Noise";}
      if( HistoCode == "D_MCs_ChNb" ){YVarHisto = "Mean Cor(s,s')";}
      if( HistoCode == "D_LFN_ChNb" ){YVarHisto = "Low Freq. Noise";}
      if( HistoCode == "D_HFN_ChNb" ){YVarHisto = "High Freq. Noise";}
      if( HistoCode == "D_SCs_ChNb" ){YVarHisto = "Sigma Cor(s,s')";}
    }

  if( HistoType == "Proj" )
    {
      if( StexNumber == 0 && SubDet == "EB" ){YVarHisto = "Number of towers";}
      if( StexNumber == 0 && SubDet == "EE" ){YVarHisto = "Number of SCs";}
      if( StexNumber != 0){YVarHisto = "Number of Xtals";}
    }

  if( HistoType == "H1Basic"){YVarHisto = "ADC count";}

  if( HistoType == "SampGlobal" ){YVarHisto = "ADC count";}

  if( HistoType == "SampProj" ){YVarHisto = "Number of events";}

  if( HistoType == "EvolProj" ){YVarHisto = "Number of runs";}

  if( HistoType == "Evol" )
    {
      if( HistoCode == "H_Ped_Date" ){YVarHisto = "Pedestal";}
      if( HistoCode == "H_TNo_Date" ){YVarHisto = "Total Noise";}
      if( HistoCode == "H_MCs_Date" ){YVarHisto = "Mean Cor(s,s')";}
      if( HistoCode == "H_LFN_Date" ){YVarHisto = "Low Freq. Noise";}
      if( HistoCode == "H_HFN_Date" ){YVarHisto = "High Freq. Noise";}
      if( HistoCode == "H_SCs_Date" ){YVarHisto = "Sigma Cor(s,s')";}
    }

//  cout << endl << "*TEcnaParHistos::GetYVarHisto(...)> HistoType = " << HistoType
//       << ", HistoCode = " << HistoCode
//       << ", StexNumber = " << StexNumber
//       << ", YVarHisto = " << YVarHisto << endl;

  return YVarHisto;

}// end of GetYVarHisto(...)

TString TEcnaParHistos::GetQuantityName(const TString chqcode)
{
// Name of the quantity as a function of the histo code

  TString chqname;
  Int_t MaxCar = fgMaxCar;
  chqname.Resize(MaxCar);
  chqname = "(no quantity name info)";

  if(chqcode == "D_NOE_ChNb"){chqname = "Number of events";}
  if(chqcode == "D_NOE_ChDs"){chqname = "Nb of events";}
  if(chqcode == "D_Ped_ChNb"){chqname = "Pedestal";}
  if(chqcode == "D_Ped_ChDs"){chqname = "Pedestal";}
  if(chqcode == "D_TNo_ChNb"){chqname = "Total Noise";}
  if(chqcode == "D_TNo_ChDs"){chqname = "Total Noise";}
  if(chqcode == "D_MCs_ChNb"){chqname = "Mean Cor(s,s')";}
  if(chqcode == "D_MCs_ChDs"){chqname = "Mean Cor(s,s')";}
  if(chqcode == "D_LFN_ChNb"){chqname = "Low Freq. Noise";}
  if(chqcode == "D_LFN_ChDs"){chqname = "Low Freq. Noise";}
  if(chqcode == "D_HFN_ChNb"){chqname = "High Freq. Noise";}
  if(chqcode == "D_HFN_ChDs"){chqname = "High Freq. Noise";}
  if(chqcode == "D_SCs_ChNb"){chqname = "Sigma Cor(s,s')";}
  if(chqcode == "D_SCs_ChDs"){chqname = "Sigma Cor(s,s')";}
  if(chqcode == "D_MSp_Samp"){chqname = "Sample means";}
  if(chqcode == "D_SSp_Samp"){chqname = "Sample sigmas";}
  if(chqcode == "D_Adc_EvDs"){chqname = "Sample ADC distribution";}
  if(chqcode == "D_Adc_EvNb"){chqname = "Sample ADC";}
  if(chqcode == "H_Ped_Date"){chqname = "Pedestal";}
  if(chqcode == "H_TNo_Date"){chqname = "Total Noise";}
  if(chqcode == "H_MCs_Date"){chqname = "Mean Cor(s,s')";}
  if(chqcode == "H_LFN_Date"){chqname = "Low Freq. Noise";}
  if(chqcode == "H_HFN_Date"){chqname = "High Freq. Noise";}
  if(chqcode == "H_SCs_Date"){chqname = "Sigma Cor(s,s')";}
  if(chqcode == "H_Ped_RuDs"){chqname = "Pedestal";}
  if(chqcode == "H_TNo_RuDs"){chqname = "Total Noise";}
  if(chqcode == "H_MCs_RuDs"){chqname = "Mean Cor(s,s')";}
  if(chqcode == "H_LFN_RuDs"){chqname = "Low Freq. Noise";}
  if(chqcode == "H_HFN_RuDs"){chqname = "High Freq. Noise";}
  if(chqcode == "H_SCs_RuDs"){chqname = "Sigma Cor(s,s')";}
  return chqname;
}

Double_t TEcnaParHistos::GetYminDefaultValue(const TString chqcode)
{
  Double_t vmin = (Double_t)0.;

  if(chqcode == "H2LFccMosMatrix"){vmin = (Double_t)-1.;}
  if(chqcode == "H2HFccMosMatrix"){vmin = (Double_t)-1.;}
  if(chqcode == "H2CorccInStins" ){vmin = (Double_t)-1.;}

  if(chqcode == "D_NOE_ChNb"){vmin = (Double_t)0.;}
  if(chqcode == "D_Ped_ChNb"){vmin = (Double_t)0.;}
  if(chqcode == "D_TNo_ChNb"){vmin = (Double_t)0.;}
  if(chqcode == "D_MCs_ChNb"){vmin = (Double_t)-1.;}
  if(chqcode == "D_LFN_ChNb"){vmin = (Double_t)0.;}
  if(chqcode == "D_HFN_ChNb"){vmin = (Double_t)0.;}
  if(chqcode == "D_SCs_ChNb"){vmin = (Double_t)-1.;}

  if(chqcode == "D_NOE_ChDs"){vmin = (Double_t)0.1;}
  if(chqcode == "D_Ped_ChDs"){vmin = (Double_t)0.1;}
  if(chqcode == "D_TNo_ChDs"){vmin = (Double_t)0.1;}
  if(chqcode == "D_MCs_ChDs"){vmin = (Double_t)0.1;}
  if(chqcode == "D_LFN_ChDs"){vmin = (Double_t)0.1;}
  if(chqcode == "D_HFN_ChDs"){vmin = (Double_t)0.1;}
  if(chqcode == "D_SCs_ChDs"){vmin = (Double_t)0.1;}

  if(chqcode == "D_MSp_Samp"){vmin = (Double_t)0.;}
  if(chqcode == "D_SSp_Samp"){vmin = (Double_t)0.;}

  if(chqcode == "D_Adc_EvNb"){vmin = (Double_t)0.;}
  if(chqcode == "D_Adc_EvDs"){vmin = (Double_t)0.;}

  if(chqcode == "H_Ped_Date"){vmin = (Double_t)0.;}
  if(chqcode == "H_TNo_Date"){vmin = (Double_t)0.;}
  if(chqcode == "H_MCs_Date"){vmin = (Double_t)-1.;}
  if(chqcode == "H_LFN_Date"){vmin = (Double_t)0.;}
  if(chqcode == "H_HFN_Date"){vmin = (Double_t)0.;}
  if(chqcode == "H_SCs_Date"){vmin = (Double_t)0.;}

  if(chqcode == "H_Ped_RuDs"){vmin = (Double_t)0.1;}
  if(chqcode == "H_TNo_RuDs"){vmin = (Double_t)0.1;}
  if(chqcode == "H_MCs_RuDs"){vmin = (Double_t)0.1;}
  if(chqcode == "H_LFN_RuDs"){vmin = (Double_t)0.1;}
  if(chqcode == "H_HFN_RuDs"){vmin = (Double_t)0.1;}
  if(chqcode == "H_SCs_RuDs"){vmin = (Double_t)0.1;}

  return vmin;
}

Double_t TEcnaParHistos::GetYmaxDefaultValue(const TString chqcode)
{
  Double_t vmax = (Double_t)0.;

  if( fFlagSubDet == "EB" )
    {
      if(chqcode == "H2LFccMosMatrix" ){vmax = (Double_t)1.;}
      if(chqcode == "H2HFccMosMatrix" ){vmax = (Double_t)1.;}
      if(chqcode == "H2CorccInStins"  ){vmax = (Double_t)1.;}

      if(chqcode == "D_NOE_ChNb"){vmax = (Double_t)500.;}
      if(chqcode == "D_Ped_ChNb"){vmax = (Double_t)500.;}
      if(chqcode == "D_TNo_ChNb"){vmax = (Double_t)2.5;}
      if(chqcode == "D_LFN_ChNb"){vmax = (Double_t)2.5;}
      if(chqcode == "D_HFN_ChNb"){vmax = (Double_t)2.5;}
      if(chqcode == "D_MCs_ChNb"){vmax = (Double_t)1.;}
      if(chqcode == "D_SCs_ChNb"){vmax = (Double_t)1.5;}
      
      if(chqcode == "D_NOE_ChDs"){vmax = (Double_t)5000.;}
      if(chqcode == "D_Ped_ChDs"){vmax = (Double_t)5000.;}
      if(chqcode == "D_TNo_ChDs"){vmax = (Double_t)5000.;}
      if(chqcode == "D_LFN_ChDs"){vmax = (Double_t)5000.;}
      if(chqcode == "D_HFN_ChDs"){vmax = (Double_t)5000.;}
      if(chqcode == "D_MCs_ChDs"){vmax = (Double_t)5000.;}
      if(chqcode == "D_SCs_ChDs"){vmax = (Double_t)5000.;}
      
      if(chqcode == "D_MSp_Samp"){vmax = (Double_t)500.;}
      if(chqcode == "D_SSp_Samp"){vmax = (Double_t)5.;}
     
      if(chqcode == "D_Adc_EvNb"){vmax = (Double_t)500.;}
      if(chqcode == "D_Adc_EvDs"){vmax = (Double_t)500.;}

      if(chqcode == "H_Ped_Date"){vmax = (Double_t)500.;}
      if(chqcode == "H_TNo_Date"){vmax = (Double_t)5.;}
      if(chqcode == "H_LFN_Date"){vmax = (Double_t)5.;}
      if(chqcode == "H_HFN_Date"){vmax = (Double_t)5.;}
      if(chqcode == "H_MCs_Date"){vmax = (Double_t)1.;}
      if(chqcode == "H_SCs_Date"){vmax = (Double_t)0.5;}

      if(chqcode == "H_Ped_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_TNo_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_LFN_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_HFN_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_MCs_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_SCs_RuDs"){vmax = (Double_t)1000.;}
    }

  if( fFlagSubDet == "EE" )
    {
      if(chqcode == "H2LFccMosMatrix"){vmax = (Double_t)1.;}
      if(chqcode == "H2HFccMosMatrix"){vmax = (Double_t)1.;}
      if(chqcode == "H2CorccInStins" ){vmax = (Double_t)0.05;}
      
      if(chqcode == "D_NOE_ChNb"){vmax = (Double_t)500.;}
      if(chqcode == "D_Ped_ChNb"){vmax = (Double_t)500.;}
      if(chqcode == "D_TNo_ChNb"){vmax = (Double_t)5.;}
      if(chqcode == "D_LFN_ChNb"){vmax = (Double_t)5.;}
      if(chqcode == "D_HFN_ChNb"){vmax = (Double_t)5.;}
      if(chqcode == "D_MCs_ChNb"){vmax = (Double_t)1.;}
      if(chqcode == "D_SCs_ChNb"){vmax = (Double_t)2.5;}

      if(chqcode == "D_NOE_ChDs"){vmax = (Double_t)1000.;}
      if(chqcode == "D_Ped_ChDs"){vmax = (Double_t)1000.;}
      if(chqcode == "D_TNo_ChDs"){vmax = (Double_t)1000.;}
      if(chqcode == "D_LFN_ChDs"){vmax = (Double_t)1000.;}
      if(chqcode == "D_HFN_ChDs"){vmax = (Double_t)1000.;}
      if(chqcode == "D_MCs_ChDs"){vmax = (Double_t)1000.;}
      if(chqcode == "D_SCs_ChDs"){vmax = (Double_t)1000.;}

      if(chqcode == "D_MSp_Samp"){vmax = (Double_t)500.;}
      if(chqcode == "D_SSp_Samp"){vmax = (Double_t)5.;}
      
      if(chqcode == "D_Adc_EvNb"){vmax = (Double_t)500.;}
      if(chqcode == "D_Adc_EvDs"){vmax = (Double_t)500.;}

      if(chqcode == "H_Ped_Date"){vmax = (Double_t)500.;}
      if(chqcode == "H_TNo_Date"){vmax = (Double_t)5.;}
      if(chqcode == "H_LFN_Date"){vmax = (Double_t)5.;}
      if(chqcode == "H_HFN_Date"){vmax = (Double_t)5.;}
      if(chqcode == "H_MCs_Date"){vmax = (Double_t)1.;}
      if(chqcode == "H_SCs_Date"){vmax = (Double_t)0.5;}

      if(chqcode == "H_Ped_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_TNo_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_LFN_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_HFN_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_MCs_RuDs"){vmax = (Double_t)1000.;}
      if(chqcode == "H_SCs_RuDs"){vmax = (Double_t)1000.;}
    }
  return vmax;
}
//================== End of TEcnaParHistos.cc ========================
