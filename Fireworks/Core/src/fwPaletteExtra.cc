#include "Fireworks/Core/src/fwPaletteExtra.h"


#include "TROOT.h"
#include "TColor.h"
#include "TMath.h"



namespace fireworks {

//static unsigned int rSize = 17;

struct PaletteExtraTopBottomIndices {
   float(* colVals)[3];
   bool whiteBg;
   FWColorManager::EPalette id;
   PaletteExtraTopBottomIndices(float(* iColors)[3], bool iIsWhiteBg, FWColorManager::EPalette iId):colVals(iColors), whiteBg(iIsWhiteBg), id(iId){};

   void set(int i, Color_t cidx) {

      if (whiteBg) {
         //if (id == FWColorManager::kFall || id == FWColorManager::kArctic) 
         const TColor * c = gROOT->GetColor(cidx);
         colVals[i][0] = c->GetRed();
         colVals[i][1] = c->GetGreen();
         colVals[i][2] = c->GetBlue();
         if (id != FWColorManager::kPurple)
         TColor::HLStoRGB(c->GetHue(), 1.2*c->GetLight(), c->GetSaturation(), colVals[i][0], colVals[i][1], colVals[i][2]);

         /*

         cidx = TColor::GetColorBright(cidx);
        TColor* cb = gROOT->GetColor(cidx);
         colVals[i+rSize][0] = cb->GetRed();
         colVals[i+rSize][1] = cb->GetGreen();
         colVals[i+rSize][2] = cb->GetBlue();
         */

      } 
      else {

         const TColor * c = gROOT->GetColor(cidx);
         colVals[i][0] = c->GetRed();
         colVals[i][1] = c->GetGreen();
         colVals[i][2] = c->GetBlue();
         /*
         cidx = TColor::GetColorDark(cidx);
 const TColor * cd = gROOT->GetColor(cidx);
         colVals[i+rSize][0] = cd->GetRed();
         colVals[i+rSize][1] = cd->GetGreen();
         colVals[i+rSize][2] = cd->GetBlue();
         */
      }
   }

//______________________________________________________________________________



   void InitArctic(){
      // set muon geometry gray
         set(0, kYellow+2);
         set(1, kMagenta-3);
         set(2,  kCyan+3);
         set(3,  kAzure+7);
         set(4,  kAzure+8);

         set(5, kAzure+3);// jets, met
         set(6, kAzure-3);
         set(7, kOrange+7);
         set(8, kAzure);// jets, met 
         set(9, kBlue-1);
         set(10, kBlue-7);
         set(11, kTeal-1); 
         set(12, kTeal-7);
         set(13, kCyan-7); 
         set(14, kAzure+10);
         // set(16, kOrange);
   }

//______________________________________________________________________________


   void InitFall()
   {
      set(0, kYellow -4); //vertices, jets
      set(1, kRed +1);
      set(2, kMagenta+3);
      set(3, kGreen + 2);
      set(4, kBlue +2);
      set(5, kOrange -5);//hcal
         set(6, kOrange -8);
         set(7, kMagenta -1);
         set(8, kRed -2); // muon, ecal, met
         set(9, kCyan +2); // tracks


      int j = 2;
      for (int i = 10; i < 15; ++i)
      {
         set(i, kRed -j);
         j++;

      }
   } 

   void InitSpring()
   {
      set(0, kRed);// jet
      set(1, kOrange-2);
      set(2, kRed-7);
      set(3, kPink+2);
      set(4, kOrange-7);
      set(5, kSpring+5);//hcal

      set(6, kSpring);
      set(7, kMagenta); // electrons
      set(8, kSpring +4); //ecal, muons
      set(9, kGreen+2 ); 
      set(10, kSpring +3);
      set(11,kGreen -5 );
      set(12,kGreen -6 );
      set(13,kGreen -7 );
      set(14, kTeal-5);
   }

void InitPurple()
   {
      set(0, kGreen-7);
      set(1, kRed +1);
      set(2, kMagenta+3);
      set(3, kCyan + 2);
      set(4, kBlue +2);
      set(5, kViolet -5);
      set(6, kViolet -6);
      set(7, kViolet -7);
      set(8, kGray+1); //ecal, muo
      set(9, kPink-8); //tracks
      set(10, kBlue-8);
      set(11, kBlue-4);
      set(12, kBlue-2);
      set(13, kViolet+8);
      set(14, kViolet +10);
   }

};



void 
GetColorValuesForPaletteExtra(float(* iColors)[3], unsigned int iSize, FWColorManager::EPalette id, bool isWhiteBg)
{
   PaletteExtraTopBottomIndices p(iColors, isWhiteBg, id);
   switch (id) {
      case FWColorManager::kArctic:
         p.InitArctic();
         break;
      case FWColorManager::kFall:
         p.InitFall();
         break;
      case FWColorManager::kSpring:
         p.InitSpring();
         break;
      
      case FWColorManager::kPurple:
         p.InitPurple();
         break;
      default:
         break;
   }


   iSize = iSize/2;
   unsigned int lastIdx = iSize -2;
   if (isWhiteBg)
   {
      for (unsigned int i = 0; i<lastIdx; ++i) {

         float s = 1.2;
         float r = TMath::Min(1.f,s* iColors[i][0]);
         float g = TMath::Min(1.f,s *iColors[i][1]);
         float b = TMath::Min(1.f,s *iColors[i][2]);

         
         iColors[i][0] = r;
         iColors[i][1] = g;
         iColors[i][2] = b;
         

         float       value = 1.7;
         r = TMath::Power(r, (2.5 - value)/2.5);
         g = TMath::Power(g, (2.5 - value)/2.5);
         b = TMath::Power(b, (2.5 - value)/2.5);
         /*
         s = 1.2;
         r = TMath::Min(1.f,s* iColors[i][0]);
         g = TMath::Min(1.f,s *iColors[i][1]);
         b = TMath::Min(1.f,s *iColors[i][2]);
         */

         r = TMath::Min(1.f, r);
         g = TMath::Min(1.f, g);
         b = TMath::Min(1.f, b);

         iColors[i + iSize][0] = r;
         iColors[i + iSize][1] = g;
         iColors[i + iSize][2] = b;
      }
   }
   else {
      for (unsigned int i = 0; i<lastIdx; ++i) {
         //   TColor* c = gROOT->GetColor(i);
         float s = 0.4;
         float r = s* iColors[i][0];
         float g = s *iColors[i][1];
         float b = s * iColors[i][2];


         // printf("[%d] (%.1f, %.1f, %.1f) => ", i,  r, g, b);
         float       value = -0.5;
         r = TMath::Power( r, (2.5 - value)/2.5);
         g = TMath::Power(g, (2.5 - value)/2.5);
         b = TMath::Power(b, (2.5 - value)/2.5);


         // printf(" (%.1f, %.1f, %.1f) \n", r, g, b);
         //TColor::HLStoRGB(c->GetHue(), c->GetSaturation(), c->GetLight()*0.8, r, g, b);
         iColors[i + iSize][0] = r;
         iColors[i + iSize][1] = g;
         iColors[i + iSize][2] = b;
  
      }
   }
}

}
