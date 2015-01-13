#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Calo/interface/FWHistSliceSelector.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Rtypes.h"
#include "TEveCaloData.h"
#include "TMath.h"
#include "TEveCalo.h"
#include "TH2F.h"


FWCaloDataHistProxyBuilder::FWCaloDataHistProxyBuilder(): m_hist(0), m_sliceSelector(0)
{
}
FWCaloDataHistProxyBuilder::~FWCaloDataHistProxyBuilder()
{
}


namespace {
    double  wrapPi(double val)
    {
        using namespace TMath;

        if (val< -Pi())
        {
            return val += TwoPi();
        }
        if (val> Pi())
        {
            return val -= TwoPi();
        }
        return val;
    }
}


void
FWCaloDataHistProxyBuilder::setCaloData(const fireworks::Context&)
{
m_caloData = context().getCaloData();
}

void
FWCaloDataHistProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
  
   if(0!=m_hist) {
      m_hist->Reset();
   }
   FWCaloDataProxyBuilderBase::itemBeingDestroyed(iItem);
}

bool
FWCaloDataHistProxyBuilder::assertCaloDataSlice()
{
   if (m_hist == 0)
   {
      // add new slice
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      m_hist = new TH2F("caloHist",
                        "caloHist",
                        fw3dlego::xbins_n - 1, fw3dlego::xbins,
                        72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      TEveCaloDataHist* ch = static_cast<TEveCaloDataHist*>(m_caloData);
      m_sliceIndex = ch->AddHistogram(m_hist);



      m_caloData->RefSliceInfo(m_sliceIndex).Setup(item()->name().c_str(), 0., 
                                                 item()->defaultDisplayProperties().color(),
                                                   item()->defaultDisplayProperties().transparency());

      // add new selector
      FWFromTEveCaloDataSelector* sel = 0;
      if (m_caloData->GetUserData())
      {
         FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*>(m_caloData->GetUserData());
         assert(0!=base);
         sel = dynamic_cast<FWFromTEveCaloDataSelector*> (base);
         assert(0!=sel);
      }
      else
      {
         sel = new FWFromTEveCaloDataSelector(m_caloData);
         //make sure it is accessible via the base class
         m_caloData->SetUserData(static_cast<FWFromEveSelectorBase*>(sel));
      }
      m_sliceSelector = instantiateSliceSelector();
      sel->addSliceSelector(m_sliceIndex, m_sliceSelector);
     
      return true;
   }
   return false;
 }


void FWCaloDataHistProxyBuilder::addEntryToTEveCaloData(float eta, float phi, float Et, bool isSelected)
{
   using namespace TMath;
   static float d = 2.5*Pi()/180;
   //    printf("comapre %f, %f \n", fw3dlego::xbins[80],  fw3dlego::xbins[61] );


   if (m_sliceSelector->aggregatePhiCells()) {
      if (Abs(eta) > fw3dlego::xbins[80])
      {
         m_hist->Fill(eta,wrapPi(phi - 3*d), Et *0.25);
         m_hist->Fill(eta,wrapPi(phi -   d), Et *0.25);
         m_hist->Fill(eta,wrapPi(phi +   d), Et *0.25);
         m_hist->Fill(eta,wrapPi(phi + 3*d), Et *0.25);
      }
      else if (Abs(eta) > fw3dlego::xbins[61])
      {
         m_hist->Fill(eta,wrapPi(phi - d), Et *0.5);
         m_hist->Fill(eta,wrapPi(phi + d), Et *0.5);
      }
      else
      {
         m_hist->Fill(eta,phi, Et);
      }
   }
   else 
   {
      m_hist->Fill(eta,phi, Et);
   }

   TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();
   if(isSelected) {
      //NOTE: I tried calling TEveCalo::GetCellList but it always returned 0, probably because of threshold issues
      // but looking at the TEveCaloHist::GetCellList code the CellId_t is just the histograms bin # and the slice
      // printf("applyChangesToAllModels ...check selected \n");


      if (m_sliceSelector->aggregatePhiCells()) {
         if (Abs(eta) > fw3dlego::xbins[80])
         {
            selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta, wrapPi(phi -3*d)),m_sliceIndex));
            selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta, wrapPi(phi -d))  ,m_sliceIndex));
            selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta, wrapPi(phi +d))  ,m_sliceIndex));
            selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta, wrapPi(phi +3*d)),m_sliceIndex));
         }
         if (Abs(eta) > fw3dlego::xbins[60])
         {
            selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta, wrapPi(phi -d)), m_sliceIndex));
            selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta, wrapPi(phi +d)), m_sliceIndex));
         }
         else
         {
            selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta,phi),m_sliceIndex));
         }
      }
   else 
   {
      selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(eta,phi),m_sliceIndex));
   }
   }
}
