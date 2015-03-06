#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimeProfileStatusBitSetter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

#include <algorithm> // for "max"
#include <math.h>
#include <TH1.h>
#include <TF1.h>


HBHETimeProfileStatusBitSetter::HBHETimeProfileStatusBitSetter()
{
  // use simple values in default constructor
  R1Min_ = 0.1;
  R1Max_ = 0.7;
  R2Min_ = 0.2;
  R2Max_ = 0.5;
  FracLeaderMin_ = 0.4;
  FracLeaderMax_ = 0.7;
  SlopeMin_ = -1.5;
  SlopeMax_ = -0.6;
  OuterMin_ = 0.9;
  OuterMax_ = 1.0;
  EnergyThreshold_=30;
}

HBHETimeProfileStatusBitSetter::HBHETimeProfileStatusBitSetter(double R1Min, double R1Max, 
							       double R2Min, double R2Max, 
							       double FracLeaderMin, double FracLeaderMax, 
							       double SlopeMin, double SlopeMax, 
							       double OuterMin, double OuterMax, 
							       double EnergyThreshold)
{
  R1Min_ = R1Min;
  R1Max_ = R1Max;
  R2Min_ = R2Min;
  R2Max_ = R2Max;
  FracLeaderMin_ = FracLeaderMin;
  FracLeaderMax_ = FracLeaderMax;
  SlopeMin_ = SlopeMin;
  SlopeMax_ = SlopeMax;
  OuterMin_ = OuterMin;
  OuterMax_ = OuterMax;
  EnergyThreshold_ = EnergyThreshold;
}

HBHETimeProfileStatusBitSetter::~HBHETimeProfileStatusBitSetter(){}









void HBHETimeProfileStatusBitSetter::hbheSetTimeFlagsFromDigi(HBHERecHitCollection * hbhe, const std::vector<HBHEDataFrame>& udigi, const std::vector<int>& RecHitIndex)
{
  
  bool Bits[4]={false, false, false, false};
  std::vector<HBHEDataFrame> digi = const_cast<std::vector<HBHEDataFrame>&>(udigi);
  std::sort(digi.begin(), digi.end(), compare_digi_energy()); // sort digis according to energies
  std::vector<double> PulseShape; // store fC values for each time slice
  int DigiSize=0;
  //  int LeadingEta=0;
  int LeadingPhi=0;
  bool FoundLeadingChannel=false;
  for(std::vector<HBHEDataFrame>::const_iterator itDigi = digi.begin(); itDigi!=digi.end(); itDigi++)
    {
      if(!FoundLeadingChannel)
	{
	  //	  LeadingEta = itDigi->id().ieta();
	  LeadingPhi = itDigi->id().iphi();
	  DigiSize=(*itDigi).size();
	  PulseShape.clear();
	  PulseShape.resize(DigiSize,0); 
	  FoundLeadingChannel=true;
	}
      if(abs(LeadingPhi - itDigi->id().iphi())<2)
	for(int i=0; i!=DigiSize; i++)
	  PulseShape[i]+=itDigi->sample(i).nominal_fC();
	    
    }


    
  if(RecHitIndex.size()>0)
    {
      double FracInLeader=-1;
      //double Slope=0; // not currently used
      double R1=-1;
      double R2=-1;
      double OuterEnergy=-1;
      double TotalEnergy=0;
      int PeakPosition=0;
      
      for(int i=0; i!=DigiSize; i++)
	{
	  if(PulseShape[i]>PulseShape[PeakPosition]) 
	    PeakPosition=i;
	  TotalEnergy+=PulseShape[i];
	}
     
     
      if(PeakPosition < (DigiSize-2))
	{
	  R1 = PulseShape[PeakPosition+1]/PulseShape[PeakPosition];
	  R2 = PulseShape[PeakPosition+2]/PulseShape[PeakPosition+1];
	}
      
      FracInLeader = PulseShape[PeakPosition]/TotalEnergy;
      
      if((PeakPosition > 0) && (PeakPosition < (DigiSize-2)))
      {
	OuterEnergy = 1. -((PulseShape[PeakPosition - 1] +
	                   PulseShape[PeakPosition]     +
	                   PulseShape[PeakPosition + 1] +
	                   PulseShape[PeakPosition + 2] )
			   / TotalEnergy);

      }
           
      /*      TH1D * HistForFit = new TH1D("HistForFit","HistForFit",DigiSize,0,DigiSize);
      for(int i=0; i!=DigiSize; i++)
	{
	  HistForFit->Fill(i,PulseShape[i]);
	  HistForFit->Fit("expo","WWQ","",PeakPosition, DigiSize-1);
	  TF1 * Fit = HistForFit->GetFunction("expo");
	  Slope = Fit->GetParameter("Slope");
	}
      delete HistForFit;
      */
      if (R1!=-1 && R2!=-1)
	Bits[0] = (R1Min_ > R1) || (R1Max_ < R1) || (R2Min_ > R2) || (R2Max_ < R2);
      if (FracInLeader!=-1)
	Bits[1] = (FracInLeader < FracLeaderMin_) || (FracInLeader > FracLeaderMax_);
      if (OuterEnergy!=-1)
	Bits[2] = (OuterEnergy < OuterMin_) || (OuterEnergy > OuterMax_);
      //  Bits[3] = (SlopeMin_ > Slope) || (SlopeMax_ < Slope);
      Bits[3] = false;

    } // if (RecHitIndex.size()>0)
  else 
    {
      
      Bits[0]=false;
      Bits[1]=false;
      Bits[2]=false;
      Bits[3]=true;
  
    } // (RecHitIndex.size()==0; no need to set Bit3 true?)
  
  for(unsigned int i=0; i!=RecHitIndex.size(); i++)
    {

      // Write calculated bit values starting from position FirstBit
      (*hbhe)[RecHitIndex.at(i)].setFlagField(Bits[0],HcalCaloFlagLabels::HSCP_R1R2);
      (*hbhe)[RecHitIndex.at(i)].setFlagField(Bits[1],HcalCaloFlagLabels::HSCP_FracLeader);
      (*hbhe)[RecHitIndex.at(i)].setFlagField(Bits[2],HcalCaloFlagLabels::HSCP_OuterEnergy);
      (*hbhe)[RecHitIndex.at(i)].setFlagField(Bits[3],HcalCaloFlagLabels::HSCP_ExpFit);
 
   }
 
  

}

