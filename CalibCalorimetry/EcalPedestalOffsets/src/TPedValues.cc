#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedValues.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <math.h>
#include <iostream>
#include <cassert>
#include "TGraphErrors.h"
#include "TAxis.h"
#include "TF1.h"

void reset (double vett[256]) 
{
   for (int i=0 ; i<256; ++i) vett[i] = 0. ; 
}


TPedValues::TPedValues (double RMSmax, int bestPedestal) :
  m_bestPedestal (bestPedestal) ,
  m_RMSmax (RMSmax) 
{
  LogDebug ("EcalPedOffset") << "entering TPedValues ctor ..." ;
  for(int i=0; i<1700;++i)
    endcapCrystalNumbers[i] = 0;
}


TPedValues::TPedValues (const TPedValues & orig) 
{
  LogDebug ("EcalPedOffset") << "entering TPedValues copyctor ..." ;
  m_bestPedestal = orig.m_bestPedestal ;
  m_RMSmax = orig.m_RMSmax ;

  for (int gain = 0 ; gain < 3 ; ++gain)
    for (int crystal = 0 ; crystal < 1700 ; ++crystal)
        for (int DAC = 0 ; DAC < 256 ; ++DAC)
          m_entries[gain][crystal][DAC] = orig.m_entries[gain][crystal][DAC] ;

  for(int i=0; i<1700;++i)
    endcapCrystalNumbers[i] = orig.endcapCrystalNumbers[i];
}



TPedValues::~TPedValues () {}
  

void TPedValues::insert (const int gainId, 
                         const int crystal, 
                         const int DAC, 
                         const int pedestal,
                         const int endcapIndex) 
{
//  assert (gainId > 0) ;
//  assert (gainId < 4) ;
  if (gainId <= 0 || gainId >= 4)
    {
      edm::LogWarning ("EcalPedOffset") << "WARNING : TPedValues : gainId " << gainId
                                        << " does not exist, entry skipped" ;
      return ;    
    }
//  assert (crystal > 0) ;
//  assert (crystal <= 1700) ;
  if (crystal <= 0 || crystal > 1700)
    {
      edm::LogWarning ("EcalPedOffset") << "WARNING : TPedValues : crystal " << crystal
                                        << " does not exist, entry skipped" ;
      return ;    
    }
//  assert (DAC >= 0) ; 
//  assert (DAC < 256) ;
  if (DAC < 0 || DAC >= 256)
    {
      edm::LogWarning ("EcalPedOffset") << "WARNING : TPedValues : DAC value " << DAC
                                        << " is out range, entry skipped" ;
      return ;    
    }
  m_entries[gainId-1][crystal-1][DAC].insert (pedestal) ;
  endcapCrystalNumbers[crystal-1] = endcapIndex;
  return ;
}
    

TPedResult TPedValues::terminate (const int & DACstart, const int & DACend) const
{
  assert (DACstart >= 0) ;
  assert (DACend <= 256) ;
//  checkEntries (DACstart, DACend) ;

  TPedResult bestDAC ;
  //! loop over gains
  for (int gainId = 1 ; gainId < 4 ; ++gainId)
    {
      //! loop over crystals
      for (int crystal = 0 ; crystal < 1700 ; ++crystal)
        {
          //! find the DAC value with the average pedestal nearest to 200
          double delta = 1000 ;
          int dummyBestDAC = -1 ;
          bool hasDigis = false;
          //! loop over DAC values
          for (int DAC = DACstart ; DAC < DACend ; ++DAC)
            {
              double average = m_entries[gainId-1][crystal][DAC].average () ;
              if (average == -1) continue ;
              hasDigis = true;
              if (m_entries[gainId-1][crystal][DAC].RMSSq () > m_RMSmax * m_RMSmax) continue ;
              if (fabs (average - m_bestPedestal) < delta &&   average>1 ) 
                {
                  delta = fabs (average - m_bestPedestal) ;
                  dummyBestDAC = DAC ;
                }
            } //! loop over DAC values

          bestDAC.m_DACvalue[gainId-1][crystal] = dummyBestDAC ;
	  
	  if ((dummyBestDAC == (DACend-1) || dummyBestDAC == -1) && hasDigis)
          {
	    int gainHuman;
	    if      (gainId ==1) gainHuman =12;
	    else if (gainId ==2) gainHuman =6;
	    else if (gainId ==3) gainHuman =1;
	    else                 gainHuman =-1;
	    edm::LogError("EcalPedOffset")
              << " TPedValues :  cannot find best DAC value for channel: "
              << endcapCrystalNumbers[crystal]
	      << " gain: " << gainHuman;
	  }
	  
        } // loop over crystals
    } // loop over gains
  return bestDAC ;
}
    

int TPedValues::checkEntries (const int & DACstart, const int & DACend) const
{
  assert (DACstart >= 0) ;
  assert (DACend <= 256) ;
  int returnCode = 0 ;
  //! loop over gains
  for (int gainId = 1 ; gainId < 4 ; ++gainId)
    {
      //! loop over crystals
      for (int crystal = 0 ; crystal < 1700 ; ++crystal)
        {
          //! loop over DAC values
          for (int DAC = DACstart ; DAC < DACend ; ++DAC)
            {
              double average = m_entries[gainId-1][crystal][DAC].average () ;
              if (average == -1) 
                {
                  ++returnCode ;
                  //! do something!
/*
                  std::cerr << "[TPedValues][checkEntries] WARNING!"
                            << "\tgainId " << gainId
                            << "\tcrystal " << crystal+1
                            << "\tDAC " << DAC
                            << " : pedestal measurement missing" 
                            << std::endl ;
*/
                }
/*
              std::cout << "[pietro][RMS]: " << m_entries[gainId-1][crystal][DAC].RMS ()     //FIXME
                        << "\t" << m_entries[gainId-1][crystal][DAC].RMSSq () //FIXME
                        << "\t" << DAC //FIXME
                        << "\t" << gainId //FIXME
                        << "\t" << crystal << std::endl ; //FIXME
*/              
            } //! loop over DAC values
        } // loop over crystals
    } // loop over gains
  return returnCode ;
}


//! create a plot of the DAC pedestal trend
int TPedValues::makePlots (TFile * rootFile, const std::string & dirName,
     const double maxSlope, const double minSlope, const double maxChi2OverNDF) const 
{
  using namespace std;
  // prepare the ROOT file
  if (!rootFile->cd (dirName.c_str ())) 
    {
      rootFile->mkdir (dirName.c_str ()) ;
      rootFile->cd (dirName.c_str ()) ;
    }
  
  // loop over the crystals
  for (int xtl=0 ; xtl<1700 ; ++xtl)
  {
    // loop over the gains
    for (int gain=0 ; gain<3 ; ++gain)
      {
        vector<double> asseX;
        vector<double> sigmaX;
        vector<double> asseY;
        vector<double> sigmaY;
        asseX.reserve(256);
        sigmaX.reserve(256);
        asseY.reserve(256);
        sigmaY.reserve(256);
        // loop over DAC values
        for (int dac=0 ; dac<256 ; ++dac)
          {
            double average = m_entries[gain][xtl][dac].average();
            if(average > -1)
            {
              double rms = m_entries[gain][xtl][dac].RMS();
              asseX.push_back(dac);
              sigmaX.push_back(0);
              asseY.push_back(average);
              sigmaY.push_back(rms);
            }
          } // loop over DAC values
        if(asseX.size() > 0)
        {
          int lastBin = 0;
          while(lastBin<(int)asseX.size()-1 && asseY[lastBin+1]>0
              && (asseY[lastBin+1]-asseY[lastBin+2])!=0)
            lastBin++;
          
          int fitRangeEnd = (int)asseX[lastBin];
          int kinkPt = 64;
          if(fitRangeEnd < 66)
            kinkPt = fitRangeEnd-4;
          TGraphErrors graph(asseX.size(),&(*asseX.begin()),&(*asseY.begin()),
              &(*sigmaX.begin()),&(*sigmaY.begin()));
          char funct[120];
          sprintf(funct,"(x<%d)*([0]*x+[1])+(x>=%d)*([2]*x+[3])",kinkPt,kinkPt);
          TF1 fitFunction("fitFunction",funct,asseX[0],fitRangeEnd);
          fitFunction.SetLineColor(2);
          
          char name[120] ;
          int gainHuman;
          if      (gain ==0) gainHuman =12;
          else if (gain ==1) gainHuman =6;
          else if (gain ==2) gainHuman =1;
          else               gainHuman =-1;
          sprintf (name,"XTL%04d_GAIN%02d",endcapCrystalNumbers[xtl],gainHuman) ;
          graph.GetXaxis()->SetTitle("DAC value");
          graph.GetYaxis()->SetTitle("Average pedestal ADC");
          graph.Fit(&fitFunction,"RWQ");
          graph.Write (name);
          
          double slope1 = fitFunction.GetParameter(0);
          double slope2 = fitFunction.GetParameter(2);

          if(fitFunction.GetChisquare()/fitFunction.GetNDF()>maxChi2OverNDF ||
              fitFunction.GetChisquare()/fitFunction.GetNDF()<0 ||
              slope1>0 || slope2>0 ||
              ((slope1<-29 || slope1>-18) && slope1<0) || 
              ((slope2<-29 || slope2>-18) && slope2<0))
          {
            edm::LogError("EcalPedOffset") << "TPedValues : TGraph for channel:" << 
              endcapCrystalNumbers[xtl] << 
              " gain:" << gainHuman << " is not linear;" << "  slope of line1:" << 
              fitFunction.GetParameter(0) << " slope of line2:" << 
              fitFunction.GetParameter(2) << " reduced chi-squared:" << 
              fitFunction.GetChisquare()/fitFunction.GetNDF();
          }
          //LogDebug("EcalPedOffset") << "TPedValues : TGraph for channel:" << xtl+1 << " gain:"
          //  << gainHuman << " has " << asseX.size() << " points...back is:" << asseX.back() 
          //  << " and front+1 is:" << asseX.front()+1;
          if((asseX.back()-asseX.front()+1)!=asseX.size())
            edm::LogError("EcalPedOffset") << "TPedValues : Pedestal average not found " <<
              "for all DAC values scanned in channel:" << endcapCrystalNumbers[xtl]
              << " gain:" << gainHuman;
        }
      } // loop over the gains
  }     // (loop over the crystals)
  
  return 0 ;
}

// Look up the crystal number in the EE schema and return it
int TPedValues::getCrystalNumber(int xtal) const
{
  return endcapCrystalNumbers[xtal];
}
