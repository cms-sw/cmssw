#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedValues.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <math.h>
#include <iostream>
#include <cassert>
#include "TGraphErrors.h"
#include "TAxis.h"

void reset (double vett[256]) 
{
   for (int i=0 ; i<256; ++i) vett[i] = 0. ; 
}


TPedValues::TPedValues (double RMSmax, int bestPedestal) :
  m_bestPedestal (bestPedestal) ,
  m_RMSmax (RMSmax) 
{
  LogDebug ("EcalPedOffset") << "entering TPedValues ctor ..." ;
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
  
}



TPedValues::~TPedValues () {}
  

void TPedValues::insert (const int gainId, 
                         const int crystal, 
                         const int DAC, 
                         const int pedestal) 
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
          //! loop over DAC values
          for (int DAC = DACstart ; DAC < DACend ; ++DAC)
            {
              double average = m_entries[gainId-1][crystal][DAC].average () ;
              if (average == -999) continue ;
              if (m_entries[gainId-1][crystal][DAC].RMSSq () > m_RMSmax * m_RMSmax) continue ;
              if (fabs (average - m_bestPedestal) < delta &&   average>1 ) 
                {
                  delta = fabs (average - m_bestPedestal) ;
                  dummyBestDAC = DAC ;
                }
            } //! loop over DAC values

          bestDAC.m_DACvalue[gainId-1][crystal] = dummyBestDAC ;
	  
	  if ( dummyBestDAC == (DACend-1) || dummyBestDAC == -1 ) {
	    int gainHuman;
	    if      (gainId ==1) gainHuman =12;
	    else if (gainId ==2) gainHuman =6;
	    else if (gainId ==3) gainHuman =1;
	    else                 gainHuman =-1;
	    
	    edm::LogWarning ("EcalPedOffset") << " TPedValues :  channel: " << (crystal+1)
					      << " gain: " << gainHuman
					      << " has offset set to: " << dummyBestDAC << "."
					      << " The maximum expected value is: " << DACend 
					      << " (need be corrected by hand? Look at plots)";
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
              if (average == -999) 
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
int TPedValues::makePlots (TFile * rootFile, const std::string & dirName) const 
{
  // prepare the ROOT file
  if (!rootFile->cd (dirName.c_str ())) 
    {
      rootFile->mkdir (dirName.c_str ()) ;
      rootFile->cd (dirName.c_str ()) ;
    }
    
  // loop over the crystals
  for (int xtl=0 ; xtl<1700 ; ++xtl)
    // loop over the gains
    for (int gain=0 ; gain<3 ; ++gain)
      {
        bool doGraph = false;
        double asseX[256] ;  reset (asseX) ;
        double sigmaX[256] ; reset (sigmaX) ;
        double asseY[256] ;  reset (asseY) ;
        double sigmaY[256] ; reset (sigmaY) ;
        // loop over DAC values
        for (int dac=0 ; dac<256 ; ++dac)
          {
            asseX[dac] = dac ;
            sigmaX[dac] = 0 ;
            asseY[dac] = m_entries[gain][xtl][dac].average () ;
            sigmaY[dac] = m_entries[gain][xtl][dac].RMS () ;
            if (asseY[dac] < -100)
              sigmaY[dac] = asseY[dac] = 0 ;
        
            // Only do the graph if one of the averages is nonzero
            if(asseY[dac] != 0)
              doGraph = true;
          } // loop over DAC values
        if(doGraph)
        {
          TGraphErrors graph (256,asseX,asseY,sigmaX,sigmaY) ;
          char name[120] ;
          int gainHuman;
          if      (gain ==0) gainHuman =12;
          else if (gain ==1) gainHuman =6;
          else if (gain ==2) gainHuman =1;
          else               gainHuman =-1;
          sprintf (name,"XTL%04d_GAIN%02d",(xtl+1),gainHuman) ;      
          graph.GetXaxis()->SetTitle("DAC value");
          graph.GetYaxis()->SetTitle("Average pedestal ADC");
          graph.Write (name);
        }
      } // loop over the gains
        // (loop over the crystals)

  return 0 ;
}
     

//! create a plot of the DAC pedestal trend
int TPedValues::makePlots (const std::string & rootFileName, const std::string & dirName) const 
{
  TFile saving (rootFileName.c_str (),"APPEND") ;
  return makePlots (&saving,dirName) ;  
}

