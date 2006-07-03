#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedValues.h"

#include <iostream>
#include "TGraphErrors.h"

void reset (double vett[256]) 
{
   for (int i=0 ; i<256; ++i) vett[i] = 0. ; 
}


TPedValues::TPedValues (double RMSmax, int bestPedestal) :
  m_bestPedestal (bestPedestal) ,
  m_RMSmax (RMSmax) 
{
  std::cout << "[TPedValues][ctor]" << std::endl ;
}


TPedValues::TPedValues (const TPedValues & orig) 
{
  std::cout << "[TPedValues][copyctor]" << std::endl ;
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
  if (gainId <= 0 || gainId > 4)
    {
      std::cerr << "ERROR : gainId " << gainId
                << " does not exist, entry skipped\n" ;
      return ;    
    }
  assert (crystal > 0) ;
  assert (crystal <= 1700) ;
  assert (DAC >= 0) ; 
  assert (DAC < 256) ;
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
              if (fabs (average - m_bestPedestal) < delta) 
                {
                  delta = fabs (average - m_bestPedestal) ;
                  dummyBestDAC = DAC ;
                }
            } //! loop over DAC values
          bestDAC.m_DACvalue[gainId-1][crystal] = dummyBestDAC ;
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
            if (asseY[dac] < -100) sigmaY[dac] = asseY[dac] = 0 ;
          } // loop over DAC values          
        TGraphErrors graph (256,asseX,asseY,sigmaX,sigmaY) ;
        char name[120] ;
        sprintf (name,"XTL%d_GAIN%d",xtl,gain) ;      
        graph.Write (name) ;
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

