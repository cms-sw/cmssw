///////////////////////////////////////////////////////////////////////////////
// File: ClusterFP420.cc
// Date: 12.2006
// Description: ClusterFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"
#include <iostream>
#include <cmath>
using namespace std;

//#define mydigidebug10
//#define mydigidebug11

//static float const_ps1s2[3] =  {0.050,.0139,0.0045};// pitch, sigma_1channelCluster, sigma_2channelsCluster - Narrow , mm
//static float constW_ps1s2[3] = {0.400,.0920,0.0280};// pitch, sigma_1channelCluster, sigma_2channelsCluster - Wide , mm

static float const_ps1s2[3] =  {0.050,.0135,0.0086};// pitch, sigma_1channelCluster, sigma_2channelsCluster - Narrow , mm
static float constW_ps1s2[3] = {0.400,.1149,0.0648};// pitch, sigma_1channelCluster, sigma_2channelsCluster - Wide , mm

// sense of xytype here is X or Y type planes. Now we are working with X only, i.e. xytype=2
ClusterFP420::ClusterFP420( unsigned int detid, unsigned int xytype, const HDigiFP420Range& range, 
			    float & cog ,float & err ) :
  detId_(detid), firstStrip_(range.first->strip())
//  detId_(detid), xytype_(xytype), firstStrip_(range.first->strip())
{
  // For the range of strips in cluster assign adc(,its numbers i->strip()) calculate cog... 
  // strip   :    
#ifdef mydigidebug11
   std::cout << "===================================== firstStrip = " << firstStrip_ << std::endl;
   std::cout << "==range.first->strip() = " << range.first->strip() << std::endl;
   std::cout << "==range.second->strip() = " << range.second->strip() << std::endl;
#endif

  amplitudes_.reserve( range.second - range.first);


  int striprange = 0;
  float sumx = 0.;
  float sumxx = 0.;
  float sumy = 0.;
  float sumyy = 0.;
  int suma = 0;

//  int lastStrip = -1;
  for (HDigiFP420Iter i=range.first; i!=range.second; i++) {
    striprange++;
#ifdef mydigidebug11
   std::cout << " current striprange = " << striprange << std::endl;
#endif
   /*
    /// check if digis consecutive: put amplitude=0 for 
      if (i!=ibeg && (difNarr(xytype,i,i-1) > 1 || difWide(xytype,i,i-1) > 1)   ){
    if (lastStrip>0 && i->strip() != lastStrip + 1) {
                 for (int j=0; j < i->strip()-(lastStrip+1); j++) {
		   amplitudes_.push_back( 0);
		 }
    }
    lastStrip = i->strip();
*/
    short amp = i->adc();       // FIXME: gain correction here

#ifdef mydigidebug11
   std::cout << " current strip = " << i->strip() << "  amp = " << amp << std::endl;
#endif

    amplitudes_.push_back(amp);
    if(xytype == 1) {
      sumx += i->stripH()*amp;
      sumy += i->stripHW()*amp;
      suma += amp;
      sumxx += (i->stripH()) * (i->stripH()) * amp;
      sumyy += (i->stripHW()) * (i->stripHW()) * amp;
    }
    else if(xytype == 2) {
      sumx += i->stripV()*amp;
      sumy += i->stripVW()*amp;
      suma += amp;
      sumxx += (i->stripV()) * (i->stripV()) * amp;
      sumyy += (i->stripVW()) * (i->stripVW()) * amp;
    }
    else {
      std::cout << " ClusterFP420: wrong xytype = " << xytype << std::endl;
    }

#ifdef mydigidebug11
   std::cout << " current sumx = " << sumx << std::endl;
   std::cout << " current sumy = " << sumy << std::endl;
   std::cout << " current suma = " << suma << std::endl;
   std::cout << " current barycenter = " << (sumx / static_cast<float>(suma) )  << std::endl;
   std::cout << " current barycenterW= " << (sumy / static_cast<float>(suma) )  << std::endl;
#endif
  } //for i


  if(suma != 0) {
    barycenter_ = sumx / static_cast<float>(suma) ;
    barycerror_ = sumxx / static_cast<float>(suma) ;
    barycerror_ = fabs(barycerror_ - barycenter_*barycenter_) ;
#ifdef mydigidebug11
    std::cout << "barycerror_ = " << barycerror_ << "barycenter_ = " << barycenter_ << std::endl;
#endif
    barycenterW_ = sumy / static_cast<float>(suma) ;
    barycerrorW_ = sumyy / static_cast<float>(suma) ;
    barycerrorW_ = fabs(barycerrorW_ - barycenterW_*barycenterW_) ;
#ifdef mydigidebug11
    std::cout << "barycerrorW_ = " << barycerrorW_ << "barycenterW_ = " << barycenterW_ << std::endl;
#endif
  }
  else{
    barycenter_ = 1000000. ;
    barycerror_ = 1000000. ;
    barycenterW_ = 1000000. ;
    barycerrorW_ = 1000000. ;
  }

  /** The barycenter of the cluster, not corrected for Lorentz shift;
   *  it can means that should not be used as position estimate for tracking.
   */
  cog = barycenter_;// cog for Narrow pixels only

#ifdef mydigidebug11
   std::cout << "AT end: barycenter_ = " << barycenter_ << std::endl;
   std::cout << "AT end:  striprange = " << striprange << std::endl;
#endif





   /*

  float sumx0 = 0.;
  float sumxx = 0.;
  lastStrip = -1;
  for (HDigiFP420Iter i=range.first; i!=range.second; i++) {
#ifdef mydigidebug11
   std::cout << " current striprange = " << striprange << std::endl;
#endif
    /// check if digis consecutive
    if (lastStrip>0 && i->strip() != lastStrip + 1) {
                 for (int j=0; j < i->strip()-(lastStrip+1); j++) {
		   amplitudes_.push_back( 0);
		 }
    }
    lastStrip = i->strip();

    short amp = i->adc();       // FIXME: gain correction here

#ifdef mydigidebug11
   std::cout << " current strip = " << i->strip() << "  amp = " << amp << std::endl;
#endif

    sumx0 += (i->strip()-cog)*amp;
    sumxx += (i->strip()-cog) * (i->strip()-cog) * amp;


#ifdef mydigidebug11
   std::cout << " 2 current sumx0 = " << sumx0 << std::endl;
   std::cout << " 2 current sumxx = " << sumxx << std::endl;
#endif
  } //for


  if(suma != 0) {
    sumx0 = sumx0 / static_cast<float>(suma) ;
    sumxx = sumxx / static_cast<float>(suma);
    
    //barycerror_ = fabs(sumxx - sumx0*sumx0) ;

    //barycerror_ = (sumxx - sumx0*sumx0) ;
    //barycerror_ *= barycerror_ ;

      barycerror_ = sumxx ;

  }
  else{
    barycerror_ = 1000000. ;
  }

*/

#ifdef mydigidebug10
   std::cout << "pitchcommon = " << const_ps1s2[0] << " sigma1= " << const_ps1s2[1]  << " sigma2= " << const_ps1s2[2]  << std::endl;
#endif

   //
  if(barycerror_ == 0.0) {
    barycerror_ = const_ps1s2[1]/const_ps1s2[0];// 
  }
  else{
    barycerror_ = const_ps1s2[2]/const_ps1s2[0];//  
  }
    barycerror_ *= barycerror_;
   //
  if(barycerrorW_ == 0.0) {
    barycerrorW_ = constW_ps1s2[1]/constW_ps1s2[0];// 
  }
  else{
    barycerrorW_ = constW_ps1s2[2]/constW_ps1s2[0];// 
  }
    barycerrorW_ *= barycerrorW_;
   //

#ifdef mydigidebug11
   std::cout << "barycerror_ = " << barycerror_ << "barycerrorW_ = " << barycerrorW_ << std::endl;
#endif
         
   // change by hands:

	// number of station
	int  mysn0 = 2;
	
	// number of planes 
	int  mypn0 = 5; // number of superplanes

	// number of station
	int  myrn0 = 6;//  6 possible sensors in superlayer
	






	// comment:              detID = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
  // unpack from detId_:

    int sScale = myrn0*mypn0, dScale = myrn0*mypn0*mysn0;
    
    int  det = (detId_-1)/dScale + 1;
    int  sector = (detId_-1- dScale*(det - 1))/sScale + 1;

#ifdef mydigidebug11
   std::cout << "sector = " << sector << " det= " << det << std::endl;
#endif

  // unpack from detId_ (OLD):
    // 	int  sScale = 2*mypn0;
    //int  sector = (detId_-1)/sScale + 1 ;



	float a = 0.00001;


	/////////////////////////////////////////////////////////////////// real configuration
	if(mysn0 == 2) {
	  if(sector==2) {
	    a = 0.0026+((0.0075-0.0026)/7.)*(mypn0-2); // at 8 m in mm 
	      }
	}
	/////////////////////////////////////////////////////////////////// for stydies:
	else if(mysn0 == 3) {
	  if(sector==2) {
	    a = 0.0011+((0.0030-0.0011)/7.)*(mypn0-2); // at 4 m  in mm 
	      }
	  else if(sector==3) {
	    a = 0.0022+((0.0068-0.0022)/7.)*(mypn0-2); // at 8 m  in mm 
	      }
	}
	else if(mysn0 == 4) {
	  if(sector==2) {
	    a = 0.0009+((0.0024-0.0009)/7.)*(mypn0-2); // at 2.7 m  in mm 
	      }
	  else if(sector==3) {
	    a = 0.0018+((0.0050-0.0018)/7.)*(mypn0-2); // at 5.4 m  in mm 
	      }
	  else if(sector==4) {
	    a = 0.0026+((0.0075-0.0026)/7.)*(mypn0-2); // at 8.1 m  in mm 
	      }
	}

	//	barycerror_+=a*a;
	//	barycerrorW_+=a*a;
	barycerror_+=a*a/const_ps1s2[0]/const_ps1s2[0];
	barycerrorW_+=a*a/constW_ps1s2[0]/constW_ps1s2[0];

    /*

  if(detId_ < 21) {
    float a = 0.0001*(int((detId_-1)/2.)+1)/pitchall;
    barycerror_+=a*a;
  }
  else if(detId_ < 41) {
    float a = 0.0001*(int((detId_-21)/2.)+1)/pitchall;
           a +=0.0036; // 2.5 m
    //          a +=0.0052; // 4 m
    //  a +=0.0131;// 8. m
    barycerror_+=a*a;
  }
  else if(detId_ < 61) {
    float a = 0.0001*(int((detId_-41)/2.)+1)/pitchall;
             a +=0.0069;// 5 m  0.0059
    //          a +=0.0101;// 8. m
    //  a +=0.0241;// 16. m
    barycerror_+=a*a;
  }
  else if(detId_ < 81) {
    float a = 0.0001*(int((detId_-61)/2.)+1)/pitchall;
          a +=0.0131;// 7.5 m   0.0111
    //       a +=0.0151;// 12. m
    //  a +=0.0301;// 24. m
    barycerror_+=a*a;
  }
*/
#ifdef mydigidebug11
   std::cout << "AT end: barycerror_ = " << barycerror_ << std::endl;
#endif

  barycerror_ = sqrt(  barycerror_ );
  err = barycerror_;

  barycerrorW_ = sqrt(  barycerrorW_ );


#ifdef mydigidebug11
   std::cout << "AT end: err = " << err<< "   detId_= " << detId_ << std::endl;
#endif

}

