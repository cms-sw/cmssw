#include "DQM/EcalCommon/interface/UtilsClient.h"

#include <string>
#include <cmath>

#include "TH1.h"
#include "TProfile.h"
#include "TClass.h"
#include "TProfile2D.h"

#include "DQMServices/Core/interface/MonitorElement.h"

void
UtilsClient::printBadChannels( const MonitorElement* me, TH1* hi, bool positive_only)
{
  if ( ! me ) {
    std::cout << "printBadChannels() failed, NULL pointer to MonitorElement !"
	      << std::endl;
    return;
  }
  if ( ! hi ) {
    std::cout << "printBadChannels() failed, NULL pointer to ROOT histogram !"
	      << std::endl;
    return;
  }
  bool title = false;
  TProfile2D* hj = dynamic_cast<TProfile2D*>(hi);
  int kx = -1;
  int ky = -1;
  for ( int ix = 1; ix <= me->getNbinsX(); ix++ ) {
    int jx = ix * hi->GetNbinsX() / me->getNbinsX();
    if ( jx == kx ) continue;
    kx = jx;
    for ( int iy = 1; iy <= me->getNbinsY(); iy++ ) {
      int jy = iy * hi->GetNbinsY() / me->getNbinsY();
      if ( jy == ky ) continue;
      ky = jy;
      if ( positive_only ) {
	if ( hi->GetBinContent(hi->GetBin(jx, jy)) <= 0 ) continue;
      } else {
	float val = me->getBinContent( ix, iy );
	//  0/3 = red/dark red
	//  1/4 = green/dark green
	//  2/5 = yellow/dark yellow
	//  6   = unknown
	if ( val == 6 ) continue;
	if ( int(val) % 3 != 0 ) continue;
      }
      if ( ! title ) {
	std::cout << " Channels failing \"" << me->getName() << "\""
		  << " (" << hi->GetName() << ") "
		  << std::endl << std::endl;
	title = true;
      }
      std::cout << " ("
		<< hi->GetXaxis()->GetBinUpEdge(jx)
		<< ", "
		<< hi->GetYaxis()->GetBinUpEdge(jy);
      if ( hj )
        std::cout << ", "
                  << hj->GetBinEntries(hj->GetBin(jx, jy));
      std::cout << ") = "
		<< hi->GetBinContent(hi->GetBin(jx, jy))
		<< " +- "
		<< hi->GetBinError(hi->GetBin(jx, jy))
		<< std::endl;
    }
  }
  if ( title ) std::cout << std::endl;
  return;
}

bool
UtilsClient::getBinStatistics( TH1* histo, const int ix, const int iy, float& num, float& mean, float& rms, float minEntries )
{
  num  = -1.; mean = -1.; rms  = -1.;

  if ( !histo ) return false;
 
  // TProfile2D does not inherit from TProfile; need two pointers
 
  TProfile *p = NULL;
  TProfile2D *p2 = NULL;
 
  int bin = histo->GetBin(ix, iy);
  char o;
 
  TClass *cl = histo->IsA();
  if( cl == TClass::GetClass("TProfile") ){
    p = dynamic_cast<TProfile *>(histo);
    num = p->GetBinEntries(bin);
    o = *( p->GetErrorOption() );
  }else if( cl == TClass::GetClass("TProfile2D") ){
    p2 = dynamic_cast<TProfile2D *>(histo);
    num = p2->GetBinEntries(bin);
    o = *( p2->GetErrorOption() );
  }else
    return false;
 
  if ( num < minEntries ) return false;
 
  mean = histo->GetBinContent(ix, iy);
  if( o == 's' )
    rms  = histo->GetBinError(ix, iy);
  else if( o == '\0' )
    rms = histo->GetBinError(ix, iy) * std::sqrt( num );
  // currently not compatible with other error options!!
 
  return true;
}
 
/*! \fn static bool getTTStatistics( TH1 *histo, const int ix, const int iy, float &num, float &mean, float &rms, float minEntries )
  \brief Returns true if the TT contains good statistical data. TT is taken as 5x5 block that contains (ix, iy)
  \param (ix, iy) crystal coordinate. For TProfile iy is ignored
  CURRENTLY ONLY FOR EB per-supermodule plots
*/
// static bool getTTStatistics(TH1 *histo, const int ix, const int iy, float &num, float &mean, float &rms, float minEntries = 1.)
// {
//   num = -1.; mean = -1.; rms = -1.;
 
//   if( !histo ) return false;
 
//   TProfile *p = NULL;
//   TProfile2D *p2 = NULL;
 
//   std::vector<int> bins;
//   std::vector<float> entries;
//   char o;
 
//   TClass *cl = histo->IsA();
//   if( cl == TClass::GetClass("TProfile") ){
 
//     int ic = histo->GetBin(ix);
//     p = static_cast<TProfile *>(histo);
 
//     o = *( p->GetErrorOption() );
 
//     int je = ((ic-1) / 20) / 5;
//     int jp = ((ic-1) % 20) / 5;
 
//     int bin;
//     for(int i = 0; i < 5; i++){
//       for(int j = 1; j <= 5; j++){
// 	bin = (je * 5 + i) * 20 + (jp-1) * 5 + j;
// 	bins.push_back( bin );
// 	entries.push_back( p->GetBinEntries( bin ) );
//       }
//     }
 
//   }else if( cl == TClass::GetClass("TProfile2D") ){
 
//     p2 = static_cast<TProfile2D *>(histo);
 
//     o = *( p2->GetErrorOption() );
 
//     int je = (ix-1) / 5;
//     int jp = (iy-1) / 5;
 
//     int bin;
//     for(int i = 1; i <= 5; i++){
//       for(int j = 1; j <= 5; j++){
// 	bin = p2->GetBin( je*5+i, jp*5+j );
// 	bins.push_back( bin );
// 	entries.push_back( p2->GetBinEntries( bin ) );
//       }
//     }
 
//   }else
//     return false;
 
//   num = 0.;
//   float tmpm, tmpr;
//   tmpm = tmpr = 0.;
 
//   int bin;
//   float ent;
//   float cont, err;
//   for(unsigned u = 0; u < bins.size(); u++){
//     bin = bins[u];
//     ent = entries[u];
 
//     num += ent;
//     tmpm += ( cont = histo->GetBinContent( bin ) ) * ent;
//     if(o == 's')
//       err = histo->GetBinError( bin );
//     else if(o == '\0')
//       err = histo->GetBinError( bin ) * std::sqrt( entries[u] );
//     tmpr += ( err * err + cont * cont ) * ent; // sumw2 of the bin
//   }
 
//   if( num < minEntries ) return false;
 
//   mean = tmpm / num;
//   rms = std::sqrt( tmpr / num - mean * mean );
 
//   return true;

// }

bool
UtilsClient::getBinQuality( const MonitorElement* me, const int ix, const int iy )
{
  if ( me ) {
    float val = me->getBinContent(ix, iy);
    //  0/3 = red/dark red
    //  1/4 = green/dark green
    //  2/5 = yellow/dark yellow
    //  6   = unknown
    if ( val == 0. || val == 2 || val == 6 ) return false;
    if ( val == 1. || val == 3 || val == 4 || val == 5 ) return true;
  }
  return false;
}

bool
UtilsClient::getBinStatus( const MonitorElement* me, const int ix, const int iy )
{
  if ( me ) {
    float val = me->getBinContent(ix, iy);
    //  0/3 = red/dark red
    //  1/4 = green/dark green
    //  2/5 = yellow/dark yellow
    //  6   = unknown
    if ( val == 0. || val == 3 ) return true;
    return false;
  }
  return false;
}

void
UtilsClient::maskBinContent( const MonitorElement* me, const int ix, const int iy )
{
  if ( me ) {
    float val = me->getBinContent(ix, iy);
    //  0/3 = red/dark red
    //  1/4 = green/dark green
    //  2/5 = yellow/dark yellow
    //  6   = unknown
    if ( val >= 0. && val <= 2. ) {
      const_cast<MonitorElement*>(me)->setBinContent(ix, iy, val+3);
    }
  }
}

int
UtilsClient::getFirstNonEmptyChannel( const TProfile2D* histo )
{
  if ( histo ) {
    int ichannel = 1;
    while ( ichannel <= histo->GetNbinsX() ) {
      double counts = histo->GetBinContent(ichannel, 1);
      if ( counts > 0 ) return( ichannel );
      ichannel++;
    }
  }
  return( 1 );
}
