#include "DQM/EcalCommon/interface/UtilsClient.h"

#include <string>

#include "TProfile2D.h"

#include "DQMServices/Core/interface/MonitorElement.h"

template<class T>
T
getHisto( const MonitorElement* me, bool clone = false, T ret = 0)
{
  if( me ) {
    TObject* ob = const_cast<MonitorElement*>(me)->getRootObject();
    if( ob ) { 
      if( clone ) {
	if( ret ) {
	  delete ret;
	}
	std::string s = "ME " + me->getName();
	ret = dynamic_cast<T>(ob->Clone(s.c_str())); 
	if( ret ) {
	  ret->SetDirectory(0);
	}
      } else {
	ret = dynamic_cast<T>(ob); 
      }
    } else {
      ret = 0;
    }
  } else {
    if( !clone ) {
      ret = 0;
    }
  }
  return ret;
}

template<class T>
void
printBadChannels( const MonitorElement* me, const T* hi, bool positive_only = false )
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
  TProfile2D* hj = dynamic_cast<TProfile2D*>(const_cast<T*>(hi));
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

template<class T>
bool
getBinStatistics( const T* histo, const int ix, const int iy, float& num, float& mean, float& rms )
{
  num  = -1.; mean = -1.; rms  = -1.;
  float n_min_bin = 1.;

  if ( histo ) {
    num = histo->GetBinEntries(histo->GetBin(ix, iy));
    if ( num >= n_min_bin ) {
      mean = histo->GetBinContent(ix, iy);
      rms  = histo->GetBinError(ix, iy);
      return true;
    }
  }
  return false;
}

bool
getBinQuality( const MonitorElement* me, const int ix, const int iy )
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
getBinStatus( const MonitorElement* me, const int ix, const int iy )
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
maskBinContent( const MonitorElement* me, const int ix, const int iy )
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
getFirstNonEmptyChannel( const TProfile2D* histo )
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
