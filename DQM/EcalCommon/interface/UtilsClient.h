// $Id: UtilsClient.h,v 1.12 2008/02/15 14:28:50 dellaric Exp $

/*!
  \file UtilsClient.h
  \brief Ecal Monitor Utils for Client
  \author B. Gobbo 
  \version $Revision: 1.12 $
  \date $Date: 2008/02/15 14:28:50 $
*/

#ifndef UtilsClient_H
#define UtilsClient_H

#include <vector>
#include <string>

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"

#include "TH1.h"
#include "TProfile2D.h"

/*! \class UtilsClient
    \brief Utilities for Ecal Monitor Client 
 */

class UtilsClient {

 public:

  /*! \fn template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0 )
      \brief Returns the histogram contained by the Monitor Element
      \param me Monitor Element
      \param clone (boolean) if true clone the histogram 
      \param ret in case of clonation delete the histogram first
   */
  template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0) {
    if( me ) {
      // std::cout << "Found '" << me->getName() <<"'" << std::endl;
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
      if( ob ) { 
	if( clone ) {
	  if( ret ) {
	    delete ret;
	  }
	  std::string s = "ME " + me->getName();
	  ret = dynamic_cast<T>((ob->operator->())->Clone(s.c_str())); 
	  if( ret ) {
	    ret->SetDirectory(0);
	  }
	} else {
	  ret = dynamic_cast<T>(ob->operator->()); 
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

  /*! \fn template<class T> static void printBadChannels( const MonitorElement* me, const T* hi, positive_only = false )
      \brief Print the bad channels
      \param me monitor element
      \param h1 histogram
      \param positive_only enable logging of channels with positive content, only
   */
  template<class T> static void printBadChannels( const MonitorElement* me, const T* hi, bool positive_only = false ) {
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
          if ( int(me->getBinContent( ix, iy )) % 3 != 0 ) continue;
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

  /*! \fn template<class T> static bool getBinStats( const T* histo, const int ix, const int iy, float& num, float& mean, float& rms )
      \brief Returns true if the bin contains good statistical data
      \param histo input ROOT histogram
      \param (ix, iy) input histogram's bin
      \param num bin's entries
      \param mean bins' mean
      \param rms bin's rms
   */
  template<class T> static bool getBinStats( const T* histo, const int ix, const int iy, float& num, float& mean, float& rms ) {
    num  = -1.; mean = -1.; rms  = -1.;
    float percent = 0.01; float n_min_bin = 10.;

    if ( histo ) {
      float n_min_tot = percent * n_min_bin * histo->GetNbinsX() * histo->GetNbinsY();
      if ( histo->GetEntries() >= n_min_tot ) {
	num = histo->GetBinEntries(histo->GetBin(ix, iy));
	if ( num >= n_min_bin ) {
	  mean = histo->GetBinContent(ix, iy);
	  rms  = histo->GetBinError(ix, iy);
	  return true;
	}
      } 
    }
    return false;
  }

  /*! \fn template<class T> static bool getBinQual( const T* histo, const int ix, const int iy )
      \brief Returns true if the bin quality is good or masked
      \param histo input ROOT histogram
      \param (ix, iy) input histogram's bins
   */
  template<class T> static bool getBinQual( const T* histo, const int ix, const int iy ) {
    if ( histo ) {
      float val = histo->getBinContent(ix, iy);
      if ( val == 0. || val == 2 ) return false;
      if ( val == 1. || val >= 3 ) return true;
    }
    return false;
  }

  /*! static int getFirstNonEmptyChannel( TProfile2D* histo )
      \brief Find the first non empty bin
      \param histo input ROOT histogram
   */
  static int getFirstNonEmptyChannel( TProfile2D* histo ) {
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

 protected:
  UtilsClient() {}

};

#endif
