// $Id: UtilsClient.h,v 1.1 2007/04/30 09:24:01 benigno Exp $

/*!
  \file UtilsClient.h
  \brief Ecal Monitor Utils for Client
  \author B. Gobbo 
  \version $Revision: 1.1 $
  \date $Date: 2007/04/30 09:24:01 $
*/

#ifndef UtilsClient_H
#define UtilsClient_H

#include <vector>
#include <string>

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"

#include "TH1.h"

/*! \class UtilsClient
    \brief Utilities for Ecal Monitor Client 
 */

class UtilsClient {

 public:

  /*! \fn template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret )
      \brief Returns the histogram contained by the Monitor Element
      \param me Monitor Element.
      \param clone (boolean) if true clone the histogram. 
      \param ret in case of clonation delete the histogram first.
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

  /*! \fn static void resetHisto( const MonitorElement* me )
      \brief Reset the ROOT object contained by the monitoring element
      \param me input Monitor Element.
   */
  static void resetHisto( const MonitorElement* me );

  /*! \fn template<class T> static void printBadChannels( const T* qth )
      \brief Print the bad channels associated to the quality test
      \param qth input QCriterionRoot.
   */
  template<class T> static void printBadChannels( const T* qth ) {
    std::vector<dqm::me_util::Channel> badChannels;
    if ( qth ) badChannels = qth->getBadChannels();
    if ( ! badChannels.empty() ) {
      std::cout << std::endl;
      std::cout << " Channels that failed \""
		<< qth->getName() << "\" "
		<< "(Algorithm: "
		<< (const_cast<T*>(qth))->getAlgoName()
		<< ")" << std::endl;
      std::cout << std::endl;
      for ( std::vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
	std::cout << " (" << it->getBinX()
		  << ", " << it->getBinY()
		  << ", " << it->getBinZ()
		  << ") = " << it->getContents()
		  << " +- " << it->getRMS()
		  << std::endl;
      }
      std::cout << std::endl;
    }
  }

  /*! \fn template<class T> static bool getBinStats( const T* histo, const int ix, const int iy, float& num, float& mean, float& rms )
      \brief Returns true if the bin contains good statistical data
      \param histo input ROOT histogram.
      \param (ix, iy) input histogram's bin.
      \param num bin's entries.
      \param mean bins' mean.
      \param rms bin's rms.
   */
  template<class T> static bool getBinStats( const T* histo, const int ix, const int iy, float& num, float& mean, float& rms ) {
    num  = -1.; mean = -1.; rms  = -1.;
    float percent = 0.9; float n_min_bin = 10.;

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
      \param histo input ROOT histogram.
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

 protected:
  UtilsClient() {}

};

#endif
