#ifndef UtilsClient_H
#define UtilsClient_H

/*!
  \file UtilsClient.h
  \brief Ecal Monitor Utils for Client
  \author B. Gobbo 
  \version $Revision: 1.22 $
  \date $Date: 2011/08/30 09:45:43 $
*/

/*! \class UtilsClient
    \brief Utilities for Ecal Monitor Client 
 */

#include "DQMServices/Core/interface/MonitorElement.h"
#include "TObject.h"

class TH1;
class TProfile;
class TProfile2D;
class TClass;

class UtilsClient {

 public:

  /*! \fn template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0 )
      \brief Returns the histogram contained by the Monitor Element
      \param me Monitor Element
      \param clone (boolean) if true clone the histogram 
      \param ret in case of clonation delete the histogram first
   */
  template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0);

  /*! \fn static void printBadChannels( const MonitorElement* me, const T* hi, bool positive_only = false )
      \brief Print the bad channels
      \param me monitor element
      \param hi histogram
      \param positive_only enable logging of channels with positive content, only
   */
  static void printBadChannels( const MonitorElement* me, TH1* hi, bool positive_only = false );

  /*! \fn static bool getBinStatistics( const T* histo, const int ix, const int iy, float& num, float& mean, float& rms )
      \brief Returns true if the bin contains good statistical data
      \param histo input ROOT histogram
      \param (ix, iy) input histogram's bin
      \param num bin's entries
      \param mean bins' mean
      \param rms bin's rms
   */
  static bool getBinStatistics( TH1* histo, const int ix, const int iy, float& num, float& mean, float& rms, float minEntries = 1. );

  /*! \fn static bool getBinQuality( const MonitorElement* me, const int ix, const int iy )
      \brief Returns true if the bin quality is good or masked
      \param me input histogram
      \param (ix, iy) input histogram's bins
   */
  static bool getBinQuality( const MonitorElement* me, const int ix, const int iy );

  /*! \fn static bool getBinStatus( const MonitorElement* me, const int ix, const int iy )
      \brief Returns true if the bin status is red/dark red
      \param me input histogram
      \param (ix, iy) input histogram's bins
   */
  static bool getBinStatus( const MonitorElement* me, const int ix, const int iy );

  /*! static void maskBinContent( const MonitorElement* me, const int ix, const int iy )
      \brief Mask the bin content
      \param histo input histogram
      \param (ix, iy) input histogram's bins
   */
  static void maskBinContent( const MonitorElement* me, const int ix, const int iy );

  /*! static int getFirstNonEmptyChannel( const TProfile2D* histo )
      \brief Find the first non empty bin
      \param histo input ROOT histogram
   */
  static int getFirstNonEmptyChannel( const TProfile2D* histo );

 private:

  UtilsClient() {}; // Hidden to force static use
  ~UtilsClient() {}; // Hidden to force static use

};

template<class T>
inline
T
UtilsClient::getHisto( const MonitorElement* me, bool clone, T ret)
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

#endif
