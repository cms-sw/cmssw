/* -*- C++ -*- */
#ifndef HcalTTPData_H
#define HcalTTPData_H

#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include <vector>

/**  \class HcalTTPData
 *
 *  Interpretive class for HcalTTPData
 *  Since this class requires external specification of the length of the data, it is implemented
 *  as an interpreter, rather than a cast-able header class.
 *
 *  $Date: 2008/04/22 17:17:24 $
 *  $Revision: 1.10 $
 *  \author J. Mans - UMD
 */

class HcalTTPData : public HcalHTRData {
 public:
  static const int TTP_INPUTS = 72;
  static const int TTP_ALGOBITS = 24;

  HcalTTPData();
  HcalTTPData(int version_to_create);
  HcalTTPData(const unsigned short* data, int length);
  HcalTTPData(const HcalTTPData&);
  
  HcalTTPData& operator=(const HcalTTPData&);

  /** \brief Check for a good event
      Requires a minimum length, matching wordcount and length, not an
      empty event */
  bool check() const;

  typedef std::vector<bool> InputBits;
  typedef std::vector<bool> AlgoBits;

  void unpack(std::vector<InputBits>& ivs, std::vector<AlgoBits>& avs) const;
    
 private:
  void determineSectionLengths(int& dataWords, 
			       int& headerWords, int& trailerWords) const;
};

#endif

