#ifndef _AlignPCLThresholds_h_
#define _AlignPCLThresholds_h_

#include "CondFormats/PCLConfig/interface/AlignPCLThreshold.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <string>
#include <vector>

using namespace std;

class AlignPCLThresholds{
 public:
  typedef map<string,AlignPCLThreshold> threshold_map;
  enum coordType {X, Y, Z, theta_X, theta_Y, theta_Z, extra_DOF, endOfTypes}; 

  AlignPCLThresholds(){}
  virtual ~AlignPCLThresholds(){}

  void setAlignPCLThreshold(string AlignableId, const AlignPCLThreshold &Threshold);
  void setAlignPCLThresholds(const int &Nrecords,const threshold_map &Thresholds);
  void setNRecords(const int &Nrecords);
                  
  const threshold_map& getThreshold_Map () const  {return m_thresholds;}
  const int& getNrecords() const {return m_nrecords;}

  AlignPCLThreshold   getAlignPCLThreshold(string AlignableId) const;
  AlignPCLThreshold & getAlignPCLThreshold(string AlignableId);
  
  float getSigCut     (string AlignableId,coordType type) const;
  float getCut        (string AlignableId,coordType type) const;
  float getMaxMoveCut (string AlignableId,coordType type) const; 
  float getMaxErrorCut(string AlignableId,coordType type) const;                     

  // overloaded methods to get all the coordinates
  array<float,6> getSigCut     (string AlignableId) const;
  array<float,6> getCut        (string AlignableId) const;
  array<float,6> getMaxMoveCut (string AlignableId) const; 
  array<float,6> getMaxErrorCut(string AlignableId) const;
  
  array<float,4> getExtraDOFCutsForAlignable(string AlignableId,const unsigned int i) const;

  double size()const {return m_thresholds.size();}
  vector<string> getAlignableList() const;

  void printAll() const;

 private:

  threshold_map m_thresholds;
  int m_nrecords;

  COND_SERIALIZABLE;

};

#endif
