//  Adapted From:
//----------------
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: EffTableReader.h,v 1.1 2009/01/07 19:02:22 kalanand Exp $
//-----------------------------------------------------------------------


#ifndef EffTableReader_h
#define EffTableReader_h

#include <string>
#include <vector>

class EffTableReader {
 public:
  class Record {
  public:
    Record () : mEtaMin (0), mEtaMax (0), mEtMax(0), mEtMin(0) {}
    Record (float fEtaMin, float fEtaMax, float fEtMax, float fEtMin, const std::vector<float>& fParameters) 
      : mEtaMin (fEtaMin), mEtaMax (fEtaMax), mEtMax (fEtMax), mEtMin (fEtMin), mParameters (fParameters) {}
    Record (const std::string& fLine);
    float etaMin() const {return mEtaMin;}
    float etaMax() const {return mEtaMax;}
    float  EtMin() const {return mEtMin;}
    float  EtMax() const {return mEtMax;}
    float etaMiddle() const {return 0.5*(etaMin()+etaMax());}
    float  EtMiddle() const {return 0.5*(EtMin()+EtMax());}
    unsigned nParameters() const {return mParameters.size();}
    float parameter (unsigned fIndex) const {return mParameters [fIndex];}
    std::vector<float> parameters () const {return mParameters;}
    int operator< (const Record& other) const {return etaMin() < other.etaMin();}
  private:
    float mEtaMin;
    float mEtaMax;
    float mEtMax;
    float mEtMin;
    std::vector<float> mParameters;
  };

  EffTableReader () {}
  EffTableReader (const std::string& fFile);
  
  /// total # of bands
  unsigned size () const {return mRecords.size();}
  /// get band index for eta and Et
  int bandIndex(float fEt, float fEta) const;
  /// get record for the band 
  const Record& record (unsigned fBand) const {return mRecords[fBand];}
  private:
  std::vector <Record> mRecords;
};

#endif
