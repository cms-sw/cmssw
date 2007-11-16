//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: SimpleJetCorrectorParameters.h,v 1.1 2007/11/01 21:50:30 fedor Exp $
//
// Generic parameters for Jet corrections
//
#ifndef SimpleJetCorrectorParameters_h
#define SimpleJetCorrectorParameters_h

#include <string>
#include <vector>

class SimpleJetCorrectorParameters {
 public:
  class Record {
  public:
    Record () : mEtaMin (0), mEtaMax (0) {}
    Record (float fEtaMin, float fEtaMax, const std::vector<float>& fParameters) 
      : mEtaMin (fEtaMin), mEtaMax (fEtaMax), mParameters (fParameters) {}
    Record (const std::string& fLine);
    float etaMin() const {return mEtaMin;}
    float etaMax() const {return mEtaMax;}
    float etaMiddle() const {return 0.5*(etaMin()+etaMax());}
    unsigned nParameters() const {return mParameters.size();}
    float parameter (unsigned fIndex) const {return mParameters [fIndex];}
    std::vector<float> parameters () const {return mParameters;}
    int operator< (const Record& other) const {return etaMin() < other.etaMin();}
  private:
    float mEtaMin;
    float mEtaMax;
    std::vector<float> mParameters;
  };

  SimpleJetCorrectorParameters () {}
  SimpleJetCorrectorParameters (const std::string& fFile);
  
  /// total # of bands
  unsigned size () const {return mRecords.size();}
  /// get band index for eta
  int bandIndex (float fEta) const;
  /// get record for the band
  const Record& record (unsigned fBand) const {return mRecords[fBand];}
  /// get vector of centers of bands
  std::vector<float> bandCenters () const;
 private:
  std::vector <Record> mRecords;
};

#endif
