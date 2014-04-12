#ifndef DataFormats_SiPixelRecHitQuality_h
#define DataFormats_SiPixelRecHitQuality_h 1

//--- pow():
#include <cmath>

//--- &&& I'm not sure why we need this. [Petar]
#include <utility>


#include "FWCore/MessageLogger/interface/MessageLogger.h"


class SiPixelRecHitQuality {
 public:
  typedef unsigned int QualWordType;
  
  
 public:

  class Packing {
  public:
    
    // Constructor: pre-computes masks and shifts from field widths
    Packing();

  public:
    QualWordType  probX_mask;
    int           probX_shift;
    float         probX_units;
    double        probX_1_over_log_units;
    char          probX_width;
    //
    QualWordType  probY_mask;
    int           probY_shift;
    float         probY_units;
    double        probY_1_over_log_units;
    char          probY_width;
    //
    QualWordType  qBin_mask;
    int           qBin_shift;
    char          qBin_width;
    //
    QualWordType  edge_mask;
    int           edge_shift;
    char          edge_width;
    //
    QualWordType  bad_mask;
    int           bad_shift;
    char          bad_width;
    //
    QualWordType  twoROC_mask;
    int           twoROC_shift;
    char          twoROC_width;
    //
    QualWordType  hasFilledProb_mask;
    int           hasFilledProb_shift;
    char          hasFilledProb_width;
    
    char spare_width;
    
    //--- Template fit probability, in X and Y directions
    //    To pack: int raw = - log(prob)/log(prob_units)
    //    Unpack : prob = prob_units^{-raw}
    //
    //--- We've obsoleted probX and probY in favor of probXY and probQ as of 09/10
    inline float probabilityX( QualWordType qualWord ) const {
      edm::LogWarning("ObsoleteVariable") << "Since 39x, probabilityX and probabilityY have been replaced by probabilityXY and probabilityQ";
      return -10;
    }
    inline float probabilityY( QualWordType qualWord ) const {
      edm::LogWarning("ObsoleteVariable") << "Since 39x, probabilityX and probabilityY have been replaced by probabilityXY and probabilityQ";
      return -10;
    }
    
    inline float probabilityXY( QualWordType qualWord ) const     {
      int raw = (qualWord >> probX_shift) & probX_mask;
      if(raw<0 || raw >16383) {
        edm::LogWarning("OutOfBounds") << "Probability XY outside the bounds of the quality word. Defaulting to Prob=0. Raw = " << raw << " QualityWord = " << qualWord;
        raw = 16383;
      }
      float prob = (raw==16383) ? 0: pow( probX_units, (float)( -raw) );
      // cout << "Bits = " << raw << " --> Prob = " << prob << endl;
      return prob;
    }
    inline float probabilityQ( QualWordType qualWord ) const     {
      int raw = (qualWord >> probY_shift) & probY_mask;
      if(raw<0 || raw >255) {
        edm::LogWarning("OutOfBounds") << "Probability Q outside the bounds of the quality word. Defaulting to Prob=0. Raw = " << raw << " QualityWord = " << qualWord;
        raw = 255;
      }
      float prob = (raw==255) ? 0 : pow( probY_units, (float)( -raw) );
      // cout << "Bits = " << raw << " --> Prob = " << prob << endl;
      return prob;
    }
    //
    //--- Charge `bin' (0,1,2,3 ~ charge, qBin==4 is a new minimum charge state, qBin=5 is unphysical, qBin6,7 = unused)
    inline int qBin( QualWordType qualWord ) const     {
      int qbin = (qualWord >> qBin_shift) & qBin_mask;
      if(qbin<0 || qbin >7) {
        edm::LogWarning("OutOfBounds") << "Qbin outside the bounds of the quality word. Defaulting to Qbin=0. Qbin = " << qbin << " QualityWord = " << qualWord;
        qbin=0;
      }
      return qbin;
    }
    //--- Quality flags (true or false):
    //--- cluster is on the edge of the module, or straddles a dead ROC
    inline bool isOnEdge( QualWordType qualWord ) const     {
      return (qualWord >> edge_shift) & edge_mask;
    }
    //--- cluster contains bad pixels, or straddles bad column or two-column
    inline bool hasBadPixels( QualWordType qualWord ) const     {
      return (qualWord >> bad_shift) & bad_mask;
    }
    //--- the cluster spans two ROCS (and thus contains big pixels)
    inline bool spansTwoROCs( QualWordType qualWord ) const     {
      return (qualWord >> twoROC_shift) & twoROC_mask;
    }
    //--- the probability is filled
    inline bool hasFilledProb( QualWordType qualWord ) const {
      return (qualWord >> hasFilledProb_shift) & hasFilledProb_mask;
    }
    
    //------------------------------------------------------
    //--- Setters: the inverse of the above.
    //------------------------------------------------------
    //
    inline void setProbabilityXY( float prob, QualWordType & qualWord ) const {
      if(prob<0 || prob>1) {
        edm::LogWarning("OutOfBounds") << "Prob XY outside the bounds of the quality word. Defaulting to Prob=0. Prob = " << prob << " QualityWord = " << qualWord;
        prob=0;
      }
      double draw = (prob<=1.6E-13) ? 16383 : - log( (double) prob ) * probX_1_over_log_units;
      unsigned int raw = (int) (draw+0.5);   // convert to integer, round correctly
      // cout << "Prob = " << prob << " --> Bits = " << raw << endl;
      qualWord |= ((raw & probX_mask) << probX_shift);
    }
    inline void setProbabilityQ( float prob, QualWordType & qualWord ) const {
      if(prob<0 || prob>1) {
        edm::LogWarning("OutOfBounds") << "Prob Q outside the bounds of the quality word. Defaulting to Prob=0. Prob = " << prob << " QualityWord = " << qualWord;
        prob=0;
      }
      double draw = (prob<=1E-5) ? 255 : - log( (double) prob ) * probY_1_over_log_units;
      unsigned int raw = (int) (draw+0.5);   // convert to integer, round correctly
      // cout << "Prob = " << prob << " --> Bits = " << raw << endl;
      qualWord |= ((raw & probY_mask) << probY_shift);
    }
    
    
    inline void setQBin( int qbin, QualWordType & qualWord ) const {
      if(qbin<0 || qbin >7) {
        edm::LogWarning("OutOfBounds") << "Qbin outside the bounds of the quality word. Defaulting to Qbin=0. Qbin = " << qbin << " QualityWord = " << qualWord;
        qbin=0;
      }
      qualWord |= ((qbin & qBin_mask) << qBin_shift);
    }
    
    inline void setIsOnEdge( bool flag, QualWordType & qualWord ) const {
      qualWord |= ((flag & edge_mask) << edge_shift);
    }
    inline void setHasBadPixels( bool flag, QualWordType & qualWord ) const {
      qualWord |= ((flag & bad_mask) << bad_shift);
    }
    inline void setSpansTwoROCs( bool flag, QualWordType & qualWord ) const {
      qualWord |= ((flag & twoROC_mask) << twoROC_shift);
    }
    inline void setHasFilledProb( bool flag, QualWordType & qualWord ) const {
      qualWord |= ((flag & hasFilledProb_mask) << hasFilledProb_shift);
    }
  };
  
public:
  static const Packing   thePacking;
};  

#endif
