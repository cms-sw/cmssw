#ifndef DataFormats_SiPixelRecHitQuality_h
#define DataFormats_SiPixelRecHitQuality_h 1

//--- pow():
#include <math.h>

//--- assert();
#include <cassert>

//--- &&& I'm not sure why we need this. [Petar]
#include <utility>

//--- uint32_t definition:
#include <boost/cstdint.hpp>

class SiPixelRecHitQuality {
 public:
  typedef uint32_t QualWordType;
  
  
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
    inline float probabilityX( QualWordType qualWord ) const     {
      int raw = (qualWord >> probX_shift) & probX_mask;
			assert(raw>=0 && raw <=2047);
			float prob = 0;
			if   (raw=2047) prob = 0;
      else             prob = pow( probX_units, (float)( -raw)) ;
      // cout << "Bits = " << raw << " --> Prob = " << prob << endl;
      return prob;
    }
    inline float probabilityY( QualWordType qualWord ) const     {
      int raw = (qualWord >> probY_shift) & probY_mask;
			assert(raw>=0 && raw <=2047);
			float prob = 0;
			if   (raw=2047) prob = 0;
      else             prob = pow( probY_units, (float)( -raw)) ;
      // cout << "Bits = " << raw << " --> Prob = " << prob << endl;
      return prob;
    }
    //
    //--- Charge `bin' (0,1,2,3 ~ charge, qBin==4 is a new minimum charge state, qBin=5 is unphysical, qBin6,7 = unused)
    inline int qBin( QualWordType qualWord ) const     {
			int qbin = (qualWord >> qBin_shift) & qBin_mask;
			assert(qbin>=0 && qbin <=7);
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
		inline void setProbabilityX( float prob, QualWordType & qualWord ) {
			assert(prob>=0 && prob<=1);
			double draw = 0;
			if   (prob <= 9E-8) draw = 2047;
			else                draw = - log( (double) prob ) * probX_1_over_log_units;
      unsigned int raw = (int) (draw+0.5);   // convert to integer, round correctly
      // cout << "Prob = " << prob << " --> Bits = " << raw << endl;
      qualWord |= ((raw & probX_mask) << probX_shift);
    }
    inline void setProbabilityY( float prob, QualWordType & qualWord ) {
			assert(prob>=0 && prob<=1);
			double draw = 0;
			if   (prob <= 9E-8) draw = 2047;
      else                draw = - log( (double) prob ) * probY_1_over_log_units;
      unsigned int raw = (int) (draw+0.5);   // convert to integer, round correctly
      // cout << "Prob = " << prob << " --> Bits = " << raw << endl;
      qualWord |= ((raw & probY_mask) << probY_shift);
    }

    
    inline void setQBin( int qbin, QualWordType & qualWord ) {
			assert(qbin>=0 && qbin<=7);
      qualWord |= ((qbin & qBin_mask) << qBin_shift);
    }
    
    inline void setIsOnEdge( bool flag, QualWordType & qualWord ) {
      qualWord |= ((flag & edge_mask) << edge_shift);
    }
    inline void setHasBadPixels( bool flag, QualWordType & qualWord ) {
      qualWord |= ((flag & bad_mask) << bad_shift);
    }
    inline void setSpansTwoROCs( bool flag, QualWordType & qualWord ) {
      qualWord |= ((flag & twoROC_mask) << twoROC_shift);
    }
		inline void setHasFilledProb( bool flag, QualWordType & qualWord ) {
			qualWord |= ((flag & hasFilledProb_mask) << hasFilledProb_shift);
		}
	
  };


 public:
  static Packing   thePacking;
};  


#endif
