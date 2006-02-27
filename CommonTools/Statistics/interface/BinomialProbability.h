#ifndef BinomialProbability_H
#define BinomialProbability_H

/** A simple class for accumulating binomial "events",
 *  i.e. events that have a yes/no outcome,
 *  and for computing the binomial error on the
 *  fraction of positive hits.
 */

class BinomialProbability {
public:

  BinomialProbability() : theHits(0), theTotal(0) {}
  
  BinomialProbability(int hits, int entries) : 
    theHits(hits), theTotal(entries) {}

  float value() const {
    return theTotal == 0 ? 0 :float(theHits) / float(theTotal);
  }
  
  float error() const {
    float p = value();
    return theTotal <= 1 ? 0 : sqrt( p*(1.f - p)/(theTotal-1));
  }
  
  int entries() const { return theTotal;}
  
  int hits() const { return theHits;}
  
  void hit() { theHits++; theTotal++;}
  
  void miss() { theTotal++;}
  
  void update( bool hit) {
    if ( hit) theHits++;
    theTotal++;
  }

private:
  
  int theHits;
  int theTotal;
  
};

#endif
