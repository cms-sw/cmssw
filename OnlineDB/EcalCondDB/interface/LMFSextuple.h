#ifndef LMFSEXTUPLE_H
#define LMFSEXTUPLE_H


/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

/**
 *   sextuple of t1, t2, t3, p1, p2, p3
 */
class LMFSextuple {
 public:
  LMFSextuple() {};
  float p[3];
  Tm    t[3];
};

#endif
