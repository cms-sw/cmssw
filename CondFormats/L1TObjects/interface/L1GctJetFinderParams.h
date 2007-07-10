#ifndef L1GCTJETFINDERPARAMS_H_
#define L1GCTJETFINDERPARAMS_H_

class L1GctJetFinderParams
{
 public:

  const unsigned CENTRAL_JET_SEED;
  const unsigned FORWARD_JET_SEED;
  const unsigned TAU_JET_SEED;
  const unsigned CENTRAL_FORWARD_ETA_BOUNDARY;

  L1GctJetFinderParams(unsigned cJetSeed=1,
                       unsigned fJetSeed=1,
                       unsigned tJetSeed=1,
                       unsigned etaBoundary=7);

  ~L1GctJetFinderParams();

};

#endif /*L1GCTJETPARAMS_H_*/

