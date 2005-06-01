#ifndef KTJET_KTDISTANCEINTERFACE_H
#define KTJET_KTDISTANCEINTERFACE_H

#ifndef STD_STRING_H
#include <string>
#define STD_STRING_H
#endif
#ifndef KTJET_KTUTIL_H
#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#define KTJET_KTUTIL_H
#endif

namespace KtJet {
/**
 \class KtDistance  
 \brief Interface class to calculate Kt for jets and pairs.  
 
 @author J.Butterworth J.Couchman B.Cox B.Waugh
*/
  class KtLorentzVector;
  class KtDistance {
  public:
    /** virtual destructor needed */
    virtual ~KtDistance() {}
    /** Jet Kt */
    virtual KtFloat operator()(const KtLorentzVector &) const = 0;
    /** Pair Kt */
    virtual KtFloat operator()(const KtLorentzVector &, const KtLorentzVector &) const = 0;
    /** Name of scheme */
    virtual std::string name() const = 0;
  };
  
}// end of namespace

#endif
