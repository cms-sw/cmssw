#ifndef JetAlgorithms_KtDistanceInterface_h
#define JetAlgorithms_KtDistanceInterface_h

#ifndef STD_STRING_H
#include <string>
#define STD_STRING_H
#endif
#include "RecoJets/JetAlgorithms/interface/KtUtil.h"

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
