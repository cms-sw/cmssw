#ifndef JetAlgorithms_KtRecomInterface_h
#define JetAlgorithms_KtRecomInterface_h


#ifndef STD_STRING_H
#include <string>
#define STD_STRING_H
#endif

class HepLorentzVectror;

namespace KtJet {
  class KtLorentzVector;
/**
 *  Interface class to combine 4-momenta
 
 @author J.Butterworth J.Couchman B.Cox B.Waugh
*/
  class KtRecom {
  public:
    /** virtual destructor needed */
    virtual ~KtRecom() {}
    /** Return merged 4-momentum */
    virtual HepLorentzVector operator()(const HepLorentzVector &, const HepLorentzVector &) const = 0;
    /** Process input 4-momentum */
    virtual KtLorentzVector operator()(const KtLorentzVector &) const = 0;
    /** Name of scheme */
    virtual std::string name() const = 0;
  };
  
}

#endif //end of namespace
