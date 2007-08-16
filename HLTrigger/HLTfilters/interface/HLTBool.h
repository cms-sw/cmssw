#ifndef HLTBool_h
#define HLTBool_h

/** \class HLTBool
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) returning always the same
 *  configurable Boolean value (good for tests)
 *
 *  $Date: 2007/07/12 08:50:55 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTBool : public HLTFilter {

  public:

    explicit HLTBool(const edm::ParameterSet&);
    ~HLTBool();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:

    /// Boolean result
    bool result_;

};

#endif //HLTBool_h
