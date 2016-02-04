#ifndef HLTBool_h
#define HLTBool_h

/** \class HLTBool
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) returning always the same
 *  configurable Boolean value (good for tests)
 *
 *  $Date: 2007/08/16 14:49:05 $
 *  $Revision: 1.1 $
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
