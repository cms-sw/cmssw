#ifndef DTFILTER_H
#define DTFILTER_H

/** \class DTFilter
 *
 * A filter for DT analysis
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 22/11/2007 14:07:37 CET $
 *
 * Modification:
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDFilter.h"

/* Collaborating Class Declarations */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* C++ Headers */
#include <bitset>

/* ====================================================================== */

/* Class DTFilter Interface */

class DTFilter : public edm::EDFilter{

  public:

/* Constructor */ 
    DTFilter(const edm::ParameterSet& config) ;

/* Destructor */ 
    virtual ~DTFilter() ;

/* Operations */
    virtual bool filter(edm::Event& e, const edm::EventSetup& s);

  private:
    enum LTCType { DT, CSC, RPC_W1, RPC_W2 };
    bool getLTC(LTCType) const;
    bool selectLTC(edm::Event& event) ;
    bool useLTC;
    bool LTC_RPC, LTC_DT, LTC_CSC;
    bool debug;
    std::bitset<6> LTC;

    bool doRunEvFiltering;
    int theRun;
    int theEv;

  protected:

};
#endif // DTFILTER_H

