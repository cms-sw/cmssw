
//-------------------------------------------------
//
//   \class L1MuGMTPattern
/**
 *   Description:  Create GMT HW test patterns
*/
//                
//   $Date: 2008/02/05 10:31:02 $
//   $Revision: 1.1 $
//
//   I. Mikulec            HEPHY Vienna
//
//--------------------------------------------------
#ifndef L1MUGMTPATTERN_H
#define L1MUGMTPATTERN_H

//---------------
// C++ Headers --
//---------------

#include <memory>
#include <string>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "FWCore/Utilities/interface/InputTag.h"

class L1MuRegionalCand;
class L1MuGMTExtendedCand;

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTPattern : public edm::EDAnalyzer {

 
  public:

    // constructor
    explicit L1MuGMTPattern(const edm::ParameterSet&);
    virtual ~L1MuGMTPattern();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    void printRegional(std::string tag, const std::vector<L1MuRegionalCand>& rmc);
    void printGMT(std::string tag, const std::vector<L1MuGMTExtendedCand>& exc);
    void printMipIso(L1CaloRegionCollection const* regions);
    void printMI(const std::vector<unsigned>* mi);
    void printCANC();
    unsigned invertQPt(unsigned);

    virtual void beginJob();
    virtual void endJob();

  private:


    edm::InputTag m_inputTag;
    edm::InputTag m_inputCaloTag;
    std::string m_outfilename;
    int m_outputType;
      
};


#endif
