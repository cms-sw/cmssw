/**
 *   
 * \author G. Franzoni
 *
 */


#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/Common/interface/Handle.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <fstream>

#include <iomanip> 


class EcalHexDisplay: public edm::EDAnalyzer {

  public:
    EcalHexDisplay(const edm::ParameterSet& ps);

  protected:
    int      verbosity_;
    int      beg_fed_id_;
    int      end_fed_id_;
    int      first_event_;
    int      last_event_;
    int      event_;
    bool     writeDcc_;
    std::string   filename_;

    void analyze(const edm::Event & e, const  edm::EventSetup& c);

  private:
    edm::InputTag fedRawDataCollectionTag_;
};
