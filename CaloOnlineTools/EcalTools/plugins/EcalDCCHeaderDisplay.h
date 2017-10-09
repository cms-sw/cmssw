/**
 * module  for displaying unpacked DCCHeader information
 *   
 * \author A. Ghezzi
 * \author S. Cooper
 * \author G. Franzoni
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <iostream>
#include <vector>



class EcalDCCHeaderDisplay: public edm::EDAnalyzer{
  
public:
  EcalDCCHeaderDisplay(const edm::ParameterSet& ps);
  
protected:

  void analyze( const edm::Event & e, const  edm::EventSetup& c);

private:
  edm::InputTag EcalDCCHeaderCollection_;

};

