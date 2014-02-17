/**\class LaserAlignmentT0Producer LaserAlignmentT0Producer.cc NewAlignment/LaserAlignmentT0Producer/src/LaserAlignmentT0Producer.cc

 Description: AlCaRECO producer (TkLAS data filter) running on T0

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Olzem
//         Created:  Wed Feb 13 17:30:40 CET 2008
// $Id: LaserAlignmentT0Producer.h,v 1.3 2010/01/06 09:38:00 mussgill Exp $
//
//


// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/Common/interface/DetSetVector.h>
#include <DataFormats/SiStripDigi/interface/SiStripDigi.h>
#include <DataFormats/SiStripDigi/interface/SiStripRawDigi.h>

//
// class decleration
//

class LaserAlignmentT0Producer : public edm::EDProducer {
public:
  explicit LaserAlignmentT0Producer( const edm::ParameterSet& );
  ~LaserAlignmentT0Producer();
  
private:
  virtual void beginJob() ;
  virtual void produce( edm::Event&, const edm::EventSetup& );
  virtual void endJob();
  void FillDetIds( void );

  // container for cfg data
  std::vector<edm::ParameterSet> digiProducerList;
  std::string digiProducer;
  std::string digiLabel;
  std::string digiType;

  // this one stores the det ids for all the 434 LAS modules
  std::vector<unsigned int> theLasDetIds;

};

