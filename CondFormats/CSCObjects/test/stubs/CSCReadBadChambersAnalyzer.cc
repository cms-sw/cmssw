#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <bitset>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

namespace edmtest
{
  class CSCReadBadChambersAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  CSCReadBadChambersAnalyzer(edm::ParameterSet const& ps ) 
      : outputToFile_( ps.getParameter<bool>("outputToFile") ),
	readBadChambers_(ps.getParameter<bool>("readBadChambers") ),
	me42installed_( ps.getParameter<bool>("me42installed") ){ }

    explicit  CSCReadBadChambersAnalyzer(int i) 
    { }
    virtual ~ CSCReadBadChambersAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

    // Test code from CSCConditions

    /// did we request reading bad channel info from db?
    bool readBadChambers() const { return readBadChambers_; }
   
    /// Is CSCDetId layer or chamber in the bad list?
    bool isInBadChamber( const CSCDetId& id ) const;

  private:

    bool outputToFile_;
    bool readBadChambers_; // flag whether or not to even attempt reading bad channel info from db
    bool me42installed_; // flag whether ME42 chambers are installed in the geometry
    const CSCBadChambers* theBadChambers;

  };
  
  void CSCReadBadChambersAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;

    int counter=0;
    std::cout <<" RUN# "<<e.id().run() <<std::endl;
    std::cout <<" EVENT# "<<e.id().event() <<std::endl;
    edm::ESHandle<CSCBadChambers> pBad;
    context.get<CSCBadChambersRcd>().get(pBad);

    theBadChambers=pBad.product();

    CSCIndexer indexer; // just to build a CSCDetId from chamber index
    


    std::cout<< "Bad Chambers:" << std::endl;

    int nbad = theBadChambers->numberOfBadChambers;
    std::cout << "No. in list = " << nbad << std::endl;

    // Iterate over all chambers via their linear index

    int countbad = 0;
    int countgood = 0;

    // One more than total number of chambers
    // Last chamber is already in ME4 but could be 41 or 42
    int lastRing = 1;
    if ( me42installed_ ) lastRing = 2;
    int totalc = indexer.startChamberIndexInEndcap(2,4,lastRing) + indexer.chambersInRingOfStation(4,lastRing);

    for( int indexc = 1; indexc!=totalc; ++indexc ) {
      counter++;

      CSCDetId id = indexer.detIdFromChamberIndex( indexc ); 
      bool bbad = isInBadChamber( id );
      std::string bbads = "no";
      if ( bbad ) {
        bbads = "yes";
        ++countbad;
      }
      else {
	++countgood;
      }
      std::cout << counter << "  " << indexc << " " << id 
           << " In bad list? " << bbads << std::endl;
    }

    std::cout << "Total number of chambers      = " << counter << std::endl;
    std::cout << "Total number of good chambers = " << countgood << std::endl;
    std::cout << "Total number of bad chambers  = " << countbad << std::endl;

    std::vector<int>::const_iterator itcham;

    /*
    // Iterate over the list of bad chambers

    for( itcham=theBadChambers->chambers.begin();itcham!=theBadChambers->chambers.end(); ++itcham ){    
      counter++;
      int indexc = *itcham; // should be the linear index of a bad chamber

      CSCDetId id = indexer.detIdFromChamberIndex( indexc ); 
      bool bbad = isInBadChamber( id );
      std::string bbads = "no";
      if ( bbad ) bbads = "yes";
      std::cout << counter << "  " << indexc << " " << id 
           << " In bad list? " << bbads << std::endl;
    }
    */

    if ( outputToFile_ ) {

      std::ofstream BadChamberFile("dbBadChamber.dat",std::ios::app);

      counter = 0;
      for( itcham=theBadChambers->chambers.begin();itcham!=theBadChambers->chambers.end(); ++itcham ){    
        counter++;
        BadChamberFile << counter << "  " << *itcham << std::endl;
      }

    }

  }

  bool CSCReadBadChambersAnalyzer::isInBadChamber( const CSCDetId& id ) const {

    if ( theBadChambers->numberOfBadChambers == 0 ) return false;

    short int iri = id.ring();
    if ( iri == 4 ) iri = 1; // reset ME1A to ME11
    CSCIndexer indexer;
    int ilin = indexer.chamberIndex( id.endcap(), id.station(), iri, id.chamber() );
    std::vector<int>::const_iterator badbegin = theBadChambers->chambers.begin();
    std::vector<int>::const_iterator badend = theBadChambers->chambers.end();
    std::vector<int>::const_iterator it = std::find( badbegin, badend, ilin );
    if ( it != badend ) return true; // id is in the list of bad chambers
    else return false;
  }

  DEFINE_FWK_MODULE(CSCReadBadChambersAnalyzer);
}

