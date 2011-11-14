#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include <DataFormats/Common/interface/DetSetVector.h>
#include <FWCore/Framework/interface/Event.h> 

// Forward declarations 
class TFile;
class TTree;

class RawDataConverter : public edm::EDAnalyzer {
  
 public:
  explicit RawDataConverter(const edm::ParameterSet&);
  ~RawDataConverter();
  
  
 private:
  enum DigiType {ZeroSuppressed, VirginRaw, ProcessedRaw, Unknown};
  virtual void beginJob() ;
  virtual void beginRun(edm::Run const &, edm::EventSetup const &) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void fillDetectorId( void );
  void ClearData( void );
  DigiType GetValidLabels( const edm::Event& iEvent ); // Check what kind of file is being processed and get valid module and instance labels returns the type of Digis that was found

  template <class T>
  void GetDigis(const edm::Event&); // Copy the Digis into the local container (theData)

  std::vector<std::string> theDigiModuleLabels;
  std::vector<std::string> theProductInstanceLabels;

  std::string CurrentModuleLabel;
  std::string CurrentInstanceLabel;

  TFile* theOutputFile;
  TTree* theOutputTree;
  LASGlobalData<std::vector<float> > theData;

  int latency;
  int eventnumber;
  int runnumber;
  int lumiBlock;
  LASGlobalData<int> detectorId;
  
};


// Copy the Digis into the local container (theData)
// Currently this has only been implemented and tested for SiStripDigis
// SiStripRawDigis and SiStripProcessedRawDigis will need some changes to work (this is the final goal)
template <class Digitype>
void RawDataConverter::GetDigis( const edm::Event& iEvent)
{
  LogDebug("RawDataConverter") << "Fill ZeroSuppressed Digis into the Tree";

  // Get the DetSetVector for the SiStripDigis 
  // This is a vector with all the modules, each module containing zero or more strips with signal (Digis)
  edm::Handle< edm::DetSetVector< Digitype > > detSetVector;  // Handle for holding the DetSetVector
  iEvent.getByLabel( CurrentModuleLabel , CurrentInstanceLabel , detSetVector );
  if( ! detSetVector.isValid() ) throw std::runtime_error("Could not find the Digis");

  // set everything in the local container to zero
  ClearData();
  
  // Fill the Digis into the Raw Data Container

  LASGlobalLoop loop;  // loop helper
  int det, ring, beam, disk, pos; // and its variables

  // loop over TEC+- (internal) modules
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    // Find the module in the DetSetVector and get a pointer (iterator) to it
    typename edm::DetSetVector< Digitype >::const_iterator theModule = detSetVector->find( detectorId.GetTECEntry( det, ring, beam, disk ) );

    if ( theModule != detSetVector->end() ) {
      // loop over all the Digis in this Module
      typename edm::DetSet< Digitype >::const_iterator theDigi;
      for (theDigi = theModule->data.begin(); theDigi != theModule->data.end(); ++theDigi ) {
	// fill the number of adc counts into the local container
	if ( theDigi->channel() < 512 ) theData.GetTECEntry( det, ring, beam, disk ).at( theDigi->channel() ) = theDigi->adc();
      }
    }
  } while( loop.TECLoop( det, ring, beam, disk ) );

  // loop TIB/TOB
  det = 2; beam = 0; pos = 0; // <- set det = 2 (TIB)
  do {
    // Find the module in the DetSetVector and get a pointer (iterator) to it
    typename edm::DetSetVector< Digitype >::const_iterator theModule = detSetVector->find( detectorId.GetTIBTOBEntry( det, beam, pos ) );

    if ( theModule != detSetVector->end() ) {
      // loop over all the Digis in this Module
      typename edm::DetSet< Digitype >::const_iterator theDigi;
      for (theDigi = theModule->data.begin(); theDigi != theModule->data.end(); ++theDigi ) {
	// fill the number of adc counts into the local container
	if ( theDigi->channel() < 512 ) theData.GetTIBTOBEntry( det, beam, pos ).at( theDigi->channel() ) = theDigi->adc();
      }
    }
  } while( loop.TIBTOBLoop( det, beam, pos ) );


  // loop TEC (AT)
  det = 0; beam = 0; disk = 0;
  do {
    // Find the module in the DetSetVector and get a pointer (iterator) to it
    typename edm::DetSetVector< Digitype >::const_iterator theModule = detSetVector->find( detectorId.GetTEC2TECEntry( det, beam, disk ) );

    if ( theModule != detSetVector->end() ) {
      // loop over all the Digis in this Module
      typename edm::DetSet< Digitype >::const_iterator theDigi;
      for (theDigi = theModule->data.begin(); theDigi != theModule->data.end(); ++theDigi ) {
	// fill the number of adc counts into the local container
	if ( theDigi->channel() < 512 ) theData.GetTEC2TECEntry( det, beam, disk ).at( theDigi->channel() ) = theDigi->adc();
      }
    }
  } while( loop.TEC2TECLoop( det, beam, disk ) );
}

