#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASGlobalDataLoop.h"
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

  void fillDetectorId( const edm::ParameterSet& );
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
  unsigned int 	unixTime;

  LASGlobalData<int> detectorId;
  
  std::string output_filename;
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

  LASGlobalDataLoop loop;
  do{
    // Find the module in the DetSetVector and get a pointer (iterator) to it
    typename edm::DetSetVector< Digitype >::const_iterator theModule = detSetVector->find( loop.GetEntry<int>(detectorId) );
    if ( theModule != detSetVector->end() ) {
      // loop over all the Digis in this Module
      typename edm::DetSet< Digitype >::const_iterator theDigi;
      for (theDigi = theModule->data.begin(); theDigi != theModule->data.end(); ++theDigi ) {
	// fill the number of adc counts into the local container
	if ( theDigi->channel() < 512 ) loop.GetEntry<std::vector<float> >(theData).at( theDigi->channel() ) = theDigi->adc();
      }
    }
  }while (loop.next() );

}

