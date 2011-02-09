#include <FWCore/Framework/interface/MakerMacros.h>

#include <DataFormats/Common/interface/DetSetVector.h>

#include <DataFormats/SiStripCommon/interface/SiStripEventSummary.h>
#include <DataFormats/SiStripDigi/interface/SiStripDigi.h>
#include <DataFormats/SiStripDigi/interface/SiStripRawDigi.h>
#include <DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h>
#include <FWCore/Framework/interface/EventSetup.h> 
#include <FWCore/Framework/interface/EventSetupRecord.h> 

#include <Alignment/LaserAlignment/interface/LASGlobalLoop.h>
//#include <Alignment/LaserAlignment/interface/LASGlobalDataLoop.h>

#include "TFile.h"
#include "TTree.h"
#include "TMath.h"

#include "RawDataConverter.h"

///
/// constructors and destructor
///
RawDataConverter::RawDataConverter( const edm::ParameterSet& iConfig ) :
  theOutputFile(0),
  theOutputTree(0),
  latency(-1),
  eventnumber(-1),
  runnumber(-1),
  lumiBlock(-1)
  
{
  output_filename = iConfig.getUntrackedParameter<std::string>( "OutputFileName" );
  theOutputFile = new TFile("TemporaryFile" , "RECREATE" );
  //theOutputFile = new TFile( iConfig.getUntrackedParameter<std::string>( "OutputFileName" ).c_str() , "RECREATE" );
  theOutputFile->cd();
  theOutputTree = new TTree( "lasRawDataTree", "lasRawDataTree" );
  theOutputTree->Branch( "lasRawData", &theData );
  theOutputTree->Branch( "latency", &latency, "latency/I" );
  theOutputTree->Branch( "eventnumber", &eventnumber, "eventnumber/I" );
  theOutputTree->Branch( "runnumber", &runnumber, "runnumber/I" );
  theOutputTree->Branch( "lumiblock", &lumiBlock, "lumiblock/I" );
  theOutputTree->Branch( "unixTime", &unixTime, "unixTime/i" );
  
  theDigiModuleLabels = iConfig.getParameter<std::vector<std::string> >( "DigiModuleLabels" );
  theProductInstanceLabels = iConfig.getParameter<std::vector<std::string> >( "ProductInstanceLabels" );

  fillDetectorId(iConfig);
}

///
///
///
RawDataConverter::~RawDataConverter() {
}


///
///
///
void RawDataConverter::beginJob() 
{  
}



///
///
///
void RawDataConverter::beginRun(edm::Run const & theRun, edm::EventSetup const & theEventSetup) 
{
  std::vector< edm::eventsetup::EventSetupRecordKey > oToFill;
  theEventSetup.fillAvailableRecordKeys (oToFill);
  std::ostringstream o;
  for(std::vector<edm::eventsetup::EventSetupRecordKey>::size_type i = 0; i < oToFill.size(); i++){
    o << oToFill[i].name() << "\n";
  }
  LogDebug("RawDataConverter") << "The size of EventSetup is: " << oToFill.size() << "\n" << o.str();
}



RawDataConverter::DigiType RawDataConverter::GetValidLabels( const edm::Event& iEvent ) // Check what kind of file is being processed and get valid module and instance labels
{
  // Clear the current labels
  CurrentModuleLabel = "";
  CurrentInstanceLabel = "";

  //Create handles for testing
  edm::Handle< edm::DetSetVector<SiStripDigi> > theStripDigis;
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > theStripRawDigis;
  edm::Handle< edm::DetSetVector<SiStripProcessedRawDigi> > theStripProcessedRawDigis;

  // Create stream foer debug message
  std::ostringstream search_message;
  search_message << "Searching for SiStripDigis\n";
  // Loop through Module and instance labels that were defined in the configuration
  for( std::vector<std::string>::iterator moduleLabel = theDigiModuleLabels.begin(); moduleLabel != theDigiModuleLabels.end(); ++moduleLabel ) {
    for( std::vector<std::string>::iterator instanceLabel = theProductInstanceLabels.begin(); instanceLabel != theProductInstanceLabels.end(); ++instanceLabel ) {

      search_message << "Checking for Module " << *moduleLabel << " Instance " << *instanceLabel << "\n";

      //First try ZeroSuppressed Digis
      iEvent.getByLabel( *moduleLabel , *instanceLabel , theStripDigis );
      if(theStripDigis.isValid()){
	search_message << "Found ZeroSuppressed\n";
	edm::LogInfo("RawDataConverter") << search_message.str();
	CurrentModuleLabel = *moduleLabel;
	CurrentInstanceLabel = *instanceLabel;
	return ZeroSuppressed;
      }

      // Next try VirginRaw Digis      
      iEvent.getByLabel( *moduleLabel , *instanceLabel , theStripRawDigis );
      if(theStripRawDigis.isValid()){
	search_message << "Found Raw\n";
	edm::LogInfo("RawDataConverter") << search_message.str();
	CurrentModuleLabel = *moduleLabel;
	CurrentInstanceLabel = *instanceLabel;
	return VirginRaw;
      }

      // Next try ProcessedRaw Digis      
      iEvent.getByLabel( *moduleLabel , *instanceLabel , theStripProcessedRawDigis );
      if(theStripProcessedRawDigis.isValid()){
	search_message << "Found ProcessedRaw\n";
	edm::LogInfo("RawDataConverter") << search_message.str();
	CurrentModuleLabel = *moduleLabel;
	CurrentInstanceLabel = *instanceLabel;
	return ProcessedRaw;
      }
    }
  }
  return Unknown;
}

///
///
///
void RawDataConverter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  // Determine the digi type to be used (only for the first time this methosd is called)
  static DigiType digitype = Unknown;  // Type of digis in this run
  if(digitype == Unknown) digitype = GetValidLabels( iEvent );   // Initialization of Digi Type

  //////////////////////////////////////////////////////////
  // Retrieve SiStripEventSummary produced by the digitizer
  //////////////////////////////////////////////////////////
  edm::Handle<SiStripEventSummary> summary;
  //iEvent.getByLabel( digiProducer, summary );
  iEvent.getByLabel( "siStripDigis", summary );
  latency = static_cast<int32_t>( summary->latency() );
  eventnumber = iEvent.id().event();
  runnumber = iEvent.run();
  lumiBlock = iEvent.luminosityBlock();
  edm::Timestamp timestamp = iEvent.time();
  unixTime = timestamp.unixTime();
  //edm::LogAbsolute("RawdataConverter") << " > run: " << runnumber << " event: " << eventnumber << " lumiBlock: " << lumiBlock << " latency: " << latency << std::endl;

  // Get the Digis as definef by digitype
  // Currently only ZeroSuppressed is implemented properly
  switch( digitype){
  case ZeroSuppressed:
    GetDigis<SiStripDigi>(iEvent);
    break;
  case VirginRaw:
    throw std::runtime_error("RawDataConverter is not yet able to process VirginRaw Data");
    break;
  case ProcessedRaw:
    throw std::runtime_error("RawDataConverter is not yet able to process ProcessedRaw Data");
    break;
  default:
    throw std::runtime_error("Did not find valid Module or Instance label");
  }

  // Push Container into the Tree
  theOutputTree->Fill();

  return;
}


///
/// Sort the events by ascending eventnumber  and write them to the output file
///
void RawDataConverter::endJob()
{

  Int_t nentries = (Int_t)theOutputTree->GetEntries();
  //Drawing variable eventnumber with no graphics option.
  //variable eventnumber stored in array fV1 (see TTree::Draw)
  theOutputTree->Draw("eventnumber","","goff");
  Int_t *index = new Int_t[nentries];
  //sort array containing eventnumber in decreasing order
  //The array index contains the entry numbers in decreasing order of eventnumber
  TMath::Sort(nentries,theOutputTree->GetV1(),index, kFALSE);

  TFile FinalFile(output_filename.c_str() , "RECREATE" );
  FinalFile.cd();

  TTree *tsorted = (TTree*)theOutputTree->CloneTree(0);

  for (Int_t i=0;i<nentries;i++) {
    theOutputTree->GetEntry(index[i]);
    tsorted->Fill();
  }
  tsorted->Write();
  delete [] index;
}


///
/// set all strips to zero
///
void RawDataConverter::ClearData( void ) {
  // Assign a vector filled with zeros to all module entries
  // The vector is const static to increase performance
  // Even more performant would be to have a complete data object that is filled with zero

  // Empty object to be assigned to all modules
  static const std::vector<float> zero_buffer(512,0);

  // loop helper
  LASGlobalDataLoop loop;
  do {
    loop.GetEntry<std::vector<float> >(theData) = zero_buffer;
  } while( loop.next() );
}


///
/// Fill the DetIds fron the cfg file into the LASGlobalData object called detectorId
///
void RawDataConverter::fillDetectorId( const edm::ParameterSet& iConfig)
{

  // the list of input digi products from the cfg
  std::vector<edm::ParameterSet> detid_list = iConfig.getParameter<std::vector<edm::ParameterSet> >( "DetIds" );
  
  // loop over all entries
  for ( std::vector<edm::ParameterSet>::iterator aDetIds = detid_list.begin(); aDetIds != detid_list.end(); ++aDetIds ) {
    int   det = aDetIds->getParameter<int>(   "det");
    int  ring = aDetIds->getParameter<int>(  "ring");
    int  beam = aDetIds->getParameter<int>(  "beam");
    int  zpos = aDetIds->getParameter<int>(  "zpos");
    int detid = aDetIds->getParameter<int>( "detid");
    //std::cout << "det " << det << "  ring " << ring << "  beam " << beam << "  zpos " << zpos << "  detid " << detid << std::endl;
    detectorId.GetEntry(det, ring, beam, zpos) = detid;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(RawDataConverter);
