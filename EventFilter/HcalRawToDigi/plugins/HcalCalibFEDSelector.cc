// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalFEDList.h"

class HcalCalibFEDSelector : public edm::EDProducer {
public:
  HcalCalibFEDSelector(const edm::ParameterSet&);
  ~HcalCalibFEDSelector();


private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  edm::InputTag rawInLabel_ ;
  std::vector<int> extraFEDs_ ; 
  
};




HcalCalibFEDSelector::HcalCalibFEDSelector(const edm::ParameterSet& iConfig)
{
  rawInLabel_ = iConfig.getParameter<edm::InputTag>("rawInputLabel");
  extraFEDs_  = iConfig.getParameter< std::vector<int> >("extraFEDsToKeep") ; 
  produces<FEDRawDataCollection>();  
}

HcalCalibFEDSelector::~HcalCalibFEDSelector()
{
}

void
HcalCalibFEDSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<FEDRawDataCollection> producedData(new FEDRawDataCollection);

  edm::Handle<FEDRawDataCollection> rawIn;
  iEvent.getByLabel(rawInLabel_,rawIn);
 
  std::vector<int> selFEDs;

  //--- Get the list of FEDs to be kept ---//
  int calibType = -1 ; 
  for (int i=FEDNumbering::MINHCALFEDID;
       i<=FEDNumbering::MAXHCALFEDID; i++) {
    const FEDRawData& fedData = rawIn->FEDData(i) ; 
    if ( fedData.size() < 24 ) continue ; // FED is empty
    int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ; 
    if ( calibType < 0 ) {
      calibType = value ; 
    } else { 
      if ( calibType != value ) 
	edm::LogWarning("HcalCalibFEDSelector") << "Conflicting calibration types found: " 
						<< calibType << " vs. " << value
						<< ".  Staying with " << calibType ; 
    }
  }

  HcalFEDList calibFeds(calibType) ; 
  selFEDs = calibFeds.getListOfFEDs() ; 
  for (unsigned int i=0; i<extraFEDs_.size(); i++) {
    bool duplicate = false ; 
    for (unsigned int j=0; j<selFEDs.size(); j++) { 
      if (extraFEDs_.at(i) == selFEDs.at(j)) {
	duplicate = true ; 
	break ; 
      }
    }
    if ( !duplicate ) selFEDs.push_back( extraFEDs_.at(i) ) ; 
  }

  // Copying:
  const FEDRawDataCollection *rdc=rawIn.product();
  
  for ( int j=0; j< FEDNumbering::lastFEDId(); ++j ) 
    {
      bool rightFED=false;
      for (uint32_t k=0; k<selFEDs.size(); k++)
	{
	  if (j==selFEDs[k])
	   {
	     rightFED=true;
	   }
       }
     if (!rightFED) continue;
     const FEDRawData & fedData = rdc->FEDData(j);
     size_t size=fedData.size();
     
     if ( size > 0 ) 
       {
       // this fed has data -- lets copy it
	 FEDRawData & fedDataProd = producedData->FEDData(j);
	 if ( fedDataProd.size() != 0 ) {
	   continue;
	 }
	 fedDataProd.resize(size);
	 unsigned char *dataProd=fedDataProd.data();
	 const unsigned char *data=fedData.data();
	 // memcpy is at-least-as-fast as assignment and can be much faster
	 memcpy(dataProd, data, size);
       }
   }

 iEvent.put(producedData);
}


// ------------ method called once each job just before starting event loop  ------------
void HcalCalibFEDSelector::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void HcalCalibFEDSelector::endJob() {
}

DEFINE_FWK_MODULE(HcalCalibFEDSelector) ; 
