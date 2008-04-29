
#include "Calibration/HcalIsolatedTrackReco/interface/SubdetFEDSelector.h"

SubdetFEDSelector::SubdetFEDSelector(const edm::ParameterSet& iConfig)
{
  getEcal_=iConfig.getParameter<bool>("getECAL");
  getStrip_=iConfig.getParameter<bool>("getSiStrip");
  getPixel_=iConfig.getParameter<bool>("getSiPixel");
  getHcal_=iConfig.getParameter<bool>("getHCAL");
  getMuon_=iConfig.getParameter<bool>("getMuon");
  getTrigger_=iConfig.getParameter<bool>("getTrigger");
  
  rawInLabel_=iConfig.getParameter<edm::InputTag>("rawInputLabel");

  produces<FEDRawDataCollection>();
  
}

SubdetFEDSelector::~SubdetFEDSelector()
{
}

void
SubdetFEDSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<FEDRawDataCollection> producedData(new FEDRawDataCollection);

  edm::Handle<FEDRawDataCollection> rawIn;
  iEvent.getByLabel(rawInLabel_,rawIn);
 
  std::vector<int> selFEDs;

  if (getEcal_)
    {
      for (int i=FEDNumbering::getEcalFEDIds().first; i<=FEDNumbering::getEcalFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::getPreShowerFEDIds().first; i<=FEDNumbering::getPreShowerFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
    }

  if (getMuon_)
    {
      for (int i=FEDNumbering::getCSCFEDIds().first; i<=FEDNumbering::getCSCFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
  for (int i=FEDNumbering::getCSCTFFEDIds().first; i<=FEDNumbering::getCSCTFFEDIds().second; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::getDTFEDIds().first; i<=FEDNumbering::getDTFEDIds().second; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::getDTTFFEDIds().first; i<=FEDNumbering::getDTTFFEDIds().second; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::getRPCFEDIds().first; i<=FEDNumbering::getRPCFEDIds().second; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::getCSCDDUFEDIds().first; i<=FEDNumbering::getCSCDDUFEDIds().second; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::getCSCContingencyFEDIds().first; i<=FEDNumbering::getCSCContingencyFEDIds().second; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::getCSCTFSPFEDIds().first; i<=FEDNumbering::getCSCTFSPFEDIds().second; i++)
        {
	  selFEDs.push_back(i);
        }
    }


  if (getHcal_)
    {
      for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
    }

  
  if (getStrip_)
    {
      for (int i=FEDNumbering::getSiStripFEDIds().first;  i<=FEDNumbering::getSiStripFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
    }

  
  if (getPixel_)
    {
      for (int i=FEDNumbering::getSiPixelFEDIds().first;  i<=FEDNumbering::getSiPixelFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
    }
  
  if (getTrigger_)
    {
      for (int i=FEDNumbering::getTriggerEGTPFEDIds().first;  i<=FEDNumbering::getTriggerEGTPFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::getTriggerGTPFEDIds().first;  i<=FEDNumbering::getTriggerGTPFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::getTriggerLTCFEDIds().first;  i<=FEDNumbering::getTriggerLTCFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::getTriggerLTCmtccFEDIds().first;  i<=FEDNumbering::getTriggerLTCmtccFEDIds().second; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::getTriggerGCTFEDIds().first;  i<=FEDNumbering::getTriggerGCTFEDIds().second; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::getTriggerLTCTriggerFEDID().first;  i<=FEDNumbering::getTriggerLTCTriggerFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::getTriggerLTCHCALFEDID().first;  i<=FEDNumbering::getTriggerLTCHCALFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::getTriggerLTCSiStripFEDID().first;  i<=FEDNumbering::getTriggerLTCSiStripFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::getTriggerLTCECALFEDID().first;  i<=FEDNumbering::getTriggerLTCECALFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::getTriggerLTCTotemCastorFEDID().first;  i<=FEDNumbering::getTriggerLTCTotemCastorFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }
      for (int i=FEDNumbering::getTriggerLTCRPCFEDID().first;  i<=FEDNumbering::getTriggerLTCRPCFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }
      
      for (int i=FEDNumbering::getTriggerLTCCSCFEDID().first;  i<=FEDNumbering::getTriggerLTCCSCFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }
      for (int i=FEDNumbering::getTriggerLTCDTFEDID().first;  i<=FEDNumbering::getTriggerLTCDTFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }
      for (int i=FEDNumbering::getTriggerLTCSiPixelFEDID().first;  i<=FEDNumbering::getTriggerLTCSiPixelFEDID().second; i++)
        {
          selFEDs.push_back(i);
        }

    }
  
  for (int i=FEDNumbering::getDAQeFEDFEDIds().first;  i<=FEDNumbering::getDAQeFEDFEDIds().second; i++)
    {
      selFEDs.push_back(i);
    }
  

  // Copying:
  const FEDRawDataCollection *rdc=rawIn.product();
  
  //   if ( ( rawData[i].provenance()->processName() != e.processHistory().rbegin()->processName() ) )
  //       continue ; // skip all raw collections not produced by the current process
  
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
//	   std::cout << " More than one FEDRawDataCollection with data in FED ";
//	   std::cout << j << " Skipping the 2nd\n";
	   continue;
	 }
	 fedDataProd.resize(size);
	 unsigned char *dataProd=fedDataProd.data();
	 const unsigned char *data=fedData.data();
	 for ( unsigned int k=0; k<size; ++k ) {
	   dataProd[k]=data[k];
	 }
       }
   }

 iEvent.put(producedData);

}


// ------------ method called once each job just before starting event loop  ------------
void SubdetFEDSelector::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void SubdetFEDSelector::endJob() {
}
