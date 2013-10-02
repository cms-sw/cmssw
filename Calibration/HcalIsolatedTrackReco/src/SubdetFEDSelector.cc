
#include "Calibration/HcalIsolatedTrackReco/interface/SubdetFEDSelector.h"

SubdetFEDSelector::SubdetFEDSelector(const edm::ParameterSet& iConfig)
{
  getEcal_=iConfig.getParameter<bool>("getECAL");
  getStrip_=iConfig.getParameter<bool>("getSiStrip");
  getPixel_=iConfig.getParameter<bool>("getSiPixel");
  getHcal_=iConfig.getParameter<bool>("getHCAL");
  getMuon_=iConfig.getParameter<bool>("getMuon");
  getTrigger_=iConfig.getParameter<bool>("getTrigger");
  
  tok_raw_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("rawInputLabel"));

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
  iEvent.getByToken(tok_raw_,rawIn);
 
  std::vector<int> selFEDs;

  if (getEcal_)
    {
      for (int i=FEDNumbering::MINECALFEDID; i<=FEDNumbering::MAXECALFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::MINPreShowerFEDID; i<=FEDNumbering::MAXPreShowerFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
    }

  if (getMuon_)
    {
      for (int i=FEDNumbering::MINCSCFEDID; i<=FEDNumbering::MAXCSCFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
  for (int i=FEDNumbering::MINCSCTFFEDID; i<=FEDNumbering::MAXCSCTFFEDID; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::MINDTFEDID; i<=FEDNumbering::MAXDTFEDID; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::MINDTTFFEDID; i<=FEDNumbering::MAXDTTFFEDID; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::MINRPCFEDID; i<=FEDNumbering::MAXRPCFEDID; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::MINCSCDDUFEDID; i<=FEDNumbering::MAXCSCDDUFEDID; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::MINCSCContingencyFEDID; i<=FEDNumbering::MAXCSCContingencyFEDID; i++)
        {
	  selFEDs.push_back(i);
        }
  for (int i=FEDNumbering::MINCSCTFSPFEDID; i<=FEDNumbering::MAXCSCTFSPFEDID; i++)
        {
	  selFEDs.push_back(i);
        }
    }


  if (getHcal_)
    {
      for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
    }

  
  if (getStrip_)
    {
      for (int i=FEDNumbering::MINSiStripFEDID;  i<=FEDNumbering::MAXSiStripFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
    }

  
  if (getPixel_)
    {
      for (int i=FEDNumbering::MINSiPixelFEDID;  i<=FEDNumbering::MAXSiPixelFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
    }
  
  if (getTrigger_)
    {
      for (int i=FEDNumbering::MINTriggerEGTPFEDID;  i<=FEDNumbering::MAXTriggerEGTPFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::MINTriggerGTPFEDID;  i<=FEDNumbering::MAXTriggerGTPFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::MINTriggerLTCFEDID;  i<=FEDNumbering::MAXTriggerLTCFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::MINTriggerLTCmtccFEDID;  i<=FEDNumbering::MAXTriggerLTCmtccFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
      for (int i=FEDNumbering::MINTriggerGCTFEDID;  i<=FEDNumbering::MAXTriggerGCTFEDID; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::MINTriggerLTCTriggerFEDID;  i<=FEDNumbering::MAXTriggerLTCTriggerFEDID; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::MINTriggerLTCHCALFEDID;  i<=FEDNumbering::MAXTriggerLTCHCALFEDID; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::MINTriggerLTCSiStripFEDID;  i<=FEDNumbering::MAXTriggerLTCSiStripFEDID; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::MINTriggerLTCECALFEDID;  i<=FEDNumbering::MAXTriggerLTCECALFEDID; i++)
        {
          selFEDs.push_back(i);
        }

      for (int i=FEDNumbering::MINTriggerLTCTotemCastorFEDID;  i<=FEDNumbering::MAXTriggerLTCTotemCastorFEDID; i++)
        {
          selFEDs.push_back(i);
        }
      for (int i=FEDNumbering::MINTriggerLTCRPCFEDID;  i<=FEDNumbering::MAXTriggerLTCRPCFEDID; i++)
        {
          selFEDs.push_back(i);
        }
      
      for (int i=FEDNumbering::MINTriggerLTCCSCFEDID;  i<=FEDNumbering::MAXTriggerLTCCSCFEDID; i++)
        {
          selFEDs.push_back(i);
        }
      for (int i=FEDNumbering::MINTriggerLTCDTFEDID;  i<=FEDNumbering::MAXTriggerLTCDTFEDID; i++)
        {
          selFEDs.push_back(i);
        }
      for (int i=FEDNumbering::MINTriggerLTCSiPixelFEDID;  i<=FEDNumbering::MAXTriggerLTCSiPixelFEDID; i++)
        {
          selFEDs.push_back(i);
        }

    }
  
  for (int i=FEDNumbering::MINDAQeFEDFEDID;  i<=FEDNumbering::MAXDAQeFEDFEDID; i++)
    {
      selFEDs.push_back(i);
    }
  

  // Copying:
  const FEDRawDataCollection *rdc=rawIn.product();
  
  //   if ( ( rawData[i].provenance()->processName() != e.processHistory().rbegin()->processName() ) )
  //       continue ; // skip all raw collections not produced by the current process
  
  for ( int j=0; j< FEDNumbering::MAXFEDID; ++j ) 
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
void SubdetFEDSelector::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void SubdetFEDSelector::endJob() {
}
