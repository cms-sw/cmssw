/** \file LaserDQM.cc
 *  DQM Monitors for Laser Alignment System
 *
 *  $Date: 2007/12/04 23:54:44 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserDQM/plugins/LaserDQM.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h" 

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



LaserDQM::LaserDQM(edm::ParameterSet const& theConf) 
  : theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
    theSearchPhiTIB(theConf.getUntrackedParameter<double>("SearchWindowPhiTIB",0.05)),
    theSearchPhiTOB(theConf.getUntrackedParameter<double>("SearchWindowPhiTOB",0.05)),
    theSearchPhiTEC(theConf.getUntrackedParameter<double>("SearchWindowPhiTEC",0.05)),
    theSearchZTIB(theConf.getUntrackedParameter<double>("SearchWindowZTIB",1.0)),
    theSearchZTOB(theConf.getUntrackedParameter<double>("SearchWindowZTOB",1.0)),
    theDigiProducersList(theConf.getParameter<Parameters>("DigiProducersList")),
    theDQMFileName(theConf.getUntrackedParameter<std::string>("DQMFileName","testDQM.root")),
    theDaqMonitorBEI()
{
  // load the configuration from the ParameterSet  
  edm::LogInfo("LaserDQM") << "==========================================================="
			   << "\n===                Start configuration                  ==="
			   << "\n    theDebugLevel              = " << theDebugLevel
			   << "\n    theSearchPhiTIB            = " << theSearchPhiTIB
			   << "\n    theSearchPhiTOB            = " << theSearchPhiTOB
			   << "\n    theSearchPhiTEC            = " << theSearchPhiTEC
			   << "\n    theSearchZTIB              = " << theSearchZTIB
			   << "\n    theSearchZTOB              = " << theSearchZTOB
			   << "\n    DQM filename               = " << theDQMFileName
			   << "\n===========================================================";

}

LaserDQM::~LaserDQM() {}

void LaserDQM::analyze(edm::Event const& theEvent, edm::EventSetup const& theSetup) 
{
  // do the Tracker Statistics
  trackerStatistics(theEvent, theSetup);
}

void LaserDQM::beginJob(const edm::EventSetup& theSetup)
{
  // get hold of DQM Backend interface
  theDaqMonitorBEI = edm::Service<DaqMonitorBEInterface>().operator->();
      
  // initialize the Monitor Elements
  initMonitors();
}

void LaserDQM::endJob(void)
{
  theDaqMonitorBEI->save(theDQMFileName.c_str());
}

void LaserDQM::fillAdcCounts(MonitorElement * theMonitor, 
			     edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator,
			     edm::DetSet<SiStripDigi>::const_iterator digiRangeIteratorEnd)
{
  // get the ROOT object from the MonitorElement
  MonitorElementT<TNamed>* theROOTObject = dynamic_cast<MonitorElementT<TNamed>*> (theMonitor);
  TH1F * theMEHistogram = dynamic_cast<TH1F *> (theROOTObject->operator->());

  // loop over all the digis in this det
  for (; digiRangeIterator != digiRangeIteratorEnd; ++digiRangeIterator) 
    {
      const SiStripDigi *digi = &*digiRangeIterator;
      
      if ( theDebugLevel > 4 ) 
	{ std::cout << " Channel " << digi->channel() << " has " << digi->adc() << " adc counts " << std::endl; }

      // fill the number of adc counts in the histogram
      if (digi->channel() < 512)
	{
	  Double_t theBinContent = theMEHistogram->GetBinContent(digi->channel()) + digi->adc();
	  theMEHistogram->SetBinContent(digi->channel(), theBinContent);
	}
    }
}

// define the SEAL module

DEFINE_FWK_MODULE(LaserDQM);
