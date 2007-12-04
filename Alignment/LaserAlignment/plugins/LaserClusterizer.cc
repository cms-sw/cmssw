/** \file LaserClusterizer.cc
 *  Clusterizer for the Laser Beams
 *
 *  $Date: 2007/03/18 19:00:20 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/plugins/LaserClusterizer.h"
#include "FWCore/Framework/interface/Event.h" 
#include "DataFormats/Common/interface/Handle.h" 
#include "FWCore/Framework/interface/EventSetup.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "DataFormats/Common/interface/DetSetVector.h" 
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h" 
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h" 

LaserClusterizer::LaserClusterizer(const edm::ParameterSet & theConf) : 
  theLaserClusterizerAlgorithm(theConf), theParameterSet(theConf)
{
  std::string alias ( theConf.getParameter<std::string>("@module_label") );

  produces<edm::DetSetVector<SiStripCluster> >().setBranchAlias( alias + "siStripClusters" );
}

// virtual destructor needed
LaserClusterizer::~LaserClusterizer() {}

void LaserClusterizer::beginJob(const edm::EventSetup& theSetup)
{
  // get the geometry of the tracker
  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
}

void LaserClusterizer::produce(edm::Event& theEvent, const edm::EventSetup& theSetup)
{
  // create empty output collection
  std::auto_ptr<edm::DetSetVector<SiStripCluster> > output(new edm::DetSetVector<SiStripCluster>);

  // retrieve producer name of the BeamProfileFitCollection
  std::string beamFitProducer = theParameterSet.getParameter<std::string>("BeamFitProducer");

  // get the BeamProfileFitCollection
  edm::Handle<LASBeamProfileFitCollection> beamFits;
  theEvent.getByLabel(beamFitProducer, beamFits);

  // retrieve producer names of the digis
  Parameters DigiProducersList = theParameterSet.getParameter<Parameters>("DigiProducersList");

  // get the digis
  edm::Handle<edm::DetSetVector<SiStripDigi> > stripDigis;
  for (Parameters::iterator itDigiProducersList = DigiProducersList.begin(); itDigiProducersList != DigiProducersList.end(); ++itDigiProducersList)
    {
      std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
      std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");

      theEvent.getByLabel(digiProducer, digiLabel, stripDigis);
      

      if ( (stripDigis->size() > 0) && (beamFits->size() > 0) )
	{
	  // invoke the laser beam clusterizer algorithm
	  theLaserClusterizerAlgorithm.run(*stripDigis,beamFits.product(),*output,theTrackerGeometry);
	}
    }

  // write the output to the event
  theEvent.put(output);
}
// define the SEAL module
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(LaserClusterizer);
