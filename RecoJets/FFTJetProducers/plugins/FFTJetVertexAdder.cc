// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetVertexAdder
// 
/**\class FFTJetVertexAdder FFTJetVertexAdder.cc RecoJets/FFTJetProducers/plugins/FFTJetVertexAdder.cc

 Description: adds a collection of fake vertices to the event record

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Jun 21 19:19:40 CDT 2012
// $Id: FFTJetVertexAdder.cc,v 1.1 2012/06/22 07:23:21 igv Exp $
//
//

#include <iostream>
#include "CLHEP/Random/RandGauss.h"

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class FFTJetVertexAdder : public edm::EDProducer
{
public:
    explicit FFTJetVertexAdder(const edm::ParameterSet&);
    ~FFTJetVertexAdder();

protected:
    // methods
    void beginJob();
    void produce(edm::Event&, const edm::EventSetup&);
    void endJob();

private:
    FFTJetVertexAdder();
    FFTJetVertexAdder(const FFTJetVertexAdder&);
    FFTJetVertexAdder& operator=(const FFTJetVertexAdder&);

    const edm::InputTag beamSpotLabel;
    const edm::InputTag existingVerticesLabel;
    const std::string outputLabel;

    const bool useBeamSpot;
    const bool addExistingVertices;

    const double fixedX;
    const double fixedY;
    const double fixedZ;

    const double sigmaX;
    const double sigmaY;
    const double sigmaZ;

    const double nDof;
    const double chi2;
    const double errX;
    const double errY;
    const double errZ;

    const unsigned nVerticesToMake;

    CLHEP::RandGauss* rGauss_;
};

//
// constructors and destructor
//
FFTJetVertexAdder::FFTJetVertexAdder(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, beamSpotLabel),
      init_param(edm::InputTag, existingVerticesLabel),
      init_param(std::string, outputLabel),
      init_param(bool, useBeamSpot),
      init_param(bool, addExistingVertices),
      init_param(double, fixedX),
      init_param(double, fixedY),
      init_param(double, fixedZ),
      init_param(double, sigmaX),
      init_param(double, sigmaY),
      init_param(double, sigmaZ),
      init_param(double, nDof),
      init_param(double, chi2),
      init_param(double, errX),
      init_param(double, errY),
      init_param(double, errZ),
      init_param(unsigned, nVerticesToMake),
      rGauss_(0)
{
    produces<reco::VertexCollection>(outputLabel);
}


FFTJetVertexAdder::~FFTJetVertexAdder()
{
    delete rGauss_;
}


// ------------ method called to produce the data  ------------
void FFTJetVertexAdder::produce(
    edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // get PFCandidates
    std::auto_ptr<reco::VertexCollection> pOutput(new reco::VertexCollection);

    double xmean = fixedX;
    double ymean = fixedY;
    double zmean = fixedZ;

    double xwidth = sigmaX;
    double ywidth = sigmaY;
    double zwidth = sigmaZ;

    if (useBeamSpot)
    {
        edm::Handle<reco::BeamSpot> beamSpotHandle;
        iEvent.getByLabel(beamSpotLabel, beamSpotHandle);
        if (!beamSpotHandle.isValid())
            throw cms::Exception("FFTJetBadConfig")
                << "ERROR in FFTJetVertexAdder:"
                " could not find beam spot information"
                << std::endl;

        xmean = beamSpotHandle->x0();
        ymean = beamSpotHandle->y0();
        zmean = beamSpotHandle->z0();

        xwidth = beamSpotHandle->BeamWidthX();
        ywidth = beamSpotHandle->BeamWidthY();
        zwidth = beamSpotHandle->sigmaZ();
    }

    reco::Vertex::Error err;
    for (unsigned i=0; i<3; ++i)
        for (unsigned j=0; j<3; ++j)
            err[i][j] = 0.0;
    err[0][0] = errX*errX;
    err[1][1] = errY*errY;
    err[2][2] = errZ*errZ;

    for (unsigned iv=0; iv<nVerticesToMake; ++iv)
    {
        const double x0 = (*rGauss_)(xmean, xwidth);
        const double y0 = (*rGauss_)(ymean, ywidth);
        const double z0 = (*rGauss_)(zmean, zwidth);
        const reco::Vertex::Point position(x0, y0, z0);
        pOutput->push_back(reco::Vertex(position, err, chi2, nDof, 0));
    }

    if (addExistingVertices)
    {
        typedef reco::VertexCollection::const_iterator IV;

        edm::Handle<reco::VertexCollection> vertices;
        iEvent.getByLabel(existingVerticesLabel, vertices);
        if (!vertices.isValid())
            throw cms::Exception("FFTJetBadConfig")
                << "ERROR in FFTJetVertexAdder:"
                " could not find existing collection of vertices"
                << std::endl;

        const IV vertend(vertices->end());
        for (IV iv=vertices->begin(); iv!=vertend; ++iv)
            pOutput->push_back(*iv);
    }

    iEvent.put(pOutput, outputLabel);
}


// ------------ method called once each job just before starting event loop
void FFTJetVertexAdder::beginJob()
{
    if (!rGauss_)
    {
        edm::Service<edm::RandomNumberGenerator> rng;
        if ( !rng.isAvailable() )
            throw cms::Exception("FFTJetBadConfig")
                << "ERROR in FFTJetVertexAdder:"
                " failed to initialize the random number generator"
                << std::endl;
        rGauss_ = new CLHEP::RandGauss(rng->getEngine());
    }
}


// ------------ method called once each job just after ending the event loop
void FFTJetVertexAdder::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetVertexAdder);
