#include <cassert>

#include "RecoJets/FFTJetProducers/interface/FFTJetInterface.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

namespace fftjetcms {

bool FFTJetInterface::storeInSinglePrecision() const 
{
    return true;
}


FFTJetInterface::FFTJetInterface(const edm::ParameterSet& ps)
    : inputLabel(ps.getParameter<edm::InputTag>("src")),
      init_param(std::string, outputLabel),
      jetType(parseJetType(ps.getParameter<std::string>("jetType"))),
      init_param(bool, doPVCorrection),
      init_param(edm::InputTag, srcPVs),
      etaDependentMagnutideFactors(
          ps.getParameter<std::vector<double> >(
              "etaDependentMagnutideFactors")),
      init_param(edm::ParameterSet, anomalous),
      init_param(bool, insertCompleteEvent),
      init_param(double, completeEventScale),
      vertex_(0.0, 0.0, 0.0)
{
      if (insertCompleteEvent && completeEventScale <= 0.0)
	throw cms::Exception("FFTJetBadConfig")
	  << "Bad scale for the complete event : must be positive"
	  << std::endl;
}


double FFTJetInterface::getEventScale() const
{
     return insertCompleteEvent ? completeEventScale : 0.0;
}


void FFTJetInterface::loadInputCollection(const edm::Event& iEvent)
{
    // Figure out if we are going to perform the vertex adjustment
    const bool adjustForVertex = doPVCorrection && jetType == CALOJET;

    // Figure out the vertex
    if (adjustForVertex)
    {
        edm::Handle<reco::VertexCollection> pvCollection;
        iEvent.getByLabel(srcPVs, pvCollection);
        if (pvCollection->empty())
            vertex_ = reco::Particle::Point(0.0, 0.0, 0.0);
        else
            vertex_ = pvCollection->begin()->position();
    }

    // Get the input collection
    iEvent.getByLabel(inputLabel, inputCollection);

    // Create the set of 4-vectors needed by the algorithm
    eventData.clear();
    candidateIndex.clear();
    unsigned index = 0;
    const reco::CandidateView::const_iterator end = inputCollection->end();
    for (reco::CandidateView::const_iterator it = inputCollection->begin();
         it != end; ++it, ++index)
    {
        const reco::Candidate& item(*it);
        if (anomalous(item))
            continue;
        if (edm::isNotFinite(item.pt()))
            continue;

        if (adjustForVertex)
        {
            const CaloTower& tower(dynamic_cast<const CaloTower&>(item));
            eventData.push_back(VectorLike(tower.p4(vertex_)));
        }
        else
            eventData.push_back(item.p4());
        candidateIndex.push_back(index);
    }
    assert(eventData.size() == candidateIndex.size());
}


void FFTJetInterface::discretizeEnergyFlow()
{
    // It is a bug to call this function before defining the energy flow grid
    assert(energyFlow.get());

    fftjet::Grid2d<Real>& g(*energyFlow);
    g.reset();

    const unsigned nInputs = eventData.size();
    const VectorLike* inp = nInputs ? &eventData[0] : 0;
    for (unsigned i=0; i<nInputs; ++i)
    {
        const VectorLike& item(inp[i]);
        g.fill(item.Eta(), item.Phi(), item.Et());
    }

    if (!etaDependentMagnutideFactors.empty())
    {
        if (etaDependentMagnutideFactors.size() != g.nEta())
            throw cms::Exception("FFTJetBadConfig")
                << "ERROR in FFTJetInterface::discretizeEnergyFlow() :"
                " number of elements in the \"etaDependentMagnutideFactors\""
                " vector is inconsistent with the grid binning"
                << std::endl;
        g.scaleData(&etaDependentMagnutideFactors[0],
                    etaDependentMagnutideFactors.size());
    }
}

}
