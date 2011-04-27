// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetPileupEstimator
// 
/**\class FFTJetPileupEstimator FFTJetPileupEstimator.cc RecoJets/FFTJetProducers/plugins/FFTJetPileupEstimator.cc

 Description: estimates the actual pileup

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Wed Apr 20 13:52:23 CDT 2011
// $Id: FFTJetPileupEstimator.cc,v 1.0 2011/04/20 00:19:43 igv Exp $
//
//

#include <cmath>

// Framework include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Data formats
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "DataFormats/JetReco/interface/FFTJetPileupSummary.h"

#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

using namespace fftjetcms;

//
// class declaration
//
class FFTJetPileupEstimator : public edm::EDProducer
{
public:
    explicit FFTJetPileupEstimator(const edm::ParameterSet&);
    ~FFTJetPileupEstimator();

protected:
    // methods
    void beginJob();
    void produce(edm::Event&, const edm::EventSetup&);
    void endJob();

private:
    FFTJetPileupEstimator();
    FFTJetPileupEstimator(const FFTJetPileupEstimator&);
    FFTJetPileupEstimator& operator=(const FFTJetPileupEstimator&);

    template<class Ptr>
    inline void checkConfig(const Ptr& ptr, const char* message)
    {
        if (ptr.get() == NULL)
            throw cms::Exception("FFTJetBadConfig") << message << std::endl;
    }

    edm::InputTag inputLabel;
    std::string outputLabel;
    double cdfvalue;
    double ptToDensityFactor;
    unsigned filterNumber;
    std::vector<double> uncertaintyZones;
    std::auto_ptr<fftjet::Functor1<double,double> > calibrationCurve;
    std::auto_ptr<fftjet::Functor1<double,double> > uncertaintyCurve;
};

//
// constructors and destructor
//
FFTJetPileupEstimator::FFTJetPileupEstimator(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, inputLabel),
      init_param(std::string, outputLabel),
      init_param(double, cdfvalue),
      init_param(double, ptToDensityFactor),
      init_param(unsigned, filterNumber),
      init_param(std::vector<double>, uncertaintyZones)
{
    calibrationCurve = fftjet_Function_parser(
        ps.getParameter<edm::ParameterSet>("calibrationCurve"));
    checkConfig(calibrationCurve, "bad calibration curve definition");

    uncertaintyCurve = fftjet_Function_parser(
        ps.getParameter<edm::ParameterSet>("uncertaintyCurve"));
    checkConfig(uncertaintyCurve, "bad uncertainty curve definition");

    produces<reco::FFTJetPileupSummary>(outputLabel);
}


FFTJetPileupEstimator::~FFTJetPileupEstimator()
{
}

//
// member functions
//

// ------------ method called to for each event  ------------
void FFTJetPileupEstimator::produce(edm::Event& iEvent,
                                    const edm::EventSetup& iSetup)
{
    edm::Handle<TH2D> input;
    iEvent.getByLabel(inputLabel, input);

    const TH2D& h(*input);
    const unsigned nScales = h.GetXaxis()->GetNbins();
    const unsigned nCdfvalues = h.GetYaxis()->GetNbins();

    const unsigned fixedCdfvalueBin = static_cast<unsigned>(
        std::floor(cdfvalue*nCdfvalues));
    if (fixedCdfvalueBin >= nCdfvalues)
    {
        throw cms::Exception("FFTJetBadConfig") 
            << "Bad cdf value" << std::endl;
    }
    if (filterNumber >= nScales)
    {
        throw cms::Exception("FFTJetBadConfig") 
            << "Bad filter number" << std::endl;
    }

    // Simple fixed-point pile-up estimate
    const double curve = h.GetBinContent(filterNumber+1U,
                                         fixedCdfvalueBin+1U);
    const double pileupRho = ptToDensityFactor*(*calibrationCurve)(curve);
    const double rhoUncert = ptToDensityFactor*(*uncertaintyCurve)(curve);

    // Determine the uncertainty zone of the estimate. The "curve"
    // has to be above or equal to uncertaintyZones[i]  but below
    // uncertaintyZones[i + 1] (the second condition is also satisfied
    // by i == uncertaintyZones.size() - 1). Of course, it is assumed
    // that the vector of zones is configured appropriately -- the zone
    // boundaries must be presented in the increasing order.
    int uncertaintyCode = -1;
    if (!uncertaintyZones.empty())
    {
        const unsigned nZones = uncertaintyZones.size();
        for (unsigned i = 0; i < nZones; ++i)
            if (curve >= uncertaintyZones[i])
            {
                if (i == nZones - 1U)
                {
                    uncertaintyCode = i;
                    break;
                }
                else if (curve < uncertaintyZones[i + 1])
                {
                    uncertaintyCode = i;
                    break;
                }
            }
    }

    std::auto_ptr<reco::FFTJetPileupSummary> summary(
        new reco::FFTJetPileupSummary(curve, pileupRho,
                                      rhoUncert, uncertaintyCode));
    iEvent.put(summary, outputLabel);
}


void FFTJetPileupEstimator::beginJob()
{
}


void FFTJetPileupEstimator::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetPileupEstimator);
