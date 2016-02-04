// -*- C++ -*-
//
// Package:    FFTJetImageRecorder
// Class:      FFTJetImageRecorder
// 
/**\class FFTJetImageRecorder FFTJetImageRecorder.cc RecoJets/JetAnalyzers/src/FFTJetImageRecorder.cc

 Description: collects the info produced by FFTJetEFlowSmoother

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Apr 21 15:52:11 CDT 2011
// $Id: FFTJetImageRecorder.cc,v 1.1 2011/06/03 05:10:07 igv Exp $
//
//

#include <cassert>
#include <sstream>
#include <numeric>

#include "TNtuple.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class FFTJetImageRecorder : public edm::EDAnalyzer
{
public:
    explicit FFTJetImageRecorder(const edm::ParameterSet&);
    ~FFTJetImageRecorder();

private:
    FFTJetImageRecorder();
    FFTJetImageRecorder(const FFTJetImageRecorder&);
    FFTJetImageRecorder& operator=(const FFTJetImageRecorder&);

    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

    edm::InputTag histoLabel;
    unsigned long counter;
};

//
// constructors and destructor
//
FFTJetImageRecorder::FFTJetImageRecorder(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, histoLabel),
      counter(0)
{
}


FFTJetImageRecorder::~FFTJetImageRecorder()
{
}


//
// member functions
//

// ------------ method called once each job just before starting event loop
void FFTJetImageRecorder::beginJob()
{
    edm::Service<TFileService> fs;
    fs->make<TNtuple>("dummy", "dummy", "var");
}


// ------------ method called to for each event  ------------
void FFTJetImageRecorder::analyze(const edm::Event& iEvent,
                                   const edm::EventSetup& iSetup)
{
    const long runnumber = iEvent.id().run();
    const long eventnumber = iEvent.id().event();

    edm::Handle<TH3F> input;
    iEvent.getByLabel(histoLabel, input);

    edm::Service<TFileService> fs;
    TH3F* copy = new TH3F(*input);

    std::ostringstream os;
    os << copy->GetName() << '_' << counter << '_'
       << runnumber << '_' << eventnumber;
    const std::string& newname(os.str());
    copy->SetNameTitle(newname.c_str(), newname.c_str());

    copy->SetDirectory(fs->getBareDirectory());

    ++counter;
}


// ------------ method called once each job just after ending the event loop
void FFTJetImageRecorder::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetImageRecorder);
