#ifndef PhysicsTools_Heppy_PdfWeightProducerTool_h
#define PhysicsTools_Heppy_PdfWeightProducerTool_h

#include "TRandom3.h"
#include <iostream>

/// TAKEN FROM http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/ElectroWeakAnalysis/Utilities/src/PdfWeightProducer.cc?&view=markup

#include <string>
#include <vector>
#include <map>
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

namespace heppy {

class PdfWeightProducerTool {
    public:
        PdfWeightProducerTool() {}
        void addPdfSet(const std::string &name) ;
        void beginJob() ;
        void processEvent(const GenEventInfoProduct & pdfstuff) ;
        const std::vector<double> & getWeights(const std::string &name) const ;
    private:
        std::vector<std::string> pdfs_; 
        std::map<std::string, std::vector<double> > weights_;    
};

};

#endif
