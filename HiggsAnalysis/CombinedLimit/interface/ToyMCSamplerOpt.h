#ifndef ROOT_ToyMCSamplerOpt_h
#define ROOT_ToyMCSamplerOpt_h

#include <memory>
#include <RooStats/ToyMCSampler.h>
struct RooProdPdf;
struct RooPoisson;

namespace toymcoptutils {
    class SinglePdfGenInfo {
        public:
            enum Mode { Binned, BinnedNoWorkaround, Poisson, Unbinned, Counting };
            SinglePdfGenInfo(RooAbsPdf &pdf, const RooArgSet& observables, bool preferBinned, const RooDataSet* protoData = NULL, int forceEvents = 0) ;
            ~SinglePdfGenInfo() ;
            RooAbsData *generate(const RooDataSet* protoData = NULL, int forceEvents = 0) ;
            RooDataSet *generateAsimov(RooRealVar *&weightVar, double weightScale = 1.0) ;
            RooDataSet *generatePseudoAsimov(RooRealVar *&weightVar, int nPoints, double weightScale = 1.0) ;
            const RooAbsPdf * pdf() const { return pdf_; }
            void setCacheTemplates(bool cache) { keepHistoSpec_ = cache; }
            Mode mode() const { return mode_; }
        private:
            Mode mode_;
            RooAbsPdf *pdf_; 
            RooArgSet observables_;
            RooAbsPdf::GenSpec *spec_;
            TH1        *histoSpec_;
            bool        keepHistoSpec_;
            RooRealVar *weightVar_;
            RooDataSet *generateWithHisto(RooRealVar *&weightVar, bool asimov, double weightScale = 1.0) ;
            RooDataSet *generateCountingAsimov() ;
            void setToExpected(RooProdPdf &prod, RooArgSet &obs) ;
            void setToExpected(RooPoisson &pois, RooArgSet &obs) ;
    };
    class SimPdfGenInfo {
        public:
            SimPdfGenInfo(RooAbsPdf &pdf, const RooArgSet& observables, bool preferBinned, const RooDataSet* protoData = NULL, int forceEvents = 0) ;
            ~SimPdfGenInfo() ;
            RooAbsData *generate(RooRealVar *&weightVar, const RooDataSet* protoData = NULL, int forceEvents = 0) ;
            RooAbsData *generateAsimov(RooRealVar *&weightVar) ;
            RooAbsData *generateEpsilon(RooRealVar *&weightVar) ;
            void setCopyData(bool copyData) { copyData_ = copyData; }
            void setCacheTemplates(bool cache) ;
        private:
            RooAbsPdf                       *pdf_; 
            RooAbsCategoryLValue            *cat_;
            RooArgSet                        observables_;
            std::vector<SinglePdfGenInfo *>  pdfs_; 
            RooArgSet                        ownedCrap_;
            std::map<std::string,RooAbsData*> datasetPieces_;
            bool                              copyData_;
            //std::map<std::string,RooDataSet*> datasetPieces_;

    }; 
}

class ToyMCSamplerOpt : public RooStats::ToyMCSampler{
    public:
        ToyMCSamplerOpt(RooStats::TestStatistic& ts, Int_t ntoys, RooAbsPdf *globalObsPdf = 0, bool generateNuisances = false) ;
        ToyMCSamplerOpt(const RooStats::ToyMCSampler &base) ;
        ToyMCSamplerOpt(const ToyMCSamplerOpt &other) ;
        ~ToyMCSamplerOpt() ;
        virtual void SetPdf(RooAbsPdf& pdf) ;
        void setGlobalObsPdf(RooAbsPdf *pdf) { globalObsPdf_ = pdf; }
        virtual RooAbsData* GenerateToyData(RooArgSet& /*nullPOI*/, double& weight) const ;
    private:
        RooAbsData* GenerateToyDataWithImportanceSampling(RooArgSet& /*nullPOI*/, double& weight) const ;

        RooAbsData* Generate(RooAbsPdf& pdf, RooArgSet& observables, const RooDataSet* protoData = NULL, int forceEvents = 0) const ;
        RooAbsPdf *globalObsPdf_;
        mutable RooDataSet *globalObsValues_; 
        mutable int globalObsIndex_;

        // We can't use the NuisanceParameterSampler because, even if public, there's no interface file for it
        mutable RooDataSet *nuisValues_; 
        mutable int nuisIndex_;

        mutable RooRealVar *weightVar_;
        mutable std::map<RooAbsPdf *, toymcoptutils::SimPdfGenInfo *> genCache_;

        mutable std::auto_ptr<RooArgSet> paramsForImportanceSampling_;
        mutable std::vector<RooArgSet *> importanceSnapshots_;
};

#endif
