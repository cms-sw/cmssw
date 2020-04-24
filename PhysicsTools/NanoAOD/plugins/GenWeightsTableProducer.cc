#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/algorithm/string.hpp"

#include <vector>
#include <unordered_map>
#include <iostream>
#include <regex>

namespace {
    ///  ---- Cache object for running sums of weights ----
    struct Counter {
        Counter() : 
            num(0), sumw(0), sumw2(0), sumPDF(), sumScale(), sumNamed(), sumPS() {}

        // the counters
        long long num;
        long double sumw;
        long double sumw2;
        std::vector<long double> sumPDF, sumScale, sumNamed, sumPS;

        void clear() { 
            num = 0; sumw = 0; sumw2 = 0;
            sumPDF.clear(); sumScale.clear(); sumNamed.clear(), sumPS.clear();
        }

        // inc the counters
        void incGenOnly(double w) { 
            num++; sumw += w; sumw2 += (w*w); 
        }

        void incPSOnly(double w0, const std::vector<double> & wPS) {
            if (!wPS.empty()) {
                if (sumPS.empty()) sumPS.resize(wPS.size(), 0);
                for (unsigned int i = 0, n = wPS.size(); i < n; ++i) sumPS[i] += (w0 * wPS[i]);
            }
        }

        void incLHE(double w0, const std::vector<double> & wScale, const std::vector<double> & wPDF, const std::vector<double> & wNamed, const std::vector<double> & wPS) {
            // add up weights
            incGenOnly(w0);
            // then add up variations
            if (!wScale.empty()) {
                if (sumScale.empty()) sumScale.resize(wScale.size(), 0);
                for (unsigned int i = 0, n = wScale.size(); i < n; ++i) sumScale[i] += (w0 * wScale[i]);
            }
            if (!wPDF.empty()) {
                if (sumPDF.empty()) sumPDF.resize(wPDF.size(), 0);
                for (unsigned int i = 0, n = wPDF.size(); i < n; ++i) sumPDF[i] += (w0 * wPDF[i]);
            }
            if (!wNamed.empty()) {
                if (sumNamed.empty()) sumNamed.resize(wNamed.size(), 0);
                for (unsigned int i = 0, n = wNamed.size(); i < n; ++i) sumNamed[i] += (w0 * wNamed[i]);
            }
            incPSOnly(w0, wPS);
        }

        void merge(const Counter & other) { 
            num += other.num; sumw += other.sumw; sumw2 += other.sumw2; 
            if (sumScale.empty() && !other.sumScale.empty()) sumScale.resize(other.sumScale.size(),0);
            if (sumPDF.empty() && !other.sumPDF.empty()) sumPDF.resize(other.sumPDF.size(),0);
            if (sumNamed.empty() && !other.sumNamed.empty()) sumNamed.resize(other.sumNamed.size(),0);
            if (sumPS.empty() && !other.sumPS.empty()) sumPS.resize(other.sumPS.size(),0);
            for (unsigned int i = 0, n = sumScale.size(); i < n; ++i) sumScale[i] += other.sumScale[i];
            for (unsigned int i = 0, n = sumPDF.size(); i < n; ++i) sumPDF[i] += other.sumPDF[i];
            for (unsigned int i = 0, n = sumNamed.size(); i < n; ++i) sumNamed[i] += other.sumNamed[i];
            for (unsigned int i = 0, n = sumPS.size(); i < n; ++i) sumPS[i] += other.sumPS[i];
        }
    };

    ///  ---- RunCache object for dynamic choice of LHE IDs ----
    struct DynamicWeightChoice {
        // choice of LHE weights
        // ---- scale ----
        std::vector<std::string> scaleWeightIDs; 
        std::string scaleWeightsDoc;
        // ---- pdf ----
        std::vector<std::string> pdfWeightIDs; 
        std::string pdfWeightsDoc;
    };

    float stof_fortrancomp(const std::string &str) {
        std::string::size_type match = str.find("d");
        if (match != std::string::npos) {
            std::string pre  = str.substr(0,match);
            std::string post = str.substr(match+1);
            return std::stof(pre) * std::pow(10.0f, std::stof(post));
        } else {
            return std::stof(str);
        }
    }
    ///  -------------- temporary objects --------------
    struct ScaleVarWeight {
        std::string wid, label;
        std::pair<float,float> scales;
        ScaleVarWeight(const std::string & id, const std::string & text, const std::string & muR, const std::string & muF) :
            wid(id), label(text), scales(stof_fortrancomp(muR), stof_fortrancomp(muF)) {}
        bool operator<(const ScaleVarWeight & other) { return (scales == other.scales ? wid < other.wid : scales < other.scales); }
    };
    struct PDFSetWeights {
        std::vector<std::string> wids;
        std::pair<unsigned int,unsigned int> lhaIDs;
        PDFSetWeights(const std::string & wid, unsigned int lhaID) : wids(1,wid), lhaIDs(lhaID,lhaID) {}
        bool operator<(const PDFSetWeights & other) const { return lhaIDs < other.lhaIDs; }
        void add(const std::string & wid, unsigned int lhaID) {
            wids.push_back(wid);
            lhaIDs.second = lhaID;
        }
        bool maybe_add(const std::string & wid, unsigned int lhaID) {
            if (lhaID == lhaIDs.second+1) {
                lhaIDs.second++;
                wids.push_back(wid);
                return true;
            } else {
                return false;
            }
        }
    };
}

class GenWeightsTableProducer : public edm::global::EDProducer<edm::StreamCache<Counter>, edm::RunCache<DynamicWeightChoice>, edm::RunSummaryCache<Counter>, edm::EndRunProducer> {
    public:
        GenWeightsTableProducer( edm::ParameterSet const & params ) :
            genTag_(consumes<GenEventInfoProduct>(params.getParameter<edm::InputTag>("genEvent"))),
            lheLabel_(params.getParameter<edm::InputTag>("lheInfo")),
            lheTag_(consumes<LHEEventProduct>(lheLabel_)),
            lheRunTag_(consumes<LHERunInfoProduct, edm::InRun>(lheLabel_)),
            namedWeightIDs_(params.getParameter<std::vector<std::string>>("namedWeightIDs")),
            namedWeightLabels_(params.getParameter<std::vector<std::string>>("namedWeightLabels")),
            lheWeightPrecision_(params.getParameter<int32_t>("lheWeightPrecision")),
            maxPdfWeights_(params.getParameter<uint32_t>("maxPdfWeights")),
            debug_(params.getUntrackedParameter<bool>("debug",false)), debugRun_(debug_.load()),
            hasIssuedWarning_(false)
        {
            produces<nanoaod::FlatTable>();
            produces<nanoaod::FlatTable>("LHEScale");
            produces<nanoaod::FlatTable>("LHEPdf");
            produces<nanoaod::FlatTable>("LHENamed");
            produces<nanoaod::FlatTable>("PS");
            produces<nanoaod::MergeableCounterTable,edm::Transition::EndRun>();
            if (namedWeightIDs_.size() != namedWeightLabels_.size()) {
                throw cms::Exception("Configuration", "Size mismatch between namedWeightIDs & namedWeightLabels");
            }
            for (const edm::ParameterSet & pdfps : params.getParameter<std::vector<edm::ParameterSet>>("preferredPDFs")) {
                const std::string & name = pdfps.getParameter<std::string>("name");
                uint32_t lhaid = pdfps.getParameter<uint32_t>("lhaid");
                preferredPDFLHAIDs_.push_back(lhaid);
                lhaNameToID_[name] = lhaid;
                lhaNameToID_[name+".LHgrid"] = lhaid;
            }
        }

        ~GenWeightsTableProducer() override {}

        void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
            // get my counter for weights
            Counter * counter = streamCache(id);

            // generator information (always available)
            edm::Handle<GenEventInfoProduct> genInfo;
            iEvent.getByToken(genTag_, genInfo);
            double weight = genInfo->weight();

            // table for gen info, always available
            auto out = std::make_unique<nanoaod::FlatTable>(1, "genWeight", true);
            out->setDoc("generator weight");
            out->addColumnValue<float>("", weight, "generator weight", nanoaod::FlatTable::FloatColumn);
            iEvent.put(std::move(out));

            // tables for LHE weights, may not be filled
            std::unique_ptr<nanoaod::FlatTable> lheScaleTab, lhePdfTab, lheNamedTab;
            std::unique_ptr<nanoaod::FlatTable> genPSTab;

            edm::Handle<LHEEventProduct> lheInfo;
            if (iEvent.getByToken(lheTag_, lheInfo)) {
                // get the dynamic choice of weights
                const DynamicWeightChoice * weightChoice = runCache(iEvent.getRun().index());
                // go fill tables
                fillLHEWeightTables(counter, weightChoice, weight, *lheInfo, *genInfo, lheScaleTab, lhePdfTab, lheNamedTab, genPSTab);
            } else {
                // Still try to add the PS weights
                fillOnlyPSWeightTable(counter, weight, *genInfo, genPSTab);
                // make dummy values 
                lheScaleTab.reset(new nanoaod::FlatTable(1, "LHEScaleWeights", true));
                lhePdfTab.reset(new nanoaod::FlatTable(1, "LHEPdfWeights", true));
                lheNamedTab.reset(new nanoaod::FlatTable(1, "LHENamedWeights", true));
                if (!hasIssuedWarning_.exchange(true)) {
                    edm::LogWarning("LHETablesProducer") << "No LHEEventProduct, so there will be no LHE Tables\n";
                }
            }

            iEvent.put(std::move(lheScaleTab), "LHEScale");
            iEvent.put(std::move(lhePdfTab), "LHEPdf");
            iEvent.put(std::move(lheNamedTab), "LHENamed");
            iEvent.put(std::move(genPSTab), "PS");
        }

        void fillLHEWeightTables(
                Counter * counter,
                const DynamicWeightChoice * weightChoice,
                double genWeight,
                const LHEEventProduct & lheProd, 
                const GenEventInfoProduct & genProd,
                std::unique_ptr<nanoaod::FlatTable> & outScale, 
                std::unique_ptr<nanoaod::FlatTable> & outPdf, 
                std::unique_ptr<nanoaod::FlatTable> & outNamed,
                std::unique_ptr<nanoaod::FlatTable> & outPS ) const
        {
            bool lheDebug = debug_.exchange(false); // make sure only the first thread dumps out this (even if may still be mixed up with other output, but nevermind)

            const std::vector<std::string> & scaleWeightIDs = weightChoice->scaleWeightIDs;
            const std::vector<std::string> & pdfWeightIDs   = weightChoice->pdfWeightIDs;

            double w0 = lheProd.originalXWGTUP();

            std::vector<double> wScale(scaleWeightIDs.size(), 1), wPDF(pdfWeightIDs.size(), 1), wNamed(namedWeightIDs_.size(), 1);
            for (auto & weight : lheProd.weights()) {
                if (lheDebug) printf("Weight  %+9.5f   rel %+9.5f   for id %s\n", weight.wgt, weight.wgt/w0,  weight.id.c_str());
                // now we do it slowly, can be optimized
                auto mScale = std::find(scaleWeightIDs.begin(), scaleWeightIDs.end(), weight.id);
                if (mScale != scaleWeightIDs.end()) wScale[mScale-scaleWeightIDs.begin()] = weight.wgt/w0;

                auto mPDF = std::find(pdfWeightIDs.begin(), pdfWeightIDs.end(), weight.id);
                if (mPDF != pdfWeightIDs.end()) wPDF[mPDF-pdfWeightIDs.begin()] = weight.wgt/w0;

                auto mNamed = std::find(namedWeightIDs_.begin(), namedWeightIDs_.end(), weight.id);
                if (mNamed != namedWeightIDs_.end()) wNamed[mNamed-namedWeightIDs_.begin()] = weight.wgt/w0;
            } 

            int vectorSize = genProd.weights().size() == 14 ? 4 : 1;
            std::vector<double> wPS(vectorSize, 1);
            if (vectorSize > 1 ) {
                for (unsigned int i=6; i<10; i++){
                    wPS[i-6] = (genProd.weights()[i])/w0;
                }
            }
            outPS.reset(new nanoaod::FlatTable(wPS.size(), "PSWeight", false));
            outPS->addColumn<float>("", wPS, vectorSize > 1 ? "PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2 " : "dummy PS weight (1.0) ", nanoaod::FlatTable::FloatColumn, lheWeightPrecision_);

            outScale.reset(new nanoaod::FlatTable(wScale.size(), "LHEScaleWeight", false));
            outScale->addColumn<float>("", wScale, weightChoice->scaleWeightsDoc, nanoaod::FlatTable::FloatColumn, lheWeightPrecision_); 

            outPdf.reset(new nanoaod::FlatTable(wPDF.size(), "LHEPdfWeight", false));
            outPdf->addColumn<float>("", wPDF, weightChoice->pdfWeightsDoc, nanoaod::FlatTable::FloatColumn, lheWeightPrecision_); 

            outNamed.reset(new nanoaod::FlatTable(1, "LHEWeight", true));
            outNamed->addColumnValue<float>("originalXWGTUP", lheProd.originalXWGTUP(), "Nominal event weight in the LHE file", nanoaod::FlatTable::FloatColumn);
            for (unsigned int i = 0, n = wNamed.size(); i < n; ++i) {
                outNamed->addColumnValue<float>(namedWeightLabels_[i], wNamed[i], "LHE weight for id "+namedWeightIDs_[i]+", relative to nominal", nanoaod::FlatTable::FloatColumn, lheWeightPrecision_);
            }
            
            counter->incLHE(genWeight, wScale, wPDF, wNamed, wPS);
        }

        void fillOnlyPSWeightTable(
                Counter * counter,
                double genWeight,
                const GenEventInfoProduct & genProd,
                std::unique_ptr<nanoaod::FlatTable> & outPS ) const
        {
            int vectorSize = genProd.weights().size() == 14 ? 4 : 1;

            std::vector<double> wPS(vectorSize, 1);
            if (vectorSize > 1 ){
                for (unsigned int i=6; i<10; i++){
                    wPS[i-6] = (genProd.weights()[i])/genWeight;
                }
            }

            outPS.reset(new nanoaod::FlatTable(wPS.size(), "PSWeight", false));
            outPS->addColumn<float>("", wPS, vectorSize > 1 ? "PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2 " : "dummy PS weight (1.0) " , nanoaod::FlatTable::FloatColumn, lheWeightPrecision_);

            counter->incGenOnly(genWeight);
            counter->incPSOnly(genWeight,wPS);
        }



        // create an empty counter
        std::shared_ptr<DynamicWeightChoice> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const override {
            edm::Handle<LHERunInfoProduct> lheInfo;

            bool lheDebug = debugRun_.exchange(false); // make sure only the first thread dumps out this (even if may still be mixed up with other output, but nevermind)
            auto weightChoice = std::make_shared<DynamicWeightChoice>();

            // getByToken throws since we're not in the endRun (see https://github.com/cms-sw/cmssw/pull/18499)
            //if (iRun.getByToken(lheRunTag_, lheInfo)) {
            if (iRun.getByLabel(lheLabel_, lheInfo)) { 
                std::vector<ScaleVarWeight> scaleVariationIDs;
                std::vector<PDFSetWeights>  pdfSetWeightIDs;
                
                std::regex weightgroupmg26x("<weightgroup\\s+(?:name|type)=\"(.*)\"\\s+combine=\"(.*)\"\\s*>");
                std::regex weightgroup("<weightgroup\\s+combine=\"(.*)\"\\s+(?:name|type)=\"(.*)\"\\s*>");
                std::regex endweightgroup("</weightgroup>");
                std::regex scalewmg26x("<weight\\s+(?:.*\\s+)?id=\"(\\d+)\"\\s*(?:lhapdf=\\d+|dyn=\\s*-?\\d+)?\\s*((?:[mM][uU][rR]|renscfact)=\"(\\S+)\"\\s+(?:[mM][uU][Ff]|facscfact)=\"(\\S+)\")(\\s+.*)?</weight>");
                std::regex scalew("<weight\\s+(?:.*\\s+)?id=\"(\\d+)\">\\s*(?:lhapdf=\\d+|dyn=\\s*-?\\d+)?\\s*((?:mu[rR]|renscfact)=(\\S+)\\s+(?:mu[Ff]|facscfact)=(\\S+)(\\s+.*)?)</weight>");
                std::regex pdfw("<weight\\s+id=\"(\\d+)\">\\s*(?:PDF set|lhapdf|PDF|pdfset)\\s*=\\s*(\\d+)\\s*(?:\\s.*)?</weight>");
                std::regex pdfwOld("<weight\\s+(?:.*\\s+)?id=\"(\\d+)\">\\s*Member \\s*(\\d+)\\s*(?:.*)</weight>");
                std::regex pdfwmg26x("<weight\\s+id=\"(\\d+)\"\\s*MUR=\"(?:\\S+)\"\\s*MUF=\"(?:\\S+)\"\\s*(?:PDF set|lhapdf|PDF|pdfset)\\s*=\\s*\"(\\d+)\"\\s*>\\s*(?:PDF=(\\d+)\\s*MemberID=(\\d+))?\\s*(?:\\s.*)?</weight>");
                std::smatch groups;
                for (auto iter=lheInfo->headers_begin(), end = lheInfo->headers_end(); iter != end; ++iter) {
                    if (iter->tag() != "initrwgt") {
                        if (lheDebug) std::cout << "Skipping LHE header with tag" << iter->tag() << std::endl;
                        continue;
                    }
                    if (lheDebug) std::cout << "Found LHE header with tag" << iter->tag() << std::endl;
                    std::vector<std::string>  lines = iter->lines();
                    bool missed_weightgroup=false; //Needed because in some of the samples ( produced with MG26X ) a small part of the header info is ordered incorrectly
                    bool ismg26x=false;
                    for (unsigned int iLine = 0, nLines = lines.size(); iLine < nLines; ++iLine) { //First start looping through the lines to see which weightgroup pattern is matched
                        boost::replace_all(lines[iLine],"&lt;", "<");
                        boost::replace_all(lines[iLine],"&gt;", ">");
                        if(std::regex_search(lines[iLine],groups,weightgroupmg26x)){
                            ismg26x=true;
                        }
                    }
                    for (unsigned int iLine = 0, nLines = lines.size(); iLine < nLines; ++iLine) {
                        if (lheDebug) std::cout << lines[iLine];
                        if (std::regex_search(lines[iLine], groups, ismg26x ? weightgroupmg26x : weightgroup) ) {
                            std::string groupname = groups.str(2);
                            if (ismg26x) groupname = groups.str(1);
                            if (lheDebug) std::cout << ">>> Looks like the beginning of a weight group for '" << groupname << "'" << std::endl;
                            if (groupname.find("scale_variation") == 0 || groupname == "Central scale variation") {
                                if (lheDebug) std::cout << ">>> Looks like scale variation for theory uncertainties" << std::endl;
                                for ( ++iLine; iLine < nLines; ++iLine) {
                                    if (lheDebug) std::cout << "    " << lines[iLine];
                                    if (std::regex_search(lines[iLine], groups, ismg26x ? scalewmg26x : scalew)) {
                                        if (lheDebug) std::cout << "    >>> Scale weight " << groups[1].str() << " for " << groups[3].str() << " , " << groups[4].str() << " , " << groups[5].str() << std::endl;
                                        scaleVariationIDs.emplace_back(groups.str(1), groups.str(2), groups.str(3), groups.str(4));
                                    } else if (std::regex_search(lines[iLine], endweightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the end of a weight group" << std::endl;
                                        if (!missed_weightgroup){
                                            break;
                                        } else missed_weightgroup=false;
                                    } else if (std::regex_search(lines[iLine], ismg26x ? weightgroupmg26x : weightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the beginning of a new weight group, I will assume I missed the end of the group." << std::endl;
                                        if (ismg26x) missed_weightgroup=true;
                                        --iLine; // rewind by one, and go back to the outer loop
                                        break;
                                    }
                                }
                            } else if (groupname == "PDF_variation" || groupname.find("PDF_variation ") == 0) {
                                if (lheDebug) std::cout << ">>> Looks like a new-style block of PDF weights for one or more pdfs" << std::endl;
                                for ( ++iLine; iLine < nLines; ++iLine) {
                                    if (lheDebug) std::cout << "    " << lines[iLine];
                                    if (std::regex_search(lines[iLine], groups, pdfw)) {
                                        unsigned int lhaID = std::stoi(groups.str(2));
                                        if (lheDebug) std::cout << "    >>> PDF weight " << groups.str(1) << " for " << groups.str(2) << " = " << lhaID << std::endl;
                                        if (pdfSetWeightIDs.empty() || ! pdfSetWeightIDs.back().maybe_add(groups.str(1),lhaID)) {
                                            pdfSetWeightIDs.emplace_back(groups.str(1),lhaID);
                                        }
                                    } else if (std::regex_search(lines[iLine], endweightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the end of a weight group" << std::endl;
                                        if (!missed_weightgroup){ 
                                            break;
                                        } else missed_weightgroup=false;
                                    } else if (std::regex_search(lines[iLine], ismg26x ? weightgroupmg26x : weightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the beginning of a new weight group, I will assume I missed the end of the group." << std::endl;
                                        if (ismg26x) missed_weightgroup=true;
                                        --iLine; // rewind by one, and go back to the outer loop
                                        break;
                                    }
                                }
                            } else if (groupname == "PDF_variation1" || groupname == "PDF_variation2") { 
                                if (lheDebug) std::cout << ">>> Looks like a new-style block of PDF weights for multiple pdfs" << std::endl;
                                unsigned int lastid = 0;
                                for ( ++iLine; iLine < nLines; ++iLine) {
                                    if (lheDebug) std::cout << "    " << lines[iLine];
                                    if (std::regex_search(lines[iLine], groups, pdfw)) {
                                        unsigned int id = std::stoi(groups.str(1));
                                        unsigned int lhaID = std::stoi(groups.str(2));
                                        if (lheDebug) std::cout << "    >>> PDF weight " << groups.str(1) << " for " << groups.str(2) << " = " << lhaID << std::endl;
                                        if (id != (lastid+1) || pdfSetWeightIDs.empty()) {
                                            pdfSetWeightIDs.emplace_back(groups.str(1),lhaID);
                                        } else {
                                            pdfSetWeightIDs.back().add(groups.str(1),lhaID);
                                        }
                                        lastid = id;
                                    } else if (std::regex_search(lines[iLine], endweightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the end of a weight group" << std::endl;
                                        if(!missed_weightgroup) {
                                            break;
                                        } else missed_weightgroup=false;
                                    } else if (std::regex_search(lines[iLine], ismg26x ? weightgroupmg26x : weightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the beginning of a new weight group, I will assume I missed the end of the group." << std::endl;
                                        if (ismg26x) missed_weightgroup=true;
                                        --iLine; // rewind by one, and go back to the outer loop
                                        break;
                                    }
                                }
                            } else if (lhaNameToID_.find(groupname) != lhaNameToID_.end()) {
                                if (lheDebug) std::cout << ">>> Looks like an old-style PDF weight for an individual pdf" << std::endl;
                                unsigned int firstLhaID = lhaNameToID_.find(groupname)->second;
                                bool first = true;
                                for ( ++iLine; iLine < nLines; ++iLine) {
                                    if (lheDebug) std::cout << "    " << lines[iLine];
                                    if (std::regex_search(lines[iLine], groups, ismg26x ? pdfwmg26x : pdfwOld)) {
                                        unsigned int member = 0;
                                        if (ismg26x==0){
                                            member = std::stoi(groups.str(2));
                                        } else {
                                            if (groups.str(4)!=""){
                                                member = std::stoi(groups.str(4));
                                             }
                                        }
                                        unsigned int lhaID = member+firstLhaID;
                                        if (lheDebug) std::cout << "    >>> PDF weight " << groups.str(1) << " for " << member << " = " << lhaID << std::endl;
                                        //if (member == 0) continue; // let's keep also the central value for now
                                        if (first) {
                                            pdfSetWeightIDs.emplace_back(groups.str(1),lhaID);
                                            first = false;
                                        } else {
                                            pdfSetWeightIDs.back().add(groups.str(1),lhaID);
                                        }
                                    } else if (std::regex_search(lines[iLine], endweightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the end of a weight group" << std::endl;
                                        if (!missed_weightgroup) {
                                            break;
                                        } else missed_weightgroup=false;
                                    } else if (std::regex_search(lines[iLine], ismg26x ? weightgroupmg26x : weightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the beginning of a new weight group, I will assume I missed the end of the group." << std::endl;
                                        if (ismg26x) missed_weightgroup=true;
                                        --iLine; // rewind by one, and go back to the outer loop
                                        break;
                                    }
                                }
                            } else {
                                for ( ++iLine; iLine < nLines; ++iLine) {
                                    if (lheDebug) std::cout << "    " << lines[iLine];
                                    if (std::regex_search(lines[iLine], groups, endweightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the end of a weight group" << std::endl;
                                        if (!missed_weightgroup){
                                            break;
                                        } else missed_weightgroup=false;
                                    } else if (std::regex_search(lines[iLine], ismg26x ? weightgroupmg26x : weightgroup)) {
                                        if (lheDebug) std::cout << ">>> Looks like the beginning of a new weight group, I will assume I missed the end of the group." << std::endl;
                                        if (ismg26x) missed_weightgroup=true;
                                        --iLine; // rewind by one, and go back to the outer loop
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    //std::cout << "============= END [ " << iter->tag() << " ] ============ \n\n" << std::endl;

                    // ----- SCALE VARIATIONS -----
                    std::sort(scaleVariationIDs.begin(), scaleVariationIDs.end());
                    if (lheDebug) std::cout << "Found " << scaleVariationIDs.size() << " scale variations: " << std::endl;
                    std::stringstream scaleDoc; scaleDoc << "LHE scale variation weights (w_var / w_nominal); ";
                    for (unsigned int isw = 0, nsw = scaleVariationIDs.size(); isw < nsw; ++isw) {
                        const auto & sw = scaleVariationIDs[isw];
                        if (isw) scaleDoc << "; ";
                        scaleDoc << "[" << isw << "] is " << sw.label;
                        weightChoice->scaleWeightIDs.push_back(sw.wid);
                        if (lheDebug) printf("    id %s: scales ren = % .2f  fact = % .2f  text = %s\n", sw.wid.c_str(), sw.scales.first, sw.scales.second, sw.label.c_str());
                    }
                    if (!scaleVariationIDs.empty()) weightChoice->scaleWeightsDoc = scaleDoc.str();

                    // ------ PDF VARIATIONS (take the preferred one) -----
                    if (lheDebug) {
                        std::cout << "Found " << pdfSetWeightIDs.size() << " PDF set errors: " << std::endl;
                        for (const auto & pw : pdfSetWeightIDs) printf("lhaIDs %6d - %6d (%3lu weights: %s, ... )\n", pw.lhaIDs.first, pw.lhaIDs.second, pw.wids.size(), pw.wids.front().c_str());
                    }
                    
                    std::stringstream pdfDoc; pdfDoc << "LHE pdf variation weights (w_var / w_nominal) for LHA IDs ";
                    bool found = false;
                    for (uint32_t lhaid : preferredPDFLHAIDs_) {
                        for (const auto & pw : pdfSetWeightIDs) {
                            if (pw.lhaIDs.first != lhaid && pw.lhaIDs.first != (lhaid+1)) continue; // sometimes the first weight is not saved if that PDF is the nominal one for the sample
                            if (pw.wids.size() == 1) continue; // only consider error sets
                            pdfDoc << pw.lhaIDs.first << " - " << pw.lhaIDs.second;
                            weightChoice->pdfWeightIDs = pw.wids;
                            if (maxPdfWeights_ < pw.wids.size()) {
                                weightChoice->pdfWeightIDs.resize(maxPdfWeights_); // drop some replicas
                                pdfDoc << ", truncated to the first " << maxPdfWeights_ << " replicas";
                            } 
                            weightChoice->pdfWeightsDoc = pdfDoc.str(); 
                            found = true; break;
                        }
                        if (found) break;
                    }
                }
            }
            return weightChoice; 
        }


        // create an empty counter
        std::unique_ptr<Counter> beginStream(edm::StreamID) const override { 
            return std::make_unique<Counter>(); 
        }
        // inizialize to zero at begin run
        void streamBeginRun(edm::StreamID id, edm::Run const&, edm::EventSetup const&) const override { 
            streamCache(id)->clear(); 
        }
        // create an empty counter
        std::shared_ptr<Counter> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override { 
            return std::make_shared<Counter>(); 
        }
        // add this stream to the summary
        void streamEndRunSummary(edm::StreamID id, edm::Run const&, edm::EventSetup const&, Counter* runCounter) const override { 
            runCounter->merge(*streamCache(id)); 
        }
        // nothing to do per se
        void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, Counter* runCounter) const override { 
        }
        // write the total to the run 
        void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&, Counter const* runCounter) const override {
            auto out = std::make_unique<nanoaod::MergeableCounterTable>();
            out->addInt("genEventCount", "event count", runCounter->num);
            out->addFloat("genEventSumw", "sum of gen weights", runCounter->sumw);
            out->addFloat("genEventSumw2", "sum of gen (weight^2)", runCounter->sumw2);

            double norm = runCounter->sumw ? 1.0/runCounter->sumw : 1;
            auto sumScales = runCounter->sumScale; for (auto & val : sumScales) val *= norm;
            out->addVFloat("LHEScaleSumw", "Sum of genEventWeight * LHEScaleWeight[i], divided by genEventSumw", sumScales);
            auto sumPDFs = runCounter->sumPDF; for (auto & val : sumPDFs) val *= norm;
            out->addVFloat("LHEPdfSumw", "Sum of genEventWeight * LHEPdfWeight[i], divided by genEventSumw", sumPDFs);
            if (!runCounter->sumNamed.empty()) { // it could be empty if there's no LHE info in the sample
                for (unsigned int i = 0, n = namedWeightLabels_.size(); i < n; ++i) {
                    out->addFloat("LHESumw_"+namedWeightLabels_[i], "Sum of genEventWeight * LHEWeight_"+namedWeightLabels_[i]+", divided by genEventSumw", runCounter->sumNamed[i] * norm);
                }
            }
            iRun.put(std::move(out));
        }
        // nothing to do here
        void globalEndRun(edm::Run const&, edm::EventSetup const&) const override { }

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
            edm::ParameterSetDescription desc;
            desc.add<edm::InputTag>("genEvent", edm::InputTag("generator"))->setComment("tag for the GenEventInfoProduct, to get the main weight");
            desc.add<edm::InputTag>("lheInfo", edm::InputTag("externalLHEProducer"))->setComment("tag for the LHE information (LHEEventProduct and LHERunInfoProduct)");

            edm::ParameterSetDescription prefpdf;
            prefpdf.add<std::string>("name");
            prefpdf.add<uint32_t>("lhaid");
            desc.addVPSet("preferredPDFs", prefpdf, std::vector<edm::ParameterSet>())->setComment("LHA PDF Ids of the preferred PDF sets, in order of preference (the first matching one will be used)");
            desc.add<std::vector<std::string>>("namedWeightIDs")->setComment("set of LHA weight IDs for named LHE weights");
            desc.add<std::vector<std::string>>("namedWeightLabels")->setComment("output names for the namedWeightIDs (in the same order)");
            desc.add<int32_t>("lheWeightPrecision")->setComment("Number of bits in the mantissa for LHE weights");
            desc.add<uint32_t>("maxPdfWeights")->setComment("Maximum number of PDF weights to save (to crop NN replicas)");
            desc.addOptionalUntracked<bool>("debug")->setComment("dump out all LHE information for one event");
            descriptions.add("genWeightsTable", desc);
        }


    protected:
        const edm::EDGetTokenT<GenEventInfoProduct> genTag_;
        const edm::InputTag lheLabel_;
        const edm::EDGetTokenT<LHEEventProduct> lheTag_;
        const edm::EDGetTokenT<LHERunInfoProduct> lheRunTag_;

        std::vector<uint32_t> preferredPDFLHAIDs_;
        std::unordered_map<std::string,uint32_t> lhaNameToID_;
        std::vector<std::string> namedWeightIDs_;
        std::vector<std::string> namedWeightLabels_;
        int lheWeightPrecision_;
        unsigned int maxPdfWeights_;

        mutable std::atomic<bool> debug_, debugRun_, hasIssuedWarning_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenWeightsTableProducer);

