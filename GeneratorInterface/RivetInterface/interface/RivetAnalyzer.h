#ifndef GeneratorInterface_RivetInterface_RivetAnalyzer
#define GeneratorInterface_RivetInterface_RivetAnalyzer

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
//#include "Rivet/RivetAIDA.fhh"
#include "Rivet/AnalysisHandler.hh"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//#include "Rivet/RivetAIDA.hh"
//#include "LWH/AIHistogramFactory.h"
//#include "LWH/VariAxis.h"

#include "Rivet/Tools/RivetYODA.hh"
#include "YODA/ROOTCnv.h"



#include <vector>
#include <string>

namespace edm{
  class ParameterSet;
  class Event;
  class EventSetup;
  class InputTag;
}

class RivetAnalyzer : public edm::EDAnalyzer
{
  public:
  RivetAnalyzer(const edm::ParameterSet&);

  virtual ~RivetAnalyzer();

  virtual void beginJob();

  virtual void endJob();  

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run&, const edm::EventSetup&);

  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  
  private:

  //void normalizeTree(AIDA::ITree& tree);
  //template<class AIDATYPE, class ROOTTYPE> ROOTTYPE* prebook(const AIDATYPE*, const std::string&);
  //template<class AIDATYPE, class ROOTTYPE> ROOTTYPE* aida2root(const AIDATYPE*, const std::string&); 
  
  void normalizeTree();
  //template<class YODATYPE, class ROOTTYPE> ROOTTYPE* prebook(const YODATYPE*, const std::string&);
  //template<class YODATYPE, class ROOTTYPE> ROOTTYPE* aida2root(const YODATYPE*, const std::string&); 
  

  edm::InputTag            _hepmcCollection;
  bool                     _useExternalWeight;
  edm::InputTag            _genEventInfoCollection;
  Rivet::AnalysisHandler   _analysisHandler;   
  bool                     _isFirstEvent;
  std::string              _outFileName;
  bool                     _doFinalize;
  bool                     _produceDQM;

  DQMStore *dbe;
  std::vector<MonitorElement *> _mes;
};


/*
//template<class AIDATYPE, class ROOTTYPE> 
template<class YODATYPE, class ROOTTYPE> 
ROOTTYPE* 
  //RivetAnalyzer::prebook(const AIDATYPE* aidah, const std::string& name){
  RivetAnalyzer::prebook(const YODATYPE* yodah, const std::string& name){
  ROOTTYPE* h = 0;
  //if (aidah->axis().isFixedBinning() ) {//equidistant binning (easier case)
  if (yodah->axis().isFixedBinning() ) {//equidistant binning (easier case)
    //int nbins = aidah->axis().bins();
    int nbins = yodah->axis().bins();
    //h = new ROOTTYPE(name.c_str(), name.c_str(), nbins, aidah->axis().lowerEdge(), aidah->axis().upperEdge());
    h = new ROOTTYPE(name.c_str(), name.c_str(), nbins, yodah->axis().lowerEdge(), yodah->axis().upperEdge());
  } else {
    //int nbins = aidah->axis().bins();
    int nbins = yodah->axis().bins();
    //need to dyn cast, IAxis lacks polymorfism
    //const LWH::VariAxis* vax = dynamic_cast<const LWH::VariAxis*>(&aidah->axis());
    const YODA::Axis1D* vax = dynamic_cast<const YODA::Axis1D*>(&yodah->axis());
    if (! vax ){
      throw cms::Exception("RivetAnalyzer") << "Cannot dynamix cast an YODA axis to VariAxis ";
    }
    double* bins = new double[nbins+1];
    for (int i=0; i<nbins; ++i) {
      bins[i] = vax->binEdges(i).first;
    }
    bins[nbins] = vax->binEdges(nbins-1).second; //take last bin right border
    h = new ROOTTYPE(name.c_str(), name.c_str(), nbins, bins);
    delete bins;
  }
  return h; 
}

template<> 
//TH1F* RivetAnalyzer::aida2root(const AIDA::IHistogram1D* aidah, const std::string& name){
TH1F* RivetAnalyzer::yoda2root(const YODA::IHistogram1D* yodah, const std::string& name){
  //TH1F* h = prebook<AIDA::IHistogram1D, TH1F>(aidah, name);
  TH1F* h = prebook<YODA::Histo1D, TH1F>(yodah, name);
  //for (int i = 0; i < aidah->axis().bins(); ++i){
  for (int i = 0; i < yodah->axis().bins(); ++i){
    //h->SetBinContent(i+1, aidah->binHeight(i));
    //h->SetBinError(i+1, aidah->binError(i));
    h->SetBinContent(i+1, yodah->binHeight(i));
    h->SetBinError(i+1, yodah->binError(i));
  }  
  return h;
}

template<>
//TProfile* RivetAnalyzer::aida2root(const AIDA::IProfile1D* aidah, const std::string& name){
TProfile* RivetAnalyzer::yoda2root(const YODA::IProfile1D* yodah, const std::string& name){
  //TProfile* h = prebook<AIDA::IProfile1D, TProfile>(aidah, name);
  TProfile* h = prebook<YODA::IProfile1D, TProfile>(yodah, name);
  //for (int i = 0; i < aidah->axis().bins(); ++i){
  for (int i = 0; i < yodah->axis().bins(); ++i){
    //h->SetBinContent(i+1, aidah->binMean(i));
    //h->SetBinError(i+1, aidah->binRms(i));
    h->SetBinContent(i+1, yodah->binMean(i));
    h->SetBinError(i+1, yodah->binRms(i));
  }
  return h;
}
*/

#endif
