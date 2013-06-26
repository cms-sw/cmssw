#ifndef UtilAlgos_HistoAnalyzer_h
#define UtilAlgos_HistoAnalyzer_h
/** \class HistoAnalyzer
 *
 * Creates histograms defined in config file 
 *
 * \author: Benedikt Hegner, DESY
 * 
 * Template parameters:
 * - C : Concrete candidate collection type
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/UtilAlgos/interface/ExpressionHisto.h"

template<typename C>
class HistoAnalyzer : public edm::EDAnalyzer {
public:
  /// constructor from parameter set
  HistoAnalyzer( const edm::ParameterSet& );
  /// destructor
  ~HistoAnalyzer();
  
protected:
  /// process an event
  virtual void analyze( const edm::Event&, const edm::EventSetup&) override;

private:
  /// label of the collection to be read in
  edm::InputTag src_;
  /// Do we weight events?
  bool usingWeights_;
  /// label of the weight collection (can be null for weights = 1)
  edm::InputTag weights_;
  /// vector of the histograms
  std::vector<ExpressionHisto<typename C::value_type>* > vhistograms;
};

template<typename C>
HistoAnalyzer<C>::HistoAnalyzer( const edm::ParameterSet& par ) : 
  src_(par.template getParameter<edm::InputTag>("src")),
  usingWeights_(par.exists("weights")),
  weights_(par.template getUntrackedParameter<edm::InputTag>("weights", edm::InputTag("fake")))
{
   edm::Service<TFileService> fs;
   std::vector<edm::ParameterSet> histograms = 
                                   par.template getParameter<std::vector<edm::ParameterSet> >("histograms");
   std::vector<edm::ParameterSet>::const_iterator it = histograms.begin();
   std::vector<edm::ParameterSet>::const_iterator end = histograms.end();

   // create the histograms from the given parameter sets 
   for (; it!=end; ++it)
   {
      ExpressionHisto<typename C::value_type>* hist = new ExpressionHisto<typename C::value_type>(*it);
      hist->initialize(*fs);
      vhistograms.push_back(hist);
   }   

}

template<typename C>
HistoAnalyzer<C>::~HistoAnalyzer() 
{
   // delete all histograms and clear the vector of pointers
   typename std::vector<ExpressionHisto<typename C::value_type>* >::iterator it = vhistograms.begin(); 
   typename std::vector<ExpressionHisto<typename C::value_type>* >::iterator end = vhistograms.end();
   for (;it!=end; ++it){
     (*it)->~ExpressionHisto<typename C::value_type>();
   }
   vhistograms.clear(); 
}

template<typename C>
void HistoAnalyzer<C>::analyze( const edm::Event& iEvent, const edm::EventSetup&) 
{
   edm::Handle<C> coll;
   iEvent.getByLabel( src_, coll);
   double weight = 1.0;
   if (usingWeights_) { 
       edm::Handle<double> weightColl;
       iEvent.getByLabel( weights_, weightColl ); 
       weight = *weightColl;
   }

   typename std::vector<ExpressionHisto<typename C::value_type>* >::iterator it = vhistograms.begin();
   typename std::vector<ExpressionHisto<typename C::value_type>* >::iterator end = vhistograms.end(); 

   for (;it!=end; ++it){
      uint32_t i = 0;
      for( typename C::const_iterator elem=coll->begin(); elem!=coll->end(); ++elem, ++i ) {
         if (!(*it)->fill( *elem, weight, i )) {
            break;
         }
      }
   }
}

#endif
