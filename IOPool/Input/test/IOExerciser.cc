// -*- C++ -*-
//
//         Package:  IOPool/Input
//           Class:  IOExerciser
// Original Author:  Brian Bockelman
//         Created:  Mon Jun  4 17:35:30 CDT 2012
// 
/*
 Description: Read out a fixed subset of an input file

 Implementation:

Much of the interaction with the framework is from EventContentAnalyzer and AsciiOutputModule

See also IOPool/Input/doc/IOExerciser-README for a more detailed description of how
to use this plugin.

*/


// system include files
#include <memory>

// user include files
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TBranch.h"
#include "TTree.h"

#include "ProductInfo.h"

class IOExerciser : public edm::OutputModule {
   public:
      explicit IOExerciser(const edm::ParameterSet&);
      ~IOExerciser();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      enum SelectionStrategy { SmallestFirst = 0,
                               LargestFirst,
                               RoundRobin };

   private:

      // ----------required OutputModule functions-----------------------------
      virtual void write(edm::EventPrincipal const &e, const edm::ModuleCallingContext*);
      virtual void writeRun(edm::RunPrincipal const&, const edm::ModuleCallingContext*){}
      virtual void writeLuminosityBlock(edm::LuminosityBlockPrincipal const&, const edm::ModuleCallingContext*){}

      virtual void respondToOpenInputFile(edm::FileBlock const& fb);

      // ----------internal implementation functions---------------------------
      void computeProducts(edm::EventPrincipal const& e);
      void fillSmallestFirst(ProductInfos const& all_products, Long64_t threshold);
      void fillLargestFirst(ProductInfos const& all_products, Long64_t threshold);
      void fillRoundRobin(ProductInfos const& all_products, Long64_t threshold);

      // ----------member data ------------------------------------------------
      bool m_fetchedProducts;
      TTree *m_eventsTree;
      ProductInfos m_products;
      ProductInfos m_all_products;
      unsigned int m_percentBranches;
      SelectionStrategy m_selectionStrategy;
      Long64_t m_currentUsage;
      const unsigned int m_triggerFactor;
      unsigned int m_triggerCount;

};

//
// constructors and destructor
//
IOExerciser::IOExerciser(const edm::ParameterSet& pset) :
   OutputModule(pset),
   m_fetchedProducts(false),
   m_eventsTree(NULL),
   m_percentBranches(pset.getUntrackedParameter<unsigned int>("percentBranches")),
   m_currentUsage(0),
   m_triggerFactor(pset.getUntrackedParameter<unsigned int>("triggerFactor")),
   m_triggerCount(0)
{
   //now do what ever initialization is needed
   std::string const& selectionStrategy = pset.getUntrackedParameter<std::string>("selectionStrategy");
   if (selectionStrategy == "smallestFirst") {
      m_selectionStrategy = SmallestFirst;
   } else if (selectionStrategy == "largestFirst") {
      m_selectionStrategy = LargestFirst;
   } else if (selectionStrategy == "roundRobin") {
      m_selectionStrategy = RoundRobin;
   } else {
      edm::Exception ex(edm::errors::Configuration);
      ex << "Invalid IOExerciser selection strategy: " << selectionStrategy;
      throw ex;
   }
   if ((m_percentBranches < 1) || (m_percentBranches > 100)) {
      edm::Exception ex(edm::errors::Configuration);
      ex << "Invalid value for percentBranches (" << m_percentBranches << "); must be between 1 and 100, inclusive";
      throw ex;
   }
}


IOExerciser::~IOExerciser()
{
}


//
// member functions
//

// ------------ method called for each event  ------------
void
IOExerciser::write(edm::EventPrincipal const& e, const edm::ModuleCallingContext* context)
{
   using namespace edm;
   if (!m_fetchedProducts)
   {
      computeProducts(e);
   }

   ModuleDescription desc;
   Event event(const_cast<EventPrincipal&>(e), desc, context);

   m_triggerCount += 1;

   int ctr = 0;

   ProductInfos &products_to_use = (m_triggerCount == m_triggerFactor) ? m_all_products : m_products;
   if (m_triggerCount == m_triggerFactor) {
      m_triggerCount = 0;
   }

   for (ProductInfos::iterator it = products_to_use.begin(); it != products_to_use.end(); ++it)
   {
      GenericHandle handle(it->className());

      event.getByLabel(it->tag(), handle);
      ctr ++;
   }
   edm::LogInfo("IOExerciser") << "IOExerciser read out " << ctr << " products.";
   
}

// ------------ method called when starting to processes a run  ------------
void 
IOExerciser::respondToOpenInputFile(edm::FileBlock const& fb)
{
   TTree *eventsTree = fb.tree();
   if (!eventsTree)
   {
      edm::Exception ex(edm::errors::ProductNotFound);
      ex << "IOExerciser was run with a TFile missing an events TTree.";
      throw ex;
   }
   m_eventsTree = eventsTree;

   m_fetchedProducts = false;
}

void
IOExerciser::computeProducts(edm::EventPrincipal const& e)
{
   using namespace edm;
   typedef std::vector<Provenance const*> Provenances;

   m_fetchedProducts = true;
   Provenances provenances;
   e.getAllProvenance(provenances);

   if (!m_eventsTree)
   {
      edm::Exception ex(edm::errors::ProductNotFound);
      ex << "IOExerciser invoked computeProducts without an events TTree.";
      throw ex;
   }

   m_all_products.clear();
   m_all_products.reserve(provenances.size());
   Long64_t totalSize = 0;
   for (Provenances::iterator itProv = provenances.begin(), itProvEnd = provenances.end();
       itProv != itProvEnd;
       ++itProv) {

      const std::string & branchName = (*itProv)->branchName();

      TBranch * branch = (TBranch*)m_eventsTree->GetBranch(branchName.c_str());
      if (!branch)
      {
         LogWarning("IOExerciser") << "Ignoring missing branch " << branchName;
         continue;
      }
      ProductInfo pi(*(*itProv), *branch);
      totalSize += pi.size();
      m_all_products.push_back(pi);
   }

   Long64_t threshold = m_percentBranches*totalSize/100;
   LogDebug("IOExerciser") << "Threshold is " << threshold << " of " << totalSize << " bytes.";

   std::sort(m_all_products.begin(), m_all_products.end(), ProductInfo::sort);

   m_products.clear();
   m_currentUsage = 0;

   switch (m_selectionStrategy)
   {
      case SmallestFirst:
         fillSmallestFirst(m_all_products, threshold);
         break;
      case LargestFirst:
         fillLargestFirst(m_all_products, threshold);
         break;
      case RoundRobin:
         fillRoundRobin(m_all_products, threshold);
          break;
   }

   LogInfo("IOExerciser") << "Reading " << m_products.size() << " of " << m_all_products.size() << " products.  Aggregate branch size is " << m_currentUsage << " of " << totalSize << " bytes.";

}

void
IOExerciser::fillSmallestFirst(ProductInfos const& all_products, Long64_t threshold)
{
   for (ProductInfos::const_iterator it = all_products.begin(), itEnd = all_products.end();
       (it != itEnd) && (m_currentUsage < threshold);
       ++it) {

       m_products.push_back(*it);
       m_currentUsage += it->size();
       LogDebug("IOExerciser") << "Adding label " << it->tag().label() << ", size " << it->size() << "; current usage is " << m_currentUsage << " of " << threshold << " bytes.";
   }
}

void
IOExerciser::fillLargestFirst(ProductInfos const& all_products, Long64_t threshold)
{
   m_currentUsage = 0;
   for (ProductInfos::const_iterator it = --all_products.end(), itBegin = all_products.begin();
       m_currentUsage < threshold;
       --it) {
   
       m_products.push_back(*it);
       m_currentUsage += it->size();
       LogDebug("IOExerciser") << "Adding label " << it->tag().label() << ", size " << it->size() << "; current usage is " << m_currentUsage << " of " << threshold << " bytes.";
       if (it == itBegin) {
          break;
       }
   }
}

void
IOExerciser::fillRoundRobin(ProductInfos const& all_products, Long64_t threshold)
{
   size_t currentSmallest = 0, currentLargest = all_products.size()-1;
   bool useSmallest = true;
   while (m_currentUsage < threshold)
   {
      if (useSmallest) {
         ProductInfo const& pi = all_products[currentSmallest];
         m_currentUsage += pi.size();
         m_products.push_back(pi);
         currentSmallest++;
         useSmallest = false;
         LogDebug("IOExerciser") << "Adding label " << pi.tag().label() << ", size " << pi.size() << "; current usage is " << m_currentUsage << " of " << threshold << " bytes.";
      } else
      {
         ProductInfo const& pi = all_products[currentLargest];
         m_currentUsage += pi.size();
         m_products.push_back(pi);
         currentLargest--;
         useSmallest = true;
         LogDebug("IOExerciser") << "Adding label " << pi.tag().label() << ", size " << pi.size() << "; current usage is " << m_currentUsage << " of " << threshold << " bytes.";
      }
   }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
IOExerciser::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   //The following says we do not know what parameters are allowed so do no validation
   // Please change this to state exactly what you do use, even if it is no parameters
   edm::ParameterSetDescription desc;
   desc.setComment("Reads a configurable subset of EDM files.");
   desc.addUntracked<unsigned int>("percentBranches", 100)
     ->setComment("Control the percent of branches IOExerciser will read out.\n"
                  "Branches are weighted by size.  Valid values are between 1 and 100.\n"
                  "Additional branches will be read out until at least this percent has\n"
                  "been read; thus, IOExerciser will likely read out more than this amount.");
   desc.addUntracked<std::string>("selectionStrategy", "smallestFirst")
     ->setComment("Control the branch selection strategy:\n"
                  "'smallestFirst' (default): Read branches in increasing order of size until limit is hit.\n"
                  "'largestFirst': Read branches in decreasing order of size until limit is hit.\n"
                  "'roundRobin': Read a small branch, then large branch.  Repeat until size limit is hit.");
   desc.addUntracked<unsigned int>("triggerFactor", 0)
      ->setComment("Controls the trigger rate.  Once every 'triggerFactor' events, IOExerciser\n"
                   "will read out all event data, not just the selected branches.  Setting to 10\n"
                   "will cause it to read out one event in 10.  Setting it to zero would mean to\n"
                   "disable trigger behavior completely.  Defaults to 0.");
   OutputModule::fillDescription(desc);
   descriptions.add("IOExerciser", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(IOExerciser);

