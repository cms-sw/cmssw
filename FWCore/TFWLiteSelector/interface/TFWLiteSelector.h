#ifndef FWCore_TFWLiteSelector_TFWLiteSelector_h
#define FWCore_TFWLiteSelector_TFWLiteSelector_h
// -*- C++ -*-
//
// Package:     TFWLiteSelector
// Class  :     TFWLiteSelector
// 
/**\class TFWLiteSelector TFWLiteSelector.h FWCore/TFWLiteSelector/interface/TFWLiteSelector.h

 Description: A 'safe' form of a TSelector which uses a Worker helper class to do the processing

 Usage:
    This is a safe form of a TSelector which separates the processing (which could happen on many 
    computers when using PROOF) from the handling of the final result (which only happens on the
    original computer which is running the ROOT job).

    The processing is handled by a worker class.  This class is expected to have the following methods
    1) a constructor which takes a 'const TList*' and a 'TList&' as its arguments.  The 'const TList*' holds
       objects The 'TList&' is used to hold all the
       'TObject' items (e.g. histograms) you want to access for the final result (e.g. the sum of all 
       histograms created by the many Workers running on different computers).  You should create the
       items in the constructor, hold onto them as member data in the Worker and 'Add' them to the TList.
       In addition, the 'TList&' can hold items sent to the workers from the TFWLiteSelector. 
    2) a method called 'process(const edm::Event&)'
       this is called for each Event
    3) a destructor which does what ever you want to have done after all the Events have finished

    You should inherit from TFWLiteSelector<...> where the template argument should be the worker you want
    to use.  You need to implement the following methods
    1) 'begin(const TList*& itemsForProcessing)'
      this is called before processing has started and before any workers get created.  If you want to pass data to
      your workers, you can create a new TList and assign it to 'itemsForProcessing' and then add the objects you 
      want passed into that list. 
      NOTE: you are responsible for deleting the created TList and for deleting all items held by the TList. The easiest
      way to do this is to add a 'std::auto_ptr<TList>' member data to your Selector and then call 'SetOwner()' on the TList.
    2) 'terminate(TList&)'
       this is called after all processing has finished.  The TList& contains all the accumulated information
       from all the workers.
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 30 21:04:46 CDT 2006
//

// system include files
class TList;

// user include files

#include "boost/shared_ptr.hpp"

#include "FWCore/TFWLiteSelector/interface/TFWLiteSelectorBasic.h"

// forward declarations
template <class TWorker>
class TFWLiteSelector : public TFWLiteSelectorBasic
{

   public:
      TFWLiteSelector() : worker_() {}
      virtual ~TFWLiteSelector() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      TFWLiteSelector(const TFWLiteSelector&); // stop default

      const TFWLiteSelector& operator=(const TFWLiteSelector&); // stop default

      virtual void preProcessing(const TList*in, TList& out) {
        worker_.reset(new TWorker(in,out));
      }
      virtual void process(const edm::Event& iEvent) {
        worker_->process(iEvent);
      }
      virtual void postProcessing(TList& out) {
        worker_->postProcess(out);
      }
      
      // ---------- member data --------------------------------
      boost::shared_ptr<TWorker> worker_;
      ClassDef(TFWLiteSelector,2)
};

#endif
