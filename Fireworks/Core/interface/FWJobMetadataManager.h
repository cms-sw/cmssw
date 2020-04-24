#ifndef Fireworks_Core_FWJobMetadataManager
#define Fireworks_Core_FWJobMetadataManager

#include "Fireworks/Core/interface/FWTypeToRepresentations.h"

#include "sigc++/signal.h"

#include <string>
#include <vector>

class FWJobMetadataUpdateRequest;
class FWTypeToRepresentations;

/** Base class which keeps track of various  job specific metadata information.
    fwlite and (eventually) full-framework derived implementations are where 
    the job is actually done.
   */
class FWJobMetadataManager
{
public:
   struct Data
   {
      std::string purpose_;
      std::string type_;
      std::string moduleLabel_;
      std::string productInstanceLabel_;
      std::string processName_;
   };
   
   FWJobMetadataManager();
   virtual ~FWJobMetadataManager();
   
   std::vector<Data> &usableData() { return m_metadata; }
   std::vector<std::string> &processNamesInJob() { return m_processNamesInJob; }

   /** Invoked when a given update request needs to happen. Will
       emit the metadataChanged_ signal when done so that observers can 
       update accordingly.
       
       Derived classes should implement the doUpdate() protected method
       to actually modify the metadata according to the request.
       
       Notice that this method is a consumer of request object and takes
       ownership of the lifetime of the request objects.
     */
   void update(FWJobMetadataUpdateRequest *request);
   
   /** This needs to be invoked to make the metadata manager keep track of
       the changes in the TypeToRepresentation.
     */
   void initReps(const FWTypeToRepresentations& iTypeAndReps);
   
   // needed by FWDetailViewManager
   virtual bool  hasModuleLabel(std::string& moduleLabel) = 0;


   sigc::signal<void>  metadataChanged_;
protected:
   /** This is the bit that needs to be implemented by a derived class 
       to update the various metadata structures.
       
       @return true if any update actually took place.
     */
   virtual bool doUpdate(FWJobMetadataUpdateRequest *) = 0;
   std::vector<Data>        m_metadata;
   std::vector<std::string> m_processNamesInJob;
   FWTypeToRepresentations *m_typeAndReps;
};

#endif
