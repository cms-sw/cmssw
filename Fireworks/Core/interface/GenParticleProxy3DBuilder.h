#ifndef Fireworks_Muons_GenParticleProxy3DBuilder_h
#define Fireworks_Muons_GenParticleProxy3DBuilder_h
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: GenParticleProxy3DBuilder.h,v 1.3 2008/07/05 22:56:33 dmytro Exp $
//

// system include files

class TEveElementList;
class FWEventItem;
class TDatabasePDG;

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

class GenParticleProxy3DBuilder : public FWRPZDataProxyBuilder
{

   public:
      GenParticleProxy3DBuilder();
      virtual ~GenParticleProxy3DBuilder() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);

      GenParticleProxy3DBuilder(const GenParticleProxy3DBuilder&); // stop default

      const GenParticleProxy3DBuilder& operator=(const GenParticleProxy3DBuilder&); // stop default

      // ---------- member data --------------------------------
      TDatabasePDG* m_pdg;
};

#endif
