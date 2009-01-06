// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronGlimpseProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWElectronGlimpseProxyBuilder.cc,v 1.1 2008/12/02 21:11:52 chrjones Exp $
//

// system include files
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Candidates/interface/addStraightLineSegment.h"

class FWElectronGlimpseProxyBuilder: public FWGlimpseSimpleProxyBuilderTemplate<reco::GsfElectron> {
      
public:
   FWElectronGlimpseProxyBuilder();
   //virtual ~FWElectronGlimpseProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
 
private:
   FWElectronGlimpseProxyBuilder(const FWElectronGlimpseProxyBuilder&); // stop default
   
   const FWElectronGlimpseProxyBuilder& operator=(const FWElectronGlimpseProxyBuilder&); // stop default
   
   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   
   // ---------- member data --------------------------------
   
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWElectronGlimpseProxyBuilder::FWElectronGlimpseProxyBuilder()
{
}

// FWElectronGlimpseProxyBuilder::FWElectronGlimpseProxyBuilder(const FWElectronGlimpseProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

//FWElectronGlimpseProxyBuilder::~FWElectronGlimpseProxyBuilder()
//{
//}

//
// assignment operators
//
// const FWElectronGlimpseProxyBuilder& FWElectronGlimpseProxyBuilder::operator=(const FWElectronGlimpseProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWElectronGlimpseProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWElectronGlimpseProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet("","");
   marker->SetLineWidth(2);
   fw::addStraightLineSegment( marker, &iData, 1.0 );
   oItemHolder.AddElement(marker);
   //add to scaler at end so that it can scale the line after all ends have been added
   scaler()->addElement(marker);   
}

//
// const member functions
//

//
// static member functions
//

REGISTER_FWGLIMPSEDATAPROXYBUILDER(FWElectronGlimpseProxyBuilder,std::vector<reco::GsfElectron>,"Electrons");
