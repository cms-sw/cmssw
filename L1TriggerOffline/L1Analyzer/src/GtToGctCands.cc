// -*- C++ -*-
//
// Package:    GtToGctCands
// Class:      GtToGctCands
// 
/**\class GtToGctCands GtToGctCands.cc L1TriggerOffline/L1Analyzer/src/GtToGctCands.cc

 Description: Convert GT candidates (electrons and jets) to GCT format

*/
//
// Original Author:  Alex Tapper
//         Created:  Mon Mar 30 17:31:03 CEST 2009
// $Id: GtToGctCands.cc,v 1.2 2009/04/01 15:10:00 georgia Exp $
//
//

#include "L1TriggerOffline/L1Analyzer/interface/GtToGctCands.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

GtToGctCands::GtToGctCands(const edm::ParameterSet& iConfig) :
  m_GTInputTag(iConfig.getParameter<edm::InputTag>("inputLabel"))
{
  using namespace l1extra;

  // For now I am making one electron collection and one jet collection with all electrons and jets from all 3 BXs.
  // This is the easiest format to analyse for CRAFT data.
  // In the future I should make different collections and treat the mutiple BXs properly, and add energy sums.
  //  produces<L1GctEmCandCollection>();
  produces<L1GctJetCandCollection>();

  // Current hack since GCT collections are not in the RECO I make the L1Extra particles here too
  produces<L1JetParticleCollection>( "Tau" ) ;
}

GtToGctCands::~GtToGctCands(){}

void GtToGctCands::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace l1extra;
  using namespace edm;
  using namespace std;
  
  // create em and jet collections
  std::auto_ptr<L1GctEmCandCollection> allElectrons (new L1GctEmCandCollection());
  std::auto_ptr<L1GctJetCandCollection> allJets (new L1GctJetCandCollection());

  // Get GT data
  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  iEvent.getByLabel(m_GTInputTag,gtrr_handle);
  L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();

  // Loop over 3BXs (shouldn't be hard coded really) and get GT cands
  // For now only make non-iso electrons and tau jet collections to be fed into L1Extra
  for (int ibx=-1; ibx<=1; ibx++) {
    const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d, ibx);
    
     std::vector<int> psbel;
     psbel.push_back(psb.aData(4));
     psbel.push_back(psb.aData(5));
     psbel.push_back(psb.bData(4));
     psbel.push_back(psb.bData(5));
     for(std::vector<int>::const_iterator ipsbel=psbel.begin(); ipsbel!=psbel.end(); ipsbel++) {
       allElectrons->push_back(L1GctEmCand((*ipsbel)&0x3f,((*ipsbel)>>10)&0x1f,(((*ipsbel)>>6)&7) * ( ((*ipsbel>>9)&1) ? -1 : 1 ),0));
     }
    
     std::vector<int> psbjet;
     psbjet.push_back(psb.aData(2));
     psbjet.push_back(psb.aData(3));
     psbjet.push_back(psb.bData(2));
     psbjet.push_back(psb.bData(3));
     for(std::vector<int>::const_iterator ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
       allJets->push_back(L1GctJetCand((*ipsbjet)&0x3f,((*ipsbjet)>>10)&0x1f,(((*ipsbjet)>>6)&7) * ( ((*ipsbjet>>9)&1) ? -1 : 1 ),1,0));
     }
  }

  // Put the new collections into the event
  iEvent.put(allElectrons);
  iEvent.put(allJets);

  // Get the collection out again... sigh
  edm::Handle<L1GctJetCandCollection> jets;
  iEvent.getByLabel("gctCandsFromGt",jets);

  // Create L1Extra collections
  auto_ptr <L1JetParticleCollection> tauJetColl(new L1JetParticleCollection);

  // Geometry
  ESHandle <L1CaloGeometry> caloGeomESH ;
  iSetup.get <L1CaloGeometryRecord>().get( caloGeomESH ) ;
  const L1CaloGeometry* caloGeom = &( *caloGeomESH ) ;

  // L1 jet scale
  ESHandle <L1CaloEtScale> jetScale ;
  iSetup.get <L1JetEtScaleRcd>().get(jetScale) ;

  // Now make the L1Extra collection
  L1GctJetCandCollection::const_iterator jetItr = jets->begin() ;
  L1GctJetCandCollection::const_iterator jetEnd = jets->end() ;
  for( int i = 0 ; jetItr != jetEnd ; ++jetItr, ++i )
    {
      if(!jetItr->empty())
        {
          double et = jetScale->et(jetItr->rank());
          L1JetParticle(gctLorentzVector(et,*jetItr,caloGeom,true),Ref<L1GctJetCandCollection>(jets,i),jetItr->bx());
        }
    }
  
  iEvent.put(tauJetColl,"Tau");
}

math::PtEtaPhiMLorentzVector
GtToGctCands::gctLorentzVector( const double& et,
					const L1GctCand& cand,
					const L1CaloGeometry* geom,
					bool central )
{
   // To keep x and y components non-zero.
   double etCorr = et + 1.e-6 ; // protect against roundoff, not only for et=0

   double eta = geom->etaBinCenter( cand.etaIndex(), central ) ;

//    double tanThOver2 = exp( -eta ) ;
//    double ez = etCorr * ( 1. - tanThOver2 * tanThOver2 ) / ( 2. * tanThOver2 );
//    double e  = etCorr * ( 1. + tanThOver2 * tanThOver2 ) / ( 2. * tanThOver2 );

   double phi = geom->emJetPhiBinCenter( cand.phiIndex() ) ;

//    return math::XYZTLorentzVector( etCorr * cos( phi ),
// 				   etCorr * sin( phi ),
// 				   ez,
// 				   e ) ;
   return math::PtEtaPhiMLorentzVector( etCorr,
					eta,
					phi,
					0. ) ;
}

