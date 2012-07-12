
#define protected public
#include "TEveBoxSet.h"
#undef protected
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/ParticleFlow/plugins/FWPFCandidateWithHitsProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/FWLite/interface/Handle.h"

namespace
{
void  addLineToLineSet(TEveStraightLineSet* ls, const float* p, int i1, int i2)
{
   i1 *= 3;
   i2 *= 3;
   ls->AddLine(p[i1], p[i1+1], p[i1+2], p[i2], p[i2+1], p[i2+2]);
}


void addBoxAsLines(TEveStraightLineSet* lineset, const float* p)
{
   for (int l = 0; l < 5; l+=4)
   {
      addLineToLineSet(lineset, p, 0+l, 1+l);
      addLineToLineSet(lineset, p, 1+l, 2+l);
      addLineToLineSet(lineset, p, 2+l, 3+l);
      addLineToLineSet(lineset, p, 3+l, 0+l);
   }
   for (int l = 0; l < 4; ++l)
      addLineToLineSet(lineset, p, 0+l, 4+l);
}
}

 //______________________________________________________________________________
void 
FWPFCandidateWithHitsProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   // init PFCandiate collection
   reco::PFCandidateCollection const * candidates = 0;
   iItem->get( candidates );
   if( candidates == 0 ) return;

 
   Int_t idx = 0;
   initCaloRecHitsCollections();
   for( reco::PFCandidateCollection::const_iterator it = candidates->begin(), itEnd = candidates->end(); it != itEnd; ++it, ++idx)
   {  
      TEveCompound* comp = createCompound();
      setupAddElement( comp, product );
      comp->IncDenyDestroy();
      // printf("products size %d/%d \n", (int)iItem->size(), product->NumChildren());

      const reco::PFCandidate& cand = *it;
      
      // track
      {
         TEveRecTrack t;
         t.fBeta = 1.;
         t.fP = TEveVector( cand.px(), cand.py(), cand.pz() );
         t.fV = TEveVector( cand.vertex().x(), cand.vertex().y(), cand.vertex().z() );
         t.fSign = cand.charge();
         TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator() );
         trk->MakeTrack();
         fireworks::setTrackTypePF( cand, trk );
         setupAddElement( trk, comp);
      }
      // hits
      {
         comp->SetMainColor(iItem->defaultDisplayProperties().color());
         addHitsForCandidate(cand, comp, vc);
      }

   }
}

//______________________________________________________________________________
void FWPFCandidateWithHitsProxyBuilder::initCaloRecHitsCollections()
{  
   // ref hcal collections
   edm::Handle<HBHERecHitCollection> handle_hits;

   m_collectionHBHE =0;
   try
   {
      edm::InputTag tag("hbhereco");
      item()->getEvent()->getByLabel(tag, handle_hits);
      if (handle_hits.isValid())
      {
         m_collectionHBHE = &*handle_hits;
      }
   }
   catch (...)
   {
      fwLog(fwlog::kWarning) <<"FWPFCandidateWithHitsProxyBuilder::build():: Failed to access hbhereco collection." << std::endl;
   }
}

//______________________________________________________________________________
void FWPFCandidateWithHitsProxyBuilder::viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const CaloRecHit*)
{
   static TEveVector vtmp;
   vtmp.Set(0.f, 0.f, 0.f);
   for( unsigned int i = 0; i < 24; i += 3 )
   {	 
      vtmp[0] += corners[i];
      vtmp[1] += corners[i + 1];
      vtmp[2] += corners[i + 2];
   }
   vtmp *= 1.f/8.f;

   if (plotEt)
   {
      scale *= vtmp.Perp()/vtmp.Mag();
   }

   // Coordinates for a scaled version of the original box
   for( unsigned int i = 0; i < 24; i += 3 )
   {	
      scaledCorners[i] = vtmp[0] + ( corners[i] - vtmp[0] ) * scale;
      scaledCorners[i + 1] = vtmp[1] + ( corners[i + 1] - vtmp[1] ) * scale;
      scaledCorners[i + 2] = vtmp[2] + ( corners[i + 2] - vtmp[2] ) * scale;
   }
}

//______________________________________________________________________________
const CaloRecHit* FWPFCandidateWithHitsProxyBuilder::getHitForDetId(uint32_t candIdx)
{

   for (HBHERecHitCollection::const_iterator it = m_collectionHBHE->begin(); it != m_collectionHBHE->end(); ++it)
   {
      unsigned int x = it->detid();
      if ( x == candIdx)
      {
         return  &(*it);
      }
   }
   return 0;
}

//______________________________________________________________________________
void FWPFCandidateWithHitsProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   /*
   std::vector<float> scaledCorners(24);

   float scale = vc->getEnergyScale()->getScaleFactor3D()/50;
   for (TEveElement::List_i i=parent->BeginChildren(); i!=parent->EndChildren(); ++i)
   {
      if ((*i)->NumChildren() > 1)
      {
         TEveElement::List_i xx =  (*i)->BeginChildren(); ++xx;
         TEveBoxSet* boxset = dynamic_cast<TEveBoxSet*>(*xx);
         TEveChunkManager* plex = boxset->GetPlex();
         //         printf("=== boxset %p N:%d Size:%d\n",(void*)plex,  plex->N(),  plex->Size());
         if (plex->N())
         {
            for (int atomIdx=0; atomIdx < plex->Size(); ++atomIdx)
            {
               TEveBoxSet::BFreeBox_t* atom = (TEveBoxSet::BFreeBox_t*)boxset->GetPlex()->Atom(atomIdx);
               // printf("atom %d %p\n", atomIdx, (void*)atom);
               // printf("%d corner %f \n", atomIdx, atom->fVertices[0][0]);
            }

         }
      }
   }
   */
}  

//______________________________________________________________________________
void FWPFCandidateWithHitsProxyBuilder::addHitsForCandidate(const reco::PFCandidate& cand, TEveElement* holder, const FWViewContext* vc)
{ 
   reco::PFCandidate::ElementsInBlocks eleInBlocks = cand.elementsInBlocks();
  
   for(unsigned elIdx=0; elIdx<eleInBlocks.size(); elIdx++)
   {
      unsigned ieTrack = 0;
      unsigned ieECAL = 0;
      unsigned ieHCAL = 0;

      reco::PFBlockRef blockRef = eleInBlocks[elIdx].first;
      unsigned indexInBlock = eleInBlocks[elIdx].second;
      edm::Ptr<reco::PFBlock> myBlock(blockRef.id(),blockRef.get(), blockRef.key());
      if (myBlock->elements()[indexInBlock].type() == 1)
         ieTrack = indexInBlock;
      if (myBlock->elements()[indexInBlock].type() == 4)
         ieECAL = indexInBlock;
      if (myBlock->elements()[indexInBlock].type() == 5)
         ieHCAL = indexInBlock;
   
 
      std::vector<float> scaledCorners(24);
      float scale = vc->getEnergyScale()->getScaleFactor3D()/50;
      if (ieHCAL) {
         reco::PFClusterRef hcalclusterRef=myBlock->elements()[ieHCAL].clusterRef();
         edm::Ptr<reco::PFCluster> myCluster(hcalclusterRef.id(),hcalclusterRef.get(), hcalclusterRef.key());
         if (myCluster.get())
         {
            const std::vector< std::pair<DetId, float> > & hitsandfracs = myCluster->hitsAndFractions();

            TEveBoxSet* boxset = new TEveBoxSet();
            boxset->Reset(TEveBoxSet::kBT_FreeBox, true, hitsandfracs.size());
            boxset->SetAntiFlick(true);
            TEveStraightLineSet* lineset = new TEveStraightLineSet();


            for ( int ihandf=0, lastIdx=(int)(hitsandfracs.size()); ihandf<lastIdx; ihandf++) 
            {
               unsigned int hitDetId = hitsandfracs[ihandf].first;
               const float* corners = context().getGeom()->getCorners(hitDetId);
               const CaloRecHit* hit = getHitForDetId(hitDetId);
               if (hit)
               {
                  viewContextBoxScale( corners, hit->energy()*scale, vc->getEnergyScale()->getPlotEt(), scaledCorners, hit);
                  boxset->AddBox( &scaledCorners[0]);
                  boxset->DigitColor(holder->GetMainColor());
                  boxset->DigitUserData((void*)hit);
                  addBoxAsLines(lineset, &scaledCorners[0]);
               }
               else
               {
                  addBoxAsLines(lineset, corners);

               }
            }
            boxset->RefitPlex();
            if (boxset->GetPlex()->Size() == 0)
               printf("Can't find matching hits with for HCAL block %d in HBHE collection. Number of hits %d.\n", elIdx, (int)hitsandfracs.size());
            else
               printf("boxset plex size N:%d Size %d: hits%d\n", boxset->GetPlex()->N(),  boxset->GetPlex()->Size(),  (int)hitsandfracs.size());
            setupAddElement(boxset, holder);
            setupAddElement(lineset, holder);

         }
         else
         {
            printf("empty cluster \n");
         }
      }
   }
}

REGISTER_FWPROXYBUILDER(FWPFCandidateWithHitsProxyBuilder, reco::PFCandidateCollection,"PF CandidatesWithHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
