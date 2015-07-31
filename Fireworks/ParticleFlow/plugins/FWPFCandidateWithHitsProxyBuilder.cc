
#define protected public
#include "TEveBoxSet.h"
#undef protected
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"
#include "TEveProjectionBases.h"

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
   const static std::string cname("particleFlowRecHitHCALUpgrade");

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

void  editLineInLineSet(TEveChunkManager::iterator& li, const float* p, int i1, int i2)
{
   TEveStraightLineSet::Line_t& line = * (TEveStraightLineSet::Line_t*) li();
   i1 *= 3;
   i2 *= 3;
   for (int i = 0; i < 3 ; ++i) {
      line.fV1[0+i] = p[i1+i];
      line.fV2[0+i] = p[i2+i];
   }

   li.next();
}

void editBoxInLineSet(TEveChunkManager::iterator& li, const float* p)
{
    
   for (int i = 0; i < 5; i+=4)
   {
      editLineInLineSet(li, p, 0+i, 1+i);

      editLineInLineSet(li, p, 1+i, 2+i);
      editLineInLineSet(li, p, 2+i, 3+i);
      editLineInLineSet(li, p, 3+i, 0+i);
   }
   for (int i = 0; i < 4; ++i)
      editLineInLineSet(li, p, 0+i, 4+i);
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
   initPFRecHitsCollections();
   for( reco::PFCandidateCollection::const_iterator it = candidates->begin(), itEnd = candidates->end(); it != itEnd; ++it, ++idx)
   {  
      TEveCompound* comp = createCompound();
      setupAddElement( comp, product );
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
void FWPFCandidateWithHitsProxyBuilder::initPFRecHitsCollections()
{  
   // ref hcal collections
   edm::Handle<reco::PFRecHitCollection> handle_hits;


   m_collectionHCAL =0;
   try
   {
      // edm::InputTag tag("particleFlowRecHitHCAL");
      edm::InputTag tag(cname);
      item()->getEvent()->getByLabel(tag, handle_hits);
      if (handle_hits.isValid())
      {
         m_collectionHCAL = &*handle_hits;
      }
      else
      {
         fwLog(fwlog::kError) <<"FWPFCandidateWithHitsProxyBuilder, item " << item()->name() <<": Failed to access collection with name " << cname << "." << std::endl;
      }
   }
   catch (...)
   {
      fwLog(fwlog::kError) <<"FWPFCandidateWithHitsProxyBuilder, item " << item()->name() <<": Failed to access collection with name " << cname << "." << std::endl;
   }
}

//______________________________________________________________________________
void FWPFCandidateWithHitsProxyBuilder::viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const reco::PFRecHit*)
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
const reco::PFRecHit* FWPFCandidateWithHitsProxyBuilder::getHitForDetId(unsigned candIdx)
{

   for (reco::PFRecHitCollection::const_iterator it = m_collectionHCAL->begin(); it != m_collectionHCAL->end(); ++it)
   {

      if ( it->detId() == candIdx)
      {
         return  &(*it);
      }
   }
   return 0;
}

//______________________________________________________________________________
void FWPFCandidateWithHitsProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   std::vector<float> scaledCorners(24);

   float scale = vc->getEnergyScale()->getScaleFactor3D()/50;
   for (TEveElement::List_i i=parent->BeginChildren(); i!=parent->EndChildren(); ++i)
   {
      if ((*i)->NumChildren() > 1)
      {
         TEveElement::List_i xx =  (*i)->BeginChildren(); ++xx;
         TEveBoxSet* boxset = dynamic_cast<TEveBoxSet*>(*xx);
         ++xx;
         TEveStraightLineSet* lineset = dynamic_cast<TEveStraightLineSet*>(*xx);
         TEveChunkManager::iterator li(lineset->GetLinePlex());
         li.next();


         TEveChunkManager* plex = boxset->GetPlex();
         if (plex->N())
         {
            for (int atomIdx=0; atomIdx < plex->Size(); ++atomIdx)
            {
              
               TEveBoxSet::BFreeBox_t* atom = (TEveBoxSet::BFreeBox_t*)boxset->GetPlex()->Atom(atomIdx);
               reco::PFRecHit* hit = (reco::PFRecHit*)boxset->GetUserData(atomIdx);
               const float* corners = item()->getGeom()->getCorners(hit->detId());
               viewContextBoxScale(corners, hit->energy()*scale, vc->getEnergyScale()->getPlotEt(), scaledCorners, hit);
               memcpy(atom->fVertices, &scaledCorners[0], sizeof(atom->fVertices));

               editBoxInLineSet(li, &scaledCorners[0]);
            }

            for (TEveProjectable::ProjList_i p = lineset->BeginProjecteds(); p != lineset->EndProjecteds(); ++p)
            {
               TEveStraightLineSetProjected* projLineSet = (TEveStraightLineSetProjected*)(*p);
               projLineSet->UpdateProjection();
            }
         }
      }
   }
}  
//______________________________________________________________________________
namespace {
TString boxset_tooltip_callback(TEveDigitSet* ds, Int_t idx)
{
   void* ud = ds->GetUserData(idx);
   if (ud);
   {
      reco::PFRecHit* hit = (reco::PFRecHit*) ud;
      // printf("idx %d %p hit data %p\n", idx, (void*)hit, ud);
      if (hit)
         return TString::Format("RecHit %d energy '%f'", idx,  hit->energy());
      else
         return "ERROR";
   }
}
}
//______________________________________________________________________________
void FWPFCandidateWithHitsProxyBuilder::addHitsForCandidate(const reco::PFCandidate& cand, TEveElement* holder, const FWViewContext* vc)
{ 
   reco::PFCandidate::ElementsInBlocks eleInBlocks = cand.elementsInBlocks();

   TEveBoxSet* boxset = 0;
   TEveStraightLineSet* lineset = 0;

   for(unsigned elIdx=0; elIdx<eleInBlocks.size(); elIdx++)
   {
      // unsigned ieTrack = 0;
      // unsigned ieECAL = 0;
      unsigned ieHCAL = 0;

      reco::PFBlockRef blockRef = eleInBlocks[elIdx].first;
      unsigned indexInBlock = eleInBlocks[elIdx].second;
      edm::Ptr<reco::PFBlock> myBlock(blockRef.id(),blockRef.get(), blockRef.key());
      /*
        if (myBlock->elements()[indexInBlock].type() == 1)
        ieTrack = indexInBlock;
        if (myBlock->elements()[indexInBlock].type() == 4)
        ieECAL = indexInBlock;
      */
      if (myBlock->elements()[indexInBlock].type() == 5)
         ieHCAL = indexInBlock;
   
 
      std::vector<float> scaledCorners(24);
      float scale = vc->getEnergyScale()->getScaleFactor3D()/50;
      if (ieHCAL &&  m_collectionHCAL) {
         reco::PFClusterRef hcalclusterRef=myBlock->elements()[ieHCAL].clusterRef();
         edm::Ptr<reco::PFCluster> myCluster(hcalclusterRef.id(),hcalclusterRef.get(), hcalclusterRef.key());
         if (myCluster.get())
         {
            const std::vector< std::pair<DetId, float> > & hitsandfracs = myCluster->hitsAndFractions();

            if (!boxset)
            {
               boxset = new TEveBoxSet();
               boxset->Reset(TEveBoxSet::kBT_FreeBox, true, hitsandfracs.size());
               boxset->SetAntiFlick(true);
               boxset->SetAlwaysSecSelect(1);
               boxset->SetPickable(1);
               boxset->SetTooltipCBFoo(boxset_tooltip_callback);
            }

            if (!lineset)
            {
               lineset = new TEveStraightLineSet();
            }

            bool hitsFound = false;
            for ( int ihandf=0, lastIdx=(int)(hitsandfracs.size()); ihandf<lastIdx; ihandf++) 
            {
               unsigned int hitDetId = hitsandfracs[ihandf].first;
               const float* corners = context().getGeom()->getCorners(hitDetId);
               const reco::PFRecHit* hit = getHitForDetId(hitDetId);
               if (hit)
               {
                  viewContextBoxScale( corners, hit->energy()*scale, vc->getEnergyScale()->getPlotEt(), scaledCorners, hit);
                  boxset->AddBox( &scaledCorners[0]);
                  // setup last box
                  boxset->DigitColor(holder->GetMainColor());
                  boxset->DigitUserData((void*)hit);
                  addBoxAsLines(lineset, &scaledCorners[0]);
                  hitsFound = true;
               }
               /*
               // AMT: don't add lines if hit is not found becuse of unconsistency of scaling.
               else
               {
                  addBoxAsLines(lineset, corners);
               }
               */
            }
            if (!hitsFound)
               fwLog(fwlog::kWarning) << Form("Can't find matching hits with for HCAL block %d in %s collection. Number of hits %d.\n", elIdx, cname.c_str(), (int)hitsandfracs.size());


         }
         else
         {
            fwLog(fwlog::kInfo) << "empty cluster \n";
         }
      }
   } // endloop cand.elementsInBlocks();


   if (boxset) {
      boxset->RefitPlex();
      setupAddElement(boxset, holder);
   }

   if (lineset) {
      setupAddElement(lineset, holder);
   }
}

REGISTER_FWPROXYBUILDER(FWPFCandidateWithHitsProxyBuilder, reco::PFCandidateCollection,"PFCandidatesWithHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
