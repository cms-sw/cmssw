#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"

#include "TEveBoxSet.h"
#include "TEveCompound.h"

class FWCaloClusterHeatMapProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloCluster>
{
 public:
   FWCaloClusterHeatMapProxyBuilder(void): maxEnergy(1e-5f) {}
   ~FWCaloClusterHeatMapProxyBuilder(void) override {}

   REGISTER_PROXYBUILDER_METHODS();

 private:
   FWCaloClusterHeatMapProxyBuilder(const FWCaloClusterHeatMapProxyBuilder &) = delete;                  // stop default
   const FWCaloClusterHeatMapProxyBuilder &operator=(const FWCaloClusterHeatMapProxyBuilder &) = delete; // stop default

   std::map<DetId, const HGCRecHit*> hitmap;
   float maxEnergy;
   
   void setItem(const FWEventItem *iItem) override;

   void build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *) override;
   void build(const reco::CaloCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *) override;
};

void FWCaloClusterHeatMapProxyBuilder::setItem(const FWEventItem *iItem)
{
   FWProxyBuilderBase::setItem(iItem);
   if (iItem)
   {
      iItem->getConfig()->assertParam("Layer", 0L, 0L, 52L);
      iItem->getConfig()->assertParam("Z+", true);
      iItem->getConfig()->assertParam("Z-", true);
   }
}

void FWCaloClusterHeatMapProxyBuilder::build(const FWEventItem *iItem,
                                             TEveElementList *product, const FWViewContext *vc)
{
   if(hitmap.empty()){
      const edm::EventBase *event = iItem->getEvent();

      edm::Handle<HGCRecHitCollection> recHitHandleEE;
      edm::Handle<HGCRecHitCollection> recHitHandleFH;
      edm::Handle<HGCRecHitCollection> recHitHandleBH;   

      event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCEERecHits" ), recHitHandleEE );
      event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCHEFRecHits" ), recHitHandleFH );
      event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCHEBRecHits" ), recHitHandleBH );

      const auto& rechitsEE = *recHitHandleEE;
      const auto& rechitsFH = *recHitHandleFH;
      const auto& rechitsBH = *recHitHandleBH;

      for (unsigned int i = 0; i < rechitsEE.size(); ++i) {
         hitmap[rechitsEE[i].detid().rawId()] = &rechitsEE[i];
         maxEnergy = fmax(maxEnergy, rechitsEE[i].energy());
      }
      for (unsigned int i = 0; i < rechitsFH.size(); ++i) {
         hitmap[rechitsFH[i].detid().rawId()] = &rechitsFH[i];
         maxEnergy = fmax(maxEnergy, rechitsFH[i].energy());   
      }
      for (unsigned int i = 0; i < rechitsBH.size(); ++i) {
         hitmap[rechitsBH[i].detid().rawId()] = &rechitsBH[i];
         maxEnergy = fmax(maxEnergy, rechitsBH[i].energy());
      }
   }

   size_t size = iItem->size();
   TEveElement::List_i pIdx = product->BeginChildren();
   for (int index = 0; index < static_cast<int>(size); ++index)
   {
      TEveElement *itemHolder = nullptr;
      if (index < product->NumChildren())
      {
         itemHolder = *pIdx;
         itemHolder->SetRnrSelfChildren(true, true);
         ++pIdx;
      }
      else
      {
         itemHolder = createCompound();
         product->AddElement(itemHolder);
      }
      if (iItem->modelInfo(index).displayProperties().isVisible())
      {
         const void *modelData = iItem->modelData(index);
         build(*reinterpret_cast<const reco::CaloCluster *>(m_helper.offsetObject(modelData)), index, *itemHolder, vc);
      }
   }
}

void FWCaloClusterHeatMapProxyBuilder::build(const reco::CaloCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *)
{
   const long layer = item()->getConfig()->value<long>("Layer");
   const bool z_plus = item()->getConfig()->value<bool>("Z+");
   const bool z_minus = item()->getConfig()->value<bool>("Z-");

   std::vector<std::pair<DetId, float>> clusterDetIds = iData.hitsAndFractions();

   bool h_hex(false);
   TEveBoxSet *hex_boxset = new TEveBoxSet();
   // hex_boxset->UseSingleColor();
   hex_boxset->SetPickable(true);
   hex_boxset->Reset(TEveBoxSet::kBT_Hex, true, 64);
   hex_boxset->SetAntiFlick(true);

   bool h_box(false);
   TEveBoxSet *boxset = new TEveBoxSet();
   // boxset->UseSingleColor();
   boxset->SetPickable(true);
   boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
   boxset->SetAntiFlick(true);

   for (std::vector<std::pair<DetId, float>>::iterator it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
        it != itEnd; ++it)
   {
      const uint8_t type = ((it->first >> 28) & 0xF);
      // HGCal
      if (type >= 8 && type <= 10)
      {
         if(hitmap.find(it->first) == hitmap.end())
            continue;

         const bool z = (it->first >> 25) & 0x1;

         // discard everything thats not at the side that we are intersted in
         if (
             ((z_plus & z_minus) != 1) &&
             (((z_plus | z_minus) == 0) || !(z == z_minus || z == !z_plus)))
            continue;

         const float *corners = item()->getGeom()->getCorners(it->first);
         const float *parameters = item()->getGeom()->getParameters(it->first);
         const float *shapes = item()->getGeom()->getShapePars(it->first);

         if (corners == nullptr || parameters == nullptr || shapes == nullptr)
         {
            continue;
         }

         const int total_points = parameters[0];
         const bool isScintillator = (total_points == 4);

         uint8_t ll = layer;
         if (layer > 0)
         {
            if (layer > 28)
            {
               if (type == 8)
               {
                  continue;
               }
               ll -= 28;
            }
            else
            {
               if (type != 8)
               {
                  continue;
               }
            }

            if (ll != ((it->first >> (isScintillator ? 17 : 20)) & 0x1F))
               continue;
         }

         float colorFactor = (fmin(hitmap[it->first]->energy(), 1.0f));

         // Scintillator
         if (isScintillator)
         {
            const int total_vertices = 3 * total_points;

            std::vector<float> pnts(24);
            for (int i = 0; i < total_points; ++i)
            {
               pnts[i * 3 + 0] = corners[i * 3];
               pnts[i * 3 + 1] = corners[i * 3 + 1];
               pnts[i * 3 + 2] = corners[i * 3 + 2];

               pnts[(i * 3 + 0) + total_vertices] = corners[i * 3];
               pnts[(i * 3 + 1) + total_vertices] = corners[i * 3 + 1];
               pnts[(i * 3 + 2) + total_vertices] = corners[i * 3 + 2] + shapes[3];
            }
            boxset->AddBox(&pnts[0]);
            boxset->DigitColor(std::min(255*4*colorFactor, 255.0f), 255-std::abs(255-4*colorFactor*255), 255-std::min(255*4*colorFactor, 255.0f));

            h_box = true;
         }
         // Silicon
         else
         {
            const int offset = 9;

            float centerX = (corners[6] + corners[6 + offset]) / 2;
            float centerY = (corners[7] + corners[7 + offset]) / 2;
            float radius = fabs(corners[6] - corners[6 + offset]) / 2;
            hex_boxset->AddHex(TEveVector(centerX, centerY, corners[2]),
                               radius, 90.0, shapes[3]);
            hex_boxset->DigitColor(std::min(255*4*colorFactor, 255.0f), 255-std::abs(255-4*colorFactor*255), 255-std::min(255*4*colorFactor, 255.0f));

            h_hex = true;
         }
      }
      // Not HGCal
      else
      {
         const float *corners = item()->getGeom()->getCorners(it->first);

         if (corners == nullptr)
         {
            continue;
         }

         h_box = true;

         std::vector<float> pnts(24);
         fireworks::energyTower3DCorners(corners, (*it).second, pnts);
         boxset->AddBox(&pnts[0]);
      }
   }

   if (h_hex)
   {
      hex_boxset->RefitPlex();
      setupAddElement(hex_boxset, &oItemHolder, false);
   }

   if (h_box)
   {
      boxset->RefitPlex();
      setupAddElement(boxset, &oItemHolder, false);
   }
}

REGISTER_FWPROXYBUILDER(FWCaloClusterHeatMapProxyBuilder, reco::CaloCluster, "Calo Cluster Heatmap", FWViewType::kISpyBit);
