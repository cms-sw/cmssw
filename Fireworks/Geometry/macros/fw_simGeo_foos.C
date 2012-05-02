//==============================================================================
// Material name/title "fixing"
//==============================================================================

#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"

#include "TClass.h"
#include "TList.h"

TGeoManager* FWGeometryTableViewManager_GetGeoManager();

TGeoElementTable *g_element_table = 0;

void fw_simGeo_fix_materials()
{
   Int_t base_element_offset = TGeoMaterial::Class()->GetDataMemberOffset("fElement");

   TString vacuum("materials:Vacuum");

   TGeoMaterial *m;
   TIter it(FWGeometryTableViewManager_GetGeoManager()->GetListOfMaterials());
   while ((m = (TGeoMaterial*) it()) != 0)
   {
      // Fixes
      if (vacuum == m->GetName())
      {
         m->SetZ(0);
      }

      TGeoMixture *mix = dynamic_cast<TGeoMixture*>(m);
      if (mix == 0)
      {
         if ( ! m->GetBaseElement())
         {
            *(TGeoElement**)(((char*)m) + base_element_offset) = g_element_table->GetElement(m->GetZ());
         }
      }
   }
}

void fw_simGeo_dump_materials(Bool_t dump_components=false)
{
   TGeoMaterial *m;
   TIter it(FWGeometryTableViewManager_GetGeoManager()->GetListOfMaterials());
   while ((m = (TGeoMaterial*) it()) != 0)
   {
      TGeoMixture *mix = dynamic_cast<TGeoMixture*>(m);
      printf("%-50s | %-40s | %2d | %.3f\n", m->GetName(), m->GetTitle(),
             mix ? mix->GetNelements() : 0, m->GetZ());
      if (dump_components)
      {
         if (mix == 0)
         {
            printf("  %4d %6s %s\n", m->GetBaseElement()->Z(), m->GetBaseElement()->GetName(), m->GetBaseElement()->GetTitle());
         }
         else
         {
            Double_t *ww = mix->GetWmixt();
            for (Int_t i = 0; i < mix->GetNelements(); ++i)
            {
               TGeoElement *e = mix->GetElement(i);
               printf("  %4d %-4s %f\n",  e->Z(), e->GetName(), ww[i]);
            }
         }
      }
   }
}

void fw_simGeo_set_material_titles(Double_t fraction=0, Bool_t long_names=false)
{
   TGeoMaterial *m;
   TIter it(FWGeometryTableViewManager_GetGeoManager()->GetListOfMaterials());
   while ((m = (TGeoMaterial*) it()) != 0)
   {
      TString tit(":");
      TGeoMixture *mix = dynamic_cast<TGeoMixture*>(m);

      if (mix == 0)
      {
         TGeoElement *e = m->GetBaseElement();
         tit += long_names ? e->GetTitle() : e->GetName();
         tit += ":";
      }
      else
      {
         Double_t *ww = mix->GetWmixt();
         for (Int_t i = 0; i < mix->GetNelements(); ++i)
         {
            if (ww[i] >= fraction)
            {
               TGeoElement *e = mix->GetElement(i);
               tit += long_names ? e->GetTitle() : e->GetName();
               tit += ":";
            }
         }
      }
      if (tit == ":") tit += ":";
      m->SetTitle(tit);
   }
}

void fw_simGeo_foos()
{
   g_element_table = new TGeoElementTable;
   g_element_table->BuildDefaultElements();

   fw_simGeo_fix_materials();
   fw_simGeo_set_material_titles(0.01);
}
