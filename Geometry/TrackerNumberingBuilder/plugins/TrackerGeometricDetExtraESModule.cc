#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PGeometricDetExtraRcd.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "ExtractStringFromDDD.h"
#include "CondDBCmsTrackerConstruction.h"

#include <memory>

class TrackerGeometricDetExtraESModule : public edm::ESProducer {
public:
  TrackerGeometricDetExtraESModule(const edm::ParameterSet& p);

  std::unique_ptr<std::vector<GeometricDetExtra> > produce(const IdealGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void putOne(std::vector<GeometricDetExtra>& gde, const GeometricDet* gd, const DDExpandedView& ev, int lev);

  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geometricDetToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
  edm::ESGetToken<PGeometricDetExtra, PGeometricDetExtraRcd> pgToken_;
  const bool fromDDD_;
  const bool fromDD4hep_;
};

using namespace edm;

TrackerGeometricDetExtraESModule::TrackerGeometricDetExtraESModule(const edm::ParameterSet& p)
    : fromDDD_(p.getParameter<bool>("fromDDD")), fromDD4hep_(p.getParameter<bool>("fromDD4hep")) {
  auto c = setWhatProduced(this);
  geometricDetToken_ = c.consumes<GeometricDet>(edm::ESInputTag());
  if (fromDDD_) {
    ddToken_ = c.consumes<DDCompactView>(edm::ESInputTag());
  } else if (fromDD4hep_) {
    dd4hepToken_ = c.consumes<cms::DDCompactView>(edm::ESInputTag());
  } else {
    pgToken_ = c.consumesFrom<PGeometricDetExtra, PGeometricDetExtraRcd>(edm::ESInputTag());
  }
}

void TrackerGeometricDetExtraESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription descDB;
  descDB.add<bool>("fromDDD", false);
  descDB.add<bool>("fromDD4hep", false);
  descriptions.add("trackerNumberingExtraGeometryDB", descDB);

  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("fromDD4hep", false);
  descriptions.add("trackerNumberingExtraGeometry", desc);

  edm::ParameterSetDescription descDD4hep;
  descDD4hep.add<bool>("fromDDD", false);
  descDD4hep.add<bool>("fromDD4hep", true);
  descriptions.add("DD4hep_trackerNumberingExtraGeometry", descDD4hep);
}

std::unique_ptr<std::vector<GeometricDetExtra> > TrackerGeometricDetExtraESModule::produce(
    const IdealGeometryRecord& iRecord) {
  auto gde = std::make_unique<std::vector<GeometricDetExtra> >();
  // get the GeometricDet which has a nav_type
  GeometricDet const& gd = iRecord.get(geometricDetToken_);
  if (fromDDD_) {
    // traverse all components from the tracker down;
    // read the DD if from DD
    const GeometricDet* tracker = &(gd);
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(ddToken_);
    DDExpandedView ev(*cpv);
    ev.goTo(tracker->navType());
    putOne((*gde), tracker, ev, 0);
    std::vector<const GeometricDet*> tc = tracker->components();
    std::vector<const GeometricDet*>::const_iterator git = tc.begin();
    std::vector<const GeometricDet*>::const_iterator egit = tc.end();
    int count = 0;
    int lev = 1;
    //  CmsTrackerStringToEnum ctst
    gde->reserve(tracker->deepComponents().size());
    for (; git != egit; ++git) {  // one level below "tracker"
      ev.goTo((*git)->navType());
      putOne((*gde), *git, ev, lev);
      std::vector<const GeometricDet*> inone = (*git)->components();
      //    std::cout << lev << " type " << (*git)->type() << " " << int((*git)->geographicalId()) << std::endl; // << " has " << inone.size() << " components." << std::endl;
      if (inone.empty())
        ++count;
      std::vector<const GeometricDet*>::const_iterator git2 = inone.begin();
      std::vector<const GeometricDet*>::const_iterator egit2 = inone.end();
      ++lev;
      for (; git2 != egit2; ++git2) {  // level 2
        ev.goTo((*git2)->navType());
        putOne((*gde), *git2, ev, lev);
        std::vector<const GeometricDet*> intwo = (*git2)->components();
        //      std::cout << lev << "\ttype " << (*git2)->type() << " " << int((*git2)->geographicalId()) << std::endl; // << " has " << intwo.size() << " components." << std::endl;
        if (intwo.empty())
          ++count;
        std::vector<const GeometricDet*>::const_iterator git3 = intwo.begin();
        std::vector<const GeometricDet*>::const_iterator egit3 = intwo.end();
        ++lev;
        for (; git3 != egit3; ++git3) {  // level 3
          ev.goTo((*git3)->navType());
          putOne((*gde), *git3, ev, lev);
          std::vector<const GeometricDet*> inthree = (*git3)->components();
          //std::cout << lev << "\t\ttype " << (*git3)->type() << " " << int((*git3)->geographicalId()) << std::endl; // << " has " << inthree.size() << " components." << std::endl;
          if (inthree.empty())
            ++count;
          std::vector<const GeometricDet*>::const_iterator git4 = inthree.begin();
          std::vector<const GeometricDet*>::const_iterator egit4 = inthree.end();
          ++lev;
          for (; git4 != egit4; ++git4) {  //level 4
            ev.goTo((*git4)->navType());
            putOne((*gde), *git4, ev, lev);
            std::vector<const GeometricDet*> infour = (*git4)->components();
            //  std::cout << lev << "\t\t\ttype " << (*git4)->type() << " " << int((*git4)->geographicalId()) << std::endl; // << " has " << infour.size() << " components." << std::endl;
            if (infour.empty())
              ++count;
            std::vector<const GeometricDet*>::const_iterator git5 = infour.begin();
            std::vector<const GeometricDet*>::const_iterator egit5 = infour.end();
            ++lev;
            for (; git5 != egit5; ++git5) {  // level 5
              ev.goTo((*git5)->navType());
              putOne((*gde), *git5, ev, lev);
              std::vector<const GeometricDet*> infive = (*git5)->components();
              //    std::cout << lev << "\t\t\t\ttype " << (*git5)->type() << " " << int((*git5)->geographicalId()) << std::endl; // << " has " << infive.size() << " components." << std::endl;
              if (infive.empty())
                ++count;
              std::vector<const GeometricDet*>::const_iterator git6 = infive.begin();
              std::vector<const GeometricDet*>::const_iterator egit6 = infive.end();
              ++lev;
              for (; git6 != egit6; ++git6) {  //level 6
                ev.goTo((*git6)->navType());
                putOne((*gde), *git6, ev, lev);
                std::vector<const GeometricDet*> insix = (*git6)->components();
                //      std::cout << lev << "\t\t\t\t\ttype " << (*git6)->type() << " " << int((*git6)->geographicalId()) << std::endl; // << " has " << insix.size() << " components." << std::endl;
                if (insix.empty()) {
                  ++count;
                } else {
                  edm::LogError("GeometricDetExtra") << "Hierarchy has exceeded hard-coded level 6 for Tracker ";
                }
              }  // level 6
              --lev;
            }  // level 5
            --lev;
          }  // level 4
          --lev;
        }  //level 3
        --lev;
      }  // level 2
      --lev;
    }
  } else if (fromDD4hep_) {
    // FIXME:
    // traverse all components from the tracker down;
    // read the DD if from DD
    const GeometricDet* tracker = &(gd);
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(dd4hepToken_);
    edm::LogInfo("GeometricDetExtra") << " Top node is  " << tracker << " " << tracker->name() << std::endl;
    edm::LogInfo("GeometricDetExtra") << "    radLength " << tracker->radLength() << "\n"
                                      << "           xi " << tracker->xi() << "\n"
                                      << " PixelROCRows " << tracker->pixROCRows() << "\n"
                                      << "   PixROCCols " << tracker->pixROCCols() << "\n"
                                      << "   PixelROC_X " << tracker->pixROCx() << "\n"
                                      << "   PixelROC_Y " << tracker->pixROCy() << "\n"
                                      << "TrackerStereoDetectors " << (tracker->stereo() ? "true" : "false") << "\n"
                                      << "SiliconAPVNumber " << tracker->siliconAPVNum() << "\n";
  } else {
    // if it is not from the DD, then just get the GDE from ES and match w/ GD.
    PGeometricDetExtra const& pgde = iRecord.getRecord<PGeometricDetExtraRcd>().get(pgToken_);
    std::map<uint32_t, const GeometricDet*> helperMap;
    const GeometricDet* tracker = &gd;
    helperMap[gd.geographicalID()] = tracker;
    std::vector<const GeometricDet*> tc = tracker->components();
    std::vector<const GeometricDet*>::const_iterator git = tc.begin();
    std::vector<const GeometricDet*>::const_iterator egit = tc.end();
    for (; git != egit; ++git) {  // one level below "tracker"
      helperMap[(*git)->geographicalID()] = (*git);
      std::vector<const GeometricDet*> inone = (*git)->components();
      std::vector<const GeometricDet*>::const_iterator git2 = inone.begin();
      std::vector<const GeometricDet*>::const_iterator egit2 = inone.end();
      for (; git2 != egit2; ++git2) {  // level 2
        helperMap[(*git2)->geographicalID()] = (*git2);
        std::vector<const GeometricDet*> intwo = (*git2)->components();
        std::vector<const GeometricDet*>::const_iterator git3 = intwo.begin();
        std::vector<const GeometricDet*>::const_iterator egit3 = intwo.end();
        for (; git3 != egit3; ++git3) {  // level 3
          helperMap[(*git3)->geographicalID()] = (*git3);
          std::vector<const GeometricDet*> inthree = (*git3)->components();
          std::vector<const GeometricDet*>::const_iterator git4 = inthree.begin();
          std::vector<const GeometricDet*>::const_iterator egit4 = inthree.end();
          for (; git4 != egit4; ++git4) {  //level 4
            helperMap[(*git4)->geographicalID()] = (*git4);
            std::vector<const GeometricDet*> infour = (*git4)->components();
            std::vector<const GeometricDet*>::const_iterator git5 = infour.begin();
            std::vector<const GeometricDet*>::const_iterator egit5 = infour.end();
            for (; git5 != egit5; ++git5) {  // level 5
              helperMap[(*git5)->geographicalID()] = (*git5);
              std::vector<const GeometricDet*> infive = (*git5)->components();
              std::vector<const GeometricDet*>::const_iterator git6 = infive.begin();
              std::vector<const GeometricDet*>::const_iterator egit6 = infive.end();
              for (; git6 != egit6; ++git6) {  //level 6
                helperMap[(*git6)->geographicalID()] = (*git6);
                if (!(*git6)->components().empty()) {
                  edm::LogError("GeometricDetExtra") << "Hierarchy has exceeded hard-coded level of 6 for Tracker ";
                }
              }  // level 6
            }    // level 5
          }      // level 4
        }        //level 3
      }          // level 2
    }

    const std::vector<PGeometricDetExtra::Item>& pgdes = pgde.pgdes_;
    gde->reserve(pgdes.size());
    std::vector<DDExpandedNode> evs;  //EMPTY
    std::string nm;                   //EMPTY
    for (const auto& pgde : pgdes) {
      //   GeometricDetExtra( GeometricDet const *gd, DetId id, GeoHistory& gh,  double vol, double dens, double wgt, double cpy, const std::string& mat, const std::string& name, bool dd=false );
      gde->emplace_back(GeometricDetExtra(helperMap[pgde._geographicalId],
                                          pgde._geographicalId,
                                          evs,
                                          pgde._volume,
                                          pgde._density,
                                          pgde._weight,
                                          pgde._copy,
                                          pgde._material,
                                          nm));
    }
  }
  return gde;
}

void TrackerGeometricDetExtraESModule::putOne(std::vector<GeometricDetExtra>& gde,
                                              const GeometricDet* gd,
                                              const DDExpandedView& ev,
                                              int lev) {
  std::string matname = ((ev.logicalPart()).material()).name().fullname();
  std::string lpname = ((ev.logicalPart()).name().fullname());
  std::vector<DDExpandedNode> evs = GeometricDetExtra::GeoHistory(ev.geoHistory().begin(), ev.geoHistory().end());
  gde.emplace_back(
      GeometricDetExtra(gd,
                        gd->geographicalId(),
                        evs,
                        ((ev.logicalPart()).solid()).volume(),
                        ((ev.logicalPart()).material()).density(),
                        ((ev.logicalPart()).material()).density() * (((ev.logicalPart()).solid()).volume() / 1000.),
                        ev.copyno(),
                        matname,
                        lpname,
                        true));
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetExtraESModule);
