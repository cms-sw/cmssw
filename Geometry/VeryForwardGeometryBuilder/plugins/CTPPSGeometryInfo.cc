/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Class to print out information on current geometry.
 **/
class CTPPSGeometryInfo : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSGeometryInfo(const edm::ParameterSet&);

  private: 
    std::string geometryType;

    bool printRPInfo, printSensorInfo;

    edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry;
    edm::ESWatcher<VeryForwardRealGeometryRecord> watcherRealGeometry;
    edm::ESWatcher<VeryForwardMisalignedGeometryRecord> watcherMisalignedGeometry;

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    static void PrintDetId(const CTPPSDetId &id, bool printDetails = true);

    void PrintGeometry(const CTPPSGeometry &, const edm::Event&);
};

//----------------------------------------------------------------------------------------------------

using namespace edm;
using namespace std;
using namespace CLHEP;

//----------------------------------------------------------------------------------------------------

CTPPSGeometryInfo::CTPPSGeometryInfo(const edm::ParameterSet& ps) :
  geometryType(ps.getUntrackedParameter<string>("geometryType", "real")),
  printRPInfo(ps.getUntrackedParameter<bool>("printRPInfo", true)),
  printSensorInfo(ps.getUntrackedParameter<bool>("printSensorInfo", true))
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSGeometryInfo::analyze(const edm::Event& event, const edm::EventSetup& es)
{
  ESHandle<CTPPSGeometry> geometry;

  if (!geometryType.compare("ideal"))
  {
    if (watcherIdealGeometry.check(es))
    {
      es.get<IdealGeometryRecord>().get(geometry);
      PrintGeometry(*geometry, event);
    }
    return;
  }

  if (!geometryType.compare("real"))
  {
    if (watcherRealGeometry.check(es))
    {
      es.get<VeryForwardRealGeometryRecord>().get(geometry);
      PrintGeometry(*geometry, event);
    }
    return;
  }

  if (!geometryType.compare("misaligned"))
  {
    if (watcherMisalignedGeometry.check(es))
    {
      es.get<VeryForwardMisalignedGeometryRecord>().get(geometry);
      PrintGeometry(*geometry, event);
    }
    return;
  }

  throw cms::Exception("CTPPSGeometryInfo") << "Unknown geometry type: `" << geometryType << "'.";
}

//----------------------------------------------------------------------------------------------------

void CTPPSGeometryInfo::PrintDetId(const CTPPSDetId &id, bool printDetails)
{
  cout << id.rawId();

  const unsigned int rpDecId = id.arm()*100 + id.station()*10 + id.rp();

  if (id.subdetId() == CTPPSDetId::sdTrackingStrip)
  {
    TotemRPDetId fid(id);
    cout << " (strip RP " << rpDecId;
    if (printDetails)
      cout <<  ", plane " << fid.plane();
    cout << ")";
  }

  if (id.subdetId() == CTPPSDetId::sdTrackingPixel)
  {
    CTPPSPixelDetId fid(id);
    cout << " (pixel RP " << rpDecId;
    if (printDetails)
      cout <<  ", plane " << fid.plane();
    cout << ")";
  }

  if (id.subdetId() == CTPPSDetId::sdTimingDiamond)
  {
    CTPPSDiamondDetId fid(id);
    cout << " (diamd RP " << rpDecId;
    if (printDetails)
      cout <<  ", plane " << fid.plane() << ", channel " << fid.channel();
    cout << ")";
  }

}

//----------------------------------------------------------------------------------------------------

void CTPPSGeometryInfo::PrintGeometry(const CTPPSGeometry &geometry, const edm::Event& event)
{
  time_t unixTime = event.time().unixTime();
  char timeStr[50];
  strftime(timeStr, 50, "%F %T", localtime(&unixTime));

  cout << ">> CTPPSGeometryInfo::PrintGeometry > new " << geometryType << " geometry found in run="
    << event.id().run() << ", event=" << event.id().event() << ", UNIX timestamp=" << unixTime
    << " (" << timeStr << ")";

  // RP geometry
  if (printRPInfo)
  {
    cout << endl << "* RPs:" << endl;
    cout << "    ce: RP center in global coordinates, in mm" << endl;
    for (auto it = geometry.beginRP(); it != geometry.endRP(); ++it)
    {
      const DDTranslation &t = it->second->translation();

      PrintDetId(CTPPSDetId(it->first), false);
      cout << fixed << setprecision(3) << " | ce=(" << t.x() << ", " << t.y() << ", " << t.z() << ")" << endl;
    }
  }
  
  // sensor geometry
  if (printSensorInfo)
  {
    cout << endl << "* sensors:" << endl;
    cout << "    ce: sensor center in global coordinates, in mm" << endl;
    cout << "    a1: local axis (1, 0, 0) in global coordinates" << endl;
    cout << "    a2: local axis (0, 1, 0) in global coordinates" << endl;
    cout << "    a3: local axis (0, 0, 1) in global coordinates" << endl;

    for (auto it = geometry.beginSensor(); it != geometry.endSensor(); ++it)
    {
      CTPPSDetId detId(it->first);
  
      Hep3Vector gl_o = geometry.localToGlobal(detId, Hep3Vector(0, 0, 0));
      Hep3Vector gl_a1 = geometry.localToGlobal(detId, Hep3Vector(1, 0, 0)) - gl_o;
      Hep3Vector gl_a2 = geometry.localToGlobal(detId, Hep3Vector(0, 1, 0)) - gl_o;
      Hep3Vector gl_a3 = geometry.localToGlobal(detId, Hep3Vector(0, 0, 1)) - gl_o;

      PrintDetId(detId);

      cout
        << " | ce=(" << gl_o.x() << ", " << gl_o.y() << ", " << gl_o.z() << ")"
        << " | a1=(" << gl_a1.x() << ", " << gl_a1.y() << ", " << gl_a1.z() << ")"
        << " | a2=(" << gl_a2.x() << ", " << gl_a2.y() << ", " << gl_a2.z() << ")"
        << " | a3=(" << gl_a3.x() << ", " << gl_a3.y() << ", " << gl_a3.z() << ")"
        << endl;
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSGeometryInfo);
