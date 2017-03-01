/****************************************************************************
*
* Authors:
*  Jan Kaspar (jan.kaspar@gmail.com) 
*  Dominik Mierzejewski <dmierzej@cern.ch>
*    
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
 
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Specific.h"

#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/DDDTotemRPConstruction.h"

#include <TMatrixD.h>

/**
 * \ingroup TotemRPGeometry
 * \brief Builds ideal, real and misaligned geometries.
 *
 * See schema of \ref TotemRPGeometry "TOTEM RP geometry classes"
 *
 * First it creates a tree of DetGeomDesc from DDCompView. For real and misaligned geometries,
 * it applies alignment corrections (RPAlignmentCorrections) found in corresponding ...GeometryRecord.
 *
 * Second, it creates TotemRPGeometry from DetGeoDesc tree.
 **/
class  TotemRPGeometryESModule : public edm::ESProducer
{
  public:
    TotemRPGeometryESModule(const edm::ParameterSet &p);
    virtual ~TotemRPGeometryESModule(); 

    std::unique_ptr<DetGeomDesc> produceIdealGD(const IdealGeometryRecord &);

    std::unique_ptr<DetGeomDesc> produceRealGD(const VeryForwardRealGeometryRecord &);
    std::unique_ptr<TotemRPGeometry> produceRealTG(const VeryForwardRealGeometryRecord &);

    std::unique_ptr<DetGeomDesc> produceMisalignedGD(const VeryForwardMisalignedGeometryRecord &);
    std::unique_ptr<TotemRPGeometry> produceMisalignedTG(const VeryForwardMisalignedGeometryRecord &);

  protected:
    unsigned int verbosity;

    void ApplyAlignments(const edm::ESHandle<DetGeomDesc> &idealGD, const edm::ESHandle<RPAlignmentCorrectionsData> &alignments,
      DetGeomDesc* &newGD);
};


using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPGeometryESModule::TotemRPGeometryESModule(const edm::ParameterSet &p)
{
  verbosity = p.getUntrackedParameter<unsigned int>("verbosity", 1);  

  setWhatProduced(this, &TotemRPGeometryESModule::produceIdealGD);

  setWhatProduced(this, &TotemRPGeometryESModule::produceRealGD);
  setWhatProduced(this, &TotemRPGeometryESModule::produceRealTG);

  setWhatProduced(this, &TotemRPGeometryESModule::produceMisalignedGD);
  setWhatProduced(this, &TotemRPGeometryESModule::produceMisalignedTG);
}

//----------------------------------------------------------------------------------------------------

TotemRPGeometryESModule::~TotemRPGeometryESModule()
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPGeometryESModule::ApplyAlignments(const ESHandle<DetGeomDesc> &idealGD, 
    const ESHandle<RPAlignmentCorrectionsData> &alignments, DetGeomDesc* &newGD)
{
  newGD = new DetGeomDesc( *(idealGD.product()) );
  deque<const DetGeomDesc *> buffer;
  deque<DetGeomDesc *> bufferNew;
  buffer.push_back(idealGD.product());
  bufferNew.push_back(newGD);

  while (buffer.size() > 0)
  {
    const DetGeomDesc *sD = buffer.front();
    DetGeomDesc *pD = bufferNew.front();
    buffer.pop_front();
    bufferNew.pop_front();

    // Is it sensor? If yes, apply full sensor alignments
    if ( pD->name().name().compare( DDD_TOTEM_RP_DETECTOR_NAME) == 0
      or pD->name().name().compare( DDD_CTPPS_DIAMONDS_DETECTOR_NAME ) == 0 )
    {
      unsigned int plId = pD->geographicalID();

      if (alignments.isValid())
      {
        const RPAlignmentCorrectionData& ac = alignments->GetFullSensorCorrection(plId);
        pD->ApplyAlignment(ac);
      }
    }

    // Is it RP box? If yes, apply RP alignments
    if (! pD->name().name().compare(DDD_TOTEM_RP_PRIMARY_VACUUM_NAME))
    {
      unsigned int rpId = pD->geographicalID();
      
      if (alignments.isValid())
      {
        const RPAlignmentCorrectionData& ac = alignments->GetRPCorrection(rpId);
        pD->ApplyAlignment(ac);
      }
    }

    // create and add children
    for (unsigned int i = 0; i < sD->components().size(); i++)
    {
      const DetGeomDesc *sDC = sD->components()[i];
      buffer.push_back(sDC);
    
      // create new node with the same information as in sDC and add it as a child of pD
      DetGeomDesc * cD = new DetGeomDesc(*sDC);
      pD->addComponent(cD);

      bufferNew.push_back(cD);
    }
  }
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> TotemRPGeometryESModule::produceIdealGD(const IdealGeometryRecord &iRecord)
{
  // get the DDCompactView from EventSetup
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get("XMLIdealGeometryESSource_CTPPS", cpv);
  
  // construct the tree of DetGeomDesc
  DDDTotemRPContruction worker;
  return std::unique_ptr<DetGeomDesc>( const_cast<DetGeomDesc*>( worker.construct(&(*cpv)) ) );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> TotemRPGeometryESModule::produceRealGD(const VeryForwardRealGeometryRecord &iRecord)
{
  // get the input GeometricalDet
  edm::ESHandle<DetGeomDesc> idealGD;
  iRecord.getRecord<IdealGeometryRecord>().get(idealGD);

  // load alignments
  edm::ESHandle<RPAlignmentCorrectionsData> alignments;
  try { iRecord.getRecord<RPRealAlignmentRecord>().get(alignments); }
  catch (...) {}

  if (alignments.isValid())
  {
    if (verbosity)
      LogVerbatim("TotemRPGeometryESModule::produceRealGD")
        << ">> TotemRPGeometryESModule::produceRealGD > Real geometry: "
        << alignments->GetRPMap().size() << " RP and "
        << alignments->GetSensorMap().size() << " sensor alignments applied.";
  } else {
    if (verbosity)
      LogVerbatim("TotemRPGeometryESModule::produceRealGD")
        << ">> TotemRPGeometryESModule::produceRealGD > Real geometry: No alignments applied.";
  }

  DetGeomDesc* newGD = NULL;
  ApplyAlignments(idealGD, alignments, newGD);
  return std::unique_ptr<DetGeomDesc>(newGD);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> TotemRPGeometryESModule::produceMisalignedGD(const VeryForwardMisalignedGeometryRecord &iRecord)
{
  // get the input GeometricalDet
  edm::ESHandle<DetGeomDesc> idealGD;
  iRecord.getRecord<IdealGeometryRecord>().get(idealGD);

  // load alignments
  edm::ESHandle<RPAlignmentCorrectionsData> alignments;
  try { iRecord.getRecord<RPMisalignedAlignmentRecord>().get(alignments); }
  catch (...) {}

  if (alignments.isValid())
  {
    if (verbosity)
      LogVerbatim("TotemRPGeometryESModule::produceMisalignedGD")
        << ">> TotemRPGeometryESModule::produceMisalignedGD > Misaligned geometry: "
        << alignments->GetRPMap().size() << " RP and "
        << alignments->GetSensorMap().size() << " sensor alignments applied.";
  } else {
    if (verbosity)
      LogVerbatim("TotemRPGeometryESModule::produceMisalignedGD")
        << ">> TotemRPGeometryESModule::produceMisalignedGD > Misaligned geometry: No alignments applied.";
  }

  DetGeomDesc* newGD = NULL;
  ApplyAlignments(idealGD, alignments, newGD);
  return std::unique_ptr<DetGeomDesc>(newGD);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<TotemRPGeometry> TotemRPGeometryESModule::produceRealTG(const VeryForwardRealGeometryRecord &iRecord)
{
  edm::ESHandle<DetGeomDesc> gD;
  iRecord.get(gD);

  return std::make_unique<TotemRPGeometry>( gD.product());
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<TotemRPGeometry> TotemRPGeometryESModule::produceMisalignedTG(const VeryForwardMisalignedGeometryRecord &iRecord)
{
  edm::ESHandle<DetGeomDesc> gD;
  iRecord.get(gD);

  return std::make_unique<TotemRPGeometry>( gD.product());
}

DEFINE_FWK_EVENTSETUP_MODULE(TotemRPGeometryESModule);
