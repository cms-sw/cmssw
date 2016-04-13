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
#include "Geometry/Records/interface/VeryForwardMeasuredGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMeasuredAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"
#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"
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

    std::unique_ptr<DDCompactView> produceMeasuredDDCV(const VeryForwardMeasuredGeometryRecord &);

    std::unique_ptr<DetGeomDesc> produceMeasuredGD(const VeryForwardMeasuredGeometryRecord &);
    std::unique_ptr<TotemRPGeometry> produceMeasuredTG(const VeryForwardMeasuredGeometryRecord &);

    std::unique_ptr<DetGeomDesc> produceRealGD(const VeryForwardRealGeometryRecord &);
    std::unique_ptr<TotemRPGeometry> produceRealTG(const VeryForwardRealGeometryRecord &);

    std::unique_ptr<DetGeomDesc> produceMisalignedGD(const VeryForwardMisalignedGeometryRecord &);
    std::unique_ptr<TotemRPGeometry> produceMisalignedTG(const VeryForwardMisalignedGeometryRecord &);

  protected:
    unsigned int verbosity;

    void ApplyAlignments(const edm::ESHandle<DetGeomDesc> &measuredGD, const edm::ESHandle<RPAlignmentCorrections> &alignments, DetGeomDesc* &newGD);
    void ApplyAlignments(const edm::ESHandle<DDCompactView> &ideal_ddcv, const edm::ESHandle<RPAlignmentCorrections> &alignments, DDCompactView *&measured_ddcv);
};


using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPGeometryESModule::TotemRPGeometryESModule(const edm::ParameterSet &p)
{
  verbosity = p.getUntrackedParameter<unsigned int>("verbosity", 1);  

  setWhatProduced(this, &TotemRPGeometryESModule::produceMeasuredDDCV);
  setWhatProduced(this, &TotemRPGeometryESModule::produceMeasuredGD);
  setWhatProduced(this, &TotemRPGeometryESModule::produceMeasuredTG);

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
    const ESHandle<RPAlignmentCorrections> &alignments, DetGeomDesc* &newGD)
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
    if (! pD->name().name().compare(DDD_TOTEM_RP_DETECTOR_NAME))
    {
      unsigned int decId = TotemRPDetId::rawToDecId(pD->geographicalID().rawId());

      if (alignments.isValid())
      {
        const RPAlignmentCorrection& ac = alignments->GetFullSensorCorrection(decId);
        pD->ApplyAlignment(ac);
      }
    }

    // Is it RP box? If yes, apply RP alignments
    if (! pD->name().name().compare(DDD_TOTEM_RP_PRIMARY_VACUUM_NAME))
    {
      unsigned int rpId = pD->copyno();

      if (alignments.isValid())
      {
        const RPAlignmentCorrection& ac = alignments->GetRPCorrection(rpId);
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

// Copies ideal_ddcv to measured_ddcv, applying the alignments if any
// WARNING: (TODO?) does not handle the "old" geometry
class MeasuredGeometryProducer {
    private:
        const DDCompactView &idealCV;
        const RPAlignmentCorrections *const alignments;
        DDLogicalPart root;
        DDCompactView *measuredCV;

        // -- Expanded view utilits ------------
        // WTF WARNING
        // DDExpandView's constructor sets DDRotation::StoreT to readonly
        //
        // Any newExpandedView() call will set this value before calling the constructor
        // Any delExpandedView() call will the restore StoreT state to _the_last_ set value
        static bool evRotationStoreState;

        // Allocate new ExpandedView and set it to point at the LogicalPart
        //
        // The LogicalPart _must_ be in the CompactView
        // Only name and ns of the LogicalPart are taken into account
        static DDExpandedView *newExpandedView(const DDCompactView &compactView, const DDLogicalPart &part) {
            evRotationStoreState = DDRotation::StoreT::instance().readOnly();
            DDExpandedView *expandedView = new DDExpandedView(compactView);
            // traverse the tree until name and ns are mached
            const string &name = part.name().name();
            const string &ns = part.name().ns();
            bool noMatch = true;

            noMatch = false;
            noMatch |= expandedView->logicalPart().name().name().compare(name);
            noMatch |= expandedView->logicalPart().name().ns().compare(ns);
            while (noMatch) {
                expandedView->next();
                noMatch = false;
                noMatch |= expandedView->logicalPart().name().name().compare(name);
                noMatch |= expandedView->logicalPart().name().ns().compare(ns);
            }
            return expandedView;
        }

        // Deallocate the ExpandedView
        //
        // Returns NULL
        static DDExpandedView *delExpandedView(DDExpandedView *expandedView) {
            delete expandedView;
            DDRotation::StoreT::instance().setReadOnly(evRotationStoreState);
            return NULL;
        }

        // -- Transformation matrix utils ------

        // Standard 4x4 tranformation matrixes are used for the alignments:
        //
        //  | R R R x |
        //  | R R R y |
        //  | R R R z |
        //  | 0 0 0 1 |
        //
        // where R are parameters of rotation matrix and x,y,z are translation parametres.
        // (Rotation and translation must be applied in this very order, as is done in CMSSW).
        // Such matrixes can be easily used to compose the transformations, 
        // e.g to describe transformation C: "first do A, then do B", multiply
        //      C = B * A
        // All tranformation matrixes are invertible.

        // Creates transformation matrix according to rotation and translation (applied in this order)
        static void translRotToTransform(const DDTranslation &translation,const DDRotationMatrix &rotation,
                TMatrixD &transform) {
            // set rotation
            double values[9];
            rotation.GetComponents(values);
            for (int i = 0; i < 9; ++i) {
                transform[i / 3][i % 3] = values[i];
            }
            // set translation
            transform[0][3] = translation.X();
            transform[1][3] = translation.Y();
            transform[2][3] = translation.Z();
            transform[3][3] = 1.;
        }

        // sets rotation and translation (applied in this order) from transformation matrix 
        static void translRotFromTransform(DDTranslation &translation, DDRotationMatrix &rotation,
                const TMatrixD &transform) {
            // set rotation
            double values[9];
            for (int i = 0; i < 9; ++i) {
                values[i] = transform[i / 3][i % 3];
            }
            rotation.SetComponents(values, values + 9);
            // set translation
            translation.SetXYZ(transform[0][3], transform[1][3], transform[2][3]);
        }

        // Gets global transform of given LogicalPart in given CompactView (uses ExpandedView to calculate)
        static void getGlobalTransform(const DDLogicalPart &part, const DDCompactView &compactView, TMatrixD &transform) {
            DDExpandedView *expandedView = newExpandedView(compactView, part);
            translRotToTransform(expandedView->translation(), expandedView->rotation(), transform);
            expandedView = delExpandedView(expandedView);
        }

        // -- Misc. utils ----------------------

        // true if part's name maches DDD_TOTEM_RP_PRIMARY_VACUUM_NAME
        static inline bool isRPBox(const DDLogicalPart &part) {
            return (! part.name().name().compare(DDD_TOTEM_RP_PRIMARY_VACUUM_NAME));
        }

        // true if part's name maches DDD_TOTEM_RP_DETECTOR_NAME
        static inline bool isDetector(const DDLogicalPart &part) {
            return (! part.name().name().compare(DDD_TOTEM_RP_DETECTOR_NAME));
        }

        // Extracts RP id from object namespace - object must be RP_Box, or RP_Hybrid (NOT RP_Silicon_Detector)
        static inline int getRPIdFromNamespace(const DDLogicalPart &part) {
            int nsLength = part.name().ns().length();
            return atoi(part.name().ns().substr(nsLength - 3, nsLength).c_str());
        }

        // Creates Detector id from RP id and Detector no
        static inline int getDetectorId(const int rpId, const int detNo) {
            return rpId * 10 + detNo;
        }

        // -------------------------------------

        // Applies alignment (translation and rotation) to transformation matrix
        //
        // translation = alignmentTranslation + translation
        // rotation    = alignmentRotation + rotation
        static void applyCorrectionToTransform(const RPAlignmentCorrection &correction, TMatrixD &transform) {
            DDTranslation translation;
            DDRotationMatrix rotation;

            translRotFromTransform(translation, rotation, transform);

            translation = correction.Translation() + translation;
            rotation    = correction.RotationMatrix() * rotation;

            translRotToTransform(translation, rotation, transform);
        }

        // Applies relative alignments to Detector rotation and translation
        void applyCorrection(const DDLogicalPart &parent, const DDLogicalPart &child, const RPAlignmentCorrection &correction,
                DDTranslation &translation, DDRotationMatrix &rotation, const bool useMeasuredParent = true) {
            TMatrixD C(4,4);    // child relative transform
            TMatrixD iP(4,4);   // ideal parent global transform
            TMatrixD mP(4,4);   // measured parent global transform
            TMatrixD F(4,4);    // final child transform

            translRotToTransform(translation, rotation, C);

            if (useMeasuredParent) 
                getGlobalTransform(parent, *measuredCV, mP);
            else 
                getGlobalTransform(parent, idealCV, mP);
            getGlobalTransform(parent, idealCV, iP);

            // global final transform
            F = iP * C;
            applyCorrectionToTransform(correction, F);
            // relative final transform
            mP.Invert();
            F = mP * F;

            translRotFromTransform(translation, rotation, F);
        }

        void positionEverythingButDetectors(void) {
            DDCompactView::graph_type::const_iterator it = idealCV.graph().begin_iter();
            DDCompactView::graph_type::const_iterator itEnd = idealCV.graph().end_iter();
            for (; it != itEnd; ++it) {
                if (!isDetector(it->to())) {
                    const DDLogicalPart from    = it->from();
                    const DDLogicalPart to      = it->to();
                    const int           copyNo      = it->edge()->copyno_;
                    const DDDivision    &division   = it->edge()->division();
                    DDTranslation       translation(it->edge()->trans());
                    DDRotationMatrix    &rotationMatrix = *(new DDRotationMatrix(it->edge()->rot()));

                    if (isRPBox(to)) {
                        const int rpId = getRPIdFromNamespace(to);
                        if (alignments != NULL) {
                            const RPAlignmentCorrection correction = alignments->GetRPCorrection(rpId);
                            applyCorrection(from, to, correction,
                                    translation, rotationMatrix, false);
                        }
                    }

                    const DDRotation rotation = DDanonymousRot(&rotationMatrix);
                    measuredCV->position(to, from, copyNo, translation, rotation, &division);
                }
            }
        }

        void positionDetectors(void) {
            DDCompactView::graph_type::const_iterator it = idealCV.graph().begin_iter();
            DDCompactView::graph_type::const_iterator itEnd = idealCV.graph().end_iter();
            for (; it != itEnd; ++it) {
                if (isDetector(it->to())) {
                    const DDLogicalPart from    = it->from();
                    const DDLogicalPart to      = it->to();
                    const int           copyNo      = it->edge()->copyno_;
                    const DDDivision    &division   = it->edge()->division();
                    DDTranslation       translation(it->edge()->trans());
                    DDRotationMatrix    &rotationMatrix = *(new DDRotationMatrix(it->edge()->rot()));

                    const int rpId  = getRPIdFromNamespace(from);
                    const int detId = getDetectorId(rpId, copyNo);
                    if (alignments != NULL) {
                        const RPAlignmentCorrection correction = alignments->GetFullSensorCorrection(detId);
                        applyCorrection(from, to, correction, 
                                translation, rotationMatrix);
                    }

                    const DDRotation rotation = DDanonymousRot(&rotationMatrix);
                    measuredCV->position(to, from, copyNo, translation, rotation, &division);
                }
            }
        }

    public:
        MeasuredGeometryProducer(const edm::ESHandle<DDCompactView> &idealCV,
                const edm::ESHandle<RPAlignmentCorrections> &alignments) : idealCV(*idealCV), alignments(alignments.isValid() ? &(*alignments) : NULL) {
            root = this->idealCV.root();
        }

        DDCompactView *&produce() {
            // create DDCompactView for measured geometry
            // notice that this class is not responsible for deleting this object
            measuredCV = new DDCompactView(root);
            // CMSSW/DetectorDescription graph interface sucks, so instead of doing a one bfs
            // we go over the tree twice (this is needed, as final detector postions are
            // dependent on new positions of RP units).
            positionEverythingButDetectors();
            positionDetectors();
            return measuredCV;
        }
};
bool MeasuredGeometryProducer::evRotationStoreState;

void TotemRPGeometryESModule::ApplyAlignments(const edm::ESHandle<DDCompactView> &ideal_ddcv,
        const edm::ESHandle<RPAlignmentCorrections> &alignments, DDCompactView *&measured_ddcv)
{
    MeasuredGeometryProducer producer(ideal_ddcv, alignments);
    measured_ddcv = producer.produce(); 
    return;
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DDCompactView> TotemRPGeometryESModule::produceMeasuredDDCV(const VeryForwardMeasuredGeometryRecord &iRecord)
{
    // get the ideal DDCompactView from EventSetup
    edm::ESHandle<DDCompactView> idealCV;
    iRecord.getRecord<IdealGeometryRecord>().get(idealCV);

    // load alignments
    edm::ESHandle<RPAlignmentCorrections> alignments;
    try {
        iRecord.getRecord<RPMeasuredAlignmentRecord>().get(alignments);
    } catch (...) {
        cout<< "Exception in TotemRPGeometryESModule::produceMeasuredDDCV"
            << " during iRecord.getRecord<RPMeasuredAlignmentRecord>().get(alignments)"<<endl;
    }

    if (alignments.isValid()) {
        if (verbosity){
            cout << ">> TotemRPGeometryESModule::produceMeasuredDDCV > Measured geometry: "
                << alignments->GetRPMap().size() << " RP and "
                << alignments->GetSensorMap().size() << " sensor alignments applied.\n";
        }
    } else {
        if (verbosity)
            cout << ">> TotemRPGeometryESModule::produceMeasuredDDCV > Measured geometry: No alignments applied.\n";
    }

    DDCompactView *measuredCV = NULL;
    ApplyAlignments(idealCV, alignments, measuredCV);
    return std::unique_ptr<DDCompactView>(measuredCV);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> TotemRPGeometryESModule::produceMeasuredGD(const VeryForwardMeasuredGeometryRecord &iRecord)
{
  // get the DDCompactView from EventSetup
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get(cpv);
  
  // construct the tree of DetGeomDesc
  DDDTotemRPContruction worker;
  return std::unique_ptr<DetGeomDesc>( const_cast<DetGeomDesc*>( worker.construct(&(*cpv)) ) );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> TotemRPGeometryESModule::produceRealGD(const VeryForwardRealGeometryRecord &iRecord)
{
  // get the input (= measured) GeometricalDet
  edm::ESHandle<DetGeomDesc> measuredGD;
  iRecord.getRecord<VeryForwardMeasuredGeometryRecord>().get(measuredGD);

  // load alignments
  edm::ESHandle<RPAlignmentCorrections> alignments;
  try { iRecord.getRecord<RPRealAlignmentRecord>().get(alignments); }
  catch (...) {}
  if (alignments.isValid()) {
    if (verbosity)
      printf(">> TotemRPGeometryESModule::produceRealGD > Real geometry: %lu RP and %lu sensor alignments applied.\n",
        alignments->GetRPMap().size(), alignments->GetSensorMap().size());
  } else {
    if (verbosity)
      printf(">> TotemRPGeometryESModule::produceRealGD > Real geometry: No alignments applied.\n");
  }

  DetGeomDesc* newGD = NULL;
  ApplyAlignments(measuredGD, alignments, newGD);
  return std::unique_ptr<DetGeomDesc>(newGD);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc> TotemRPGeometryESModule::produceMisalignedGD(const VeryForwardMisalignedGeometryRecord &iRecord)
{
  // get the input (= measured) GeometricalDet
  edm::ESHandle<DetGeomDesc> measuredGD;
  iRecord.getRecord<VeryForwardMeasuredGeometryRecord>().get(measuredGD);

  // load alignments
  edm::ESHandle<RPAlignmentCorrections> alignments;
  try { iRecord.getRecord<RPMisalignedAlignmentRecord>().get(alignments); }
  catch (...) {}
  if (alignments.isValid()) {
      printf(">> TotemRPGeometryESModule::produceMisalignedGD > Misaligned geometry: %lu RP and %lu sensor alignments applied.\n",
        alignments->GetRPMap().size(), alignments->GetSensorMap().size());
  } else {
    if (verbosity)
      printf(">> TotemRPGeometryESModule::produceMisalignedGD > Misaligned geometry: No alignments applied.\n");
  }

  DetGeomDesc* newGD = NULL;
  ApplyAlignments(measuredGD, alignments, newGD);
  return std::unique_ptr<DetGeomDesc>(newGD);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<TotemRPGeometry> TotemRPGeometryESModule::produceMeasuredTG(const VeryForwardMeasuredGeometryRecord &iRecord)
{
  edm::ESHandle<DetGeomDesc> gD;
  iRecord.get(gD);
  
  return std::make_unique<TotemRPGeometry>( gD.product() );
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
