/****************************************************************************
*
* Authors:
*  Jan Kaspar (jan.kaspar@gmail.com) 
*  Dominik Mierzejewski <dmierzej@cern.ch>
*    
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
 
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/src/LogicalPart.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

#include <TMatrixD.h>

// Copies ideal_ddcv to measured_ddcv, applying the alignments if any
// WARNING: (TODO?) does not handle the "old" geometry
class MeasuredGeometryProducer
{
  private:
    const DDCompactView &idealCV;
    const RPAlignmentCorrectionsData *const alignments;
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
    static void translRotToTransform(const DDTranslation &translation, const DDRotationMatrix &rotation, TMatrixD &transform) {
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
    static void translRotFromTransform(DDTranslation &translation, DDRotationMatrix &rotation, const TMatrixD &transform) {
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
      delExpandedView(expandedView);
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
    static inline TotemRPDetId getRPIdFromNamespace(const DDLogicalPart &part) {
      const int nsLength = part.name().ns().length();
      const unsigned int rpDecId = atoi(part.name().ns().substr(nsLength - 3, nsLength).c_str());

      const unsigned int armIdx = rpDecId / 100;
      const unsigned int stIdx = (rpDecId / 10) % 10;
      const unsigned int rpIdx = rpDecId % 10;

      return TotemRPDetId(armIdx, stIdx, rpIdx);
    }

    // -------------------------------------

    // Applies alignment (translation and rotation) to transformation matrix
    //
    // translation = alignmentTranslation + translation
    // rotation    = alignmentRotation + rotation
    static void applyCorrectionToTransform(const RPAlignmentCorrectionData &correction, TMatrixD &transform) {
      DDTranslation translation;
      DDRotationMatrix rotation;

      translRotFromTransform(translation, rotation, transform);

      translation = correction.getTranslation() + translation;
      rotation    = correction.getRotationMatrix() * rotation;

      translRotToTransform(translation, rotation, transform);
    }

    // Applies relative alignments to Detector rotation and translation
    void applyCorrection(const DDLogicalPart &parent, const DDLogicalPart &child, const RPAlignmentCorrectionData &correction, DDTranslation &translation, DDRotationMatrix &rotation, const bool useMeasuredParent = true) {
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
            if (alignments != NULL) {
              TotemRPDetId rpId = getRPIdFromNamespace(to);
              const RPAlignmentCorrectionData correction = alignments->GetRPCorrection(rpId);
              applyCorrection(from, to, correction, translation, rotationMatrix, false);
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

          if (alignments != NULL) {
            TotemRPDetId detId = getRPIdFromNamespace(from);
            detId.setPlane(copyNo);

            const RPAlignmentCorrectionData correction = alignments->GetFullSensorCorrection(detId);
            applyCorrection(from, to, correction, translation, rotationMatrix);
          }

          const DDRotation rotation = DDanonymousRot(&rotationMatrix);
          measuredCV->position(to, from, copyNo, translation, rotation, &division);
        }
      }
    }

  public:
    MeasuredGeometryProducer(const edm::ESHandle<DDCompactView> &idealCV,
    const edm::ESHandle<RPAlignmentCorrectionsData> &alignments) :
      idealCV(*idealCV), alignments(alignments.isValid() ? &(*alignments) : NULL) {
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
