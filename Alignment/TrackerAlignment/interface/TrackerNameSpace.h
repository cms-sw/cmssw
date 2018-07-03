#ifndef Alignment_TrackerAlignment_TrackerNameSpace_H
#define Alignment_TrackerAlignment_TrackerNameSpace_H

#include "CondFormats/Alignment/interface/Definitions.h"


class TrackerTopology;
class TrackerAlignmentLevelBuilder;

namespace align {
  class TrackerNameSpace {

    /// grant access for the TrackerAlignmentLevelBuilder (in global namespace)
    friend class ::TrackerAlignmentLevelBuilder;

  public:
    TrackerNameSpace(const TrackerTopology*);

    TrackerNameSpace(const TrackerNameSpace&) = default;
    TrackerNameSpace& operator=(const TrackerNameSpace&) = default;
    TrackerNameSpace(TrackerNameSpace&&) = default;
    TrackerNameSpace& operator=(TrackerNameSpace&&) = default;

    virtual ~TrackerNameSpace() = default;

    class TPB {
      /// grant access for the enclosing TrackerNameSpace
      friend class TrackerNameSpace;

      /// grant access for the TrackerAlignmentLevelBuilder (in global namespace)
      friend class ::TrackerAlignmentLevelBuilder;

    public:
      TPB(const TrackerTopology*);
      TPB(const TPB&) = default;
      TPB& operator=(const TPB&) = default;
      TPB(TPB&&) = default;
      TPB& operator=(TPB&&) = default;
      virtual ~TPB() = default;

      /// Module number increases with z from 1 to 8.
      unsigned int moduleNumber(align::ID) const;

      /// Ladder number increases from 1 at the top to 2 * lpqc at the bottom
      /// of each half cylinder.
      unsigned int ladderNumber(align::ID) const;

      /// Layer number increases with rho from 1 to 3.
      unsigned int layerNumber(align::ID) const;

      /// Half barrel number is 1 at left side (-x) and 2 at right side (+x).
      unsigned int halfBarrelNumber(align::ID) const;

      /// Barrel number is 1 for all align::ID's which belong to this barrel
      unsigned int barrelNumber(align::ID) const;

    private:
      const TrackerTopology* trackerTopology_;

      /// Number of ladders for each quarter cylinder.
      std::vector<unsigned int> lpqc_;
    };

    class TPE {
      /// grant access for the enclosing TrackerNameSpace
      friend class TrackerNameSpace;

      /// grant access for the TrackerAlignmentLevelBuilder (in global namespace)
      friend class ::TrackerAlignmentLevelBuilder;

    public:
      TPE(const TrackerTopology*);
      TPE(const TPE&) = default;
      TPE& operator=(const TPE&) = default;
      TPE(TPE&&) = default;
      TPE& operator=(TPE&&) = default;
      virtual ~TPE() = default;

      /// Module number increases with rho; from 1 to 4.
      unsigned int moduleNumber(align::ID) const;

      /// Panel number is 1 for 4 modules, 2 for 3 modules.
      unsigned int panelNumber(align::ID) const;

      /// Blade number increases from 1 at the top to 12 at the bottom
      /// of each half disk.
      unsigned int bladeNumber(align::ID) const;

      /// Half disk number increases with |z| from 1 to 3.
      unsigned int halfDiskNumber(align::ID) const;

      /// Half cylinder number is 1 at left side (-x) and 2 at right side (+x).
      unsigned int halfCylinderNumber(align::ID) const;

      /// Endcap number is 1 for -z and 2 for +z.
      unsigned int endcapNumber(align::ID) const;

    private:
      const TrackerTopology* trackerTopology_;

      /// no. of blades per quarter disk
      unsigned int bpqd_;
    };

    class TIB {
      /// grant access for the enclosing TrackerNameSpace
      friend class TrackerNameSpace;

      /// grant access for the TrackerAlignmentLevelBuilder (in global namespace)
      friend class ::TrackerAlignmentLevelBuilder;

    public:
      TIB(const TrackerTopology*);
      TIB(const TIB&) = default;
      TIB& operator=(const TIB&) = default;
      TIB(TIB&&) = default;
      TIB& operator=(TIB&&) = default;
      virtual ~TIB() = default;

      /// Module number increases with |z| from 1 to 3.
      unsigned int moduleNumber(align::ID) const;

      /// String number increases with |phi| from right (1) to left (sphs)
      /// of each half shell.
      unsigned int stringNumber(align::ID) const;

      /// Surface number is 1 for inner and 2 for outer.
      unsigned int surfaceNumber(align::ID) const;

      /// Half shell number is 1 for bottom (-y) and 2 for top (+y).
      unsigned int halfShellNumber(align::ID) const;

      /// Layer number increases with rho from 1 to 8.
      unsigned int layerNumber(align::ID) const;

      /// Half barrel number is 1 at -z side and 2 at +z side.
      unsigned int halfBarrelNumber(align::ID) const;

      /// Barrel number is 1 for all align::ID's which belong to this barrel
      unsigned int barrelNumber(align::ID) const;

    private:
      const TrackerTopology* trackerTopology_;

      /// Number of strings for each surface of a half shell.
      std::vector<unsigned int> sphs_;

    };

    class TOB {
      /// grant access for the enclosing TrackerNameSpace
      friend class TrackerNameSpace;

      /// grant access for the TrackerAlignmentLevelBuilder (in global namespace)
      friend class ::TrackerAlignmentLevelBuilder;

    public:
      TOB(const TrackerTopology*);
      TOB(const TOB&) = default;
      TOB& operator=(const TOB&) = default;
      TOB(TOB&&) = default;
      TOB& operator=(TOB&&) = default;
      virtual ~TOB() = default;

      /// Module number increases with |z| from 1 to 6.
      unsigned int moduleNumber(align::ID) const;

      /// Rod number increases with phi.
      unsigned int rodNumber(align::ID) const;

      /// Layer number increases with rho from 1 to 6.
      unsigned int layerNumber(align::ID) const;

      /// HalfBarrel number is 1 at -z side and 2 at +z side.
      unsigned int halfBarrelNumber(align::ID) const;

      /// Barrel number is 1 for all align::ID's which belong to this barrel
      unsigned int barrelNumber(align::ID) const;

    private:
      const TrackerTopology* trackerTopology_;
    };

    class TID {
      /// grant access for the enclosing TrackerNameSpace
      friend class TrackerNameSpace;

      /// grant access for the TrackerAlignmentLevelBuilder (in global namespace)
      friend class ::TrackerAlignmentLevelBuilder;

    public:
      TID(const TrackerTopology*);
      TID(const TID&) = default;
      TID& operator=(const TID&) = default;
      TID(TID&&) = default;
      TID& operator=(TID&&) = default;
      virtual ~TID() = default;

      /// Module number increases with phi.
      unsigned int moduleNumber(align::ID) const;

      /// Side number is 1 for back ring and 2 for front (towards IP).
      unsigned int sideNumber(align::ID) const;

      /// Ring number increases with rho from 1 to 3.
      unsigned int ringNumber(align::ID) const;

      /// Disk number increases with |z| from 1 to 3.
      unsigned int diskNumber(align::ID) const;

      /// Endcap number is 1 at -z side and 2 at +z side.
      unsigned int endcapNumber(align::ID) const;
    private:
      const TrackerTopology* trackerTopology_;
    };

    class TEC {
      /// grant access for the enclosing TrackerNameSpace
      friend class TrackerNameSpace;

      /// grant access for the TrackerAlignmentLevelBuilder (in global namespace)
      friend class ::TrackerAlignmentLevelBuilder;

    public:
      TEC(const TrackerTopology*);
      TEC(const TEC&) = default;
      TEC& operator=(const TEC&) = default;
      TEC(TEC&&) = default;
      TEC& operator=(TEC&&) = default;
      virtual ~TEC() = default;

      /// Module number increases (decreases) with phi for +z (-z) endcap.
      unsigned int moduleNumber(align::ID) const;

      /// Ring number increases with rho.
      unsigned int ringNumber(align::ID) const;

      /// Petal number increases with phi from 1 to 8.
      unsigned int petalNumber(align::ID) const;

      /// Side number is 1 for back disk and 2 for front (towards IP).
      unsigned int sideNumber(align::ID) const;

      /// Disk number increases with |z| from 1 to 9.
      unsigned int diskNumber(align::ID) const;

      /// Endcap number is 1 at -z side and 2 at +z side.
      unsigned int endcapNumber(align::ID) const;

    private:
      const TrackerTopology* trackerTopology_;
    };


    const TrackerTopology* trackerTopology() const { return trackerTopology_; }
    const TPB& tpb() const { return tpb_; }
    const TPE& tpe() const { return tpe_; }
    const TIB& tib() const { return tib_; }
    const TOB& tob() const { return tob_; }
    const TID& tid() const { return tid_; }
    const TEC& tec() const { return tec_; }

  private:
    const TrackerTopology* trackerTopology_;
    TPB tpb_;
    TPE tpe_;
    TIB tib_;
    TOB tob_;
    TID tid_;
    TEC tec_;
  };
};
#endif /* Alignment_TrackerAlignment_TrackerNameSpace_H */
