#ifndef RecoTauTag_TauTagTools_PFTauDiscriminants
#define RecoTauTag_TauTagTools_PFTauDiscriminants

#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantBase.h"

/*
 * Discriminants.h
 *
 * Author: Evan K. Friis, UC Davis; friis@physics.ucdavis.edu
 *
 * A non-inclusive (for now) set of discriminants to be used for TauID.
 *
 * These discriminants inherit from the base class found in PFTauDiscriminantBase.h
 *
 * The arguments given in the constructor are:
 *      DiscriminantBase<TYPE>(name, ROOT Branch Name, Is Required, Is Multiple, Default Value)
 *
 * Implementation is done by defining the abstract doComputatation(...) method (see src/Discriminants.cc)
 * The return value(s) should be inserted into 'result', a vector of type TYPE.  Note that even if the value returns
 * a single value, it (and only it) shoudl be inserted into the vector.  The vector is automatically cleared by the discriminant
 * manager.
 *
 * Note on adding discriminants: If you get weird vtable errors during linking, make sure that you have implemented the destructor! 
 * i.e. ~DecayMode(){}; versus ~DecayMode();
 *
 * TODO: make these macros...
 *
 */

namespace PFTauDiscriminants {

typedef reco::Particle::LorentzVector LorentzVector;

//forward declarations

class DecayMode : public DiscriminantBase<int> {
   public:
      DecayMode():DiscriminantBase<int>("DecayMode", "I", true, false, -1){};
      ~DecayMode(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<int>& result);
};

class OutlierNCharged : public DiscriminantBase<int> {
   public:
      OutlierNCharged():DiscriminantBase<int>("OutlierNCharged", "I", true, false, -1){};
      ~OutlierNCharged(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<int>& result);
};

class OutlierN : public DiscriminantBase<int> {
   public:
      OutlierN():DiscriminantBase<int>("OutlierN", "I", true, false, -1){};
      ~OutlierN(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<int>& result);
};


class Pt : public DiscriminantBase<double>  {
   public:
      Pt():DiscriminantBase<double>("Pt", "D", true, false, 0.0){};
      ~Pt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class Eta : public DiscriminantBase<double>  {
   public:
      Eta():DiscriminantBase<double>("Eta", "D", true, false, 0.0){};
      ~Eta(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class MainTrackPt : public DiscriminantBase<double>  {
   public:
      MainTrackPt():DiscriminantBase<double>("MainTrackPt", "D", true, false, -1){};
      ~MainTrackPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class MainTrackAngle : public DiscriminantBase<double>  {
   public:
      MainTrackAngle():DiscriminantBase<double>("MainTrackAngle", "D", true, false, -1){};
      ~MainTrackAngle(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class TrackPt : public DiscriminantBase<double> {
   public:
      TrackPt():DiscriminantBase<double>("TrackPt", "vector<double>", false, true, 0.0){};
      ~TrackPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class PiZeroPt : public DiscriminantBase<double> {
   public:
      PiZeroPt():DiscriminantBase<double>("PiZeroPt", "vector<double>", false, true, 0.0){};
      ~PiZeroPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

// any objects in the PFTauDecayMode that were moved filtered
class FilteredObjectPt : public DiscriminantBase<double> {
   public:
      FilteredObjectPt():DiscriminantBase<double>("FilteredObjectPt", "vector<double>", false, true, 0.0){};
      ~FilteredObjectPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

//  Matches to PiZeroPt, each element gives the corresponding # of photons in each PiZero
class GammaOccupancy : public DiscriminantBase<double> {
   public:
      GammaOccupancy():DiscriminantBase<double>("GammaOccupancy", "vector<double>", false, true, 0.0){}
      ~GammaOccupancy(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

// In same order as PiZeroPt.  Can be matched to PiZeros using PiZeroPt and GammaOccupancy
class GammaPt : public DiscriminantBase<double> {
   public:
      GammaPt():DiscriminantBase<double>("GammaPt", "vector<double>", false, true, 0.0){}
      ~GammaPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};


class TrackAngle : public DiscriminantBase<double> {
   public:
      TrackAngle():DiscriminantBase<double>("TrackAngle", "vector<double>", false, true, 0.0){};
      ~TrackAngle(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class PiZeroAngle : public DiscriminantBase<double> {
   public:
      PiZeroAngle():DiscriminantBase<double>("PiZeroAngle", "vector<double>", false, true, 0.0){};
      ~PiZeroAngle(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class Dalitz : public DiscriminantBase<double> {
   public:
      Dalitz():DiscriminantBase<double>("Dalitz", "vector<double>", false, true, 0.0){};
      ~Dalitz(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

// takes invariant mass of all objects in signal cone
class InvariantMassOfSignal : public DiscriminantBase<double> {
   public:
      InvariantMassOfSignal():DiscriminantBase<double>("InvariantMassOfSignal", "D", true, false, 0.0){};
      ~InvariantMassOfSignal(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

// takes invariant mass of all objects in signal cone + filtered objects
class InvariantMassOfSignalWithFiltered : public DiscriminantBase<double> {
   public:
      InvariantMassOfSignalWithFiltered():DiscriminantBase<double>("InvariantMassOfSignalWithFiltered", "D", true, false, 0.0){};
      ~InvariantMassOfSignalWithFiltered(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};


// returns vector of invariant masses of larger and larger subsets of all signal objects e.g. result[2] is
// the invariant mass of the lead track with the next highest Pt object
class InvariantMass : public DiscriminantBase<double> {
   public:
      InvariantMass():DiscriminantBase<double>("InvariantMass", "vector<double>", false, true, 0.0){};
      ~InvariantMass(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class OutlierPt : public DiscriminantBase<double> {
   public:
      OutlierPt():DiscriminantBase<double>("OutlierPt", "vector<double>", false, true, 0.0){};
      ~OutlierPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class OutlierSumPt : public DiscriminantBase<double> {
   public:
      OutlierSumPt():DiscriminantBase<double>("OutlierSumPt", "D", true, false, 0.0){};
      ~OutlierSumPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class OutlierMass : public DiscriminantBase<double> {
   public:
      OutlierMass():DiscriminantBase<double>("OutlierMass", "D", true, false, 0.0){};
      ~OutlierMass(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class OutlierAngle : public DiscriminantBase<double> {
   public:
      OutlierAngle():DiscriminantBase<double>("OutlierAngle", "vector<double>", false, true, 0.0){};
      ~OutlierAngle(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class ChargedOutlierPt : public DiscriminantBase<double> {
   public:
      ChargedOutlierPt():DiscriminantBase<double>("ChargedOutlierPt", "vector<double>", false, true, 0.0){};
      ~ChargedOutlierPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class ChargedOutlierSumPt : public DiscriminantBase<double> {
   public:
      ChargedOutlierSumPt():DiscriminantBase<double>("ChargedOutlierSumPt", "D", true, false, 0.0){};
      ~ChargedOutlierSumPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class ChargedOutlierAngle : public DiscriminantBase<double> {
   public:
      ChargedOutlierAngle():DiscriminantBase<double>("ChargedOutlierAngle", "vector<double>", false, true, 0.0){};
      ~ChargedOutlierAngle(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class NeutralOutlierPt : public DiscriminantBase<double> {
   public:
      NeutralOutlierPt():DiscriminantBase<double>("NeutralOutlierPt", "vector<double>", false, true, 0.0){};
      ~NeutralOutlierPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class NeutralOutlierSumPt : public DiscriminantBase<double> {
   public:
      NeutralOutlierSumPt():DiscriminantBase<double>("NeutralOutlierSumPt", "D", true, false, 0.0){};
      ~NeutralOutlierSumPt(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};

class NeutralOutlierAngle : public DiscriminantBase<double> {
   public:
      NeutralOutlierAngle():DiscriminantBase<double>("NeutralOutlierAngle", "vector<double>", false, true, 0.0){};
      ~NeutralOutlierAngle(){};
   protected:
      void doComputation(PFTauDiscriminantManager* input, std::vector<double>& result);
};


}
#endif





